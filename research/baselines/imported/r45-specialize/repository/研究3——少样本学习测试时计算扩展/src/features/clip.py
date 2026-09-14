# -*- coding: utf-8 -*-
"""CLIP 特征提取（改造自研究1 src/clip_branch.py，清理 patch_size 硬编码）。

- 加载统一走 open_clip；优先本地 ~/.cache/torch/hub/checkpoints/{RN50.pt,ViT-B-16.pt}，
  缺失时从 openaipublic 直连下载（open_clip 原生逻辑），网络不可用时显式报错（无 mock）。
- 提供：CLS 全局特征（分类线）、文本特征（zero-shot）、patch 特征（AD 语义热图，仅 ViT）。
- sanity_check()：校验权重非随机（OpenAI 预训练 logit_scale.exp()≈100，随机 init≈14.3）。
"""
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..config import CLIP_CKPT_DIR

# 模型名 → 本地权重文件名（OpenAI 官方格式）
_LOCAL_CKPT = {
    "ViT-B-16": "ViT-B-16.pt",
    "RN50": "RN50.pt",
    "ViT-B-32": "ViT-B-32.pt",
    "RN101": "RN101.pt",
}

# sanity check 阈值：OpenAI 预训练 logit_scale.exp()≈100，随机 init≈14.29
_SANE_LOGIT_SCALE_EXP = 50.0


class CLIPExtractor:
    """冻结 CLIP 特征提取器（全局 / 文本 / patch 三种特征）。"""

    def __init__(self, model_name: str = "ViT-B-16", device: str = "cuda"):
        import open_clip

        self.device = device
        self.model_name = model_name
        pretrained = self._resolve_pretrained(model_name)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.embed_dim = self.model.text_projection.shape[-1]
        # patch_size 从权重推断（杜绝研究1 中"配置一个值、类内写死 14"的问题）
        visual = self.model.visual
        if hasattr(visual, "patch_size") and visual.patch_size is not None:
            self.patch_size = int(visual.patch_size[0] if isinstance(visual.patch_size, tuple) else visual.patch_size)
        elif hasattr(visual, "conv1"):  # open_clip ViT: conv1 stride = patch size
            self.patch_size = int(visual.conv1.stride[0])
        else:
            self.patch_size = -1  # RN50 无 patch 概念
        self.is_vit = hasattr(visual, "transformer")
        self._text_cache: dict = {}
        self.sanity_check()

    @staticmethod
    def _resolve_pretrained(model_name: str) -> str:
        """本地 .pt 优先；否则交给 open_clip 下载（openaipublic 直连）。"""
        fname = _LOCAL_CKPT.get(model_name)
        if fname:
            local = CLIP_CKPT_DIR / fname
            if local.exists():
                return str(local)
        return "openai"  # open_clip 内置下载（失败会抛错，不静默降级）

    def sanity_check(self) -> None:
        """确认权重非随机初始化。"""
        logit_scale = float(self.model.logit_scale.exp().item())
        if self.is_vit and logit_scale < _SANE_LOGIT_SCALE_EXP:
            raise RuntimeError(
                f"CLIP {self.model_name} 疑似随机权重：logit_scale.exp()={logit_scale:.2f} "
                f"(OpenAI 预训练应≈100)"
            )
        # 两个不同文本的嵌入不应几乎相同（随机权重塌缩检测）
        t = self.encode_text(["a photo of a cat", "a satellite image of a forest"])
        sim = float(F.cosine_similarity(t[0:1], t[1:2]).item())
        if sim > 0.999:
            raise RuntimeError(f"CLIP {self.model_name} 文本嵌入塌缩（cos={sim:.4f}），疑似坏权重")
        print(f"    [CLIP] {self.model_name} sanity OK: logit_scale.exp()={logit_scale:.1f}, 文本区分度 cos={sim:.3f}")

    # ---------------- 文本 ----------------
    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        """文本 → L2 归一化嵌入 [n, D]。"""
        with torch.no_grad():
            tokens = self.tokenizer(list(prompts)).to(self.device)
            feats = self.model.encode_text(tokens)
            return F.normalize(feats.float(), dim=-1)

    def class_text_features(self, classnames: Sequence[str], templates: Optional[Sequence[str]] = None) -> torch.Tensor:
        """类名 × prompt 模板集成 → [C, D]（带缓存）。"""
        templates = templates or ["a photo of a {}."]
        key = "|".join(classnames) + "##" + "|".join(templates)
        if key not in self._text_cache:
            per_cls = []
            for cname in classnames:
                feats = self.encode_text([t.format(cname) for t in templates])
                per_cls.append(feats.mean(dim=0))
            self._text_cache[key] = F.normalize(torch.stack(per_cls), dim=-1)
        return self._text_cache[key]

    # ---------------- 图像（内部统一走 token 前向，支持任意分辨率） ----------------
    @torch.no_grad()
    def _forward_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """ViT 手动前向 → post-ln_post tokens [B, N+1, D_v]（含 pos-emb 插值）。"""
        if not self.is_vit:
            raise NotImplementedError("patch/token 级前向仅支持 ViT 类 CLIP；RN50 请用 encode_image 全局特征")
        v = self.model.visual
        x = image.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(v.conv1.weight.dtype)
        x = v.conv1(x)                                    # [B, D, gh, gw]
        gh, gw = x.shape[-2:]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)   # [B, N, D]
        cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        pos = v.positional_embedding.to(x.dtype)
        n_side = int((pos.shape[0] - 1) ** 0.5)
        if (gh, gw) != (n_side, n_side):                  # 变分辨率：双三次插值 pos-emb
            cls_pos, patch_pos = pos[:1], pos[1:]
            patch_pos = patch_pos.reshape(1, n_side, n_side, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(gh, gw), mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, x.shape[-1])
            pos = torch.cat([cls_pos, patch_pos], dim=0)
        x = x + pos
        x = v.ln_pre(x)
        # open_clip>=3 的 Transformer 接收 NLD 并内部转置，勿手动 permute
        # （研究1 时代的 LND 手动转置在此版本会错位成跨 batch 注意力）
        x = v.transformer(x)
        x = v.ln_post(x)
        return x.float()

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """图像 → L2 归一化 CLS 特征 [B, D]（任意分辨率，pos-emb 自动插值）。"""
        if self.is_vit:
            tokens = self._forward_tokens(image)
            feats = tokens[:, 0] @ self.model.visual.proj
        else:
            feats = self.model.encode_image(image.to(self.device))
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_image_patches(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """图像 → patch 特征 [B, N, D]（已投影到图文联合空间）与网格 (gh, gw)。"""
        tokens = self._forward_tokens(image)
        x = image if image.dim() == 4 else image.unsqueeze(0)
        gh, gw = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        patches = tokens[:, 1:, :] @ self.model.visual.proj
        return F.normalize(patches.float(), dim=-1), (gh, gw)

    def zero_shot_logits(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """logit_scale · (q · text.T)。"""
        return float(self.model.logit_scale.exp().item()) * (image_feats @ text_feats.T)

    def anomaly_map(
        self,
        image: torch.Tensor,
        text_normal: torch.Tensor,
        text_abnormal: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """AD 语义热图：patch 级 (sim_abnormal − sim_normal)，返回 ([gh,gw], 图像分)。"""
        patches, (gh, gw) = self.encode_image_patches(image)
        sim_n = patches @ text_normal.mean(dim=0, keepdim=True).T   # [B, N, 1]
        sim_a = patches @ text_abnormal.mean(dim=0, keepdim=True).T
        scores = (sim_a - sim_n).squeeze(-1)[0]                     # [N]
        amap = scores.reshape(gh, gw)
        return amap, float(amap.max().item())


if __name__ == "__main__":
    # 单元验证（需要真实权重；自动下载缺失权重）
    import numpy as np
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = CLIPExtractor("ViT-B-16", device=device)
    img = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype("uint8"))
    x = ext.preprocess(img).unsqueeze(0).to(device)
    f = ext.encode_image(x)
    assert f.shape == (1, ext.embed_dim) and abs(f.norm().item() - 1) < 1e-3
    # 回归校验：不同图像的特征必须可区分（防 transformer 输入布局错位塌缩）
    img2 = Image.fromarray((np.random.rand(280, 320, 3) * 255).astype("uint8"))
    x2 = ext.preprocess(img2).unsqueeze(0).to(device)
    f2 = ext.encode_image(x2)
    cos = float(F.cosine_similarity(f, f2).item())
    assert cos < 0.999, f"不同图像特征塌缩（cos={cos}），疑似前向布局错误"
    # 文本先验 zero-shot logits 形状
    t = ext.class_text_features(["cat", "dog"])
    logits = ext.zero_shot_logits(f, t)
    assert logits.shape == (1, 2)
    # 变分辨率 patch 提取（r 轴）
    x448 = F.interpolate(x, size=(448, 448), mode="bilinear", align_corners=False)
    patches, (gh, gw) = ext.encode_image_patches(x448)
    assert (gh, gw) == (28, 28) and patches.shape[1] == 784
    print(f"[clip] 单元验证通过: patch_size={ext.patch_size}, embed_dim={ext.embed_dim}, patch448={patches.shape}")
