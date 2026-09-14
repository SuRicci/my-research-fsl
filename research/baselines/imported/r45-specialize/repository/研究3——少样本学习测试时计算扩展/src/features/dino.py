# -*- coding: utf-8 -*-
"""DINOv2 特征提取（改造自研究1 src/dino_branch.py；**已删除全部 mock fallback**）。

- 自包含 ViT 实现（PatchEmbed/Block/Attention/LayerScale），加载本地 .pth（strict=True），
  任何加载失败显式 raise（研究3 红线：禁用 mock，防"结果不具实际意义"污染）。
- pos-emb 双三次插值支持变分辨率（r 轴）。
- 提供：CLS 全局特征（分类线）、patch 特征（AD 记忆库）。
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..config import DINOV2_VITS14_PATH

_ARCHS: Dict[str, Dict] = {
    "dinov2_vits14": {"embed_dim": 384, "depth": 12, "num_heads": 6, "mlp_ratio": 4},
    "dinov2_vitb14": {"embed_dim": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4},
    "dinov2_vitl14": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "mlp_ratio": 4},
}

_LOCAL_WEIGHTS: Dict[str, Path] = {
    "dinov2_vits14": DINOV2_VITS14_PATH,
}


class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class LayerScale(torch.nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = torch.nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class MLP(torch.nn.Module):
    """与 DINOv2 官方键对齐的 MLP（fc1/fc2；研究1 用 Sequential 导致 strict 加载失败）。"""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(torch.nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 norm_layer=torch.nn.LayerNorm, init_values=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else torch.nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else torch.nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVisionTransformer(torch.nn.Module):
    """自包含 ViT，与 DINOv2 官方权重键严格对齐（strict=True 加载）。"""

    def __init__(self, img_size=518, patch_size=14, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4.0, init_values=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = torch.nn.Dropout(0.0)
        self.blocks = torch.nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, init_values=init_values) for _ in range(depth)
        ])
        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = torch.nn.Identity()

    def interpolate_pos_encoding(self, x, w, h):
        """变分辨率 pos-emb 双三次插值（DINOv2 官方实现同款）。"""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos = self.pos_embed[:, 0]
        patch_pos = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0, h0 = w // self.patch_embed.patch_size, h // self.patch_embed.patch_size
        patch_pos = F.interpolate(
            patch_pos.reshape(1, int(N**0.5), int(N**0.5), dim).permute(0, 3, 1, 2),
            size=(h0, w0), mode="bicubic", align_corners=False,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos.unsqueeze(0), patch_pos), dim=1)

    def prepare_tokens(self, x):
        B, _, w, h = x.shape
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward_features(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return {"x_norm_clstoken": x[:, 0], "x_norm_patchtokens": x[:, 1:]}

    def forward(self, x):
        return self.forward_features(x)


def build_dinov2(model_name: str = "dinov2_vits14", weight_path: Optional[Path] = None) -> DinoVisionTransformer:
    """构建并加载本地 DINOv2 权重；失败显式 raise（禁止 mock 降级）。"""
    if model_name not in _ARCHS:
        raise ValueError(f"未知 DINOv2 架构: {model_name}（可选: {list(_ARCHS)}）")
    path = Path(weight_path) if weight_path else _LOCAL_WEIGHTS.get(model_name)
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"DINOv2 {model_name} 本地权重缺失: {path}；请先下载（研究3 禁止 mock 权重）"
        )
    cfg = _ARCHS[model_name]
    model = DinoVisionTransformer(
        embed_dim=cfg["embed_dim"], depth=cfg["depth"],
        num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
    )
    state_dict = torch.load(str(path), map_location="cpu")
    # 官方 backbone 权重含 mask_token（仅预训练掩码用，推理不需要），剔除后严格对齐
    state_dict.pop("mask_token", None)
    model.load_state_dict(state_dict, strict=True)
    return model


class DINOExtractor:
    """冻结 DINOv2 特征提取器（CLS + patch）。"""

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cuda",
                 weight_path: Optional[str] = None):
        self.device = device
        self.model_name = model_name
        self.model = build_dinov2(model_name, weight_path)
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.patch_size = self.model.patch_embed.patch_size
        self.embed_dim = self.model.embed_dim
        self.sanity_check()

    def sanity_check(self) -> None:
        """权重非随机的廉价检验：cls_token 初始化为全零，训练后必非零。"""
        cls_abs = float(self.model.cls_token.abs().sum().item())
        pos_std = float(self.model.pos_embed.std().item())
        if cls_abs < 1e-6 or pos_std < 1e-4:
            raise RuntimeError(
                f"DINOv2 {self.model_name} 疑似随机权重: |cls_token|={cls_abs:.4f}, "
                f"pos_embed std={pos_std:.5f}（随机 init 恰为 cls_token=0）"
            )
        print(f"    [DINO] {self.model_name} sanity OK: |cls_token|={cls_abs:.2f}, pos_embed std={pos_std:.4f}")

    @torch.no_grad()
    def extract(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """图像 → (CLS [B,D], patches [B,N,D], (gh,gw))，均 L2 归一化。"""
        x = image.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        out = self.model.forward_features(x)
        gh, gw = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        cls = F.normalize(out["x_norm_clstoken"].float(), dim=-1)
        patches = F.normalize(out["x_norm_patchtokens"].float(), dim=-1)
        return cls, patches, (gh, gw)

    @torch.no_grad()
    def extract_cls(self, image: torch.Tensor) -> torch.Tensor:
        return self.extract(image)[0]

    @torch.no_grad()
    def extract_patches(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        _, p, grid = self.extract(image)
        return p, grid


if __name__ == "__main__":
    # 单元验证（需真实权重；mock 已删除，加载失败会直接 raise）
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = DINOExtractor("dinov2_vits14", device=device)
    x = torch.rand(1, 3, 224, 224, device=device)
    cls, patches, grid = ext.extract(x)
    assert cls.shape == (1, 384) and patches.shape == (1, 256, 384) and grid == (16, 16)
    # 变分辨率（r 轴）：pos-emb 插值
    x448 = torch.rand(1, 3, 448, 448, device=device)
    cls448, patches448, grid448 = ext.extract(x448)
    assert grid448 == (32, 32) and patches448.shape[1] == 1024
    # 确定性：同输入同输出
    cls_b, _, _ = ext.extract(x)
    assert torch.allclose(cls, cls_b)
    print(f"[dino] 单元验证通过: 224→{patches.shape}, 448→{patches448.shape}")
