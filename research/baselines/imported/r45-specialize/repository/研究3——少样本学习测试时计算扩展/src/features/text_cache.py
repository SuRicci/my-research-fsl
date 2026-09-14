# -*- coding: utf-8 -*-
"""zero-shot 文本先验通路：类名 × prompt 模板 → 文本特征，一次算完落盘 cache/text/。

- 仅 CLIP backbone 有文本分支（DINOv2 无文本 → 显式报错，由调用方跳过 requires_text 方法）。
- 模板口径与 CLIP/Tip-Adapter 文献一致：各数据集单个领域模板（dataset），
  或 ImageNet 80 模板集成（imagenet80，与 open_clip/CLIP 官方清单一致）。
- 缓存键含数据集名 + 模板集 + 类名 md5，跨进程/跨 seed 复用（文本特征与 episode 无关）。
- logit_scale 固定 100（OpenAI CLIP 的 logit_scale.exp()≈100，见 features/clip.py sanity）。
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .cache_io import FeatureCache

# backbone 名 → open_clip 模型名（与 run/common.py BACKBONES 的 clip 分支保持一致；
# 文本通路仅支持 clip 类 backbone）
_CLIP_MODEL = {
    "clip_vitb16": "ViT-B-16",
    "clip_rn50": "RN50",
}

# ImageNet 80 模板集成（CLIP, Radford et al. 2021；与 open_clip/TPT 官方清单逐字一致）
IMAGENET80_TEMPLATES: List[str] = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

# 各数据集领域模板（CLIP 论文/Tip-Adapter/CoOp 惯例的单个模板）
DATASET_TEMPLATES: Dict[str, List[str]] = {
    "eurosat": ["a centered satellite photo of {}."],
    "dtd": ["{} texture."],
    "cub200": ["a photo of a {}, a type of bird."],
    "flowers102": ["a photo of a {}, a type of flower."],
    "cifar_fs": ["a photo of a {}."],
    "cifar100": ["a photo of a {}."],
}
DEFAULT_TEMPLATE = ["a photo of a {}."]

DEFAULT_LOGIT_SCALE = 100.0  # OpenAI CLIP logit_scale.exp()≈100


def templates_for(dataset: str, template_set: str = "dataset") -> List[str]:
    """按数据集与模板集名返回 prompt 模板列表。"""
    if template_set == "imagenet80":
        return IMAGENET80_TEMPLATES
    if template_set == "dataset":
        return DATASET_TEMPLATES.get(dataset, DEFAULT_TEMPLATE)
    raise KeyError(f"未知模板集: {template_set}（可选: dataset, imagenet80）")


def has_text(backbone: str, classnames: Sequence[str]) -> bool:
    """该 (backbone, 数据集) 是否可提供有效文本先验。

    miniImageNet 类名为 wnid（如 n01443537），CLIP 无法理解 → 视为无文本先验。
    """
    if backbone not in _CLIP_MODEL:
        return False
    if not classnames:
        return False
    # wnid 形态：n + 8 位数字（miniImageNet/ImageNet synset id）
    if all(c.startswith("n") and len(c) == 9 and c[1:].isdigit() for c in classnames):
        return False
    return True


def text_features_for_dataset(dataset: str, classnames: Sequence[str],
                              backbone: str = "clip_vitb16", device: str = "cuda",
                              template_set: str = "dataset",
                              logit_scale: float = DEFAULT_LOGIT_SCALE
                              ) -> Tuple[torch.Tensor, float]:
    """数据集全类文本特征（[C, D]，L2 归一化，float32 CPU tensor）与 logit 尺度。

    首次计算后落盘 cache/text/{backbone}/（FeatureCache 命名空间），后续直接读缓存。
    """
    if backbone not in _CLIP_MODEL:
        raise KeyError(f"文本先验仅支持 CLIP backbone: {backbone}（可选: {list(_CLIP_MODEL)}）")
    templates = templates_for(dataset, template_set)
    cls_hash = hashlib.md5("|".join(classnames).encode("utf-8")).hexdigest()[:8]
    key = f"{dataset}|{template_set}|{cls_hash}"
    fc = FeatureCache("text", backbone)
    if fc.has(key):
        return fc.get(key), logit_scale
    from .clip import CLIPExtractor

    extractor = CLIPExtractor(_CLIP_MODEL[backbone], device=device)
    feats = extractor.class_text_features(classnames, templates)  # [C, D] 已归一化
    fc.put(key, feats)
    fc.flush()
    print(f"    [text] {dataset}/{template_set}: {len(classnames)} 类 × {len(templates)} 模板 → 已缓存 {key}")
    return feats.float().cpu(), logit_scale


def episode_zs_logits_fn(text_feats: torch.Tensor, logit_scale: float = DEFAULT_LOGIT_SCALE):
    """由 episode 各类文本特征 [n_way, D] 构造 zs_logits_fn: qv [Q, D] → [Q, n_way]。"""
    def fn(qv: torch.Tensor) -> torch.Tensor:
        return logit_scale * (qv @ text_feats.T)
    return fn


if __name__ == "__main__":
    # 单元验证：模板清单完整性 + has_text 判定（不加载模型）
    assert len(IMAGENET80_TEMPLATES) == 80, "ImageNet 模板应为 80 条"
    assert templates_for("eurosat") == ["a centered satellite photo of {}."]
    assert templates_for("miniimagenet") == DEFAULT_TEMPLATE
    assert len(templates_for("eurosat", "imagenet80")) == 80
    assert has_text("clip_vitb16", ["annual crop", "forest"])
    assert not has_text("clip_vitb16", ["n01443537", "n02138441"]), "wnid 应判为无文本先验"
    assert not has_text("dinov2_vits14", ["cat"]), "DINOv2 无文本分支"
    print("[text_cache] 模板/判定单元验证通过")
