# -*- coding: utf-8 -*-
"""只读桥接研究3：数据集加载、编码器；图库特征缓存放实验12自己的 cache/。

边界声明：对研究3目录只读；图库（train split）特征在研究3缓存中不存在，
编码一次后写入本目录 cache/gallery/，遵守"特征只算一次"。
"""
import sys
from pathlib import Path

import torch

_R3_ROOT = Path(__file__).resolve().parents[2] / "研究3——少样本学习测试时计算扩展"
if not _R3_ROOT.is_dir():
    raise RuntimeError(f"研究3目录不存在: {_R3_ROOT}")
if str(_R3_ROOT) not in sys.path:
    sys.path.insert(0, str(_R3_ROOT))

from src.config import DINOV2_VITS14_PATH  # noqa: E402
from src.data.cls_datasets import load_cls_dataset  # noqa: E402
from src.features.cache_io import FeatureCache  # noqa: E402

E12_ROOT = Path(__file__).resolve().parents[1]
E12_CACHE = E12_ROOT / "cache"
RES = 224

_NORM = {
    "clip_vitb16": ((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
    "dinov2_vits14": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def center_transform(backbone: str):
    from torchvision import transforms
    mean, std = _NORM[backbone]
    return transforms.Compose([
        transforms.Resize(RES, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RES),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


_extractor = {}


def get_extractor(backbone: str, device: str = "cuda"):
    if backbone in _extractor:
        return _extractor[backbone]
    if backbone == "clip_vitb16":
        from src.features.clip import CLIPExtractor
        ext = CLIPExtractor("ViT-B-16", device=device)
    elif backbone == "dinov2_vits14":
        from src.features.dino import DINOExtractor
        ext = DINOExtractor("dinov2_vits14", device=device,
                            weight_path=str(DINOV2_VITS14_PATH))
    else:
        raise KeyError(f"未知 backbone: {backbone}")
    _extractor[backbone] = ext
    return ext


def encode(backbone: str, images: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    ext = get_extractor(backbone, device)
    if backbone == "clip_vitb16":
        return ext.encode_image(images).float().cpu()
    return ext.extract_cls(images).float().cpu()


def gallery_cache(scope: str, backbone: str) -> FeatureCache:
    return FeatureCache(f"gallery/{scope}", backbone, root=E12_CACHE)


def gallery_key(backbone: str, scope: str, idx: int) -> str:
    return f"{backbone}|{scope}#{idx}|v0|r{RES}"


# ---------------- 查询侧：只读研究3的 cls 缓存 ----------------

import json  # noqa: E402
from typing import Dict, List  # noqa: E402

import numpy as np  # noqa: E402

from src.config import CACHE_ROOT as R3_CACHE_ROOT  # noqa: E402


def load_labels(scope: str) -> np.ndarray:
    """查询侧数据集标签（与研究3加载口径一致；scope 形如 miniimagenet_test）。"""
    name = scope
    split = "test"
    for suf in ("_test", "_val", "_train"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            split = suf[1:]
            break
    ds = load_cls_dataset(name, split)
    return np.asarray(ds.labels, dtype=np.int64)


def load_classnames(scope: str):
    """(labels, classnames)；classnames[i] 是该视图下标签 i 的名字。
    跨 split/跨数据集比较污染时必须经类名映射（标签 id 空间可能不同：
    例如 cifar_fs_test 重映射到 0..19，cifar100 train 仍是 0..99）。"""
    name = scope
    split = "test"
    for suf in ("_test", "_val", "_train"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            split = suf[1:]
            break
    ds = load_cls_dataset(name, split)
    return np.asarray(ds.labels, dtype=np.int64), list(ds.classnames)


def query_cached_index_set(scope: str, backbone: str) -> set:
    idx_path = R3_CACHE_ROOT / "cls" / scope / backbone / "index.json"
    if not idx_path.exists():
        raise FileNotFoundError(f"研究3缓存索引缺失: {idx_path}")
    with open(idx_path, "r", encoding="utf-8") as f:
        keys = json.load(f).keys()
    prefix = f"{backbone}|{scope}#"
    suffix = f"|v0|r{RES}"
    return {int(k[len(prefix):-len(suffix)]) for k in keys
            if k.startswith(prefix) and k.endswith(suffix)}


def query_cached_pools(labels: np.ndarray, scope: str,
                       backbone: str) -> Dict[int, np.ndarray]:
    have = query_cached_index_set(scope, backbone)
    return {int(c): np.array(sorted(i for i in np.nonzero(labels == c)[0]
                                    if int(i) in have), dtype=np.int64)
            for c in np.unique(labels)}


def load_query_features(scope: str, backbone: str,
                        indices: List[int]) -> torch.Tensor:
    fc = FeatureCache(f"cls/{scope}", backbone, root=R3_CACHE_ROOT,
                      read_only=True)
    keys = [f"{backbone}|{scope}#{i}|v0|r{RES}" for i in indices]
    return fc.get_many(keys)
