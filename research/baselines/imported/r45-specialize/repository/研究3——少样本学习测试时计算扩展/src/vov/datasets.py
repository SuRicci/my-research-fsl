# -*- coding: utf-8 -*-
"""canonical dataset id → 数据集 split / 缓存 scope 的唯一出处（二十轮 R3-P0-5）。

缓存目录实际布局（cache/cls/，已核实）：
    miniimagenet_test / dtd_test / cifar_fs_test / cub200_test / flowers102 / eurosat

协议与 jobs 一律使用 canonical id（miniimagenet / dtd / cifar_fs / cub200 /
flowers102 / eurosat）；cache scope = 对应 ClsDataset.name（特征缓存键 img_id
前缀与之一致）。**禁止调用者手工拼接 `_test`**——所有 runner/jobs/结果都经
本模块解析，结果中同时落 canonical id 与 cache scope 两个字段。
"""
from typing import Dict, List

# canonical id → {"split": 加载 split, "cache_scope": ClsDataset.name / cache/cls 子目录}
CANONICAL_DATASETS: Dict[str, Dict[str, str]] = {
    "miniimagenet": {"split": "test", "cache_scope": "miniimagenet_test"},
    "dtd": {"split": "test", "cache_scope": "dtd_test"},
    "cifar_fs": {"split": "test", "cache_scope": "cifar_fs_test"},
    "cub200": {"split": "test", "cache_scope": "cub200_test"},
    "flowers102": {"split": "test", "cache_scope": "flowers102"},
    "eurosat": {"split": "test", "cache_scope": "eurosat"},
}

# cross-fitting 的四个确认 fold（v3.5：160 run = 4×2×2×10）
CONFIRM_FOLDS: List[str] = ["miniimagenet", "dtd", "cifar_fs", "cub200"]
# 开发/止损集（80 run = 2×2×2×10）
DEV_FOLDS: List[str] = ["miniimagenet", "dtd"]


def canonical_id(name: str) -> str:
    """校验并返回 canonical id；未知名立即 KeyError（fail closed）。

    显式拒绝手工加 `_test` 的调用（如 `dtd_test`）——那是 cache scope，不是
    canonical id，防止映射被旁路。
    """
    if name in CANONICAL_DATASETS:
        return name
    if name.endswith("_test") and name[:-5] in CANONICAL_DATASETS:
        raise KeyError(
            f"{name!r} 是 cache scope 而非 canonical id——请传 {name[:-5]!r} "
            f"（禁止调用者手工加 _test，R3-P0-5）")
    raise KeyError(f"未知 canonical dataset: {name}（可选: {list(CANONICAL_DATASETS)}）")


def cache_scope(name: str) -> str:
    """canonical id → cache scope（= ClsDataset.name = cache/cls/<scope>）。"""
    return CANONICAL_DATASETS[canonical_id(name)]["cache_scope"]


def split_of(name: str) -> str:
    return CANONICAL_DATASETS[canonical_id(name)]["split"]


def load_dataset(name: str):
    """canonical id → ClsDataset（split 由本映射锁定，调用者不得自选）。"""
    from ..data.cls_datasets import load_cls_dataset

    cid = canonical_id(name)
    ds = load_cls_dataset(cid, split_of(cid))
    if ds.name != cache_scope(cid):
        raise RuntimeError(
            f"ClsDataset.name {ds.name!r} != 锁定 cache scope {cache_scope(cid)!r}"
            f"——缓存键前缀会错位（fail closed）")
    return ds
