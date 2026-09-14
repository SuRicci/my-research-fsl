# -*- coding: utf-8 -*-
"""全局配置：yaml 加载 + 路径常量 + 种子（模式借鉴研究1 src/config.py，全部重写）。"""
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# 项目根 = 本文件上两级（研究3/）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_ROOT = PROJECT_ROOT / "data"
CACHE_ROOT = PROJECT_ROOT / "cache"
RESULTS_ROOT = PROJECT_ROOT / "results"

# 记账单位：1 次 CLIP-B/16@224 前向 ≈ 17.5 GFLOPs（实验计划 §1.1）
C0_GFLOPS = 17.5

DEFAULT_SEED = 42

# CLIP 本地权重目录（torch hub checkpoints；RN50.pt / ViT-B-16.pt 优先）
CLIP_CKPT_DIR = Path(os.path.expanduser("~/.cache/torch/hub/checkpoints"))
# DINOv2 本地权重（只读引用研究1 的已下载文件，不拷贝不修改）
DINOV2_VITS14_PATH = (
    PROJECT_ROOT.parent / "研究1——CLIP结合DinoV2" / "pretrained" / "dinov2_vits14_pretrain.pth"
)

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)   # CLIP 官方归一化
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
DINO_MEAN = (0.485, 0.456, 0.406)                      # DINOv2 归一化
DINO_STD = (0.229, 0.224, 0.225)


def load_yaml(name: str) -> Dict[str, Any]:
    """加载 configs/ 下的 yaml（名字可带或不带 .yaml 后缀）。"""
    path = CONFIG_DIR / (name if name.endswith(".yaml") else f"{name}.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """全局随机种子（python/numpy/torch，若可用）。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(obj: Dict[str, Any], out_dir: Path, name: str) -> Path:
    """结果 JSON 落盘（自动加时间戳，UTF-8）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{timestamp()}.json"

    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)
    return path


@dataclass
class DatasetCfg:
    """单个分类数据集的解析结果。"""
    name: str
    path: Path
    num_classes: int
    extra: Dict[str, Any] = field(default_factory=dict)


def resolve_dataset_cfg(name: str, kind: str = "classification") -> DatasetCfg:
    """从 datasets.yaml 解析某数据集的绝对路径与元信息。"""
    cfg = load_yaml("datasets")
    root = Path(cfg.get("root", "data"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    table = cfg[kind]
    if name not in table:
        raise KeyError(f"未知数据集: {kind}/{name}（可选: {list(table)}）")
    entry = dict(table[name])
    raw = entry.pop("path", name)
    p = Path(raw)
    if not p.is_absolute():
        # 相对路径基于 datasets.yaml 的 root（默认 data/）；
        # 支持 "../研究1..." 这类跳出 data/ 的只读引用
        p = (root / p).resolve()
    num_classes = entry.pop("num_classes", 0)
    return DatasetCfg(name=name, path=p, num_classes=num_classes, extra=entry)


@dataclass
class RunPaths:
    """一次运行的标准输出位置。"""
    results_dir: Path = RESULTS_ROOT
    cache_dir: Path = CACHE_ROOT


if __name__ == "__main__":
    # 最小自检：yaml 可读、路径存在性如实报告
    for y in ["datasets", "budgets", "methods"]:
        d = load_yaml(y)
        print(f"[config] {y}.yaml 顶层键: {list(d)}")
    print(f"[config] PROJECT_ROOT={PROJECT_ROOT}")
    print(f"[config] DINOv2 权重存在: {DINOV2_VITS14_PATH.exists()} ({DINOV2_VITS14_PATH})")
    print(f"[config] CLIP 权重目录: {CLIP_CKPT_DIR}")
    for pt in ["RN50.pt", "ViT-B-16.pt"]:
        print(f"[config]   {pt}: {(CLIP_CKPT_DIR / pt).exists()}")
    set_seed(0)
    a = np.random.rand(3)
    set_seed(0)
    assert np.allclose(a, np.random.rand(3)), "种子不可复现"
    print("[config] set_seed 可复现 OK")
