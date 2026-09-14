# -*- coding: utf-8 -*-
"""E2：AD 预算轴扫描（实验计划 §5.1）。

记忆库 coreset 保留率 × 邻居数 k × 视图数 V × 分辨率 r 的预算网格，
按类输出图像级 AUROC/AP/F1-max（PRO 仅在 --pixel 时计算，CPU 密集）。

smoke 示例：
  conda run -n torch python -m src.run.run_scaling_ad \
    --dataset mvtec --category bottle --shot 1 --res 224 \
    --coreset-ratios 0.25,1.0 --neighbors 1,3
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torchvision import transforms

from src.ad.memory_bank import MemoryBank
from src.ad.metrics import compute_ap, compute_auroc, compute_f1_max
from src.config import DINO_MEAN, DINO_STD, RESULTS_ROOT, load_yaml, save_json, set_seed
from src.data.ad_datasets import (AnomalyDataset, get_few_shot_samples,
                                  get_mvtec_path, get_test_dataset)
from src.features.cache_io import FeatureCache
from src.features.dino import DINOExtractor
from src.views import cache_key


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E2 AD 预算轴扫描")
    p.add_argument("--dataset", default="mvtec")
    p.add_argument("--category", default="bottle")
    p.add_argument("--shot", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--res", type=int, default=224)
    p.add_argument("--coreset-ratios", default="0.01,0.05,0.25,1.0")
    p.add_argument("--neighbors", default="1,3,9")
    p.add_argument("--views", default="1", help="视图数档位（旋转增强；当前 smoke 固定 1）")
    p.add_argument("--max-test", type=int, default=0, help=">0 时截断测试集（快速验证）")
    p.add_argument("--masking", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--from-cache", action="store_true",
                   help="读 cache/ad/<dataset> 的 v0 patch 特征（extract_features_ad.py 产物），"
                        "不跑 backbone；coreset/k/res 轴完全由缓存覆盖")
    p.add_argument("--out", default=str(RESULTS_ROOT))
    return p.parse_args(argv)


def _float_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    acfg = load_yaml("methods")["ad"]

    root = get_mvtec_path() if args.dataset == "mvtec" else None
    if root is None:
        raise NotImplementedError("目前仅实现 mvtec（visa 待数据到位）")

    def img_id(image_path: str) -> str:
        rel = Path(image_path).relative_to(Path(root) / args.category).as_posix()
        return f"{args.dataset}#{args.category}/{rel}"

    mb = MemoryBank(None, use_masking=bool(args.masking),
                    rotation_angles=acfg["rotation_angles"],
                    aggregation=acfg["aggregation"])

    if args.from_cache:
        # 缓存路径：v0 patch 特征全部读盘，旋转增强在 patch 网格级施加
        fc = FeatureCache(f"ad/{args.dataset}", "dinov2_vits14")

        def pkey(iid: str, angle: int = 0) -> str:
            return cache_key(iid, angle, args.res, "dinov2_vits14") + "|patch"

        train_ds = AnomalyDataset(str(root), args.category, split="train")
        rng = random.Random(args.seed)          # 与 get_few_shot_samples 同种子同顺序
        indices = list(range(len(train_ds)))
        rng.shuffle(indices)
        angles = [int(a) for a in acfg["rotation_angles"]]
        support_patches = [[fc.get(pkey(img_id(train_ds.samples[i]["image_path"]), ang))
                            for ang in angles]
                           for i in indices[:args.shot]]
        n_patch = support_patches[0][0].shape[0]
        gh = int(round(n_patch ** 0.5))
        assert gh * gh == n_patch, f"patch 网格非方形: {n_patch}"
        mb.build_from_patches(support_patches, (gh, gh))
        print(f"  [AD] {args.category}: {args.shot}-shot 记忆库={mb.bank.shape[0]} patch "
              f"(from-cache, masking={bool(args.masking)}, 旋转×{len(acfg['rotation_angles'])})")

        test = get_test_dataset(str(root), args.category, image_size=args.res)
        n_test = len(test) if args.max_test <= 0 else min(args.max_test, len(test))
        labels = np.array([test.samples[i]["label"] for i in range(n_test)])
        test_patches = [fc.get(pkey(img_id(test.samples[i]["image_path"])))
                        for i in range(n_test)]

        rows = []
        for ratio in _float_list(args.coreset_ratios):
            for k in _int_list(args.neighbors):
                scores = np.array([mb.score_patches(p, (gh, gh), coreset_ratio=ratio,
                                                    n_neighbors=k, coreset_seed=args.seed)[1]
                                   for p in test_patches])
                row = {"category": args.category, "shot": args.shot,
                       "coreset_ratio": ratio, "n_neighbors": k, "res": args.res,
                       "image_auroc": round(compute_auroc(scores, labels), 4),
                       "image_ap": round(compute_ap(scores, labels), 4),
                       "image_f1_max": round(compute_f1_max(scores, labels), 4),
                       "n_test": int(n_test)}
                rows.append(row)
                print(f"  coreset={ratio:5.2f} k={k}  AUROC={row['image_auroc']:.4f} "
                      f"AP={row['image_ap']:.4f} F1={row['image_f1_max']:.4f}")

        out = save_json({"task": "E2_scaling_ad", "args": vars(args), "rows": rows},
                        Path(args.out), f"scaling_ad_{args.dataset}_{args.category}")
        print(f"[E2] 结果已落盘: {out}")
        return rows

    ext = DINOExtractor("dinov2_vits14", device=args.device)
    mb.extractor = ext
    tf = transforms.Compose([
        transforms.Resize((args.res, args.res)),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])
    support = get_few_shot_samples(str(root), args.category, args.shot,
                                   image_size=args.res, transform=tf, seed=args.seed)
    mb.build(support)
    print(f"  [AD] {args.category}: {args.shot}-shot 记忆库={mb.bank.shape[0]} patch "
          f"(masking={bool(args.masking)}, 旋转×{len(acfg['rotation_angles'])})")

    test = get_test_dataset(str(root), args.category, image_size=args.res, transform=tf)
    n_test = len(test) if args.max_test <= 0 else min(args.max_test, len(test))
    labels = np.array([test.samples[i]["label"] for i in range(n_test)])

    rows = []
    for ratio in _float_list(args.coreset_ratios):
        for k in _int_list(args.neighbors):
            scores = []
            for i in range(n_test):
                img = test[i]["image"]
                _, s = mb.score_image(img, coreset_ratio=ratio, n_neighbors=k,
                                      coreset_seed=args.seed)
                scores.append(s)
            scores = np.array(scores)
            row = {"category": args.category, "shot": args.shot,
                   "coreset_ratio": ratio, "n_neighbors": k, "res": args.res,
                   "image_auroc": round(compute_auroc(scores, labels), 4),
                   "image_ap": round(compute_ap(scores, labels), 4),
                   "image_f1_max": round(compute_f1_max(scores, labels), 4),
                   "n_test": int(n_test)}
            rows.append(row)
            print(f"  coreset={ratio:5.2f} k={k}  AUROC={row['image_auroc']:.4f} "
                  f"AP={row['image_ap']:.4f} F1={row['image_f1_max']:.4f}")

    out = save_json({"task": "E2_scaling_ad", "args": vars(args), "rows": rows},
                    Path(args.out), f"scaling_ad_{args.dataset}_{args.category}")
    print(f"[E2] 结果已落盘: {out}")
    return rows


if __name__ == "__main__":
    main()
