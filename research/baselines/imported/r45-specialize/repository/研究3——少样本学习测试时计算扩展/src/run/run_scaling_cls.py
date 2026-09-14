# -*- coding: utf-8 -*-
"""E1：分类预算轴扫描（实验计划 §5.1）。

在 episodic 协议下对方法 × 预算点（V/k/T/r 网格）扫描精度，输出
精度-预算表（JSON 落盘）。特征走磁盘缓存（只算一次）。

smoke 示例：
  conda run -n torch python -m src.run.run_scaling_cls \
    --dataset miniimagenet --backbone clip_vitb16 \
    --n-way 5 --shot 1 --episodes 20 \
    --methods tip_adapter laplacian_shot --V 1,2,4 --T 5
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 允许直接脚本运行

import torch

from src.budgets import Budget
from src.config import RESULTS_ROOT, load_yaml, save_json, set_seed
from src.data.cls_datasets import load_cls_dataset
from src.data.splits import EpisodeGenerator
from src.run.common import (aggregate_episode_metrics, budget_cost_c0,
                            episode_feature_indices, extract_features,
                            infer_episode, score_episode)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E1 分类预算轴扫描")
    p.add_argument("--dataset", default="miniimagenet")
    p.add_argument("--split", default="test")
    p.add_argument("--backbone", default="clip_vitb16")
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--shot", type=int, default=1)
    p.add_argument("--q-per-cls", type=int, default=15)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", nargs="+", default=["tip_adapter"])
    p.add_argument("--V", default="1", help="逗号分隔的视图数档位")
    p.add_argument("--k", default="-1", help="逗号分隔的检索宽度档位")
    p.add_argument("--T", default="0", help="逗号分隔的传导轮数档位")
    p.add_argument("--r", default="224", help="逗号分隔的分辨率档位")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default=str(RESULTS_ROOT))
    return p.parse_args(argv)


def _int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    mcfg = load_yaml("methods")
    tip_kw = dict(mcfg["cache_based"]["tip_adapter"])
    tr_kw = dict(mcfg["transductive"]["laplacian_shot"])

    ds = load_cls_dataset(args.dataset, args.split)
    gen = EpisodeGenerator(ds.labels, n_way=args.n_way, k_shot=args.shot,
                           q_per_cls=args.q_per_cls, seed=args.seed)
    episodes = gen.generate(args.episodes)

    V_max = max(_int_list(args.V))
    res = max(_int_list(args.r))  # smoke 只支持单分辨率；多分辨率需逐档特征（后接调度）
    indices = episode_feature_indices(episodes)
    feats = extract_features(ds, indices, V_max, res, args.backbone,
                             device=args.device, batch_size=args.batch_size)

    zs_fn = None  # miniImageNet 类名为 wnid，无文本先验；文本先验仅用于有类名的数据集

    rows = []
    for method in args.methods:
        kw = dict(tip_kw)
        kw.update(tr_kw)
        for V in _int_list(args.V):
            for k in _int_list(args.k):
                for T in _int_list(args.T):
                    budget = Budget(V=V, k=k, T=T if method in ("laplacian_shot", "alpha_tim") else 0,
                                    r=res)
                    results = []
                    for ep in episodes:
                        sf = torch.stack([feats[i] for i in ep.support_idx])
                        qf = torch.stack([feats[i] for i in ep.query_idx])
                        sy = torch.from_numpy(ep.support_labels)
                        qy = torch.from_numpy(ep.query_labels)
                        r = infer_episode(method, qf, sf, sy, args.n_way, budget,
                                          zs_logits_fn=zs_fn, **kw)
                        results.append(score_episode(r, qy))
                    agg = aggregate_episode_metrics(results)
                    cost = budget_cost_c0(budget, args.backbone,
                                          n_support=args.n_way * args.shot,
                                          n_query=args.n_way * args.q_per_cls,
                                          n_class=args.n_way)
                    row = {"method": method, "budget": budget.to_dict(),
                           "cost_c0": round(cost, 3), **{k2: round(v, 4) for k2, v in agg.items()}}
                    rows.append(row)
                    print(f"  {method:16s} {budget.tag():18s} cost={cost:6.2f}C0 "
                          f"acc={agg['acc_mean']:.4f}±{agg['acc_mean_ci95']:.4f} "
                          f"vote={agg['acc_vote']:.4f} oracle={agg['acc_oracle']:.4f}")

    # 文件名含 shot/seed/episodes：多 seed 并行 worker 同秒落盘不互相覆盖
    out = save_json({"task": "E1_scaling_cls", "args": vars(args), "rows": rows},
                    Path(args.out), f"scaling_cls_{args.dataset}_{args.backbone}"
                                    f"_s{args.shot}_seed{args.seed}_ep{args.episodes}")
    print(f"[E1] 结果已落盘: {out}")
    return rows


if __name__ == "__main__":
    main()
