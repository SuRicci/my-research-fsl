# -*- coding: utf-8 -*-
"""E3：难度信号 benchmark（实验计划 §5.1，核心图③数据源）。

在 episodic 协议固定预算下推理，逐 query 收集 S1–S5 与秩融合信号，
评估「信号 vs 错分」AUROC 与「信号 vs oracle 边际收益」Spearman：
- 既报全体 query 汇聚（pooled）的值，也按 episodic 惯例报逐 episode 均值±95%CI
  （analysis/stats.py: mean_ci；episode 内标签/增益无方差时跳过并记 n_valid）。
- S5 跨分支一致性需要第二 backbone 后验（--signal-backbone，默认 dinov2_vits14），
  同方法同预算在第二 backbone 缓存特征上推理得到 p_dino。

示例：
  conda run -n torch python -m src.run.run_signals \
    --dataset miniimagenet --backbone clip_vitb16 --episodes 200 --shot 1 --V 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from src.analysis.stats import mean_ci
from src.budgets import Budget
from src.config import RESULTS_ROOT, load_yaml, save_json, set_seed
from src.data.cls_datasets import load_cls_dataset
from src.data.splits import EpisodeGenerator
from src.run.common import (episode_feature_indices, extract_features,
                            infer_episode, score_episode)
from src.signals.difficulty import compute_cls_signals


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E3 难度信号 benchmark")
    p.add_argument("--dataset", default="miniimagenet")
    p.add_argument("--split", default="test")
    p.add_argument("--backbone", default="clip_vitb16")
    p.add_argument("--signal-backbone", default="dinov2_vits14",
                   help="S5 跨分支信号的第二 backbone；空串则关闭 S5")
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--shot", type=int, default=1)
    p.add_argument("--q-per-cls", type=int, default=15)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="tip_adapter")
    p.add_argument("--V", type=int, default=4, help="视图数（S4 视图方差需要 V≥2）")
    p.add_argument("--k", type=int, default=-1)
    p.add_argument("--T", type=int, default=0, help="传导轮数（laplacian_shot 等需 >0）")
    p.add_argument("--r", type=int, default=224)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default=str(RESULTS_ROOT))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    mcfg = load_yaml("methods")
    kw = dict(mcfg["cache_based"]["tip_adapter"])
    kw.update(mcfg["transductive"]["laplacian_shot"])
    nn_k = mcfg["signals"].get("nn_ratio_k", 5)

    ds = load_cls_dataset(args.dataset, args.split)
    gen = EpisodeGenerator(ds.labels, n_way=args.n_way, k_shot=args.shot,
                           q_per_cls=args.q_per_cls, seed=args.seed)
    episodes = gen.generate(args.episodes)
    indices = episode_feature_indices(episodes)
    feats = extract_features(ds, indices, args.V, args.r, args.backbone,
                             device=args.device, batch_size=args.batch_size)
    feats_b = None
    if args.signal_backbone:
        feats_b = extract_features(ds, indices, args.V, args.r, args.signal_backbone,
                                   device=args.device, batch_size=args.batch_size)

    budget = Budget(V=args.V, k=args.k, T=args.T, r=args.r)
    sig_vals: dict = {}
    wrong_eps, oracle_gain_eps = [], []
    for ep in episodes:
        sf = torch.stack([feats[i] for i in ep.support_idx])   # [S, V, D]
        qf = torch.stack([feats[i] for i in ep.query_idx])     # [Q, V, D]
        sy = torch.from_numpy(ep.support_labels)
        qy = torch.from_numpy(ep.query_labels)
        n_class = args.n_way

        # 聚合推理（均值后验）
        res = score_episode(infer_episode(args.method, qf, sf, sy, n_class, budget, **kw), qy)
        # 单视图（v=0）推理供 S3 距离信号
        qv, sv = qf[:, 0], sf[:, 0]
        dist = 1.0 - qv @ sv.T                                   # 余弦距离 [Q, S]
        # 逐视图后验（S4 用）：重算每视图概率
        p_views = []
        for v in range(args.V):
            rv = infer_episode(args.method, qf[:, v:v + 1], sf[:, v:v + 1], sy, n_class,
                               Budget(V=1, k=args.k, T=args.T, r=args.r), **kw)
            p_views.append(rv.probs)
        p_views = torch.stack(p_views, dim=1)                    # [Q, V, C]
        # S5：第二 backbone 同方法同预算的聚合后验
        p_other = None
        if feats_b is not None:
            sf_b = torch.stack([feats_b[i] for i in ep.support_idx])
            qf_b = torch.stack([feats_b[i] for i in ep.query_idx])
            p_other = infer_episode(args.method, qf_b, sf_b, sy, n_class, budget, **kw).probs

        sigs = compute_cls_signals(res.probs, dist=dist, p_views=p_views,
                                   p_other_branch=p_other, nn_ratio_k=nn_k)
        wrong = (res.probs.argmax(1) != qy).float()
        # oracle 边际收益：oracle 对而均值后验错 → 升 V 可获得的收益
        oracle_gain = ((res.per_view_pred == qy.unsqueeze(1)).any(dim=1)
                       & (res.probs.argmax(1) != qy)).float()
        for kname, v in sigs.items():
            sig_vals.setdefault(kname, []).append(v)
        wrong_eps.append(wrong)
        oracle_gain_eps.append(oracle_gain)

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    wrong_all = torch.cat(wrong_eps).numpy()
    oracle_gain_all = torch.cat(oracle_gain_eps).numpy()
    n_ep = len(wrong_eps)

    rows = []
    for kname, chunks in sig_vals.items():
        s_all = torch.cat(chunks).numpy()
        # pooled（全体 query 汇聚）
        auroc_pool = roc_auc_score(wrong_all, s_all) if len(np.unique(wrong_all)) > 1 else float("nan")
        sp_pool = spearmanr(s_all, oracle_gain_all).correlation
        # 逐 episode + 95% CI（episodic 惯例；无方差 episode 跳过）
        aurocs, sps = [], []
        for i in range(n_ep):
            w = wrong_eps[i].numpy()
            g = oracle_gain_eps[i].numpy()
            s = chunks[i].numpy()
            if len(np.unique(w)) > 1 and np.std(s) > 0:
                aurocs.append(float(roc_auc_score(w, s)))
            if len(np.unique(g)) > 1 and np.std(s) > 0:
                sp = spearmanr(s, g).correlation
                if sp == sp:
                    sps.append(float(sp))
        ac = mean_ci(aurocs)
        sc = mean_ci(sps)
        rows.append({
            "signal": kname,
            "auroc_vs_error": round(float(auroc_pool), 4),
            "auroc_ep_mean": round(ac["mean"], 4), "auroc_ep_ci95": round(ac["ci95"], 4),
            "auroc_n_valid": ac["n"],
            "spearman_vs_oracle_gain": round(float(sp_pool), 4) if sp_pool == sp_pool else None,
            "spearman_ep_mean": round(sc["mean"], 4), "spearman_ep_ci95": round(sc["ci95"], 4),
            "spearman_n_valid": sc["n"],
        })
        print(f"  {kname:18s} AUROC={auroc_pool:.4f} (ep {ac['mean']:.4f}±{ac['ci95']:.4f}, "
              f"n={ac['n']})  Spearman={sp_pool:.4f} (ep {sc['mean']:.4f}±{sc['ci95']:.4f}, "
              f"n={sc['n']})")

    out = save_json({"task": "E3_signals", "args": vars(args),
                     "error_rate": round(float(wrong_all.mean()), 4),
                     "oracle_gain_rate": round(float(oracle_gain_all.mean()), 4),
                     "rows": rows},
                    Path(args.out),
                    f"signals_{args.dataset}_{args.backbone}_{args.method}_s{args.shot}")
    print(f"[E3] 结果已落盘: {out}")
    return rows


if __name__ == "__main__":
    main()
