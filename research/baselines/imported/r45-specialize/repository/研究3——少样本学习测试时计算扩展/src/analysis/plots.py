# -*- coding: utf-8 -*-
"""审稿六图（实验计划 §5.4）的绘图函数（matplotlib，Agg 后端）。

① plot_scaling_grid      主 scaling 图（log-x，多数据集小图矩阵）
② plot_oracle_gap        oracle 上界 vs 投票（饱和缺口）
③ plot_bucket_panel      难度分桶面板（收益集中难样本）
④ plot_iso_flops         iso-FLOPs 策略对比
⑤ plot_frontier          compute-optimal frontier（均匀 vs 自适应）
⑥ plot_alloc_hist        预算分配直方图 + 精度-vs-平均预算曲线
每个函数收 records（dict 列表）+ 输出路径，落盘 PNG。保持轻依赖、可直接 import。
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scaling_grid(per_dataset: Dict[str, List[Dict]], out_path: Path,
                      x_key: str = "cost", y_key: str = "acc") -> Path:
    """① 主 scaling 图：per_dataset = {数据集名: [{cost, acc, method}, ...]}。"""
    names = sorted(per_dataset)
    n = len(names)
    ncol = min(3, max(1, n))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, name in zip(axes.flat, names):
        recs = per_dataset[name]
        by_method: Dict[str, List[Dict]] = {}
        for r in recs:
            by_method.setdefault(r.get("method", "?"), []).append(r)
        for m, rs in by_method.items():
            rs = sorted(rs, key=lambda r: r[x_key])
            ax.plot([r[x_key] for r in rs], [r[y_key] for r in rs], "o-", ms=3, label=m)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("FLOPs (C0)")
        ax.set_ylabel(y_key)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    return _save(fig, out_path)


def plot_oracle_gap(costs: Sequence[float], acc_vote: Sequence[float],
                    acc_oracle: Sequence[float], out_path: Path, title: str = "") -> Path:
    """② oracle 上界 vs 投票（饱和缺口）。"""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(costs, acc_vote, "o-", label="vote/mean")
    ax.plot(costs, acc_oracle, "s--", label="oracle (any view)")
    ax.fill_between(costs, acc_vote, acc_oracle, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs (C0)")
    ax.set_ylabel("acc")
    ax.set_title(title or "oracle gap")
    ax.legend()
    return _save(fig, out_path)


def plot_bucket_panel(bucket_gain: Dict[str, Sequence[float]], tiers: Sequence[str],
                      out_path: Path) -> Path:
    """③ 难度分桶面板：bucket_gain = {easy/mid/hard: 各档精度}。"""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for bucket, accs in bucket_gain.items():
        ax.plot(range(len(accs)), accs, "o-", label=bucket)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels(tiers, rotation=30)
    ax.set_ylabel("acc")
    ax.set_title("gain by difficulty bucket")
    ax.legend()
    return _save(fig, out_path)


def plot_iso_flops(records: List[Dict], out_path: Path,
                   x_key: str = "cost", y_key: str = "acc") -> Path:
    """④ iso-FLOPs 策略对比：records 含 method/cost/acc。"""
    fig, ax = plt.subplots(figsize=(5, 4))
    by_method: Dict[str, List[Dict]] = {}
    for r in records:
        by_method.setdefault(r.get("method", "?"), []).append(r)
    for m, rs in by_method.items():
        rs = sorted(rs, key=lambda r: r[x_key])
        ax.plot([r[x_key] for r in rs], [r[y_key] for r in rs], "o-", ms=4, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs (C0)")
    ax.set_ylabel(y_key)
    ax.set_title("iso-FLOPs comparison")
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def plot_frontier(frontier_uniform: List[Dict], frontier_adaptive: List[Dict],
                  out_path: Path) -> Path:
    """⑤ compute-optimal frontier：均匀 vs 自适应。"""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for fr, label, style in [(frontier_uniform, "uniform", "o-"),
                             (frontier_adaptive, "adaptive", "s-")]:
        fr = sorted(fr, key=lambda r: r["cost"])
        ax.plot([r["cost"] for r in fr], [r.get("acc", 1 - r.get("error", 0)) for r in fr],
                style, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("FLOPs (C0)")
    ax.set_ylabel("acc")
    ax.set_title("compute-optimal frontier")
    ax.legend()
    return _save(fig, out_path)


def plot_alloc_hist(assign: Sequence[int], costs: Sequence[float],
                    budget_curve: List[Dict], out_path: Path) -> Path:
    """⑥ 预算分配直方图 + 精度-vs-平均预算曲线（双子图）。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))
    ax1.hist(list(assign), bins=len(set(assign)), rwidth=0.8)
    ax1.set_xlabel("tier")
    ax1.set_ylabel("#queries")
    ax1.set_title("allocation histogram")
    bc = sorted(budget_curve, key=lambda r: r["budget"])
    ax2.plot([r["budget"] for r in bc], [r["acc"] for r in bc], "o-")
    ax2.set_xlabel("mean budget (C0)")
    ax2.set_ylabel("acc")
    ax2.set_title("acc vs budget")
    return _save(fig, out_path)


if __name__ == "__main__":
    # 单元验证：六图各画一次，确认文件落盘
    import tempfile

    out = Path(tempfile.mkdtemp()) / "fig.png"
    recs = {"ds1": [{"cost": c, "acc": 0.5 + 0.02 * np.log(c), "method": "m1"}
                    for c in [1, 2, 4, 8]]}
    plot_scaling_grid(recs, out)
    assert out.exists()
    plot_oracle_gap([1, 2, 4], [0.6, 0.65, 0.68], [0.7, 0.75, 0.78], out)
    plot_bucket_panel({"easy": [0.9, 0.9], "hard": [0.4, 0.6]}, ["t1", "t2"], out)
    plot_iso_flops(recs["ds1"] + [{"cost": 2, "acc": 0.62, "method": "m2"}], out)
    plot_frontier([{"cost": 1, "acc": 0.5}, {"cost": 4, "acc": 0.6}],
                  [{"cost": 1, "acc": 0.52}, {"cost": 4, "acc": 0.65}], out)
    plot_alloc_hist([0, 0, 1, 2, 2], [1, 2, 4], [{"budget": 1, "acc": 0.5}], out)
    print(f"[plots] 单元验证通过: 六图落盘 OK → {out}")
