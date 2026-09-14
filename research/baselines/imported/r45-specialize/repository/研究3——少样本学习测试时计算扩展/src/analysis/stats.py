# -*- coding: utf-8 -*-
"""统计工具：95% CI（episodic 惯例）与配对 bootstrap 显著性。"""
from typing import Dict, Sequence, Tuple

import numpy as np


def mean_ci(values: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    """均值 ± 95% CI（episodic FSL 惯例：mean ± 1.96·std/√n）。"""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    half = 1.96 * v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return {"mean": float(v.mean()), "ci95": float(half), "n": int(n)}


def paired_bootstrap(a: Sequence[float], b: Sequence[float],
                     n_boot: int = 10000, seed: int = 0) -> Dict[str, float]:
    """配对 bootstrap：检验 mean(a) 是否显著优于 mean(b)（同单元配对，如逐 episode）。"""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    assert len(x) == len(y)
    rng = np.random.RandomState(seed)
    n = len(x)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs.append(float((x[idx] - y[idx]).mean()))
    diffs = np.asarray(diffs)
    p = float((diffs <= 0).mean())        # 单侧：a ≤ b 的概率
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": float((x - y).mean()), "ci95": (float(lo), float(hi)), "p_one_sided": p}


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    a = 0.7 + 0.05 * rng.randn(100)
    b = 0.68 + 0.05 * rng.randn(100)
    r = mean_ci(a)
    assert r["ci95"] > 0 and r["n"] == 100
    pb = paired_bootstrap(a, b, n_boot=2000)
    assert pb["p_one_sided"] < 0.05, "应有显著差异"
    pb2 = paired_bootstrap(a, a[rng.permutation(100)], n_boot=2000)
    assert pb2["p_one_sided"] > 0.3, "打乱配对（零假设）不应显著"
    print(f"[stats] 单元验证通过: mean={r['mean']:.3f}±{r['ci95']:.3f}, "
          f"paired p={pb['p_one_sided']:.4f} / 对照 p={pb2['p_one_sided']:.3f}")
