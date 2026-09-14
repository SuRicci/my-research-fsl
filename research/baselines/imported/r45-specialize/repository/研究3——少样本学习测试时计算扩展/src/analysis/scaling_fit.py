# -*- coding: utf-8 -*-
"""幂律 scaling 拟合（实验计划 §3.3-8）：error L = a·C^(−α) + L∞，Huber + L-BFGS。

对 (预算 C, 误差 L) 序列在 log 域拟合幂律；报 α、R²、参数置信区间（渐近协方差）。
另含：compute-optimal frontier 包络、oracle 缺口度量。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PowerLawFit:
    """拟合结果：L = a·C^(−alpha) + L_inf。"""
    a: float
    alpha: float
    l_inf: float
    r2: float
    param_std: Tuple[float, float, float]   # (a, alpha, l_inf) 的渐近标准差
    n_points: int

    def predict(self, c: np.ndarray) -> np.ndarray:
        return self.a * np.power(c, -self.alpha) + self.l_inf


def fit_power_law(costs: Sequence[float], errors: Sequence[float],
                  huber_delta: float = 1.0) -> PowerLawFit:
    """Huber 损失 + L-BFGS 拟合 L = a·C^(−α) + L∞。

    参数化用 log 空间（a=exp(pa), α=exp(pα)）保证正定；L∞ 自由。
    """
    from scipy.optimize import minimize

    C = np.asarray(costs, dtype=np.float64)
    L = np.asarray(errors, dtype=np.float64)
    assert len(C) == len(L) and len(C) >= 4, "幂律拟合至少需要 4 个点"
    assert (C > 0).all(), "成本必须为正"

    def model(p):
        pa, palpha, l_inf = p
        return np.exp(pa) * np.power(C, -np.exp(palpha)) + l_inf

    def huber(res):
        d = huber_delta * max(1e-8, float(np.std(L)))
        abs_r = np.abs(res)
        quad = np.minimum(abs_r, d)
        lin = abs_r - quad
        return float(np.sum(0.5 * quad ** 2 + d * lin))

    def objective(p):
        return huber(model(p) - L)

    # 初值：L∞=最小误差×0.9，α 用两端点 log-log 斜率
    l0 = max(0.0, L.min() * 0.9)
    i0, i1 = int(np.argmin(C)), int(np.argmax(C))
    slope = max(1e-3, (np.log(max(L[i0] - l0, 1e-4)) - np.log(max(L[i1] - l0, 1e-4)))
                / (np.log(C[i1]) - np.log(C[i0]) + 1e-12))
    a0 = max(1e-4, (L[i0] - l0) * C[i0] ** slope)
    p0 = np.array([np.log(a0), np.log(slope), l0])

    res = minimize(objective, p0, method="L-BFGS-B",
                   bounds=[(np.log(1e-6), np.log(1e6)),
                           (np.log(1e-3), np.log(10.0)),
                           (0.0, L.min() + 1e-6)])
    pa, palpha, l_inf = res.x
    a, alpha = float(np.exp(pa)), float(np.exp(palpha))
    pred = model(res.x)
    ss_res = float(np.sum((L - pred) ** 2))
    ss_tot = float(np.sum((L - L.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot

    # 渐近协方差：JᵀJ 逆 × 残差方差
    eps = 1e-6
    J = np.zeros((len(C), 3))
    for j in range(3):
        dp = np.zeros(3); dp[j] = eps
        J[:, j] = (model(res.x + dp) - model(res.x - dp)) / (2 * eps)
    dof = max(1, len(C) - 3)
    sigma2 = ss_res / dof
    try:
        cov = sigma2 * np.linalg.inv(J.T @ J + 1e-10 * np.eye(3))
        std = tuple(float(np.sqrt(max(cov[i, i], 0.0))) for i in range(3))
    except np.linalg.LinAlgError:
        std = (float("nan"),) * 3
    return PowerLawFit(a=a, alpha=alpha, l_inf=float(l_inf), r2=float(r2),
                       param_std=std, n_points=len(C))


def compute_frontier(records: List[Dict]) -> List[Dict]:
    """compute-optimal frontier：每预算点取误差最小的记录（包络）。

    records: 含 "cost"（C0）与 "error"（或 "acc" 自动转 error=1-acc）的 dict 列表。
    返回按成本升序、误差单调不增的包络子序列。
    """
    pts = []
    for r in records:
        err = r.get("error", 1.0 - r.get("acc", 0.0))
        pts.append((float(r["cost"]), float(err), r))
    pts.sort(key=lambda x: (x[0], x[1]))
    frontier, best = [], np.inf
    for cost, err, r in pts:
        if err < best - 1e-12:
            frontier.append(r)
            best = err
    return frontier


def oracle_gap(acc_vote: Sequence[float], acc_oracle: Sequence[float]) -> Dict[str, float]:
    """oracle 缺口（饱和缺口）：oracle 精度（任一视图对即对）与投票精度的差。"""
    v = np.asarray(acc_vote)
    o = np.asarray(acc_oracle)
    return {"gap_mean": float((o - v).mean()), "gap_max": float((o - v).max()),
            "headroom_ratio": float((o - v).mean() / max(1e-12, (1 - v).mean()))}


if __name__ == "__main__":
    # 单元验证：对合成幂律数据应恢复参数
    rng = np.random.RandomState(0)
    C = np.array([1, 2, 4, 8, 16, 32], dtype=float)
    true = dict(a=0.5, alpha=0.3, l_inf=0.05)
    L = true["a"] * C ** (-true["alpha"]) + true["l_inf"] + rng.normal(0, 0.005, len(C))
    fit = fit_power_law(C, L)
    assert abs(fit.alpha - true["alpha"]) < 0.1, f"α 偏差过大: {fit.alpha}"
    assert abs(fit.l_inf - true["l_inf"]) < 0.05
    assert fit.r2 > 0.95
    # frontier：误差随成本单调不增的包络
    recs = [{"cost": c, "error": e} for c, e in zip([1, 2, 4, 8], [0.4, 0.35, 0.36, 0.3])]
    fr = compute_frontier(recs)
    assert [r["cost"] for r in fr] == [1, 2, 8], "frontier 应剔除被支配点 (4, 0.36)"
    g = oracle_gap([0.6, 0.7], [0.8, 0.85])
    assert 0 < g["gap_mean"] < 0.3
    print(f"[scaling_fit] 单元验证通过: alpha={fit.alpha:.3f}(真0.3), L_inf={fit.l_inf:.3f}(真0.05), R2={fit.r2:.4f}")
