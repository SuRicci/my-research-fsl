# -*- coding: utf-8 -*-
"""难度信号（实验计划 §1.3，全部零开销：只依赖后验/检索几何/多视图，免标签免训练）。

分类侧（输入为 query 后验 p [Q,C]、query-support 检索距离、多视图后验等）：
- S1 熵 H(p)（仅消融/复现 Yang 负结果对照）
- S2 top-2 边际 d2 = 1 − (p1 − p2)
- S3 NN 距离比 d3 = dist(q,NN1)/dist(q,NN2)（→1 为难）；变体 dist1/mean_k dist
- S4 视图方差 Var_v(p^(v)) 的迹（需 V≥2，辅助）
- S5 跨分支一致性 d5 = JS(p_clip, p_dino) 或 1{ŷ_clip≠ŷ_dino}
- 融合主信号 d = mean(rank(d2), rank(d3), rank(d5))（批内秩化，免标定）

AD 侧（输入为异常分数/双分支热图/patch 分数）：
- A1 |s − τ| 边际；A2 CLIP/DINO 热图分歧（相关距离）；A3 patch 分数局部方差。

约定：所有信号均为"越大越难"，输出 [Q] 的一维数组。
"""
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- 分类信号
def s1_entropy(p: torch.Tensor) -> torch.Tensor:
    """S1 熵：-Σ p log p（p 为概率 [Q, C]）。"""
    p = p.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=1)


def s2_top2_margin(p: torch.Tensor) -> torch.Tensor:
    """S2 top-2 边际：1 − (p1 − p2)，越大越难。"""
    top2 = p.topk(min(2, p.shape[1]), dim=1).values
    if top2.shape[1] == 1:
        return torch.zeros(p.shape[0], device=p.device)
    return 1.0 - (top2[:, 0] - top2[:, 1])


def s3_nn_ratio(dist: torch.Tensor, k: int = 1) -> torch.Tensor:
    """S3 NN 距离比：dist 为 query 到 support/缓存的最近距离 [Q, N]（已升序或未排序均可）。

    k=1: dist1/dist2；k>1: dist1/mean(dist2..dist_{k+1})。比值 →1 为难。
    """
    d_sorted = dist.sort(dim=1, descending=False).values
    n = d_sorted.shape[1]
    if n < 2:
        return torch.ones(d_sorted.shape[0], device=dist.device)
    d1 = d_sorted[:, 0].clamp_min(0.0)
    hi = min(k + 1, n - 1)
    denom = d_sorted[:, 1:hi + 1].mean(dim=1) if hi >= 1 else d_sorted[:, 1]
    return d1 / denom.clamp_min(1e-12)


def s4_view_variance(p_views: torch.Tensor) -> torch.Tensor:
    """S4 视图方差：p_views [Q, V, C] 各视图后验的方差迹（V<2 时全零）。"""
    if p_views.dim() != 3 or p_views.shape[1] < 2:
        return torch.zeros(p_views.shape[0], device=p_views.device)
    return p_views.var(dim=1).sum(dim=1)


def _js_div(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """逐样本 JS 散度。"""
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=1)
    kl_qm = (q * (q / m).log()).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def s5_branch_consistency(p_clip: torch.Tensor, p_dino: torch.Tensor,
                          mode: str = "js") -> torch.Tensor:
    """S5 跨分支分歧：mode='js' 为 JS(p_clip, p_dino)；'disagree' 为预测不一致 0/1。"""
    if mode == "disagree":
        return (p_clip.argmax(1) != p_dino.argmax(1)).float()
    return _js_div(p_clip, p_dino)


def _average_rank(s: torch.Tensor) -> torch.Tensor:
    """平均秩（0..n-1）：同值元素取段内秩均值，不用输入位置决胜
    （十五轮评审 P0-1：argsort(argsort) 对同值按位置强行赋不同秩，
    置换同值样本后秩不跟随 → 破坏置换等变；S5 0/1 分歧与单调分箱
    会制造大量同值，非理论边角）。"""
    n = len(s)
    order = torch.argsort(s)                       # 稳定排序（同值按位置，但不影响秩值）
    sorted_s = s[order]
    uniq, counts = torch.unique_consecutive(sorted_s, return_counts=True)
    starts = torch.cat([torch.zeros(1, dtype=torch.long, device=s.device),
                        counts.cumsum(0)[:-1]])
    seg_mean = (starts.float() + (counts.float() - 1) / 2)  # 每段平均秩
    rank_sorted = seg_mean.repeat_interleave(counts)
    out = torch.empty(n, dtype=torch.float, device=s.device)
    out[order] = rank_sorted
    return out


def rank_fuse(signals: Sequence[torch.Tensor]) -> torch.Tensor:
    """秩融合主信号：各信号批内秩化（0..1，同值取平均秩）后取均值。"""
    ranks = []
    for s in signals:
        r = _average_rank(s)
        ranks.append(r / max(1, len(s) - 1))
    return torch.stack(ranks, dim=0).mean(dim=0)


def compute_cls_signals(p: torch.Tensor, dist: Optional[torch.Tensor] = None,
                        p_views: Optional[torch.Tensor] = None,
                        p_other_branch: Optional[torch.Tensor] = None,
                        nn_ratio_k: int = 5) -> Dict[str, torch.Tensor]:
    """一站式计算可用信号（缺输入的信号跳过）。p 为概率后验 [Q, C]。"""
    out = {"s1_entropy": s1_entropy(p), "s2_top2": s2_top2_margin(p)}
    if dist is not None:
        out["s3_nn_ratio"] = s3_nn_ratio(dist, k=1)
        out["s3_nn_ratio_mk"] = s3_nn_ratio(dist, k=nn_ratio_k)
    if p_views is not None:
        out["s4_view_var"] = s4_view_variance(p_views)
    if p_other_branch is not None:
        out["s5_branch_js"] = s5_branch_consistency(p, p_other_branch, "js")
        out["s5_branch_dis"] = s5_branch_consistency(p, p_other_branch, "disagree")
    # 融合主信号：S2+S3+S5（缺哪个跳哪个）
    fuse_keys = [k for k in ["s2_top2", "s3_nn_ratio", "s5_branch_js"] if k in out]
    if len(fuse_keys) >= 2:
        out["fused"] = rank_fuse([out[k] for k in fuse_keys])
    return out


# --------------------------------------------------------------------------- AD 信号
def ad_score_margin(scores: torch.Tensor, tau: float) -> torch.Tensor:
    """A1：|s − τ| 的**负值**作为难度（越贴阈值越难 → 输出 −|s−τ| 再取负号归一到越大越难）。"""
    return -(scores - tau).abs()


def ad_branch_divergence(map_clip: np.ndarray, map_dino: np.ndarray) -> float:
    """A2：双分支热图分歧 = 1 − Spearman 相关（拉平后）。"""
    from scipy.stats import spearmanr

    a = map_clip.flatten()
    b = map_dino.flatten()
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    rho = spearmanr(a, b).correlation
    if rho is None or np.isnan(rho):
        return 0.0
    return float(1.0 - rho)


def ad_patch_local_var(patch_scores: np.ndarray, grid: tuple, win: int = 3) -> float:
    """A3：patch 分数局部方差均值（热图 [gh, gw] 上 win×win 窗口方差的均值）。"""
    gh, gw = grid
    h = patch_scores.reshape(gh, gw)
    t = torch.from_numpy(h).float().unsqueeze(0).unsqueeze(0)
    k = torch.ones(1, 1, win, win) / (win * win)
    mean = F.conv2d(t, k, padding=win // 2)
    mean2 = F.conv2d(t * t, k, padding=win // 2)
    var = (mean2 - mean * mean).clamp_min(0.0)
    return float(var.mean().item())


def bucketize(signal: torch.Tensor, n_buckets: int = 3) -> torch.Tensor:
    """按信号批内分位数分桶（返回每桶 id 0..n-1，0=最易）。"""
    qs = torch.quantile(signal, torch.linspace(0, 1, n_buckets + 1, device=signal.device))
    qs[0], qs[-1] = qs[0] - 1e-9, qs[-1] + 1e-9
    return torch.bucketize(signal, qs[1:-1])


if __name__ == "__main__":
    # 单元验证：形状与取值域
    torch.manual_seed(0)
    Q, C, N = 30, 5, 20
    logits = torch.randn(Q, C)
    p = F.softmax(logits, dim=1)
    dist = torch.rand(Q, N) * 0.5  # 余弦距离 ∈ [0,2]
    pv = F.softmax(logits.unsqueeze(1) + 0.5 * torch.randn(Q, 4, C), dim=-1)
    p2 = F.softmax(logits + torch.randn(Q, C), dim=-1)

    out = compute_cls_signals(p, dist=dist, p_views=pv, p_other_branch=p2)
    for k, v in out.items():
        assert v.shape == (Q,), f"{k} 形状错误: {v.shape}"
        assert torch.isfinite(v).all(), f"{k} 含非法值"
    # 取值域检查
    assert (out["s1_entropy"] >= 0).all() and (out["s1_entropy"] <= np.log(C) + 1e-4).all()
    assert (out["s2_top2"] >= 0).all() and (out["s2_top2"] <= 1).all()
    assert (out["s3_nn_ratio"] >= 0).all()
    assert (out["s5_branch_js"] >= 0).all() and (out["s5_branch_js"] <= np.log(2) + 1e-4).all()
    assert (out["fused"] >= 0).all() and (out["fused"] <= 1).all()
    # 极端：近似 one-hot 后验 → S2≈0（最易）
    p_hard = torch.zeros(2, C); p_hard[0, 1] = 20.0
    assert s2_top2_margin(F.softmax(p_hard, dim=1))[0].item() < 1e-3
    # 分桶
    b = bucketize(out["fused"], 3)
    assert b.min() >= 0 and b.max() <= 2 and len(b) == Q
    # AD 信号
    hm = np.random.rand(16, 16)
    assert ad_patch_local_var(hm, (16, 16)) >= 0
    d = ad_branch_divergence(hm, hm)
    assert abs(d) < 1e-6, "自相关分歧应为 0"
    m = ad_score_margin(torch.tensor([0.5, 0.9]), 0.5)
    assert m[0] == 0 and m[1] < 0
    print(f"[difficulty] 单元验证通过: 信号 {sorted(out)}; 形状/取值域 OK")
