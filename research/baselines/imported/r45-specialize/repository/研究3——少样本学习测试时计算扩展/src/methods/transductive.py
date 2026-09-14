# -*- coding: utf-8 -*-
"""传导式方法（P0/P1，跑在缓存特征上）：LaplacianShot / α-TIM。

统一约定：
- 输入 p0 [Q, C] 为缓存法得到的初始后验（logits），support 标签 one-hot；
- rounds = T（预算轴）：迭代上限即预算，T=0 时返回 p0 本身；
- 输出细化后的 logits [Q, C]。

参考：LaplacianShot (Ziko et al., ICML 2020)；α-TIM (Veilleux et al., 2021)。
PT+MAP（P1）已实现于 methods/p1_baselines.py（本文件不再占位）。
"""
from typing import Optional

import torch
import torch.nn.functional as F


def _onehot(labels: torch.Tensor, n_class: int) -> torch.Tensor:
    return F.one_hot(labels.long(), n_class).float()


def _affinity_knn(feats: torch.Tensor, knn: int) -> torch.Tensor:
    """余弦亲和矩阵 + 对称 kNN 稀疏化（自环置零）。feats 需已 L2 归一化。"""
    W = feats @ feats.T
    n = W.shape[0]
    W.fill_diagonal_(0.0)
    if 0 < knn < n - 1:
        topv, topi = W.topk(knn, dim=1)
        mask = torch.zeros_like(W).scatter_(1, topi, 1.0)
        W = W * mask
        W = torch.clamp(W, min=0.0)
        W = torch.maximum(W, W.T)  # 对称化
    return W


def laplacian_shot_logits(p0_logits: torch.Tensor, support_feats: torch.Tensor,
                          support_labels: torch.Tensor, query_feats: torch.Tensor,
                          rounds: int, lam: float = 0.5, knn: int = 10) -> torch.Tensor:
    """LaplacianShot 低档传导：Z ← (Y + λ·W·Z) ⊘ (1 + λ·d)，Jacobi 迭代 T 轮即止。

    p0_logits: [Q, C] 初始 logits（转概率后作初值）；support 标签作为钉住的 Y。
    图建在 query∪support 上（support 节点标签在每轮后重置为真值——laplacian 正则标准做法）。
    """
    if rounds <= 0:
        return p0_logits
    n_class = p0_logits.shape[1]
    all_feats = torch.cat([support_feats, query_feats], dim=0)
    W = _affinity_knn(all_feats, knn)
    d = W.sum(dim=1, keepdim=True)                       # 度
    Y = torch.zeros(all_feats.shape[0], n_class, device=p0_logits.device)
    Y[:support_feats.shape[0]] = _onehot(support_labels, n_class)
    Z = torch.cat([_onehot(support_labels, n_class),
                   F.softmax(p0_logits, dim=1)], dim=0)  # 初值
    denom = 1.0 + lam * d
    for _ in range(rounds):
        Z = (Y + lam * (W @ Z)) / denom
        Z[:support_feats.shape[0]] = Y[:support_feats.shape[0]]  # 钉住 support
    return Z[support_feats.shape[0]:].clamp_min(1e-12).log()  # 回 logits 形式


def alpha_tim_logits(p0_logits: torch.Tensor, support_feats: torch.Tensor,
                     support_labels: torch.Tensor, query_feats: torch.Tensor,
                     rounds: int, alpha: float = 1.0, lr: float = 0.1,
                     temperature: float = 1.0) -> torch.Tensor:
    """α-TIM 中档传导：support CE + α·条件熵 − 边际熵，对 query logits 做梯度下降。

    简化实现（免训练、纯特征空间）：可学习参数为 query logits 本身（等价 TIM 的
    对偶坐标下降的一阶版本）；rounds = 迭代上限（T 预算轴）。
    """
    if rounds <= 0:
        return p0_logits
    n_class = p0_logits.shape[1]
    sup_y = _onehot(support_labels, n_class)
    # support 上的类原型（固定），query logits 为优化变量
    protos = torch.stack([support_feats[support_labels == c].mean(dim=0)
                          for c in range(n_class)])
    protos = F.normalize(protos, dim=-1)
    z = p0_logits.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    sim = F.normalize(query_feats, dim=-1) @ protos.T / temperature  # [Q, C]
    for _ in range(rounds):
        opt.zero_grad()
        p = F.softmax(z, dim=1)
        # 条件熵（自信度）项：鼓励低熵
        cond_ent = -(p * (p + 1e-12).log()).sum(dim=1).mean()
        # 边际熵项：鼓励类均衡（α-散度近似，α=1 退化为标准 TIM 互信息）
        marginal = p.mean(dim=0)
        marg_ent = -(marginal * (marginal + 1e-12).log()).sum()
        # 与原型相似度一致性（弱 CE 锚点，防止漂移）
        anchor = F.cross_entropy(z, sim.argmax(dim=1))
        loss = cond_ent - alpha * marg_ent + 0.1 * anchor
        loss.backward()
        opt.step()
    return z.detach()


TRANSDUCTIVE_METHODS = ["laplacian_shot", "alpha_tim"]


def run_transductive(name: str, p0_logits: torch.Tensor, support_feats: torch.Tensor,
                     support_labels: torch.Tensor, query_feats: torch.Tensor,
                     rounds: int, **kw) -> torch.Tensor:
    """统一分发入口；rounds = T 预算轴。"""
    if name == "laplacian_shot":
        return laplacian_shot_logits(p0_logits, support_feats, support_labels, query_feats,
                                     rounds, lam=kw.get("lam", 0.5), knn=kw.get("knn", 10))
    if name == "alpha_tim":
        return alpha_tim_logits(p0_logits, support_feats, support_labels, query_feats,
                                rounds, alpha=kw.get("alpha", 1.0), lr=kw.get("lr", 0.1),
                                temperature=kw.get("temperature", 1.0))
    raise KeyError(f"未知传导方法: {name}（可选: {TRANSDUCTIVE_METHODS}）")


if __name__ == "__main__":
    # 单元验证：合成特征，T=0 恒等、T>0 不改形状且不应劣化可分簇
    torch.manual_seed(0)
    C, D, S, Q = 5, 64, 4, 75
    centers = F.normalize(torch.randn(C, D), dim=-1) * 3
    sup = F.normalize(centers.repeat_interleave(S, 0) + 0.3 * torch.randn(C * S, D), dim=-1)
    qry = F.normalize(centers.repeat_interleave(Q // C, 0) + 0.3 * torch.randn(Q, D), dim=-1)
    sup_y = torch.arange(C).repeat_interleave(S)
    qry_y = torch.arange(C).repeat_interleave(Q // C)
    from .cache_based import tip_adapter_logits

    p0 = tip_adapter_logits(qry, sup, sup_y, C, alpha=1.0, beta=5.5)
    acc0 = (p0.argmax(1) == qry_y).float().mean().item()

    # T=0 恒等
    z0 = laplacian_shot_logits(p0, sup, sup_y, qry, rounds=0)
    assert torch.equal(z0, p0), "T=0 应恒等"
    z1 = laplacian_shot_logits(p0, sup, sup_y, qry, rounds=5)
    assert z1.shape == (Q, C)
    acc1 = (z1.argmax(1) == qry_y).float().mean().item()
    assert acc1 >= acc0 - 0.05, f"LaplacianShot 显著劣化: {acc0}->{acc1}"
    # T 是预算轴：T=1 与 T=5 应不同
    z1a = laplacian_shot_logits(p0, sup, sup_y, qry, rounds=1)
    assert not torch.allclose(z1a, z1), "T 轴不生效"

    z2 = alpha_tim_logits(p0, sup, sup_y, qry, rounds=10)
    assert z2.shape == (Q, C) and torch.isfinite(z2).all()
    acc2 = (z2.argmax(1) == qry_y).float().mean().item()
    print(f"[transductive] 单元验证通过: p0={acc0:.3f} lapT5={acc1:.3f} atimT10={acc2:.3f}")
