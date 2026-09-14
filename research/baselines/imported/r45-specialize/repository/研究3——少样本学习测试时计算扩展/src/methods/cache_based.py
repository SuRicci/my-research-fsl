# -*- coding: utf-8 -*-
"""缓存/度量类方法（P0 自实现，全部跑在缓存特征上，纯特征空间运算）。

zero-shot CLIP / kNN / SimpleShot / ProtoNet / Tip-Adapter / GDA。
统一接口：输入 L2 归一化特征与标签，输出 logits [Q, C]。
k 预算轴：Tip-Adapter/kNN 用 top-k 检索宽度；其余方法用缓存子采样等价。
"""
from typing import Optional

import torch
import torch.nn.functional as F


def _onehot(labels: torch.Tensor, n_class: int) -> torch.Tensor:
    return F.one_hot(labels.long(), n_class).float()


def zero_shot_logits(query_feats: torch.Tensor, text_feats: torch.Tensor,
                     logit_scale: float = 100.0) -> torch.Tensor:
    """CLIP zero-shot：logit_scale · (q · W_text.T)。text_feats [C, D] 需已归一化。"""
    return logit_scale * (query_feats @ text_feats.T)


def knn_logits(query_feats: torch.Tensor, support_feats: torch.Tensor,
               support_labels: torch.Tensor, n_class: int,
               k: int = -1, temperature: float = 1.0) -> torch.Tensor:
    """kNN（余弦）：k=-1 全量；softmax(sim/T) 加权 one-hot 投票。"""
    sim = query_feats @ support_feats.T                      # [Q, N]
    if 0 < k < sim.shape[1]:
        topv, topi = sim.topk(k, dim=1)
        mask = torch.zeros_like(sim).scatter_(1, topi, 1.0)
        sim = sim * mask - (1 - mask) * 1e9
    w = F.softmax(sim / temperature, dim=1)
    return w @ _onehot(support_labels, n_class)              # [Q, C]（对数前概率即 logits）


def simpleshot_logits(query_feats: torch.Tensor, support_feats: torch.Tensor,
                      support_labels: torch.Tensor, n_class: int) -> torch.Tensor:
    """SimpleShot/CLAMP：减 support 均值 → L2 → 最近类心（负平方欧氏距离为 logits）。"""
    mean = support_feats.mean(dim=0, keepdim=True)
    q = F.normalize(query_feats - mean, dim=-1)
    s = F.normalize(support_feats - mean, dim=-1)
    centroids = torch.stack([s[support_labels == c].mean(dim=0) for c in range(n_class)])
    centroids = F.normalize(centroids, dim=-1)
    return q @ centroids.T


def protonet_logits(query_feats: torch.Tensor, support_feats: torch.Tensor,
                    support_labels: torch.Tensor, n_class: int,
                    cosine: bool = True) -> torch.Tensor:
    """ProtoNet：类原型最近邻。"""
    protos = torch.stack([support_feats[support_labels == c].mean(dim=0) for c in range(n_class)])
    if cosine:
        return F.normalize(query_feats, dim=-1) @ F.normalize(protos, dim=-1).T
    return -torch.cdist(query_feats, protos) ** 2


def tip_adapter_logits(query_feats: torch.Tensor, support_feats: torch.Tensor,
                       support_labels: torch.Tensor, n_class: int,
                       zs_logits: Optional[torch.Tensor] = None,
                       alpha: float = 1.0, beta: float = 5.5, k: int = -1) -> torch.Tensor:
    """Tip-Adapter（免训练版）：logits = zs + alpha · exp(-beta(1-sim)) · L。

    k 轴 = 检索宽度：top-k 亲和之外的缓存条目置零（等价检索剪枝）。
    zs_logits 为 None 时退化为纯缓存后验（乘 logit 尺度保持量纲一致）。
    """
    affinity = query_feats @ support_feats.T                 # [Q, N]，特征已归一化
    if 0 < k < affinity.shape[1]:
        topv, topi = affinity.topk(k, dim=1)
        mask = torch.zeros_like(affinity).scatter_(1, topi, 1.0)
        affinity = affinity * mask - (1 - mask)              # 置 -1 → exp(-2β)≈0
    cache_logits = torch.exp(-beta * (1 - affinity)) @ _onehot(support_labels, n_class)
    if zs_logits is None:
        return alpha * cache_logits
    return zs_logits + alpha * cache_logits


def gda_logits(query_feats: torch.Tensor, support_feats: torch.Tensor,
               support_labels: torch.Tensor, n_class: int, reg: float = 1e-4) -> torch.Tensor:
    """GDA（共享协方差高斯判别 = LDA 闭式解），Tip-Adapter 谱系的判别式代表。"""
    D = support_feats.shape[1]
    mus = torch.stack([support_feats[support_labels == c].mean(dim=0) for c in range(n_class)])
    centered = support_feats - mus[support_labels]
    cov = centered.T @ centered / max(1, support_feats.shape[0] - 1)
    cov = cov + reg * torch.eye(D, device=cov.device)
    prec = torch.linalg.pinv(cov)
    # log N(x; mu_c, Σ) ∝ -0.5 (x-μ)ᵀ Σ⁻¹ (x-μ) + log π_c
    pi = torch.bincount(support_labels, minlength=n_class).float() / support_labels.shape[0]
    xm = query_feats.unsqueeze(1) - mus.unsqueeze(0)         # [Q, C, D]
    maha = torch.einsum("qcd,de,qce->qc", xm, prec, xm)
    return -0.5 * maha + torch.log(pi + 1e-12).unsqueeze(0)


def subsample_cache(support_feats: torch.Tensor, support_labels: torch.Tensor,
                    keep_per_class: int, seed: int = 0):
    """k 轴的另一种形态：每类缓存子采样（缓存规模档位）。"""
    g = torch.Generator().manual_seed(seed)
    keep = []
    for c in range(int(support_labels.max().item()) + 1):
        idx = torch.nonzero(support_labels == c).squeeze(-1)
        if len(idx) > keep_per_class:
            idx = idx[torch.randperm(len(idx), generator=g)[:keep_per_class]]
        keep.append(idx)
    sel = torch.cat(keep)
    return support_feats[sel], support_labels[sel]


METHODS = ["zero_shot", "knn", "simpleshot", "protonet", "tip_adapter", "gda"]


def run_method(name: str, query_feats: torch.Tensor, support_feats: torch.Tensor,
               support_labels: torch.Tensor, n_class: int, k: int = -1,
               zs_logits: Optional[torch.Tensor] = None, **kw) -> torch.Tensor:
    """统一分发入口（供 run/ 脚本按名字调用）。"""
    if name == "zero_shot":
        if zs_logits is None:
            raise ValueError("zero_shot 需要 zs_logits（文本先验）")
        return zs_logits
    if name == "knn":
        return knn_logits(query_feats, support_feats, support_labels, n_class, k=k,
                          temperature=kw.get("temperature", 1.0))
    if name == "simpleshot":
        return simpleshot_logits(query_feats, support_feats, support_labels, n_class)
    if name == "protonet":
        return protonet_logits(query_feats, support_feats, support_labels, n_class)
    if name == "tip_adapter":
        return tip_adapter_logits(query_feats, support_feats, support_labels, n_class,
                                  zs_logits=zs_logits, alpha=kw.get("alpha", 1.0),
                                  beta=kw.get("beta", 5.5), k=k)
    if name == "gda":
        return gda_logits(query_feats, support_feats, support_labels, n_class,
                          reg=kw.get("reg", 1e-4))
    raise KeyError(f"未知方法: {name}（可选: {METHODS}）")


if __name__ == "__main__":
    # 单元验证：合成特征上各方法形状/合理性（两类可分高斯簇应接近满分）
    torch.manual_seed(0)
    C, D, S, Q = 5, 64, 4, 60
    centers = F.normalize(torch.randn(C, D), dim=-1) * 3
    sup = F.normalize(centers.repeat_interleave(S, 0) + 0.3 * torch.randn(C * S, D), dim=-1)
    qry = F.normalize(centers.repeat_interleave(Q // C, 0) + 0.3 * torch.randn(Q, D), dim=-1)
    sup_y = torch.arange(C).repeat_interleave(S)
    qry_y = torch.arange(C).repeat_interleave(Q // C)

    for name in METHODS:
        if name == "zero_shot":
            zs = zero_shot_logits(qry, F.normalize(centers, dim=-1))
            logits = run_method(name, qry, sup, sup_y, C, zs_logits=zs)
        else:
            logits = run_method(name, qry, sup, sup_y, C, k=4)
        assert logits.shape == (Q, C), f"{name} 形状错误"
        acc = (logits.argmax(1) == qry_y).float().mean().item()
        assert acc > 0.9, f"{name} 在可分簇上精度异常: {acc}"
        print(f"[cache_based] {name}: acc={acc:.3f} (shape OK)")
    # top-k 检索宽度：k=1 与全量结果应不同（预算轴生效）
    l1 = tip_adapter_logits(qry, sup, sup_y, C, alpha=1.0, beta=5.5, k=1)
    l2 = tip_adapter_logits(qry, sup, sup_y, C, alpha=1.0, beta=5.5, k=-1)
    assert not torch.allclose(l1, l2)
    print("[cache_based] top-k 检索宽度生效 OK")
