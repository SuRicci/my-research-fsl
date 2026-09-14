# -*- coding: utf-8 -*-
"""检索增强方法族：同一冻结特征、同一查询集，只改支持集的构造。

方法（raw 之外都共享固定的检索深度 r，计入成本）：
- raw：纯支持集原型（协议基线）；
- uniform_aug：每类取图库 top-r 近邻（对该类支持质心的余弦），均匀并入原型；
- weighted_aug：候选按闭式可靠性加权后并入：
    w = sigmoid(α·margin + β·(coh − coh_mean)) · 1/(1 + γ·hub)
  其中 coh=与该类支持质心的余弦，margin=coh − 他类质心最大余弦，
  hub=候选在图库中自身近邻密度（枢纽代理，越高越可疑）；
  (α,β,γ,r) 只在开发 episode 上选定，评估集冻结；
- weighted_fallback：weighted_aug + 低覆盖类回退纯支持集
  （该类 top-r 平均 coh < τ_cov 时回退，τ_cov 同样只在开发集选定）；
- true25_diag：同类目真 25-shot 参照（超预算诊断，不进主表排名）。

污染口径：miniIN/CIFAR-FS 图库与测试类不相交（污染=语义他类近邻）；
DTD 图库与测试类共享（可直接测真标签污染率）。
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@dataclass
class AugConfig:
    r: int = 32
    alpha: float = 4.0
    beta: float = 2.0
    gamma: float = 1.0
    tau_cov: float = 0.05
    hub_k: int = 16


def proto_from(S: torch.Tensor, y: torch.Tensor, n_way: int) -> torch.Tensor:
    S = _normalize(S)
    mus = []
    for c in range(n_way):
        m = S[y == c]
        mus.append(m.mean(dim=0) if len(m) else torch.zeros(S.shape[1]))
    return _normalize(torch.stack(mus))


def predict(protos: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return (_normalize(X) @ protos.T).argmax(dim=1)


def raw(S, yS, X, n_way, G=None, cfg=None):
    return predict(proto_from(S, yS, n_way), X)


def true_k_diag(S_big, yS_big, X, n_way):
    """超预算参照：直接使用更多同类目标注（诊断 headroom，不进主表）。"""
    return predict(proto_from(S_big, yS_big, n_way), X)


def retrieve(S: torch.Tensor, yS: torch.Tensor, G: torch.Tensor,
             n_way: int, r: int):
    """每类按支持质心取图库 top-r；返回 (cand[N,r,D], coh[N,r], idx[N,r], protos)。"""
    protos = proto_from(S, yS, n_way)
    Gn = _normalize(G)
    sim = protos @ Gn.T                      # [N, G]
    r_eff = min(r, G.shape[0])
    top = sim.topk(r_eff, dim=1)
    cand = Gn[top.indices]                   # [N, r, D]
    return cand, top.values, top.indices, protos


def precompute_hub(G: torch.Tensor, k: int, chunk: int = 4096,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu"
                   ) -> torch.Tensor:
    """全图库逐点的 kNN 平均相似度（hubness 代理），一次性计算 [G]。
    结果只依赖图库与编码器，不依赖 episode——可预算一次后复用。"""
    Gn = _normalize(G).to(device)
    n = Gn.shape[0]
    k_eff = min(k + 1, n)  # 含自身，稍后剔除
    out = torch.zeros(n)
    for st in range(0, n, chunk):
        sim = Gn[st:st + chunk] @ Gn.T       # [c, G]
        top = sim.topk(k_eff, dim=1).values
        out[st:st + chunk] = (top.sum(dim=1) - 1.0).cpu() / (k_eff - 1)
    return out


def hubness(Gn: torch.Tensor, cand: torch.Tensor, k: int) -> torch.Tensor:
    """候选在图库内的近邻密度（kNN 平均相似度），[n_way, r]。大=枢纽。
    逐 episode 重算的慢路径；正式跑用 precompute_hub + 索引。"""
    k_eff = min(k, Gn.shape[0])
    n_way, r, _ = cand.shape
    out = []
    for c in range(n_way):
        sim = cand[c] @ Gn.T                 # [r, G]
        out.append(sim.topk(k_eff, dim=1).values.mean(dim=1))
    return torch.stack(out)


def weighted_proto(S, yS, G, n_way, cfg: AugConfig,
                   fallback: bool, hub_vec: torch.Tensor = None
                   ) -> Tuple[torch.Tensor, Dict]:
    cand, coh, idx, protos = retrieve(S, yS, G, n_way, cfg.r)
    n_way_, r_eff, _ = cand.shape
    # margin：与自有质心的余弦 − 与他类质心的最大余弦
    margins = torch.zeros(n_way_, r_eff)
    for c in range(n_way_):
        s = cand[c] @ protos.T                   # [r, N_way]
        other = torch.cat([s[:, :c], s[:, c + 1:]], dim=1).max(dim=1).values \
            if n_way_ > 1 else torch.zeros(r_eff)
        margins[c] = s[:, c] - other
    if hub_vec is None:
        hub = hubness(_normalize(G), cand, cfg.hub_k)
    else:
        hub = hub_vec[idx]
    w = torch.sigmoid(cfg.alpha * margins +
                      cfg.beta * (coh - coh.mean(dim=1, keepdim=True)))
    w = w / (1.0 + cfg.gamma * hub)
    used_fallback = torch.zeros(n_way_, dtype=torch.bool)
    new_protos = []
    for c in range(n_way_):
        sc = _normalize(S[yS == c])
        cov = float(coh[c].mean())
        if fallback and cov < cfg.tau_cov:
            new_protos.append(protos[c])
            used_fallback[c] = True
            continue
        num = sc.sum(dim=0) + (w[c].unsqueeze(1) * cand[c]).sum(dim=0)
        new_protos.append(num)
    return _normalize(torch.stack(new_protos)), \
        {"weights": w, "coh": coh, "hub": hub, "fallback": used_fallback}


def uniform_aug(S, yS, X, n_way, G, cfg: AugConfig, hub_vec=None):
    cand, _coh, _idx, _p = retrieve(S, yS, G, n_way, cfg.r)
    protos = []
    for c in range(n_way):
        sc = _normalize(S[yS == c])
        protos.append(torch.cat([sc, cand[c]]).mean(dim=0))
    return predict(_normalize(torch.stack(protos)), X)


def weighted_aug(S, yS, X, n_way, G, cfg: AugConfig, hub_vec=None):
    protos, _ = weighted_proto(S, yS, G, n_way, cfg, fallback=False,
                               hub_vec=hub_vec)
    return predict(protos, X)


def weighted_fallback(S, yS, X, n_way, G, cfg: AugConfig, hub_vec=None):
    protos, _info = weighted_proto(S, yS, G, n_way, cfg, fallback=True,
                                   hub_vec=hub_vec)
    return predict(protos, X)


def diagnostics(S, yS, n_way, G, cfg: AugConfig, hub_vec=None) -> Dict:
    """供 runner 记录权重/覆盖/回退诊断（不影响预测）。"""
    _, info = weighted_proto(S, yS, G, n_way, cfg, fallback=True,
                             hub_vec=hub_vec)
    return {"fallback_rate": float(info["fallback"].float().mean()),
            "mean_coh": float(info["coh"].mean()),
            "mean_hub": float(info["hub"].mean())}


METHODS = ("raw", "uniform_aug", "weighted_aug", "weighted_fallback")
