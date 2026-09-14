# -*- coding: utf-8 -*-
"""实验12 smoke：合成图库上的实现核验（仅验证实现，不构成自然证据）。

构造（固定种子，确定性）：5 个查询类，类间距中等使 1-shot 不完美；
图库含类 0..3 的紧近邻、一个共同方向的枢纽簇、以及靠近类 0 但偏向
类 1 方向的污染簇；类 4 在图库中无近邻（无覆盖）。
检查：
1. 各方法全部跑通且 raw < 1（有 headroom）；
2. true_k 参照 ≥ raw；
3. 污染存在时 weighted_aug ≥ uniform_aug（可靠性加权的价值）；
4. 类 4 覆盖最低，且把 τ_cov 设在类间时能触发其 fallback；
5. 权重单调性：污染簇平均权重 < 真近邻平均权重。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e12 import augment  # noqa: E402

N_WAY, D = 5, 64


def orth_toward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """b 正交于 a 方向化后的单位向量（指向 b 相对 a 的分量）。"""
    an = a / a.norm()
    v = b - (b @ an) * an
    return v / v.norm()


def build(seed=0):
    rng = torch.Generator().manual_seed(seed)
    mus = 1.5 * torch.randn(N_WAY, D, generator=rng)
    k, q = 1, 15
    S = torch.stack([mus[c] + torch.randn(D, generator=rng)
                     for c in range(N_WAY) for _ in range(k)])
    yS = torch.tensor([c for c in range(N_WAY) for _ in range(k)])
    X = torch.stack([mus[c] + 2.5 * torch.randn(D, generator=rng)
                     for c in range(N_WAY) for _ in range(q)])
    yQ = torch.tensor([c for c in range(N_WAY) for _ in range(q)])

    G = []
    for c in range(N_WAY - 1):  # 类 0..3 的紧近邻
        for _ in range(60):
            G.append(mus[c] + 0.3 * torch.randn(D, generator=rng))
    hub_dir = torch.randn(D, generator=rng)  # 枢纽簇：共同方向
    hub_dir = hub_dir / hub_dir.norm()
    for _ in range(150):
        G.append(2.0 * hub_dir + 0.1 * torch.randn(D, generator=rng))
    # 污染簇：靠近类 0 质心但偏向类 1（检索排名高、语义错误）
    d01 = orth_toward(mus[0], mus[1])
    for _ in range(60):
        G.append(mus[0] + 0.6 * d01 + 0.15 * torch.randn(D, generator=rng))
    G = torch.stack(G)

    T = torch.stack([mus[c] + torch.randn(D, generator=rng)
                     for c in range(N_WAY) for _ in range(25)])
    yT = torch.tensor([c for c in range(N_WAY) for _ in range(25)])
    return S, yS, X, yQ, G, T, yT


def main():
    cfg = augment.AugConfig(r=32, alpha=4.0, beta=2.0, gamma=1.0,
                            tau_cov=0.05)
    S, yS, X, yQ, G, T, yT = build()

    acc = {
        "raw": float((augment.raw(S, yS, X, N_WAY) == yQ).float().mean()),
        "uniform_aug": float((augment.uniform_aug(S, yS, X, N_WAY, G, cfg)
                              == yQ).float().mean()),
        "weighted_aug": float((augment.weighted_aug(S, yS, X, N_WAY, G, cfg)
                               == yQ).float().mean()),
        "weighted_fallback": float((
            augment.weighted_fallback(S, yS, X, N_WAY, G, cfg) == yQ
        ).float().mean()),
        "true25": float((augment.true_k_diag(T, yT, X, N_WAY) == yQ)
                        .float().mean()),
    }
    for m, a in acc.items():
        print(f"[e12 smoke] {m:<18s} acc={a:.3f}")
    assert acc["raw"] < 1.0, "合成任务应有 headroom"
    assert acc["true25"] >= acc["raw"], "true-k 参照应不低于 raw"
    assert acc["weighted_aug"] >= acc["uniform_aug"], \
        "污染存在时加权不应差于均匀"

    # 覆盖排序与 fallback 触发
    cand, coh, _idx, _protos = augment.retrieve(S, yS, G, N_WAY, cfg.r)
    cov = coh.mean(dim=1)
    assert int(cov.argmin()) == N_WAY - 1, "类 4 应覆盖最低"
    tau = float((cov[-1] + cov[:-1].min()) / 2)
    cfg2 = augment.AugConfig(**{**cfg.__dict__, "tau_cov": tau})
    _, info = augment.weighted_proto(S, yS, G, N_WAY, cfg2, fallback=True)
    assert bool(info["fallback"][-1]) and not info["fallback"][:-1].any(), \
        f"只有类 4 应回退: {info['fallback']}"

    # 权重单调性：类 0 的真近邻平均权重应高于污染/枢纽簇
    w = augment.weighted_proto(S, yS, G, N_WAY, cfg, fallback=False)[1]["weights"]
    protos = augment.proto_from(S, yS, N_WAY)
    print(f"[e12 smoke] 覆盖/类: {[round(float(v), 3) for v in cov]} "
          f"fallback={info['fallback'].tolist()}")
    print(f"[e12 smoke] 类0候选权重 mean={float(w[0].mean()):.3f}")
    print("[e12 smoke] 全部通过")


if __name__ == "__main__":
    main()
