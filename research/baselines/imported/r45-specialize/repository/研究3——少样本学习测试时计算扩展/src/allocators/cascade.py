# -*- coding: utf-8 -*-
"""阈值级联分配器（anytime 协议 baseline，Shallow-Deep 式）。

逐 query：先在最低档推理并取难度信号；信号低于阈值 τ（免标签分位数标定）即输出，
否则升到下一档重估，直到顶档。返回每个 query 的最终档位索引。
"""
from typing import List, Sequence

import torch

from ..signals.calibration import quantile_thresholds


class CascadeAllocator:
    """阈值级联：tau_quantile 决定早停比例（越大越多样本早停、越省预算）。"""

    def __init__(self, n_tiers: int, tau_quantile: float = 0.67):
        assert n_tiers >= 2
        self.n_tiers = n_tiers
        self.tau_quantile = tau_quantile

    def assign(self, signals_per_tier: Sequence[torch.Tensor]) -> torch.Tensor:
        """输入各档推理后得到的难度信号列表（第 i 档对全体 query 的信号 [Q]）。

        简化离线版：假设各档信号已算好（特征缓存使得逐档重估廉价）。
        返回每 query 最终档位索引 [Q]（int64）。
        """
        n_q = len(signals_per_tier[0])
        assign = torch.full((n_q,), self.n_tiers - 1, dtype=torch.int64)
        alive = torch.ones(n_q, dtype=torch.bool)
        for t in range(self.n_tiers - 1):
            sig = signals_per_tier[t]
            tau = quantile_thresholds(sig[alive], self.tau_quantile) if alive.any() else 0.0
            stop = alive & (sig <= tau)     # 够"易"→ 停在该档
            assign[stop] = t
            alive = alive & ~stop
            if not alive.any():
                break
        return assign


if __name__ == "__main__":
    torch.manual_seed(0)
    # 构造 3 档信号：低档对一半样本足够自信
    sig_easy = torch.rand(50) * 0.3
    sig_hard = 0.7 + torch.rand(50) * 0.3
    s0 = torch.cat([sig_easy, sig_hard])
    s1 = torch.rand(100) * 0.5
    s2 = torch.rand(100) * 0.5
    alloc = CascadeAllocator(n_tiers=3, tau_quantile=0.67)
    a = alloc.assign([s0, s1, s2])
    assert a.shape == (100,) and a.min() >= 0 and a.max() <= 2
    # 易样本应大量停在 0 档
    assert (a[:50] == 0).float().mean() > 0.5
    print(f"[cascade] 单元验证通过: 档位分布={torch.bincount(a, minlength=3).tolist()}")
