# -*- coding: utf-8 -*-
"""难度分桶 × 固定预算分配器（anytime/budgeted-batch 主方法之一，Snell/DSC 式）。

按融合难度信号把 query 批内分 tertile（easy/mid/hard），每桶映射到固定档位
（如 easy→最低档、hard→最高档）。档位映射可来自配置或 LOO 标定结果。
"""
from typing import Sequence

import torch

from ..signals.difficulty import bucketize


class BucketAllocator:
    """分桶分配：tier_map[bucket] = 档位索引。"""

    def __init__(self, tier_map: Sequence[int], n_buckets: int = 3):
        assert len(tier_map) == n_buckets
        self.tier_map = list(tier_map)
        self.n_buckets = n_buckets

    def assign(self, signal: torch.Tensor) -> torch.Tensor:
        """signal [Q]（越大越难）→ 每 query 档位索引 [Q]。"""
        buckets = bucketize(signal, self.n_buckets)
        lut = torch.tensor(self.tier_map, dtype=torch.int64, device=signal.device)
        return lut[buckets]

    def mean_tier(self, signal: torch.Tensor) -> float:
        return float(self.assign(signal).float().mean().item())


if __name__ == "__main__":
    torch.manual_seed(0)
    sig = torch.rand(90)
    alloc = BucketAllocator([0, 2, 4], n_buckets=3)
    a = alloc.assign(sig)
    assert set(a.unique().tolist()) <= {0, 2, 4}
    # 难桶样本应比易桶拿到更高档
    hi = sig.topk(30).indices
    lo = sig.topk(30, largest=False).indices
    assert a[hi].float().mean() > a[lo].float().mean()
    print(f"[bucket] 单元验证通过: 档位分布={torch.bincount(a, minlength=5).tolist()}, mean={alloc.mean_tier(sig):.2f}")
