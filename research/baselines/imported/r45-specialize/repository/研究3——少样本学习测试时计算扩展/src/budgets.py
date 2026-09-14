# -*- coding: utf-8 -*-
"""统一预算模型（实验计划 §1.1）：预算向量 b=(V,k,T,r) 的 FLOPs 折算与记账器。

FLOPs(q; b) = V · (r/224)² · F_bb  +  F_cache(k)  +  T · F_trans(Q)
- F_bb   : 主干前向（主导项），以 CLIP-B/16@224=17.5G 为记账单位 C0
- F_cache: k 条缓存余弦检索 = D 维 matmul，相对主干 <0.01%（近零成本轴）
- F_trans: 每轮 Q×(N·C) 级特征运算，episodic 场景可忽略（近零成本轴）
同时记录墙钟时间与峰值显存（iso-cost 口径）。
"""
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import C0_GFLOPS

# 各 backbone 在 224 输入下的单次前向 GFLOPs（公开报告量级，落地后可实测校准）
BACKBONE_GFLOPS_224: Dict[str, float] = {
    "clip_vitb16": 17.5,
    "clip_rn50": 4.1,
    "dinov2_vits14": 4.6,
    "dinov2_vitb14": 17.4,
}

BASE_RES = 224  # 分辨率折算基准


@dataclass(frozen=True)
class Budget:
    """一个预算点 b=(V,k,T,r)。k=-1 表示全量缓存。"""
    V: int = 1       # 每样本增广视图数
    k: int = -1      # 检索 top-k / 缓存子采样（-1=全量）
    T: int = 0       # 传导迭代轮数
    r: int = 224     # 输入分辨率

    def to_dict(self) -> Dict[str, int]:
        return {"V": self.V, "k": self.k, "T": self.T, "r": self.r}

    def tag(self) -> str:
        return f"V{self.V}_k{self.k}_T{self.T}_r{self.r}"


def backbone_flops_g(backbone: str, res: int = BASE_RES) -> float:
    """主干单次前向 GFLOPs，随分辨率平方缩放（ViT token 数 ∝ (r/patch)²）。"""
    if backbone not in BACKBONE_GFLOPS_224:
        raise KeyError(f"未知 backbone: {backbone}（可选: {list(BACKBONE_GFLOPS_224)}）")
    return BACKBONE_GFLOPS_224[backbone] * (res / BASE_RES) ** 2


def flops_g(
    budget: Budget,
    backbone: str = "clip_vitb16",
    n_cache: int = 0,
    feat_dim: int = 512,
    n_query: int = 75,
    n_support: int = 5,
    n_class: int = 5,
) -> float:
    """单 query 平均成本折算（GFLOPs）。

    视图聚合只计 query 侧 V 次前向（support 特征离线一次、跨 query 摊销≈0）。
    检索/传导为特征空间 matmul，量级远小于主干但如实记账。
    """
    f_forward = budget.V * backbone_flops_g(backbone, budget.r)
    k_eff = n_cache if budget.k in (-1, 0) else min(budget.k, n_cache)
    # 余弦检索：k_eff 次 D 维点积（乘加计 2 FLOPs）
    f_cache = 2.0 * k_eff * feat_dim / 1e9
    # 传导每轮：图传播 W@Z 为 n_all²×n_class 稀疏 matmul（不涉特征维）
    n_all = n_query + n_support
    f_trans = budget.T * 2.0 * n_all * n_all * n_class / 1e9
    return f_forward + f_cache + f_trans


def flops_in_c0(flops_g_value: float) -> float:
    """折算为记账单位 C0（= C0_GFLOPS GFLOPs）。"""
    return flops_g_value / C0_GFLOPS


class BudgetLedger:
    """记账器：累计 FLOPs / 前向次数 / 墙钟 / 峰值显存。

    用法：
        ledger = BudgetLedger("clip_vitb16")
        with ledger.measure(n_forwards=8, res=224):
            ... 推理 ...
        ledger.add_feature_ops(budget, ...)   # 特征空间运算单独记
        print(ledger.summary())
    """

    def __init__(self, backbone: str = "clip_vitb16"):
        self.backbone = backbone
        self.total_flops_g = 0.0
        self.total_forwards = 0
        self.wall_time_s = 0.0
        self.peak_mem_mb = 0.0
        self._t0: Optional[float] = None

    @staticmethod
    def _gpu_mem_mb() -> float:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() / 1024**2
        except Exception:
            pass
        return 0.0

    def measure(self, n_forwards: int, res: int = BASE_RES):
        """上下文管理器：记录一段含 n_forwards 次主干前向的区间。"""
        ledger = self

        class _Ctx:
            def __enter__(self):
                ledger._t0 = time.perf_counter()
                return self

            def __exit__(self, *exc):
                dt = time.perf_counter() - ledger._t0
                ledger.wall_time_s += dt
                ledger.total_forwards += n_forwards
                ledger.total_flops_g += n_forwards * backbone_flops_g(ledger.backbone, res)
                ledger.peak_mem_mb = max(ledger.peak_mem_mb, ledger._gpu_mem_mb())
                return False

        return _Ctx()

    def add_feature_ops(self, flops_g_value: float) -> None:
        """特征空间运算（检索/传导）单独记账。"""
        self.total_flops_g += flops_g_value

    def summary(self) -> Dict[str, float]:
        return {
            "backbone": self.backbone,
            "flops_g": round(self.total_flops_g, 4),
            "flops_c0": round(flops_in_c0(self.total_flops_g), 4),
            "forwards": self.total_forwards,
            "wall_time_s": round(self.wall_time_s, 3),
            "peak_mem_mb": round(self.peak_mem_mb, 1),
        }


if __name__ == "__main__":
    # 单元验证：折算表与记账器
    b = Budget(V=1, k=-1, T=0, r=224)
    f = flops_g(b, "clip_vitb16", n_cache=5, n_query=75)
    assert abs(f - 17.5) < 0.01, f"基档应≈1 C0，实得 {f}"
    assert abs(flops_in_c0(f) - 1.0) < 1e-3

    # V=2, r=448 → 2 × (448/224)² = 8 C0
    b2 = Budget(V=2, k=-1, T=0, r=448)
    f2 = flops_g(b2, "clip_vitb16")
    assert abs(flops_in_c0(f2) - 8.0) < 1e-6, f"实得 {flops_in_c0(f2)}"

    # k/T 为近零成本轴：k=64、T=50 相对主干 <0.1%
    b3 = Budget(V=1, k=64, T=50, r=224)
    f3 = flops_g(b3, "clip_vitb16", n_cache=64, n_query=75, n_support=5, n_class=5)
    extra = f3 - 17.5
    assert extra / 17.5 < 1e-3, f"k/T 轴应近零成本，实得 +{extra/17.5:.2%}"

    # 记账器
    led = BudgetLedger("clip_rn50")
    with led.measure(n_forwards=4, res=224):
        time.sleep(0.01)
    s = led.summary()
    assert abs(s["flops_g"] - 4 * 4.1) < 1e-6 and s["forwards"] == 4
    assert s["wall_time_s"] >= 0.01
    print("[budgets] 折算/记账单元验证全部通过:", s)
