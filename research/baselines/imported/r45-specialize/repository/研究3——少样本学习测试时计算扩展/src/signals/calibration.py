# -*- coding: utf-8 -*-
"""免标签标定（实验计划 §1.4）：support 留一标定 / 分位数标定。

用途：估计"每桶 × 每档"的期望增益 g(tier | bucket)，供分配器在不接触
query 标签的情况下决策。
- LOO 标定：support 集上留一交叉验证，用同样管线逐档评估 → 每档的期望精度曲线；
  再按 query 的难度信号分桶，用 support 信号的分桶边界近似（尺度对齐靠批内秩化）。
- 分位数标定：直接按 query 批内信号分位数定阈值（更粗糙但更稳）。
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .difficulty import bucketize


@dataclass
class CalibrationTable:
    """标定结果：每个档位 tier 的期望精度（整体与分桶）。

    n_valid/n_skipped 记录 LOO 有效/跳过样本数（1-shot 等退化场景诊断用）；
    method ∈ {"loo", "quantile", "loo_degenerate"}。
    """
    tier_acc: List[float]                      # [n_tiers] 留一平均精度
    bucket_acc: List[List[float]] = field(default_factory=list)  # [n_buckets][n_tiers]
    bucket_edges: List[float] = field(default_factory=list)      # 信号分桶边界
    method: str = "loo"
    n_valid: int = -1                          # LOO 有效样本数（-1 = 非 LOO）
    n_skipped: int = 0                         # LOO 因空类跳过的样本数

    def gain(self, bucket: int, tier: int) -> float:
        """g(tier | bucket)；无分桶信息时退化为整体精度。"""
        if self.bucket_acc:
            return self.bucket_acc[bucket][tier]
        return self.tier_acc[tier]

    def n_tiers(self) -> int:
        return len(self.tier_acc)


def loo_calibrate(support_feats: torch.Tensor, support_labels: torch.Tensor,
                  infer_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor],
                  tiers: Sequence[int], signal_fn: Callable[[torch.Tensor], torch.Tensor],
                  n_buckets: int = 3, min_valid: int = 1,
                  fallback_tier_logits: Optional[Sequence[torch.Tensor]] = None) -> CalibrationTable:
    """support 留一标定。

    infer_fn(support_feats_loo, support_labels_loo, query_feats_held, tier) -> logits [1, C]
        对每个档位 tier 评估留一样本；signal_fn 从 logits 提取难度信号。
    返回 CalibrationTable（tier_acc = 各档留一精度）。

    空类跳过保护：1-shot 下留一会使该类支撑为空（推理结构性不可能），
    此类样本直接跳过不计入统计（否则空类均值产生 NaN/系统性偏差）。
    有效样本 < min_valid（如纯 1-shot 全部被跳过）时 LOO 结构性退化：
    提供 fallback_tier_logits（query 批各档 logits）则回退到分位数标定，
    否则返回全零增益的退化表（分配器不升档，安全）。
    """
    n = support_feats.shape[0]
    correct = {t: [] for t in tiers}
    sig_by_tier: Dict[int, List[float]] = {t: [] for t in tiers}
    n_skipped = 0
    for i in range(n):
        keep = torch.arange(n) != i
        sf, sy = support_feats[keep], support_labels[keep]
        held = int(support_labels[i].item())
        if int((sy == held).sum().item()) == 0:
            n_skipped += 1  # 留一后该类无支撑 → 跳过（空类保护）
            continue
        qf = support_feats[i:i + 1]
        for t in tiers:
            logits = infer_fn(sf, sy, qf, t)
            pred = int(logits.argmax(1).item())
            correct[t].append(pred == held)
            sig_by_tier[t].append(float(signal_fn(logits).item()))
    n_valid = n - n_skipped
    if n_valid < min_valid:
        if fallback_tier_logits is not None:
            table = quantile_calibrate(fallback_tier_logits, n_buckets=n_buckets)
            table.n_valid, table.n_skipped = n_valid, n_skipped
            return table
        return CalibrationTable(tier_acc=[0.0 for _ in tiers],
                                method="loo_degenerate",
                                n_valid=n_valid, n_skipped=n_skipped)
    tier_acc = [float(np.mean(correct[t])) for t in tiers]

    # 分桶精度：用最低档信号分桶（桶定义与在线 query 侧一致）
    base_sig = torch.tensor(sig_by_tier[tiers[0]])
    buckets = bucketize(base_sig, n_buckets).tolist()
    bucket_acc = []
    for b in range(n_buckets):
        row = []
        for t in tiers:
            vals = [c for c, bb in zip(correct[t], buckets) if bb == b]
            row.append(float(np.mean(vals)) if vals else float(np.mean(correct[t])))
        bucket_acc.append(row)
    edges = torch.quantile(base_sig, torch.linspace(0, 1, n_buckets + 1)).tolist()
    return CalibrationTable(tier_acc=tier_acc, bucket_acc=bucket_acc,
                            bucket_edges=edges[1:-1], method="loo",
                            n_valid=n_valid, n_skipped=n_skipped)


def quantile_calibrate(per_tier_logits: Sequence[torch.Tensor],
                       n_buckets: int = 3) -> CalibrationTable:
    """分位数标定（免标签回退）：query 批信号分位数定桶边界，
    用各档平均置信度（softmax 最大概率）作为精度的免标签代理。

    用于 LOO 结构性不可用的场景（如 1-shot 留一后空类）。代理增益可为负
    （高档位置信度反而下降时），贪心分配器遇非正增益即停，行为安全。
    """
    from .difficulty import s2_top2_margin

    tiers_logits = [l for l in per_tier_logits]
    probs = [torch.softmax(l, dim=1) for l in tiers_logits]
    conf = [p.max(dim=1).values for p in probs]           # 每档逐样本置信度
    base_sig = s2_top2_margin(probs[0])                    # 难度信号（最低档）
    edges = torch.quantile(base_sig, torch.linspace(0, 1, n_buckets + 1)).tolist()
    buckets = bucketize(base_sig, n_buckets).tolist()
    tier_acc = [float(c.mean().item()) for c in conf]
    bucket_acc = []
    for b in range(n_buckets):
        idx = [i for i, bb in enumerate(buckets) if bb == b]
        row = [float(c[idx].mean().item()) if idx else tier_acc[t]
               for t, c in enumerate(conf)]
        bucket_acc.append(row)
    return CalibrationTable(tier_acc=tier_acc, bucket_acc=bucket_acc,
                            bucket_edges=edges[1:-1], method="quantile")


def quantile_thresholds(signal: torch.Tensor, tau_quantile: float = 0.67) -> float:
    """分位数标定：阈值级联用（免标签）。"""
    return float(torch.quantile(signal, tau_quantile).item())


if __name__ == "__main__":
    # 单元验证：合成可分簇上 LOO 标定应给出高精度且不崩溃
    import torch.nn.functional as F

    torch.manual_seed(0)
    C, D, S = 5, 64, 6
    centers = F.normalize(torch.randn(C, D), dim=-1) * 3
    sup = F.normalize(centers.repeat_interleave(S, 0) + 0.3 * torch.randn(C * S, D), dim=-1)
    sup_y = torch.arange(C).repeat_interleave(S)

    from ..methods.cache_based import protonet_logits
    from .difficulty import s2_top2_margin

    def infer(sf, sy, qf, tier):
        # tier 假装是"迭代档"：这里用温度扰动模拟档位差异
        return protonet_logits(qf, sf, sy, C) / (1.0 + 0.1 * tier)

    def sig(logits):
        return s2_top2_margin(F.softmax(logits, dim=1))[0]

    table = loo_calibrate(sup, sup_y, infer, tiers=[0, 1, 2], signal_fn=sig, n_buckets=3)
    assert table.n_tiers() == 3 and len(table.bucket_acc) == 3
    assert all(0.0 <= a <= 1.0 for a in table.tier_acc)
    assert table.tier_acc[0] > 0.8, f"可分簇上 LOO 精度异常: {table.tier_acc}"
    assert table.method == "loo" and table.n_valid == C * S and table.n_skipped == 0
    th = quantile_thresholds(torch.rand(100), 0.67)
    assert 0.0 < th < 1.0
    print(f"[calibration] LOO 验证通过: tier_acc={table.tier_acc}, edges={[round(e,3) for e in table.bucket_edges]}")

    # 1-shot 退化：每类仅 1 样本，留一后该类空支撑 → 全部跳过，回退/退化
    sup1 = F.normalize(centers + 0.05 * torch.randn(C, D), dim=-1)   # [C, D]
    sup1_y = torch.arange(C)
    q_logits = [torch.randn(30, C) * s for s in (1.0, 1.3, 1.8)]  # 档位↑置信度↑
    t1 = loo_calibrate(sup1, sup1_y, infer, tiers=[0, 1, 2], signal_fn=sig)
    assert t1.method == "loo_degenerate" and t1.n_valid == 0 and t1.n_skipped == C
    assert all(a == 0.0 for a in t1.tier_acc), "退化表应为零增益（不升档，安全）"
    t2 = loo_calibrate(sup1, sup1_y, infer, tiers=[0, 1, 2], signal_fn=sig,
                       fallback_tier_logits=q_logits)
    assert t2.method == "quantile" and t2.n_skipped == C and len(t2.bucket_acc) == 3
    assert torch.isfinite(torch.tensor(t2.tier_acc)).all(), "分位数回退含非法值"
    # 部分退化（2-shot 中混 1-shot 类）：仅跳过空类样本，其余正常统计
    sup2 = torch.cat([sup, sup1[:1]], dim=0)
    sup2_y = torch.cat([sup_y, torch.tensor([C])])                    # 新增 1-shot 类
    t3 = loo_calibrate(sup2, sup2_y, infer, tiers=[0, 1], signal_fn=sig)
    assert t3.method == "loo" and t3.n_skipped == 1 and t3.n_valid == C * S
    print(f"[calibration] 1-shot 退化/回退验证通过: degenerate n_skipped={t1.n_skipped}, "
          f"quantile tier_conf={[round(a,3) for a in t2.tier_acc]}, 部分退化 n_valid={t3.n_valid}")
