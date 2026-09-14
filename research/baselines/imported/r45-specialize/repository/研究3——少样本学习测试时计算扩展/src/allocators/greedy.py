# -*- coding: utf-8 -*-
"""边际收益贪心分配器（budgeted-batch 协议主方法，实验计划 §1.4，UAB 式）。

输入：难度信号、档位成本表（统一折算 FLOPs/C0）、整批总预算 B；
用 LOO 标定表估计每桶×每档期望增益 g(tier|bucket)，迭代执行
"单位预算增益最大"的升级，直到预算耗尽。
对照分配器（uniform/always-min/always-max/random/entropy-early-stop/oracle）
在 run_adaptive.py 中以本模块原语组装。
"""
from typing import Optional, Sequence

import torch

from ..signals.calibration import CalibrationTable
from ..signals.difficulty import bucketize


class GreedyAllocator:
    """边际收益贪心：整批 query 共享总预算 B（单位 C0）。"""

    def __init__(self, tier_costs: Sequence[float], max_tier: Optional[int] = None):
        """tier_costs: 每档相对基档的每 query 成本（C0 单位），升序。"""
        self.costs = [float(c) for c in tier_costs]
        self.max_tier = max_tier if max_tier is not None else len(self.costs) - 1
        # 最近一次 assign 的名义/实际预算（十六轮 P1-1；提前 return 时保持 None）
        self.last_stats: Optional[dict] = None

    def assign(self, signal: torch.Tensor, calib: CalibrationTable,
               total_budget: float, n_buckets: int = 3) -> torch.Tensor:
        """返回每 query 档位索引 [Q]。

        signal: 难度信号 [Q]（越大越难）；calib: LOO 标定表；
        total_budget: 整批总预算（C0，= 每 query 平均预算 × Q）。
        """
        n_q = len(signal)
        buckets = bucketize(signal, n_buckets)
        cur = torch.zeros(n_q, dtype=torch.int64)          # 全部置最低档
        base_cost = self.costs[0] * n_q
        spent = base_cost
        if base_cost > total_budget:
            self.last_stats = {"nominal_budget": float(total_budget),
                               "actual_cost": float(spent),
                               "budget_utilization": float(spent / total_budget) if total_budget > 0 else None,
                               "unused_budget": float(total_budget - spent)}
            return cur  # 预算不够最低档：全留 0 档（调用方应保证 B ≥ Q·c1）

        def marginal_gain(i: int) -> float:
            b = int(buckets[i].item())
            t = int(cur[i].item())
            if t >= self.max_tier:
                return -1.0
            dg = calib.gain(b, t + 1) - calib.gain(b, t)
            dc = self.costs[t + 1] - self.costs[t]
            return dg / max(dc, 1e-12)

        blocked = torch.zeros(n_q, dtype=torch.bool)   # 预算放不下的并列组（永久封锁）
        while True:
            gains = torch.tensor([marginal_gain(i) for i in range(n_q)])
            # 只考虑预算内可升级且未封锁的
            feasible = []
            for i in range(n_q):
                if blocked[i]:
                    continue
                t = int(cur[i].item())
                if t >= self.max_tier:
                    continue
                dc = self.costs[t + 1] - self.costs[t]
                if spent + dc <= total_budget + 1e-9:
                    feasible.append(i)
            if not feasible:
                break
            gi = gains[torch.tensor(feasible)]
            gmax = float(gi.max().item())
            if gmax <= 0:
                break  # 无正增益即停（预算可剩余）
            # 二十八轮（28号文）+ 十五轮评审（P0-1）决胜规则：
            # - 信号驱动标定（bucket_acc 非空）：**并列组整体升级或整体放弃**——
            #   同增益（同 bucket 同 tier）的 query 构成并列组，整组升一级若超出
            #   剩余预算则整组封锁（宁可留少量预算不用），再考虑次高增益组。
            #   组内不加任何位置/信号二次排序 → exact tie 下仍置换等变
            #   （组成员资格只随增益=query 身份走）。
            # - 无分桶标定（fixed_prior 退化负对照）：保持输入序 argmax 决胜，
            #   其顺序依赖性是受控对照的既定特征（见 gonogo_protocol_v3 §5）。
            if calib.bucket_acc:
                upgraded_any = False
                # 并列组按增益降序处理；exact tie 用 1e-12 容差聚类
                order = torch.argsort(gi, descending=True)
                sorted_g = gi[order]
                j = 0
                while j < len(order):
                    g0 = float(sorted_g[j].item())
                    k = j
                    while k < len(order) and abs(float(sorted_g[k].item()) - g0) <= 1e-12:
                        k += 1
                    grp = [feasible[int(order[x].item())] for x in range(j, k)]
                    if g0 <= 0:
                        break
                    dc_grp = sum(self.costs[int(cur[i].item()) + 1] - self.costs[int(cur[i].item())]
                                 for i in grp)
                    if spent + dc_grp <= total_budget + 1e-9:
                        for i in grp:
                            cur[i] += 1
                        spent += dc_grp
                        upgraded_any = True
                        # 十六轮 P0-1：每成功升级一个并列组立即退出组循环、
                        # 回到外层重算全部 marginal gain——升级会改变该组的下一档
                        # 收益，不重算会把预算先给当前次高组而非真正的边际最优
                        # （评审 3-query 反例：[0,1,1] 效用 0.15 vs 正确 [0,0,2] 0.28）
                        break
                    else:
                        for i in grp:
                            blocked[i] = True   # 预算只会减少 → 永久封锁该组
                    j = k
                if not upgraded_any:
                    break
            else:
                best = feasible[int(gi.argmax().item())]
                t = int(cur[best].item())
                spent += self.costs[t + 1] - self.costs[t]
                cur[best] = t + 1
        # 十六轮 P1-1：名义 vs 实际预算落盘（整组放弃/无正增益可致 utilization<1，
        # 比较时必须用实际成本，报告「上限 B、实际 x×」）
        self.last_stats = {"nominal_budget": float(total_budget),
                           "actual_cost": float(spent),
                           "budget_utilization": float(spent / total_budget) if total_budget > 0 else None,
                           "unused_budget": float(total_budget - spent)}
        return cur

    def expected_cost(self, assign: torch.Tensor) -> float:
        """分配结果的整批总成本（C0）。"""
        lut = torch.tensor(self.costs)
        return float(lut[assign].sum().item())


def uniform_assign(n_q: int, tier: int) -> torch.Tensor:
    """对照：均匀分配（同预算平铺到固定档）。"""
    return torch.full((n_q,), tier, dtype=torch.int64)


def random_assign(n_q: int, costs: Sequence[float], total_budget: float,
                  n_tiers: int, seed: int = 0) -> torch.Tensor:
    """对照：随机分配（同期望预算打乱档位，隔离信号真实贡献）。"""
    g = torch.Generator().manual_seed(seed)
    assign = torch.zeros(n_q, dtype=torch.int64)
    spent = costs[0] * n_q
    order = torch.randperm(n_q, generator=g)
    for i in order:
        while True:
            t = int(assign[i].item())
            if t >= n_tiers - 1:
                break
            if torch.rand(1, generator=g).item() < 0.5:  # 随机决定是否升档
                dc = costs[t + 1] - costs[t]
                if spent + dc <= total_budget:
                    assign[i] = t + 1
                    spent += dc
                    continue
            break
    return assign


if __name__ == "__main__":
    from ..signals.calibration import CalibrationTable

    torch.manual_seed(0)
    n_q = 60
    sig = torch.rand(n_q)
    # 构造标定表：难桶（bucket 2）升档增益大，易桶增益≈0
    calib = CalibrationTable(
        tier_acc=[0.6, 0.65, 0.68],
        bucket_acc=[[0.9, 0.9, 0.9], [0.6, 0.7, 0.75], [0.3, 0.5, 0.7]],
    )
    costs = [1.0, 2.0, 4.0]
    alloc = GreedyAllocator(costs, max_tier=2)
    a = alloc.assign(sig, calib, total_budget=1.5 * n_q, n_buckets=3)
    assert a.shape == (n_q,)
    # 预算约束必须满足
    assert alloc.expected_cost(a) <= 1.5 * n_q + 1e-6
    # 难样本应拿到更高档
    hi = sig.topk(20).indices
    lo = sig.topk(20, largest=False).indices
    assert a[hi].float().mean() > a[lo].float().mean(), "贪心未向难样本倾斜"
    # 随机对照成本不超预算
    r = random_assign(n_q, costs, 1.5 * n_q, 3, seed=0)
    lut = torch.tensor(costs)
    assert lut[r].sum() <= 1.5 * n_q + 1e-6
    print(f"[greedy] 单元验证通过: 成本={alloc.expected_cost(a):.1f}/{1.5*n_q:.0f}, "
          f"档位分布={torch.bincount(a, minlength=3).tolist()}")
