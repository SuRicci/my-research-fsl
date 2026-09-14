# -*- coding: utf-8 -*-
"""动态 value-of-view 分配器原语（gonogo_protocol_v3_5 §4/§4b，十九轮）。

**受控特征数据流**：z 分量不由任意外部 derive 回调生成（可旁路审计）——
`read_view()` 的返回值是衍生特征的**唯一数据源**，z₁/z₂/z₄ 由本模块内固定
函数按 §4b 锁定公式计算，导出函数不接受 qid 或 store。

观测单位：视图 v 的观测 obs = {"posterior": [...], "nn_dist": float}
（posterior = 该视图后验概率向量；nn_dist = 与 support 集最近邻距离的代理）。

接口（协议锁定，v3.5）：
    QueryState(qid, current_tier, observed_views, features, posterior_history, observations)
    score_all_fn(states) -> {qid: ScoreResult}   # batch rank 口径（net_flip.score_all）
    observe_fn(state) -> meta  # 只读取新获准视图观测、固定导出 z、返回审计元数据
    allocator.allocate(states, score_all_fn, observe_fn, total_budget) -> trace

不变量（tests/test_dynamic_vov.py 固定）：
- 每升一个并列组 → observe → 立即重算全部 score（marginal-greedy 动态版）；
- store access log 是信息访问唯一通道：已读视图集合恒等于付费视图集合；
- 无外部 derive 入口；z 数值与「手工从 read_view 返回值重算」逐项一致；
- 并列组（score 容差 1e-12 同值）整体升级/整体放弃 → 按 qid 对齐的置换等变；
- 决策 trace 记录 qid/旧 tier/新 tier/可见特征键/score/Δcost/source/
  read_views/input_hash/feature_version；
- 全部 query×transition 尝试写入 last_audit（含未升级项），终态五类计数。
"""
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

TIER_VIEWS = (1, 2, 4, 8)                     # tier 索引 → 视图数
FEATURE_VERSION = "vov_z_v2"                  # §4b 公式版本（v3.3 的回调版为 v1，从未启用）


# ---------------------------------------------------------------- §4b 固定导出

def _margin(p: Sequence[float]) -> float:
    s = sorted(p, reverse=True)
    return s[0] - s[1] if len(s) > 1 else s[0]


def _entropy_norm(p: Sequence[float]) -> float:
    c = len(p)
    h = -sum(x * math.log2(x) for x in p if x > 0.0)
    return h / math.log2(c) if c > 1 else 0.0


def _jsd(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen–Shannon 散度（log 底 2，∈[0,1]）。"""
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def kl(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0.0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _cos(p: Sequence[float], q: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(p, q))
    np_ = math.sqrt(sum(a * a for a in p))
    nq = math.sqrt(sum(b * b for b in q))
    return dot / (np_ * nq) if np_ > 0 and nq > 0 else 0.0


def derive_z1(obs0: dict) -> List[float]:
    """z₁ = [margin(p₀), entropy(p₀)/log₂C, nn_dist₀]（one-view，3 维）。"""
    p = obs0["posterior"]
    return [_margin(p), _entropy_norm(p), float(obs0["nn_dist"])]


def derive_z2(obs0: dict, obs1: dict) -> List[float]:
    """z₂ = [JSD(p₀,p₁), nn₁−nn₀, cos(p₀,p₁)]（两视图摘要，3 维）。"""
    p0, p1 = obs0["posterior"], obs1["posterior"]
    return [_jsd(p0, p1), float(obs1["nn_dist"]) - float(obs0["nn_dist"]),
            _cos(p0, p1)]


def derive_z4(obs: Sequence[dict]) -> List[float]:
    """z₄ = [mean_{i<j} JSD, mean(nn₂,nn₃)−mean(nn₀,nn₁), mean_{i<j} cos]（4 视图）。"""
    assert len(obs) == 4, "z4 需要恰好 4 个已授权观测"
    ps = [o["posterior"] for o in obs]
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    mean_jsd = sum(_jsd(ps[i], ps[j]) for i, j in pairs) / len(pairs)
    mean_cos = sum(_cos(ps[i], ps[j]) for i, j in pairs) / len(pairs)
    dnn = (float(obs[2]["nn_dist"]) + float(obs[3]["nn_dist"])) / 2.0 \
        - (float(obs[0]["nn_dist"]) + float(obs[1]["nn_dist"])) / 2.0
    return [mean_jsd, dnn, mean_cos]


# ---------------------------------------------------------------- 状态与审计库

@dataclass
class QueryState:
    """逐 query 分配状态。features 只含已获准视图派生分量（键 "z1"/"z2"/"z4"）；
    observations 保存已授权原始观测（view 索引 → obs），是 z 的唯一来源。"""
    qid: str
    current_tier: int = 0
    observed_views: int = 1
    features: Dict[str, List[float]] = field(default_factory=dict)
    posterior_history: List[dict] = field(default_factory=list)
    observations: Dict[int, dict] = field(default_factory=dict)


class AuditedFeatureStore:
    """审计型特征库：信息访问唯一通道。每次读取记录 (qid, view)。

    feats[qid][view] = {"posterior": [...], "nn_dist": float}。
    **没有外部 derive 入口**——z 分量只能由本模块固定函数从 read_view
    返回值计算（v3.4 §4b，十八轮 P0-1：外部回调可闭包旁路审计）。
    """

    def __init__(self, feats: Dict[str, Dict[int, dict]]):
        self._feats = feats
        self.access_log: List = []   # (qid, view)

    def read_view(self, qid: str, view: int) -> dict:
        self.access_log.append((qid, view))
        return self._feats[qid][view]


def _hash_observations(new_views: Sequence[int], obs: Dict[int, dict]) -> str:
    payload = [[v, obs[v]["posterior"], obs[v]["nn_dist"]] for v in new_views]
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def make_init_fn(store: AuditedFeatureStore) -> Callable[[QueryState], dict]:
    """分配前初始化：读取 view0（已含在基档付费内）并固定导出 z1。"""

    def init(state: QueryState) -> dict:
        state.observations[0] = store.read_view(state.qid, 0)
        state.features["z1"] = derive_z1(state.observations[0])
        state.posterior_history.append({"tier": 0, "views": [0]})
        return {"read_views": [0],
                "input_hash": _hash_observations([0], state.observations),
                "feature_version": FEATURE_VERSION}

    return init


def make_observe_fn(store: AuditedFeatureStore) -> Callable[[QueryState], dict]:
    """升级 observe：只读取新获准视图，返回审计元数据（read_views/input_hash/
    feature_version）。升到 tier1（2 视图）导出 z2；升到 tier2（4 视图）导出 z4；
    升到 tier3（8 视图）无新 z 分量（无 z8 模型），仅记录付费读取。
    """

    def observe(state: QueryState) -> dict:
        new_tier = state.current_tier + 1
        target_views = TIER_VIEWS[new_tier]
        new_views = list(range(state.observed_views, target_views))
        for v in new_views:
            state.observations[v] = store.read_view(state.qid, v)
        if new_tier == 1:
            state.features["z2"] = derive_z2(state.observations[0],
                                             state.observations[1])
        elif new_tier == 2:
            state.features["z4"] = derive_z4([state.observations[v]
                                              for v in range(4)])
        state.posterior_history.append({"tier": new_tier, "views": new_views})
        return {"read_views": list(new_views),
                "input_hash": _hash_observations(new_views, state.observations),
                "feature_version": FEATURE_VERSION}

    return observe


# ---------------------------------------------------------------- 分配器

class DynamicVoVAllocator:
    """动态边际收益贪心：整批共享预算；score 随状态演化，每步全量重算。

    v3.5 起 score 走 **batch 接口**（P0-1：episode 内 rank 是批级运算）：
    `score_all_fn(states_subset) -> {qid: ScoreResult}`，由
    `net_flip.score_all` 或等价物提供；逐步 score 审计挂 `last_audit`。
    """

    def __init__(self, tier_costs: Sequence[float], tie_tol: float = 1e-12):
        self.costs = [float(c) for c in tier_costs]
        self.max_tier = len(self.costs) - 1
        self.tie_tol = tie_tol
        self.last_stats: Optional[dict] = None
        self.last_audit: Optional[dict] = None

    def allocate(self, states: List[QueryState],
                 score_all_fn: Callable[[List[QueryState]], Dict[str, object]],
                 observe_fn: Callable[[QueryState], dict],
                 total_budget: float) -> List[dict]:
        """执行分配，返回决策 trace（每升级一步一条，含 source）。

        score_all_fn 必须是纯函数（不读取新视图、不改状态），对传入的
        eligible 子集逐 query 返回 ScoreResult；observe_fn 在组成员 tier
        更新前调用（先取新视图观测，再记 tier），其返回的审计元数据原样
        并入 trace。所有 query×transition 尝试（含未升级项）写入
        `last_audit["attempts"]`；终态五类计数写入 `last_audit["counts"]`。
        """
        n = len(states)
        spent = self.costs[0] * n
        trace: List[dict] = []
        attempts: List[dict] = []
        if spent > total_budget:
            self._set_stats(total_budget, spent)
            self._set_audit(attempts, states, [False] * n)
            return trace

        blocked = [False] * n
        while True:
            eligible = [(i, s) for i, s in enumerate(states)
                        if not blocked[i] and s.current_tier < self.max_tier]
            if not eligible:
                break
            # 每步全量重算（动态：上一步 observe 可能改变了某些 state 的 score）
            results = score_all_fn([s for _, s in eligible])
            cand = []   # (score, idx)
            for i, s in eligible:
                res = results[s.qid]
                attempts.append({"qid": s.qid, "transition": s.current_tier,
                                 "source": res.source, "value": res.value})
                if res.value <= 0:
                    continue
                dc = self.costs[s.current_tier + 1] - self.costs[s.current_tier]
                if spent + dc > total_budget + 1e-9:
                    blocked[i] = True   # 预算只减不增 → 永久不可行
                else:
                    cand.append((res.value, i))
            if not cand:
                break
            cand.sort(key=lambda x: -x[0])
            upgraded = False
            j = 0
            while j < len(cand):
                g0 = cand[j][0]
                k = j
                while k < len(cand) and abs(cand[k][0] - g0) <= self.tie_tol:
                    k += 1
                grp = [i for _, i in cand[j:k]]
                dc_grp = sum(self.costs[states[i].current_tier + 1]
                             - self.costs[states[i].current_tier] for i in grp)
                if spent + dc_grp <= total_budget + 1e-9:
                    for i in grp:
                        s = states[i]
                        old_tier = s.current_tier
                        vis_keys = sorted(s.features.keys())
                        src = results[s.qid].source
                        meta = observe_fn(s) or {}   # 先取新获准视图观测（审计在内）
                        s.current_tier = old_tier + 1
                        s.observed_views = TIER_VIEWS[s.current_tier]
                        dc = self.costs[s.current_tier] - self.costs[old_tier]
                        spent += dc
                        trace.append({"qid": s.qid, "old_tier": old_tier,
                                      "new_tier": s.current_tier,
                                      "visible_features": vis_keys,
                                      "score": g0, "delta_cost": dc,
                                      "source": src,
                                      "read_views": meta.get("read_views", []),
                                      "input_hash": meta.get("input_hash"),
                                      "feature_version": meta.get("feature_version")})
                    upgraded = True
                    break   # 每升一组立即重算（十六轮 P0-1 语义，动态版）
                for i in grp:
                    blocked[i] = True           # 预算只会减少 → 永久封锁
                j = k
            if not upgraded:
                break

        self._set_stats(total_budget, spent)
        self._set_audit(attempts, states, blocked)
        return trace

    def _set_stats(self, total_budget: float, spent: float) -> None:
        self.last_stats = {"nominal_budget": float(total_budget),
                           "actual_cost": float(spent),
                           "budget_utilization": float(spent / total_budget)
                           if total_budget > 0 else None,
                           "unused_budget": float(total_budget - spent)}

    def _set_audit(self, attempts: List[dict], states: List[QueryState],
                   blocked: List[bool]) -> None:
        """终态五类计数（v3.5 P0-3；每 query 按最后一次 attempt 归类，互斥）：

        na_no_upgrade（模型 NA）> missing_no_upgrade（特征缺失）>
        nonpositive_utility（模型打分 ≤0）> budget_blocked（正值但预算不可行）
        > net_flip_model（正常参与分配）。
        """
        counts = {"net_flip_model": 0, "na_no_upgrade": 0,
                  "missing_no_upgrade": 0, "nonpositive_utility": 0,
                  "budget_blocked": 0}
        last: Dict[str, dict] = {}
        for a in attempts:
            last[a["qid"]] = a
        for i, s in enumerate(states):
            a = last.get(s.qid)
            if a is None:
                continue                    # 起步即 max_tier（当前配置不发生）
            if a["source"] == "na_no_upgrade":
                counts["na_no_upgrade"] += 1
            elif a["source"] == "missing_feature_no_upgrade":
                counts["missing_no_upgrade"] += 1
            elif a["value"] <= 0:
                counts["nonpositive_utility"] += 1
            elif blocked[i] and s.current_tier < self.max_tier:
                counts["budget_blocked"] += 1
            else:
                counts["net_flip_model"] += 1
        self.last_audit = {"attempts": attempts, "counts": counts}
