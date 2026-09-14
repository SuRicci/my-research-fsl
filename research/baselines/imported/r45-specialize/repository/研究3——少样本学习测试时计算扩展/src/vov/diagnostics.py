# -*- coding: utf-8 -*-
"""v3.5 eval 诊断（二十轮 R3-P0-9/11）：逐 tier 聚合预测、逐 transition 信号
（AUROC/Spearman/BSS/覆盖率/事件率/校准，**episode 内 rank 口径**）、
matched-cost oracle gap、policy-reachable vs all-query 分布。

所有函数只消费 query log 行与已加载模型，不接触特征缓存。
"""
import math
from typing import Dict, List, Optional, Sequence

from ..allocators.dynamic_vov import TIER_VIEWS
from ..allocators.net_flip import (CLASSES, MIN_CLASS_COUNT, ModelStatus,
                                   NetFlipTransitionModel,
                                   average_rank_columns)
from .pipeline import build_transition_samples


# ---------------------------------------------------------------- 逐 tier 预测

def per_tier_predictions(rows: Sequence[dict]) -> Dict[str, Dict[int, int]]:
    """每 query × 每 tier 的均值后验 argmax 预测：{qid: {tier: pred}}。"""
    by_q: Dict[str, Dict[int, dict]] = {}
    for r in rows:
        by_q.setdefault(r["qid"], {})[r["view"]] = r
    out: Dict[str, Dict[int, int]] = {}
    for qid, bv in by_q.items():
        C = len(bv[0]["posterior"])
        preds: Dict[int, int] = {}
        for tier, V in enumerate(TIER_VIEWS):
            if not all(v in bv for v in range(V)):
                continue
            mean_p = [sum(bv[v]["posterior"][c] for v in range(V)) / V
                      for c in range(C)]
            preds[tier] = int(max(range(C), key=lambda c: mean_p[c]))
        out[qid] = preds
    return out


# ---------------------------------------------------------------- 逐 transition 信号

def _auroc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    from scipy.stats import spearmanr

    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(spearmanr(x, y).statistic)


def _calibration_bins(p_plus: Sequence[float], y: Sequence[int],
                      n_bins: int = 10) -> List[dict]:
    """P(+1|z) 校准：分箱均值预测 vs 经验 +1 率。"""
    bins = [{"lo": i / n_bins, "hi": (i + 1) / n_bins, "n": 0,
             "mean_pred": None, "empirical": None} for i in range(n_bins)]
    acc_p = [0.0] * n_bins
    acc_y = [0.0] * n_bins
    for p, t in zip(p_plus, y):
        b = min(int(p * n_bins), n_bins - 1)
        bins[b]["n"] += 1
        acc_p[b] += p
        acc_y[b] += 1.0 if t == 1 else 0.0
    for i, b in enumerate(bins):
        if b["n"]:
            b["mean_pred"] = acc_p[i] / b["n"]
            b["empirical"] = acc_y[i] / b["n"]
    return bins


def transition_diagnostics(rows: Sequence[dict],
                           models: Dict[int, NetFlipTransitionModel]
                           ) -> Dict[str, dict]:
    """逐 transition 信号诊断（all-query 口径样本 + episode 内 rank）：

    对每个 transition t：status、n、事件率（y=−1/0/+1 经验频率）、
    P(+1)/P(−1) 区分度 AUROC、utility−y 的 Spearman、BSS（参考 = 模型携带的
    训练折经验先验）、Brier、log-loss、P(+1) 校准曲线、覆盖率。
    NA 模型按协议不作信号门槛判定（只落 status/事件率）。
    """
    samples = build_transition_samples(rows)
    n_query = len({r["qid"] for r in rows})
    out: Dict[str, dict] = {}
    for t in (0, 1, 2):
        sel = [s for s in samples if s["transition"] == t]
        m = models.get(t)
        entry: Dict[str, object] = {
            "status": (m.status if m is not None else "missing"),
            "n_samples": len(sel),
            "sample_coverage": (len(sel) / n_query) if n_query else None,
        }
        if sel:
            n = len(sel)
            entry["event_rates"] = {
                str(c): sum(1 for s in sel if s["y"] == c) / n
                for c in CLASSES}
        if m is None or m.status != ModelStatus.READY or not sel:
            out[str(t)] = entry                     # NA 格不判定（协议 §6a）
            continue
        # episode 内 rank（评估口径；与训练 causal 快照同为 episode 内变换）
        by_ep: Dict[str, List[dict]] = {}
        for s in sel:
            by_ep.setdefault(s["episode_id"], []).append(s)
        Z: List[List[float]] = []
        y: List[int] = []
        for ep in sorted(by_ep):
            grp = by_ep[ep]
            for s, zr in zip(grp, average_rank_columns(
                    [s["z_raw"] for s in grp])):
                Z.append(zr)
                y.append(s["y"])
        p_plus, p_minus, util = [], [], []
        for z in Z:
            p = m.predict_proba(z)
            p_plus.append(p[CLASSES.index(1)])
            p_minus.append(p[CLASSES.index(-1)])
            util.append(p[CLASSES.index(1)] - p[CLASSES.index(-1)])
        entry.update({
            "n_train": m.n_train,
            "converged": m.converged,
            "auroc_plus": _auroc([1 if t_ == 1 else 0 for t_ in y], p_plus),
            "auroc_minus": _auroc([1 if t_ == -1 else 0 for t_ in y], p_minus),
            "spearman_utility": _spearman(util, y),
            "bss": m.bss(Z, y),
            "brier": m.brier(Z, y),
            "log_loss": m.log_loss(Z, y),
            "calibration_plus": _calibration_bins(p_plus, y),
            "min_class_count_threshold": MIN_CLASS_COUNT,
        })
        out[str(t)] = entry
    return out


# ---------------------------------------------------------------- oracle gap

def oracle_matched_gap(rows: Sequence[dict], assign: Dict[str, int],
                       tier_costs: Sequence[float]) -> dict:
    """matched-cost oracle gap（R3-P0-9）：每 episode 内，oracle 在与 vov 实际
    花费**完全相同**的预算下重分配（每 query 升到「能答对的最低档」，按 Δ成本
    升序贪心修复错例直到预算尽），报告 acc_oracle_matched − acc_vov。

    另报非受限 oracle（每 query 取最低正确档）的精度/成本作参考上界。
    成本单位为 view 数（P1-7 口径）。
    """
    preds = per_tier_predictions(rows)
    by_ep: Dict[str, List[str]] = {}
    label: Dict[str, Optional[int]] = {}
    for r in rows:
        if r["view"] == 0:
            by_ep.setdefault(r["episode_id"], []).append(r["qid"])
            label[r["qid"]] = r["true_label"]
    gaps, gaps_unlim = [], []
    acc_v_list, acc_o_list, acc_ou_list = [], [], []
    for ep in sorted(by_ep):
        qids = sorted(by_ep[ep])
        spent = sum(float(tier_costs[assign[q]]) for q in qids)
        # vov 实际精度
        labeled = [q for q in qids if label[q] is not None]
        if not labeled:
            continue
        acc_v = sum(preds[q][assign[q]] == label[q] for q in labeled) / len(labeled)
        # 每 query 的最低正确档与修复成本
        fix = {}   # qid -> (min_correct_tier or None, delta_cost)
        for q in qids:
            y = label[q]
            if y is None:
                continue
            t_star = next((t for t in range(len(TIER_VIEWS))
                           if preds[q].get(t) == y), None)
            fix[q] = (t_star, float(tier_costs[t_star] - tier_costs[0])
                      if t_star is not None else None)
        # 非受限 oracle：全部取最低正确档（不可修复 = 四档全错）
        n_unfixable = sum(1 for q in labeled if fix[q][0] is None)
        acc_ou = (len(labeled) - n_unfixable) / len(labeled)
        # matched-cost oracle：从全基档出发，按 Δ成本升序贪心修复
        base_cost = float(tier_costs[0]) * len(qids)
        room = spent - base_cost
        fixed = 0
        cands = sorted((fix[q][1], q) for q in labeled
                       if fix[q][0] is not None and fix[q][1] > 0)
        for dc, q in cands:
            if dc <= room + 1e-9:
                room -= dc
                fixed += 1
        n_wrong0 = sum(1 for q in labeled if preds[q].get(0) != label[q])
        acc_o = (len(labeled) - n_wrong0 + fixed) / len(labeled)
        acc_v_list.append(acc_v)
        acc_o_list.append(acc_o)
        acc_ou_list.append(acc_ou)
        gaps.append(acc_o - acc_v)
        gaps_unlim.append(acc_ou - acc_v)
    n = max(len(gaps), 1)
    return {"gap_matched_cost": sum(gaps) / n if gaps else 0.0,
            "gap_unrestricted": sum(gaps_unlim) / n if gaps_unlim else 0.0,
            "acc_vov_mean": sum(acc_v_list) / n if acc_v_list else 0.0,
            "acc_oracle_matched": sum(acc_o_list) / n if acc_o_list else 0.0,
            "acc_oracle_unrestricted": sum(acc_ou_list) / n
            if acc_ou_list else 0.0,
            "n_episode": len(gaps),
            "cost_unit": "views",
            "note": "matched-cost oracle = 同花费下按 Δ成本贪心修复错例的下界"}


def reachable_vs_all(rows: Sequence[dict], assign: Dict[str, int],
                     difficulty: Dict[str, float]) -> dict:
    """policy-reachable（被升档 query）vs all-query 的分布对比（R3-P0-9/11）：
    难度（1−margin）与基档错误率的对比，披露策略选择的子集是否系统性偏斜。
    """
    by_q = {r["qid"]: r for r in rows if r["view"] == 0}
    up = [q for q in by_q if assign.get(q, 0) > 0]
    allq = sorted(by_q)

    def _stat(qs):
        if not qs:
            return {"n": 0}
        ds = sorted(difficulty[q] for q in qs)
        wrong0 = sum(1 for q in qs
                     if by_q[q]["true_label"] is not None
                     and by_q[q]["pred"] != by_q[q]["true_label"])
        n_lab = sum(1 for q in qs if by_q[q]["true_label"] is not None)
        return {"n": len(qs),
                "difficulty_mean": sum(ds) / len(ds),
                "difficulty_median": ds[len(ds) // 2],
                "base_error_rate": (wrong0 / n_lab) if n_lab else None}

    return {"upgraded": _stat(up), "all_query": _stat(allq),
            "note": "upgraded = final_tier>0（policy-reachable 子集）"}
