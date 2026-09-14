# -*- coding: utf-8 -*-
"""v3.5 确认/开发结果分析器（二十轮 R3-P0-9 重写）。

输入：run_eval 结果 JSON（schema_version=vov_result_v2，含 dataset/method/
backbone/seed/budget_mult/transition_diagnostics/oracle 等）。

口径锁定：
- 按 (dataset, method, backbone) 分组；**缺 dataset 字段即报错**（禁止把
  protocol_sha256 当 fold 的旧行为）；
- BSS 逐 run 取 run_eval 的 episode 内 rank 口径值；门限用 **run 级
  bootstrap 95% CI 下界**（v3.4 §6：BSS>0 且 CI 下界 > −0.05）；
- 主判定（GO）：primary 4 格（tip_adapter/protonet × clip_vitb16/dinov2_vits14）
  逐一过闸——B=1.5 上 Δ(vov−hard_first) 点估计 > +0.10pt 且 hierarchical
  （fold→run）bootstrap CI 下界 > 0、Δ(vov−random_matched) 点估计 > 0、
  逐 transition 信号门槛（AUROC≥0.60 / Spearman≥0.10 / BSS 规则 / NA 格不判定）；
- 存在性空间（matched-cost oracle gap 均值）≤ 0.10pt → 自动 NO-GO；
- 主 B=1.5 + 副预算 {1.25,1.75,2.5,3,6} 曲线与逐格表。
"""
import math
from typing import Dict, List, Optional, Sequence

from .pipeline import BUDGET_MULTS, PRIMARY_BUDGET, hierarchical_bootstrap

PRIMARY_METHODS = ["tip_adapter", "protonet"]
PRIMARY_BACKBONES = ["clip_vitb16", "dinov2_vits14"]

# v3.4 §6 预声明门槛
DELTA_HARD_MIN_PT = 0.10          # 点估计 > +0.10pt
AUROC_MIN = 0.60
SPEARMAN_MIN = 0.10
BSS_MIN = 0.0
BSS_CI_LO_MIN = -0.05
EXISTENCE_MIN_PT = 0.10           # 存在性空间（oracle gap）≤ 0.10pt → NO-GO

REQUIRED_FIELDS = ["dataset", "method", "backbone", "budget_mult", "acc_vov",
                   "delta_vs_hard", "delta_vs_random"]


def _validate_result(r: dict, src: str = "") -> None:
    for k in REQUIRED_FIELDS:
        if k not in r:
            raise ValueError(
                f"结果 {src or '<dict>'} 缺字段 {k}——analyzer 拒绝"
                f"（R3-P0-9：不得回退到 protocol_sha256 当 fold）")


def _bootstrap_mean_ci(vals: Sequence[float], n_boot: int, seed: int
                       ) -> Optional[Dict[str, float]]:
    """run 级 bootstrap：均值与 95% 百分位 CI（描述性；fold 内 run 间）。"""
    import numpy as np

    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if len(v) == 0:
        return None
    rng = np.random.RandomState(seed)
    means = np.array([v[rng.randint(0, len(v), len(v))].mean()
                      for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(v.mean()), "ci95": (float(lo), float(hi)),
            "n_run": int(len(v))}


def _cell_signal_gates(runs: List[dict], n_boot: int, seed: int) -> dict:
    """逐 transition 信号门槛聚合（AUROC/Spearman/BSS/覆盖/事件率/校准）。"""
    gates: Dict[str, dict] = {}
    for t in (0, 1, 2):
        diags = [r["transition_diagnostics"][str(t)] for r in runs
                 if r.get("transition_diagnostics", {}).get(str(t))]
        if not diags:
            gates[str(t)] = {"status": "no_data", "gate_pass": None}
            continue
        status = diags[0].get("status")
        if status != "ready":
            gates[str(t)] = {"status": status, "n_runs": len(diags),
                             "event_rates": diags[0].get("event_rates"),
                             "gate_pass": None,
                             "note": "NA 格不作信号门槛判定（协议 §6a）"}
            continue
        auroc_p = [d.get("auroc_plus") for d in diags]
        auroc_m = [d.get("auroc_minus") for d in diags]
        spear = [d.get("spearman_utility") for d in diags]
        bss = [d.get("bss") for d in diags]
        cov = [d.get("sample_coverage") for d in diags]

        def _m(xs):
            xs = [x for x in xs if x is not None]
            return (sum(xs) / len(xs)) if xs else None

        bss_boot = _bootstrap_mean_ci(bss, n_boot, seed + 17 + t)
        auroc_p_m, auroc_m_m = _m(auroc_p), _m(auroc_m)
        spear_m, bss_m, cov_m = _m(spear), _m(bss), _m(cov)
        checks = {
            "auroc_plus>=0.60": (auroc_p_m is not None
                                 and auroc_p_m >= AUROC_MIN),
            "auroc_minus>=0.60": (auroc_m_m is not None
                                  and auroc_m_m >= AUROC_MIN),
            "spearman>=0.10": (spear_m is not None and spear_m >= SPEARMAN_MIN),
            "bss>0": (bss_m is not None and bss_m > BSS_MIN),
            "bss_ci95_lo>-0.05": (bss_boot is not None
                                  and bss_boot["ci95"][0] > BSS_CI_LO_MIN),
        }
        gates[str(t)] = {
            "status": "ready", "n_runs": len(diags),
            "auroc_plus": auroc_p_m, "auroc_minus": auroc_m_m,
            "spearman_utility": spear_m, "bss": bss_m,
            "bss_run_bootstrap": bss_boot,
            "sample_coverage": cov_m,
            "event_rates": diags[0].get("event_rates"),
            "checks": checks,
            "gate_pass": all(checks.values())}
    return gates


def analyze_results(results: Sequence[dict], n_boot: int = 2000,
                    seed: int = 0) -> dict:
    """主入口：结果 dict 列表 → v3.5 §6 判定报告。"""
    for i, r in enumerate(results):
        _validate_result(r, src=f"results[{i}]")

    # 按 (dataset, method, backbone) 分组
    by_group: Dict[tuple, List[dict]] = {}
    for r in results:
        by_group.setdefault((r["dataset"], r["method"], r["backbone"]),
                            []).append(r)

    # ---------------- 逐格预算曲线（主 1.5 + 副预算全集） ----------------
    group_tables: Dict[str, dict] = {}
    for (ds, me, bb), runs in sorted(by_group.items()):
        by_b: Dict[float, List[dict]] = {}
        for r in runs:
            by_b.setdefault(float(r["budget_mult"]), []).append(r)
        curve = []
        for b in sorted(by_b):
            rs = by_b[b]
            n = len(rs)
            curve.append({
                "budget_mult": b, "n_runs": n,
                "acc_vov": sum(r["acc_vov"] for r in rs) / n,
                "acc_hard_first": sum(r["acc_hard_first"] for r in rs) / n,
                "acc_random_matched": sum(r["acc_random_matched"] for r in rs) / n,
                "delta_vs_hard": sum(r["delta_vs_hard"] for r in rs) / n,
                "delta_vs_random": sum(r["delta_vs_random"] for r in rs) / n,
                "oracle_gap_matched": sum(r["oracle"]["gap_matched_cost"]
                                          for r in rs if r.get("oracle")) / n,
                "random_matched_perm_std":
                    sum(r.get("acc_random_matched_perm_std", 0.0)
                        for r in rs) / n})
        group_tables[f"{ds}__{me}__{bb}"] = {"curve": curve, "n_runs": len(runs)}

    # ---------------- primary 4 格 GO/NO-GO（fold = dataset 分层） ----------------
    cells: Dict[str, dict] = {}
    for me in PRIMARY_METHODS:
        for bb in PRIMARY_BACKBONES:
            cell = f"{me}__{bb}"
            folds: Dict[str, List[dict]] = {}
            for (ds, m2, b2), runs in by_group.items():
                if m2 == me and b2 == bb:
                    folds[ds] = [r for r in runs
                                 if float(r["budget_mult"]) == PRIMARY_BUDGET]
            if not folds:
                cells[cell] = {"verdict": "NO-GO",
                               "reasons": ["该 primary 格子无任何 B=1.5 run"]}
                continue
            pri = [r for rs in folds.values() for r in rs]
            dh = {f: [r["delta_vs_hard"] for r in rs]
                  for f, rs in folds.items()}
            boot = hierarchical_bootstrap(dh, n_boot=n_boot, seed=seed)
            pt_hard = 100.0 * boot["diff"]                    # pt 单位
            dr = [r["delta_vs_random"] for r in pri]
            pt_rand = 100.0 * (sum(dr) / len(dr))
            gates = _cell_signal_gates(pri, n_boot, seed)
            gap = [r["oracle"]["gap_matched_cost"] for r in pri
                   if r.get("oracle")]
            existence_pt = 100.0 * (sum(gap) / len(gap)) if gap else 0.0
            checks = {
                "delta_hard_point>+0.10pt": pt_hard > DELTA_HARD_MIN_PT,
                "delta_hard_hier_ci95_lo>0": boot["ci95"][0] > 0.0,
                "delta_random_point>0": pt_rand > 0.0,
                "signal_gates_all_transitions": all(
                    g.get("gate_pass") in (True, None)   # NA 格不判定
                    for g in gates.values()),
                "existence_space>0.10pt": existence_pt > EXISTENCE_MIN_PT,
            }
            reasons = [k for k, v in checks.items() if not v]
            cells[cell] = {
                "verdict": "GO" if not reasons else "NO-GO",
                "reasons": reasons,
                "primary_budget": PRIMARY_BUDGET,
                "delta_vs_hard_pt": pt_hard,
                "delta_vs_hard_hier_bootstrap": boot,
                "delta_vs_random_pt": pt_rand,
                "existence_space_oracle_gap_pt": existence_pt,
                "signal_gates": gates,
                "n_fold": len(folds), "n_runs": len(pri),
                "folds": sorted(folds)}

    overall = "GO" if (cells and all(c["verdict"] == "GO"
                                     for c in cells.values())) else "NO-GO"
    # 存在性空间总闸：任一格 ≤ θ 已计入 checks；此处给全局摘要
    existence = {c: cells[c].get("existence_space_oracle_gap_pt")
                 for c in cells}
    return {"verdict": overall,
            "budget_mults_locked": list(BUDGET_MULTS),
            "primary_budget": PRIMARY_BUDGET,
            "primary_cells": cells,
            "group_tables": group_tables,
            "existence_space_pt_by_cell": existence,
            "n_results": len(results),
            "note": ("hierarchical/run 级 bootstrap 均为描述性不确定度；"
                     "NA transition 格不作信号门槛判定（协议 §6a）；"
                     "成本单位 = view 数（非真实 FLOPs，P1-7）")}
