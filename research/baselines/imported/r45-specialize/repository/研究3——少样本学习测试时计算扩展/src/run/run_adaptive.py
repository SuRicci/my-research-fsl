# -*- coding: utf-8 -*-
"""E4：自适应预算分配（实验计划 §5.1，双协议）。

anytime：always-min/max、uniform 各档、分桶、熵早停级联、随机、贪心@均值预算、oracle
的 精度 vs 平均预算 点（iso-FLOPs 散点）；
budgeted-batch：uniform/random/greedy 在 B ∈ {1×,1.5×,2×,4×,8×} 下的精度
（§3.3 协议 6；乘率默认读 configs/budgets.yaml 的 batch_budget_multipliers）。

档位轴默认取 V（视图数）维度：tiers = V∈{1,2,4,8}（成本 = 各档每 query FLOPs，C0）。
greedy 的标定：support LOO；1-shot 下 LOO 结构性退化 → 分位数回退
（signals/calibration.py，calib_methods 与 greedy 升档诊断随结果落盘）。

示例：
  conda run -n torch python -m src.run.run_adaptive \
    --dataset miniimagenet --backbone clip_vitb16 --episodes 200 --shot 1 --tiers 1,2,4,8
"""
import argparse
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F

from src.allocators.bucket import BucketAllocator
from src.allocators.cascade import CascadeAllocator
from src.allocators.greedy import GreedyAllocator, random_assign, uniform_assign
from src.budgets import Budget
from src.config import RESULTS_ROOT, load_yaml, save_json, set_seed
from src.data.cls_datasets import load_cls_dataset
from src.data.splits import EpisodeGenerator
from src.features.text_cache import (episode_zs_logits_fn, has_text,
                                     text_features_for_dataset)
from src.run.common import (budget_cost_c0, episode_feature_indices,
                            extract_features, infer_episode, score_episode)
from src.signals.calibration import loo_calibrate
from src.signals.difficulty import s1_entropy, s2_top2_margin, rank_fuse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E4 自适应预算分配")
    p.add_argument("--dataset", default="miniimagenet")
    p.add_argument("--split", default="test")
    p.add_argument("--backbone", default="clip_vitb16")
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--shot", type=int, default=1)
    p.add_argument("--q-per-cls", type=int, default=15)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="tip_adapter")
    p.add_argument("--tiers", default="1,2,4,8", help="档位轴：V 视图数档位（升序）")
    p.add_argument("--k", type=int, default=-1)
    p.add_argument("--r", type=int, default=224)
    p.add_argument("--budget-mults", default="",
                   help="空串则读 budgets.yaml 的 batch_budget_multipliers")
    p.add_argument("--calib-1shot", default="cross_ep", choices=["cross_ep", "fixed_prior"],
                   help="1-shot 标定：cross_ep=前25集带标签历史（成本记账并披露）；"
                        "fixed_prior=纯免标签无历史（固定增益先验）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default=str(RESULTS_ROOT))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    mcfg = load_yaml("methods")
    kw = dict(mcfg["cache_based"]["tip_adapter"])
    kw.update(mcfg["transductive"]["laplacian_shot"])

    tiers = [int(x) for x in args.tiers.split(",")]
    # 传导法工作点（L6/计划 §5.1）：laplacian_shot/alpha_tim 的 V 轴在 T=5 工作点评估
    work_T = 5 if args.method in ("laplacian_shot", "alpha_tim") else 0
    if args.budget_mults.strip():
        mults = [float(x) for x in args.budget_mults.split(",")]
    else:
        mults = [float(x) for x in load_yaml("budgets")["batch_budget_multipliers"]]

    ds = load_cls_dataset(args.dataset, args.split)
    gen = EpisodeGenerator(ds.labels, n_way=args.n_way, k_shot=args.shot,
                           q_per_cls=args.q_per_cls, seed=args.seed)
    episodes = gen.generate(args.episodes)

    # 八轮评审 R3-1：CLIP backbone + 可读类名时接入标准 Tip-Adapter 文本先验
    #（logits = zs + alpha·cache，与实验计划公式一致）；DINOv2 或 wnid 类名（miniImageNet）
    # 无文本先验 → cache-only 实现，输出中如实命名 method_impl。
    text_all, logit_scale = None, 1.0
    if has_text(args.backbone, ds.classnames):
        text_all, logit_scale = text_features_for_dataset(
            args.dataset, ds.classnames, args.backbone, device=args.device)
        print(f"  [E4] 文本先验已接入（{args.dataset}/{args.backbone}，标准 Tip-Adapter 路径）")
    else:
        print(f"  [E4] 无文本先验（{args.dataset}/{args.backbone}）→ cache-only 实现")
    use_text = text_all is not None
    zs_list = []
    for ep in episodes:
        if use_text:
            text_ep = text_all[torch.from_numpy(ep.classes).long()]
            zs_list.append(episode_zs_logits_fn(text_ep, logit_scale))
        else:
            zs_list.append(None)
    method_impl = args.method if use_text else f"{args.method}__cache_only"
    indices = episode_feature_indices(episodes)
    feats = extract_features(ds, indices, max(tiers), args.r, args.backbone,
                             device=args.device, batch_size=args.batch_size)

    # 每档成本（C0/ query）
    costs = [budget_cost_c0(Budget(V=t, k=args.k, T=work_T, r=args.r), args.backbone,
                            n_support=args.n_way * args.shot,
                            n_query=args.n_way * args.q_per_cls, n_class=args.n_way)
             for t in tiers]
    base_budget = costs[0]  # 均匀基线预算 = 全置最低档
    n_tiers = len(tiers)
    n_q_total = sum(len(e.query_idx) for e in episodes)

    # 预计算：每 episode × 每档 的逐 query 预测与后验（缓存特征，向量运算）
    per_ep = []
    for ep_i, ep in enumerate(episodes):
        sf = torch.stack([feats[i] for i in ep.support_idx])
        qf = torch.stack([feats[i] for i in ep.query_idx])
        sy = torch.from_numpy(ep.support_labels)
        qy = torch.from_numpy(ep.query_labels)
        entry = {"qy": qy, "tiers": []}
        for t in tiers:
            r = score_episode(infer_episode(args.method, qf, sf, sy, args.n_way,
                                            Budget(V=t, k=args.k, T=work_T, r=args.r),
                                            zs_logits_fn=zs_list[ep_i], **kw), qy)
            sig = rank_fuse([s2_top2_margin(r.probs), s1_entropy(r.probs)])
            entry["tiers"].append({"preds": r.probs.argmax(1), "signal": sig,
                                   "probs": r.probs})
        per_ep.append(entry)

    # 标定（每 episode 一次，anytime 贪心与全部预算档共享；LOO 退化→分位数回退）
    def _infer(sf, sy, qf, tier, zs_fn=None):
        """标定推理（八轮评审 R3-2 修复）：与正式推理同方法、同「逐视图后验再平均」
        聚合路径，仅视图数取前 tiers[tier] 个；文本先验随 episode 传入。
        旧实现固定用 ProtoNet 代理，与被评估方法（tip_adapter/laplacian_shot）失配。"""
        v = tiers[tier]
        r = infer_episode(args.method, qf[:, :v], sf[:, :v], sy, args.n_way,
                          Budget(V=v, k=args.k, T=work_T, r=args.r),
                          zs_logits_fn=zs_fn, **kw)
        return (r.probs + 1e-12).log()

    def _sig(logits):
        return s2_top2_margin(F.softmax(logits, dim=1))[0]

    # R3-4 修复（2026-08-21 二轮评审）：1-shot 下旧 fallback 用当前 query 批的全档位
    # logits 标定 = 分配前偷看高预算输出。两种合规标定（--calib-1shot 选择）：
    # - cross_ep：跨 episode 标定。前 N_CAL 个 episode 的完整信息（含标签）拟合分位数表
    #   ——部署假设=存在带标签的历史校准集，标签与推理成本单独记账并随结果落盘（B4）；
    # - fixed_prior：纯免标签无历史——固定增益先验（每档 +1pt 的披露超参），
    #   不看任何 query 高档输出、不用任何标签。
    N_CAL = 25
    calibs, calib_methods = [], set()
    calib_audit = []  # 逐 episode 标定审计（B5）：method/n_valid/n_skipped/fallback_used
    if args.shot == 1:
        from src.signals.calibration import CalibrationTable
        if args.calib_1shot == "cross_ep":
            cal_ep = per_ep[:N_CAL]
            sigs = torch.cat([e["tiers"][0]["signal"] for e in cal_ep])
            edges = torch.quantile(sigs, torch.linspace(0, 1, 4)).tolist()[1:-1]
            tier_acc = [float(np.mean([float((e["tiers"][t]["preds"] == e["qy"]).float().mean())
                                       for e in cal_ep])) for t in range(n_tiers)]
            # 分桶×档位精度：桶边界用标定集信号的全局分位数
            def _bucket_of(e):
                s = e["tiers"][0]["signal"]
                return torch.bucketize(s, torch.tensor(edges))
            bucket_acc = []
            for b in range(3):
                row = []
                for t in range(n_tiers):
                    vals = [float((e["tiers"][t]["preds"][bkt == b] == e["qy"][bkt == b]).float().mean())
                            for e in cal_ep if (bkt := _bucket_of(e)).eq(b).any()]
                    row.append(float(np.mean(vals)) if vals else tier_acc[t])
                bucket_acc.append(row)
            fixed_calib = CalibrationTable(tier_acc=tier_acc, bucket_acc=bucket_acc,
                                           bucket_edges=edges, method="cross_ep_quantile",
                                           n_valid=sum(len(e["qy"]) for e in cal_ep))
            calib_methods.add("cross_ep_quantile")
            eval_start = N_CAL  # 前 25 个 episode 是标定集，不进主指标
        else:  # fixed_prior：纯免标签无历史，披露增益先验
            g = [0.0, 0.01, 0.02, 0.03][:n_tiers]   # 每档 +1pt 披露超参
            base = 0.5                              # 基准精度占位（greedy 只用差值）
            fixed_calib = CalibrationTable(tier_acc=[base + x for x in g],
                                           method="fixed_prior")
            calib_methods.add("fixed_prior")
            # 四轮评审 B2：所有 1-shot 协议统一在 episode 25-199 评估（fixed_prior
            # 不需要校准，但同样跳过前 25 个以保证逐 episode 可比）
            eval_start = N_CAL
    else:
        eval_start = 0
    # 四轮评审 B4：校准阶段成本记账（cross_ep 假设两阶段执行：标定集全档位前向+标签）
    calib_cost = {"n_calib_episodes": 0, "n_calib_queries": 0, "n_calib_labels": 0,
                  "calib_forward_c0": 0.0, "calib_wall_time_s": None}
    if args.shot == 1 and args.calib_1shot == "cross_ep":
        cal_ep = per_ep[:N_CAL]
        n_q_cal = sum(len(e["qy"]) for e in cal_ep)
        calib_cost = {"n_calib_episodes": N_CAL, "n_calib_queries": int(n_q_cal),
                      "n_calib_labels": int(n_q_cal),
                      # 两阶段执行下标定需全档前向：最高档成本 × 标定 query 数
                      "calib_forward_c0": float(costs[-1] * n_q_cal),
                      # 标定在模拟预计算内完成，无可归因墙钟；记账用理论口径
                      "calib_wall_time_s": None}
    for ep_i, entry in enumerate(per_ep):
        if args.shot == 1:
            calibs.append(fixed_calib)
            calib_audit.append({"episode": ep_i, "calib_method": fixed_calib.method,
                                "n_valid": fixed_calib.n_valid, "n_skipped": 0,
                                "fallback_used": False})
            continue
        sf = torch.stack([feats[i] for i in episodes[ep_i].support_idx])  # [S,V,D]
        sy = torch.from_numpy(episodes[ep_i].support_labels)
        fallback = [(entry["tiers"][t]["probs"] + 1e-12).log() for t in range(n_tiers)]
        calib = loo_calibrate(sf, sy, partial(_infer, zs_fn=zs_list[ep_i]),
                              list(range(n_tiers)), _sig,
                              n_buckets=3, fallback_tier_logits=fallback)
        calib_methods.add(calib.method)
        calibs.append(calib)
        calib_audit.append({"episode": ep_i, "calib_method": calib.method,
                            "n_valid": calib.n_valid, "n_skipped": calib.n_skipped,
                            "fallback_used": calib.method != "loo"})

    def eval_assign(assign_fn, label):
        """assign_fn(entry, ep_i) -> tier 索引 [Q]；返回评估集平均精度与平均预算。
        1-shot 跨 episode 标定时前 N_CAL 个 episode 是标定集，不进主指标（R3-4）。"""
        accs, spent, n_q_eval = [], 0.0, 0
        for ep_i in range(eval_start, len(per_ep)):
            entry = per_ep[ep_i]
            a = assign_fn(entry, ep_i)
            qy = entry["qy"]
            preds = torch.stack([entry["tiers"][t]["preds"][i]
                                 for i, t in enumerate(a.tolist())])
            accs.append(float((preds == qy).float().mean().item()))
            spent += sum(costs[t] for t in a.tolist())
            n_q_eval += len(qy)
        return {"allocator": label, "acc": round(float(np.mean(accs)), 4),
                "mean_cost_c0": round(spent / max(n_q_eval, 1), 3)}

    galloc = GreedyAllocator(costs, max_tier=n_tiers - 1)
    mean_b = sum(costs) / len(costs)

    # ---------------- anytime 协议 ----------------
    rows = []
    rows.append(eval_assign(lambda e, i: uniform_assign(len(e["qy"]), 0), "always_min"))
    for t in range(1, n_tiers - 1):
        rows.append(eval_assign(lambda e, i, t=t: uniform_assign(len(e["qy"]), t),
                                f"uniform_v{tiers[t]}"))
    rows.append(eval_assign(lambda e, i: uniform_assign(len(e["qy"]), n_tiers - 1),
                            "always_max"))
    # uniform（同期望预算平铺到不超预算的最高档）
    t_uni_mean = max([t for t in range(n_tiers) if costs[t] <= mean_b + 1e-9], default=0)
    rows.append(eval_assign(lambda e, i: uniform_assign(len(e["qy"]), t_uni_mean), "uniform"))
    mid = n_tiers // 2
    bucket_map = [0, mid, n_tiers - 1] if n_tiers >= 3 else list(range(n_tiers))
    balloc = BucketAllocator(bucket_map, n_buckets=min(3, n_tiers))
    rows.append(eval_assign(lambda e, i: balloc.assign(e["tiers"][0]["signal"]), "bucket"))
    # 熵早停（复现 Yang 负结果对照）：级联 + S1 熵信号
    calloc = CascadeAllocator(n_tiers, tau_quantile=mcfg["allocators"]["cascade"]["tau_quantile"])
    rows.append(eval_assign(
        lambda e, i: calloc.assign([e["tiers"][t]["signal"] for t in range(n_tiers)]),
        "entropy_cascade"))
    # 随机分配（同期望预算打乱，隔离信号真实贡献）
    def _rand(e, i):
        return random_assign(len(e["qy"]), costs, mean_b * len(e["qy"]), n_tiers,
                             seed=args.seed)
    rows.append(eval_assign(_rand, "random"))
    # 贪心 @ 均值预算（anytime 可比点）
    rows.append(eval_assign(
        lambda e, i: galloc.assign(e["tiers"][0]["signal"], calibs[i],
                                   total_budget=mean_b * len(e["qy"])),
        "greedy"))
    # oracle 上界（按是否答对分配最低可行档）
    def _oracle(e, i):
        a = torch.zeros(len(e["qy"]), dtype=torch.int64)
        for t in range(n_tiers):
            wrong = e["tiers"][t]["preds"] != e["qy"]
            a[wrong] = min(t + 1, n_tiers - 1)
        return a
    rows.append(eval_assign(_oracle, "oracle"))

    for r in rows:
        print(f"  anytime {r['allocator']:16s} acc={r['acc']:.4f} cost={r['mean_cost_c0']:.2f}C0")

    # ---------------- budgeted-batch 协议（§3.3：B ∈ {1×,1.5×,2×,4×,8×}） ----------------
    # R3-3（2026-08-21 二轮评审）：新增与 greedy 逐 episode 精确同直方图的对照——
    # random_matched（同 tier 计数随机映射 query）与 easy_first（同直方图、信号升序升档），
    # 严格隔离难度信号价值；成本逐 episode 与 greedy 完全相等（断言检查）。
    bb_rows = []
    rng_match = torch.Generator().manual_seed(args.seed + 777)
    for mult in mults:
        res_b = {"budget_mult": mult}
        match_acc = defaultdict(list)   # easy_first / random_matched（按 mult 隔离）
        for name in ["uniform", "random", "greedy"]:
            accs, spent, tiers_g = [], 0.0, []
            n_q_eval = 0
            for ep_i in range(eval_start, len(per_ep)):
                entry = per_ep[ep_i]
                n_q = len(entry["qy"])
                B = base_budget * mult * n_q
                if name == "uniform":
                    per_q = base_budget * mult
                    a = uniform_assign(n_q, max([t for t in range(n_tiers)
                                                 if costs[t] <= per_q + 1e-9], default=0))
                elif name == "random":
                    a = random_assign(n_q, costs, B, n_tiers, seed=args.seed)
                else:
                    a = galloc.assign(entry["tiers"][0]["signal"], calibs[ep_i], total_budget=B)
                    tiers_g.append(a.float())
                preds = torch.stack([entry["tiers"][t]["preds"][i]
                                     for i, t in enumerate(a.tolist())])
                accs.append(float((preds == entry["qy"]).float().mean().item()))
                spent += sum(costs[t] for t in a.tolist())
                n_q_eval += n_q
                # matched baselines（仅 greedy 行）：同直方图重排
                if name == "greedy":
                    counts = torch.bincount(a, minlength=n_tiers)
                    tier_seq = torch.repeat_interleave(torch.arange(n_tiers), counts)
                    # random_matched：20 次独立排列（四轮评审 B7：分离匹配随机性与数据随机性）
                    rm_ep = []
                    for _p in range(20):
                        perm = torch.randperm(n_q, generator=rng_match)
                        a_rm = tier_seq[perm]
                        preds_rm = torch.stack([entry["tiers"][t]["preds"][i]
                                                for i, t in enumerate(a_rm.tolist())])
                        rm_ep.append(float((preds_rm == entry["qy"]).float().mean().item()))
                        assert abs(sum(costs[t] for t in a_rm.tolist())
                                   - sum(costs[t] for t in a.tolist())) < 1e-6
                    match_acc["random_matched"].append(float(np.mean(rm_ep)))
                    match_acc["random_matched_perm_std"].append(float(np.std(rm_ep)))
                    sig = entry["tiers"][0]["signal"]
                    # 同直方图两向排序对照（信号大=难）：
                    # hard_first=最难的 query 优先升档（免标签，只用基档信号排序）；
                    # easy_first=最易的优先升档
                    for label, desc in (("easy_first", False), ("hard_first", True)):
                        a_x = torch.zeros(n_q, dtype=torch.int64)
                        order = torch.argsort(sig, descending=desc)
                        pos = 0
                        for t in range(n_tiers - 1, -1, -1):
                            c = int(counts[t].item())
                            a_x[order[pos:pos + c]] = t
                            pos += c
                        preds_x = torch.stack([entry["tiers"][t]["preds"][i]
                                               for i, t in enumerate(a_x.tolist())])
                        match_acc[label].append(
                            float((preds_x == entry["qy"]).float().mean().item()))
            res_b[f"{name}_acc"] = round(float(np.mean(accs)), 4)
            res_b[f"{name}_cost_c0"] = round(spent / max(n_q_eval, 1), 3)
            if name == "greedy":
                tg = torch.cat(tiers_g)
                res_b["greedy_mean_tier"] = round(float(tg.mean().item()), 3)
                res_b["greedy_frac_upgraded"] = round(float((tg > 0).float().mean().item()), 4)
                res_b["random_matched_acc"] = round(float(np.mean(match_acc["random_matched"])), 4)
                res_b["random_matched_perm_std"] = round(
                    float(np.mean(match_acc["random_matched_perm_std"])), 4)
                res_b["easy_first_acc"] = round(float(np.mean(match_acc["easy_first"])), 4)
                res_b["hard_first_acc"] = round(float(np.mean(match_acc["hard_first"])), 4)
        bb_rows.append(res_b)
        print(f"  budgeted-batch B={mult:4.1f}x uniform={res_b['uniform_acc']:.4f} "
              f"random={res_b['random_acc']:.4f} greedy={res_b['greedy_acc']:.4f} "
              f"randMatched={res_b.get('random_matched_acc', 0):.4f} easyFirst={res_b.get('easy_first_acc', 0):.4f}")

    out = save_json({"task": "E4_adaptive", "args": vars(args),
                     # 八轮评审 R3-1：方法实现与文本先验如实落盘
                     # （tip_adapter 无文本先验时为 cache-only 亲和分类器，非标准 Tip-Adapter）
                     "method_impl": method_impl,
                     "text_prior": bool(use_text),
                     "anytime": rows, "budgeted_batch": bb_rows,
                     "tier_costs_c0": costs, "calib_methods": sorted(calib_methods),
                     "eval_start_episode": eval_start,
                     "n_calib_episodes": calib_cost["n_calib_episodes"],
                     # 四轮评审 B3/B4：标签来源与校准成本显式落盘
                     "label_source": ("前 25 个 episode 的 query 标签（带标签历史在线校准——"
                                      "非标准免标签设定）"
                                      if (args.shot == 1 and args.calib_1shot == "cross_ep")
                                      else ("无（免标签固定先验）"
                                            if args.calib_1shot == "fixed_prior"
                                            else "support 标签（LOO，FSL 设定自带）")),
                     "calib_cost": calib_cost,
                     "calib_audit": calib_audit,
                     "matched_note": "random_matched(20 次排列均值)/easy_first/hard_first 与 greedy "
                                     "逐 episode 同直方图（成本严格相等）"},
                    Path(args.out),
                    f"adaptive_{args.dataset}_{args.backbone}_{args.method}_s{args.shot}")
    print(f"[E4] 结果已落盘: {out}（标定方式: {sorted(calib_methods)}）")
    return {"anytime": rows, "budgeted_batch": bb_rows}


if __name__ == "__main__":
    main()
