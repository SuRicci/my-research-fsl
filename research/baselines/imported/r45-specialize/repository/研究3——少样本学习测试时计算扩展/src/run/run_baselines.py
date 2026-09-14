# -*- coding: utf-8 -*-
"""E5：baseline 对比表（iso-FLOPs，实验计划 §5.1）。

固定预算档（默认 V=1/k=全量/T=5/r=224），横向对比方法谱系：
P0：zero-shot（有文本类名时）/kNN/SimpleShot/ProtoNet/Tip-Adapter/GDA
+ 传导代表 LaplacianShot/α-TIM；
P1（自实现，读缓存特征）：PT+MAP/APE/TDA/TransCLIP/MuSC（methods/p1_baselines.py）。

文本先验：有类名的数据集（cifar_fs/eurosat/flowers102/dtd/cub200）经
features/text_cache.py 统一计算文本特征（落盘 cache/text/），requires_text 方法
（zero_shot/ape/tda/transclip/musc）在无文本先验时（miniimagenet 的 wnid 类名、
非 CLIP backbone）自动跳过；tip_adapter 有文本先验时按论文口径融合 zs 项。

示例：
  conda run -n torch python -m src.run.run_baselines \
    --dataset miniimagenet --backbone clip_vitb16 --episodes 20
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.budgets import Budget
from src.config import RESULTS_ROOT, load_yaml, save_json, set_seed
from src.data.cls_datasets import load_cls_dataset
from src.data.splits import EpisodeGenerator
from src.features.text_cache import (episode_zs_logits_fn, has_text,
                                     text_features_for_dataset)
from src.methods.cache_based import METHODS as CACHE_METHODS
from src.methods.p1_baselines import P1_METHODS
from src.methods.transductive import TRANSDUCTIVE_METHODS
from src.run.common import (aggregate_episode_metrics, budget_cost_c0,
                            episode_feature_indices, extract_features,
                            infer_episode, score_episode)

P0_DEFAULT = ["knn", "simpleshot", "protonet", "tip_adapter", "gda", "laplacian_shot"]
P1_DEFAULT = ["pt_map", "ape", "tda", "transclip", "musc"]
ALL_METHODS = CACHE_METHODS + TRANSDUCTIVE_METHODS + list(P1_METHODS)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E5 baseline 对比表（iso-FLOPs）")
    p.add_argument("--dataset", default="miniimagenet")
    p.add_argument("--split", default="test")
    p.add_argument("--backbone", default="clip_vitb16")
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--shot", type=int, default=1)
    p.add_argument("--q-per-cls", type=int, default=15)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", nargs="+", default=P0_DEFAULT)
    p.add_argument("--V", type=int, default=1)
    p.add_argument("--k", type=int, default=-1)
    p.add_argument("--T", type=int, default=5, help="传导/迭代类方法的迭代轮数档位")
    p.add_argument("--r", type=int, default=224)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--prompts", default=None,
                   help="文本模板集覆盖（dataset/imagenet80），默认读 configs/methods.yaml text.template_set")
    p.add_argument("--out", default=str(RESULTS_ROOT))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    mcfg = load_yaml("methods")
    kw = dict(mcfg["cache_based"]["tip_adapter"])
    kw.update(mcfg["transductive"]["laplacian_shot"])
    kw.update(mcfg["transductive"]["alpha_tim"])
    p1_cfg = mcfg.get("p1", {})
    text_cfg = mcfg.get("text", {})
    template_set = args.prompts or text_cfg.get("template_set", "dataset")
    logit_scale = float(text_cfg.get("logit_scale", 100.0))

    ds = load_cls_dataset(args.dataset, args.split)
    gen = EpisodeGenerator(ds.labels, n_way=args.n_way, k_shot=args.shot,
                           q_per_cls=args.q_per_cls, seed=args.seed)
    episodes = gen.generate(args.episodes)
    indices = episode_feature_indices(episodes)
    feats = extract_features(ds, indices, args.V, args.r, args.backbone,
                             device=args.device, batch_size=args.batch_size)

    # 文本先验：有类名且为 CLIP backbone 时，全类文本特征一次算完（读缓存）
    text_all = None
    want_text = any(m in P1_METHODS and P1_METHODS[m]["requires_text"]
                    or m in ("zero_shot", "tip_adapter") for m in args.methods)
    if want_text and has_text(args.backbone, ds.classnames):
        text_all, logit_scale = text_features_for_dataset(
            args.dataset, ds.classnames, args.backbone, device=args.device,
            template_set=template_set, logit_scale=logit_scale)
    elif want_text:
        print(f"  [E5] {args.dataset}/{args.backbone} 无可用文本先验，"
              f"requires_text 方法将跳过（tip_adapter 退化为纯缓存后验）")

    rows = []
    for method in args.methods:
        if method not in ALL_METHODS:
            print(f"  [E5] 跳过未实现方法: {method}")
            continue
        needs_text = method == "zero_shot" or \
            (method in P1_METHODS and P1_METHODS[method]["requires_text"])
        if needs_text and text_all is None:
            print(f"  [E5] 跳过 {method}（requires_text，当前无文本先验）")
            continue
        uses_T = method in TRANSDUCTIVE_METHODS or \
            (method in P1_METHODS and P1_METHODS[method]["uses_T"])
        budget = Budget(V=args.V, k=args.k, T=args.T if uses_T else 0, r=args.r)
        mkw = dict(kw)
        mkw.update(p1_cfg.get(method, {}))
        results = []
        for ep in episodes:
            sf = torch.stack([feats[i] for i in ep.support_idx])
            qf = torch.stack([feats[i] for i in ep.query_idx])
            sy = torch.from_numpy(ep.support_labels)
            qy = torch.from_numpy(ep.query_labels)
            if text_all is not None:
                text_ep = text_all[torch.from_numpy(ep.classes).long()]
                zs_fn = episode_zs_logits_fn(text_ep, logit_scale)
            else:
                text_ep, zs_fn = None, None
            r = infer_episode(method, qf, sf, sy, args.n_way, budget,
                              zs_logits_fn=zs_fn, text_feats=text_ep, **mkw)
            results.append(score_episode(r, qy))
        agg = aggregate_episode_metrics(results)
        cost = budget_cost_c0(budget, args.backbone, args.n_way * args.shot,
                              args.n_way * args.q_per_cls, args.n_way)
        row = {"method": method, "budget": budget.to_dict(), "cost_c0": round(cost, 3),
               **{k: round(v, 4) for k, v in agg.items()}}
        rows.append(row)
        print(f"  {method:16s} cost={cost:5.2f}C0 acc={agg['acc_mean']:.4f}"
              f"±{agg['acc_mean_ci95']:.4f}")

    rows.sort(key=lambda r: -r["acc_mean"])
    out = save_json({"task": "E5_baselines", "args": vars(args), "rows": rows},
                    Path(args.out), f"baselines_{args.dataset}_{args.backbone}")
    print(f"[E5] 结果已落盘: {out}")
    return rows


if __name__ == "__main__":
    main()
