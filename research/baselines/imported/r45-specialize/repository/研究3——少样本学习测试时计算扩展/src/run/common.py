# -*- coding: utf-8 -*-
"""run/ 脚本的共享管线：特征提取（走缓存）+ episode 推理 + 预算记账。

工程红线：特征只算一次——所有实验经 FeatureCache 读缓存特征做向量运算。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..budgets import Budget, flops_g, flops_in_c0
from ..data.cls_datasets import ClsDataset
from ..data.splits import Episode
from ..features.cache_io import FeatureCache
from ..methods.cache_based import run_method
from ..methods.p1_baselines import P1_METHODS, run_p1_method
from ..methods.transductive import run_transductive, TRANSDUCTIVE_METHODS
from ..views import ViewSampler, cache_key

BACKBONES = {
    "clip_vitb16": {"kind": "clip", "model": "ViT-B-16"},
    "clip_rn50": {"kind": "clip", "model": "RN50"},
    "dinov2_vits14": {"kind": "dino", "model": "dinov2_vits14"},
}


def get_extractor(backbone: str, device: str = "cuda"):
    """按 backbone 名构造特征提取器（真实权重，无 mock）。"""
    if backbone not in BACKBONES:
        raise KeyError(f"未知 backbone: {backbone}（可选: {list(BACKBONES)}）")
    spec = BACKBONES[backbone]
    if spec["kind"] == "clip":
        from ..features.clip import CLIPExtractor
        return CLIPExtractor(spec["model"], device=device)
    from ..features.dino import DINOExtractor
    return DINOExtractor(spec["model"], device=device)


def _img_id(dataset: ClsDataset, idx: int) -> str:
    return f"{dataset.name}#{idx}"


def extract_features(dataset: ClsDataset, indices: Sequence[int], V: int, res: int,
                     backbone: str, device: str = "cuda", batch_size: int = 64,
                     scope: str = "cls") -> Dict[int, torch.Tensor]:
    """提取 (img_idx, v<V, res) 的 CLS 特征并写缓存；返回 {idx: [V, D] tensor}。

    已缓存的键直接读取（断点续跑），不重跑 backbone。
    """
    fc = FeatureCache(f"{scope}/{dataset.name}", backbone)
    extractor = None
    sampler = ViewSampler("clip" if BACKBONES[backbone]["kind"] == "clip" else "dinov2")
    out: Dict[int, List[torch.Tensor]] = {i: [None] * V for i in indices}

    # 先收集缓存命中
    todo = []
    for i in indices:
        iid = _img_id(dataset, i)
        for v in range(V):
            k = cache_key(iid, v, res, backbone)
            if fc.has(k):
                out[i][v] = fc.get(k)
            else:
                todo.append((i, v, k))
    if todo:
        extractor = get_extractor(backbone, device)
        # 按 (img, view) 批量前向
        for st in range(0, len(todo), batch_size):
            chunk = todo[st:st + batch_size]
            imgs = []
            for i, v, k in chunk:
                pil = dataset.get_image(i)
                imgs.append(sampler.make_transform(_img_id(dataset, i), v, res)(pil))
            x = torch.stack(imgs).to(device)
            feats = extractor.encode_image(x) if BACKBONES[backbone]["kind"] == "clip" \
                else extractor.extract_cls(x)
            for (i, v, k), f in zip(chunk, feats):
                fc.put(k, f)
                out[i][v] = f.float().cpu()
        fc.flush()
    return {i: torch.stack(vs) for i, vs in out.items()}


def episode_feature_indices(episodes: Sequence[Episode]) -> List[int]:
    idx = set()
    for e in episodes:
        idx.update(e.support_idx.tolist())
        idx.update(e.query_idx.tolist())
    return sorted(idx)


@dataclass
class EpisodeResult:
    acc_mean: float      # 均值后验聚合精度
    acc_vote: float      # 多数投票精度
    acc_oracle: float    # oracle：任一视图正确即对（V=1 时等于单视图精度）
    probs: torch.Tensor  # 聚合后验 [Q, C]
    per_view_pred: torch.Tensor  # [Q, V]


def infer_episode(name: str, qf: torch.Tensor, sf: torch.Tensor,
                  support_labels: torch.Tensor, n_class: int,
                  budget: Budget, zs_logits_fn=None, text_feats=None, **method_kw) -> EpisodeResult:
    """单 episode 在给定预算下的推理（纯特征空间运算）。

    qf [Q, V, D] / sf [S, V, D]；V 轴取前 budget.V 个视图。
    视图聚合：多数投票 + 均值后验双报，另报 oracle（实验计划 §1.2）。
    text_feats: episode 各类的文本特征 [n_way, D]（仅 ape 等需要文本特征的 P1 方法用）。
    """
    V = min(budget.V, qf.shape[1])
    n_q = qf.shape[0]
    per_view_pred, per_view_prob = [], []
    for v in range(V):
        qv, sv = qf[:, v], sf[:, v]
        zs = zs_logits_fn(qv) if zs_logits_fn is not None else None
        if name in P1_METHODS:
            logits = run_p1_method(name, qv, sv, support_labels, n_class,
                                   zs_logits=zs, text_feats=text_feats,
                                   k=budget.k, rounds=budget.T, **method_kw)
        elif name in TRANSDUCTIVE_METHODS:
            p0 = run_method("tip_adapter", qv, sv, support_labels, n_class, k=budget.k,
                            zs_logits=zs, **method_kw)
            logits = run_transductive(name, p0, sv, support_labels, qv,
                                      rounds=budget.T, **method_kw)
        else:
            logits = run_method(name, qv, sv, support_labels, n_class, k=budget.k,
                                zs_logits=zs, **method_kw)
        prob = F.softmax(logits, dim=1)
        per_view_prob.append(prob)
        per_view_pred.append(prob.argmax(1))
    pv = torch.stack(per_view_pred, dim=1)               # [Q, V]
    probs = torch.stack(per_view_prob, dim=0).mean(dim=0)  # 均值后验
    return EpisodeResult(acc_mean=0.0, acc_vote=0.0, acc_oracle=0.0,
                         probs=probs, per_view_pred=pv)


def score_episode(res: EpisodeResult, query_labels: torch.Tensor) -> EpisodeResult:
    """填充三种精度指标。"""
    y = query_labels
    res.acc_mean = float((res.probs.argmax(1) == y).float().mean().item())
    # 多数投票（平局取最小类号，确定性）
    votes = F.one_hot(res.per_view_pred, res.probs.shape[1]).float().sum(dim=1)
    res.acc_vote = float((votes.argmax(1) == y).float().mean().item())
    any_correct = (res.per_view_pred == y.unsqueeze(1)).any(dim=1)
    res.acc_oracle = float(any_correct.float().mean().item())
    return res


def budget_cost_c0(budget: Budget, backbone: str, n_support: int, n_query: int,
                   n_class: int, feat_dim: int = 512) -> float:
    """单 query 平均成本（C0 单位）。"""
    return flops_in_c0(flops_g(budget, backbone, n_cache=n_support,
                               feat_dim=feat_dim, n_query=n_query,
                               n_support=n_support, n_class=n_class))


def aggregate_episode_metrics(results: List[EpisodeResult]) -> Dict[str, float]:
    from ..analysis.stats import mean_ci

    out = {}
    for key in ["acc_mean", "acc_vote", "acc_oracle"]:
        vals = [getattr(r, key) for r in results]
        ci = mean_ci(vals)
        out[key] = ci["mean"]
        out[f"{key}_ci95"] = ci["ci95"]
    return out
