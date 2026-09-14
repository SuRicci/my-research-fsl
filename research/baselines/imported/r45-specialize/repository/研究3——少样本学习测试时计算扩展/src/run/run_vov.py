# -*- coding: utf-8 -*-
"""v3.5 value-of-view runner（十九轮 P0-4；二十轮 R3-P0 运行热修版）：dump /
train / eval 三模式 + argparse CLI（`python -m src.run.run_vov <dump|train|eval>`）。

权限纪律（v3.4 §4b-5 / v3.5 P0-8）：观测只经 `CachedViewProvider` 逐
(qid, view) 导出 posterior+nn_dist；score/derive 从不接触完整特征 tensor；
FeatureCache 全程 read_only=True，cache miss 立即失败（fail closed）。

二十轮热修要点（不改科学协议语义）：
- R3-P0-1/2：`run_eval`/`roll_reachable` 逐 episode 独立分配（独立 rank 池与
  预算池）；预算 API 只收 `budget_mult`，episode 预算 = budget_mult × 基档 ×
  本 episode query 数，调用者不得传绝对值；
- R3-P0-3：`run_dump` 用 `len(ep.classes)`（Episode 无 n_class 字段）；
- R3-P0-5：dataset 一律 canonical id（src/vov/datasets.py 唯一出处），
  cache scope 由映射决定，禁止调用者手工加 `_test`；
- R3-P0-6：`CachedViewProvider.episode_observations()` 复用
  `src.run.common.infer_episode` 同一方法路径（含 zs_logits/文本先验），
  method_impl 如实写 `tip_adapter` / `tip_adapter_cache_only`；
- R3-P0-7/8：train 入口血缘校验、eval 逐项验证 manifest/hash 链后才读 query；
- R3-P0-11：eval 落盘 .trace/.audit/.assignments 三类审计产物；
- R3-P0-12（方案1）：训练样本用 rollout 首次可行动作时刻的 causal ranked z
  快照（见 pipeline.roll_reachable）。

模式：
- dump：dataset loader → EpisodeGenerator → CachedViewProvider 逐 view 产出
  posterior/nn_dist/pred → query log（JSONL 原子写 + .meta.json 指纹侧车）；
- train：query log → transition 样本 → 逐级 cross-fitted（冻结前级 roll
  reachable，禁回填）→ 3 个 NetFlip 模型 + 血缘 manifest（含 method/backbone）；
- eval：query log + 模型目录（manifest/hash/血缘逐项验证，fail closed）→
  逐 episode DynamicVoVAllocator 分配 → 同 tier 直方图逐 episode 重放
  hard-first / random_matched（20 次固定排列）对照 → trace + score audit +
  assignments + 逐 transition 信号诊断 + oracle gap 落盘。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from ..allocators.dynamic_vov import (TIER_VIEWS, AuditedFeatureStore,
                                      DynamicVoVAllocator, QueryState,
                                      make_init_fn, make_observe_fn)
from ..allocators.net_flip import (ModelStatus, NetFlipTransitionModel,
                                   score_all)
from ..vov.pipeline import (assert_no_leak, atomic_write_json,
                            atomic_write_jsonl, build_transition_samples,
                            env_fingerprint, file_sha256, git_commit,
                            matched_histogram_assign, model_manifest,
                            protocol_hash, read_jsonl, roll_reachable,
                            train_transition_model, validate_query_log)

# 视图数成本（tier 索引 → 累计视图）。**注意口径（二十轮 P1-7）**：这是 view
# 计数单位，不是真实 FLOPs/C0——共享 support 视图成本与方法后处理成本未计入；
# 结果字段一律标 cost_unit="views"，正式报告不得称「真实 FLOPs」。
TIER_COSTS = [1.0, 2.0, 4.0, 8.0]

RESULT_SCHEMA_VERSION = "vov_result_v2"


# ---------------------------------------------------------------- 权限 provider

class CachedViewProvider:
    """权限包装的观测提供者：逐 (qid, view) 导出 {posterior, nn_dist}。

    内部持有 read_only FeatureCache 与方法名；外部（allocator/score/derive）
    只能拿到导出的观测 dict，永远接触不到完整特征 tensor（v3.4 §4b-5）。

    二十轮 R3-P0-5/6：
    - `dataset` 是 canonical id（src/vov/datasets.py），cache scope 由映射
      决定并随结果落盘；禁止手工传 `dtd_test` 这类 scope 名；
    - 逐 view 后验复用 `src.run.common.infer_episode` 同一方法路径：CLIP +
      可读类名时接入 zs 文本先验（标准 Tip-Adapter），无文本先验/DINO 时
      method_impl 明确写 `tip_adapter_cache_only`（与旧 run_adaptive.py 一致）。
    """

    def __init__(self, dataset: str, backbone: str, method: str,
                 cache_root: Optional[Path] = None, res: int = 224,
                 dataset_obj=None, device: str = "cuda",
                 method_params: Optional[dict] = None):
        from ..config import CACHE_ROOT, load_yaml
        from ..features.cache_io import FeatureCache
        from ..vov.datasets import cache_scope, canonical_id

        self.dataset = canonical_id(dataset)          # 手工 _test 在此被拒绝
        self.cache_scope = cache_scope(self.dataset)
        self.backbone = backbone
        self.method = method
        self.res = res
        root = Path(cache_root) if cache_root is not None else CACHE_ROOT
        self.fc = FeatureCache(f"cls/{self.cache_scope}", backbone, root=root,
                               read_only=True)        # miss 即 KeyError（fail）
        # 方法超参加载（与 run_adaptive.py 同出处：configs/methods.yaml）
        if method_params is not None:
            self.method_kw = dict(method_params)
        else:
            mcfg = load_yaml("methods")
            self.method_kw = dict(mcfg["cache_based"].get(method, {}))
        # 文本先验（R3-P0-6）：与旧 run_adaptive.py 同一判定与加载路径
        self.text_all = None
        self.logit_scale = 1.0
        self.text_prior = False
        if dataset_obj is not None:
            from ..features.text_cache import (has_text,
                                               text_features_for_dataset)
            if has_text(backbone, dataset_obj.classnames):
                self.text_all, self.logit_scale = text_features_for_dataset(
                    self.dataset, dataset_obj.classnames, backbone,
                    device=device)
                self.text_prior = True
        self.method_impl = self._resolve_method_impl()

    def _resolve_method_impl(self) -> str:
        """method_impl 口径（R3-P0-6）：tip_adapter 有文本先验 → 'tip_adapter'；
        无文本先验/DINO → 'tip_adapter_cache_only'；其余方法不消费文本先验，
        如实写方法名本身（text_prior 字段单独披露）。"""
        if self.method == "tip_adapter" and not self.text_prior:
            return "tip_adapter_cache_only"
        return self.method

    def _feats(self, idx: int, view: int) -> torch.Tensor:
        from ..views import cache_key
        return self.fc.get(cache_key(f"{self.cache_scope}#{idx}", view,
                                     self.res, self.backbone))

    def episode_observations(self, episode_id: str, support_idx, support_labels,
                             query_idx, query_labels, n_class: int,
                             max_views: int = 8,
                             episode_classes=None) -> List[dict]:
        """单 episode 逐 view 推理，返回 query log 行（split/dataset 等由调用方补）。

        后验计算 = `infer_episode(method, qf[:, v:v+1], sf[:, v:v+1], ...,
        Budget(V=1), zs_logits_fn=...)` 逐 view 切片调用——与旧 run_adaptive
        的 `_infer` 切片口径完全一致（同一方法分发、同一 softmax 路径）。
        """
        from ..budgets import Budget
        from ..features.text_cache import episode_zs_logits_fn
        from ..run.common import infer_episode

        zs_fn = None
        if self.text_prior and episode_classes is not None:
            text_ep = self.text_all[torch.as_tensor(
                [int(c) for c in episode_classes]).long()]
            zs_fn = episode_zs_logits_fn(text_ep, self.logit_scale)
        sf = torch.stack([self._feats(int(i), v) for i in support_idx
                          for v in range(max_views)])     # 仅 provider 内部可见
        sf = sf.view(len(support_idx), max_views, -1)
        sy = torch.as_tensor(support_labels)
        rows = []
        for qi, qidx in enumerate(query_idx):
            qid = f"{episode_id}#q{qi}"
            true_label = (None if query_labels is None
                          else int(query_labels[qi]))
            qf = torch.stack([self._feats(int(qidx), v)
                              for v in range(max_views)])      # [V, D]
            for v in range(max_views):
                res = infer_episode(self.method, qf[None, v:v + 1, :],
                                    sf[:, v:v + 1], sy, n_class,
                                    Budget(V=1, k=-1, T=0, r=self.res),
                                    zs_logits_fn=zs_fn, **self.method_kw)
                prob = res.probs[0].tolist()
                sim = (qf[v:v + 1] @ sf[:, v].T)[0]            # 特征已归一化
                nn = float(1.0 - sim.max().item())
                rows.append({"qid": qid, "episode_id": episode_id,
                             "view": v, "posterior": prob, "nn_dist": nn,
                             "pred": int(max(range(n_class),
                                             key=lambda c: prob[c])),
                             "true_label": true_label,
                             "method_impl": self.method_impl,
                             "text_prior": bool(self.text_prior)})
        return rows


# ---------------------------------------------------------------- dump

def _episode_split_sha256(episodes: Sequence) -> str:
    """episode 划分指纹：全部 support/query 索引与标签的 canonical hash。"""
    payload = []
    for ep in episodes:
        payload.append({"s": [int(i) for i in ep.support_idx],
                        "sy": [int(x) for x in ep.support_labels],
                        "q": [int(i) for i in ep.query_idx],
                        "qy": ([int(x) for x in ep.query_labels]
                               if ep.query_labels is not None else None),
                        "c": [int(c) for c in ep.classes]})
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run_dump(provider: CachedViewProvider, episodes: Sequence, dataset: str,
             method: str, backbone: str, seed: int, split: str,
             out_path: Path, configs_dir: Optional[Path] = None,
             max_views: int = 8) -> dict:
    """dump 模式：episodes（Episode 对象列表）→ query log（原子写）+
    `.meta.json` 指纹侧车（cache scope/index hash、episode split hash、
    method_impl/text_prior）。返回指纹 dict。"""
    from ..vov.datasets import canonical_id

    dataset = canonical_id(dataset)
    rows: List[dict] = []
    for ei, ep in enumerate(episodes):
        ep_id = f"{dataset}__{method}__{backbone}__s{seed}__e{ei}"
        ep_rows = provider.episode_observations(
            ep_id, ep.support_idx, ep.support_labels, ep.query_idx,
            getattr(ep, "query_labels", None), len(ep.classes),  # R3-P0-3
            max_views=max_views, episode_classes=ep.classes)
        for r in ep_rows:
            r.update({"dataset": dataset, "method": method, "backbone": backbone,
                      "seed": int(seed), "split": split})
        rows.extend(ep_rows)
    atomic_write_jsonl(out_path, rows)
    fp = {"protocol_sha256": protocol_hash(configs_dir) if configs_dir else "unknown",
          "method": method, "method_impl": provider.method_impl,
          "text_prior": bool(provider.text_prior),
          "method_params": dict(provider.method_kw),
          "backbone": backbone, "seed": int(seed),
          "split": split, "dataset": dataset,
          "cache_scope": provider.cache_scope,
          "cache_index_sha256": provider.fc.index_hash(),
          "n_rows": len(rows), "n_episodes": len(episodes),
          "episode_split_sha256": _episode_split_sha256(episodes),
          "query_log_sha256": file_sha256(out_path),
          "query_log_schema": "vov_query_log_v2",
          "git_commit": git_commit(), "env": env_fingerprint()}
    atomic_write_json(Path(str(out_path) + ".meta.json"), fp)
    return fp


# ---------------------------------------------------------------- train

def _validate_train_lineage(rows: Sequence[dict], target_fold: Optional[str],
                            train_datasets: Sequence[str],
                            seed_ids: Sequence[int],
                            method: Optional[str],
                            backbone: Optional[str]) -> dict:
    """run_train 入口血缘校验（R3-P0-7，全部 fail closed）：

    - 行内 dataset ⊆ 声明 train_datasets，且 target fold 不在训练集/日志中；
    - 行内 method/backbone 全日志单一，且与要训练的格子一致（参数显式传入时）；
    - 行内 seed ⊆ seed_ids；
    - (qid, episode_id, view) 无重复（由 validate_query_log 保证）。
    """
    meta = validate_query_log(rows)
    declared = set(train_datasets)
    actual = set(meta["datasets"])
    if not actual <= declared:
        raise RuntimeError(
            f"日志实际 dataset {sorted(actual)} 超出声明 train_datasets "
            f"{sorted(declared)}——血缘不符，fail closed（R3-P0-7）")
    if target_fold is not None:
        if target_fold in declared:
            raise RuntimeError(
                f"target fold {target_fold} 出现在 train_datasets——泄漏，fail closed")
        if target_fold in actual:
            raise RuntimeError(
                f"target fold {target_fold} 出现在训练日志中——泄漏，fail closed")
    for key, given in (("methods", method), ("backbones", backbone)):
        vals = meta[key]
        if len(vals) != 1:
            raise RuntimeError(
                f"训练日志 {key} 不单一: {vals}——格子隔离失败，fail closed")
        if given is not None and vals[0] != given:
            raise RuntimeError(
                f"训练日志 {key}={vals[0]} 与声明格子 {given} 不一致——fail closed")
    if not set(meta["seeds"]) <= set(seed_ids):
        raise RuntimeError(
            f"日志 seed {meta['seeds']} 超出声明 seed_ids {sorted(seed_ids)}"
            f"——fail closed")
    return meta


def run_train(query_log_paths: Sequence[Path], out_dir: Path,
              target_fold: Optional[str], train_datasets: Sequence[str],
              seed_ids: Sequence[int], budget_mult: float,
              version: str, method: Optional[str] = None,
              backbone: Optional[str] = None,
              configs_dir: Optional[Path] = None) -> Dict[int, str]:
    """train 模式：逐级 cross-fitted 训练（冻结前级 roll reachable，禁回填）。

    预算口径（R3-P0-2）：`budget_mult` 为每 query 乘率，episode 预算在
    roll_reachable 内按 `budget_mult × 基档 × 本 episode query 数` 计算；
    **不再接受绝对总成本**。
    训练特征口径（R3-P0-12 方案1）：transition t 的样本用「冻结前级 +
    后级 NA 占位」rollout 中、各 query 首次进入 t 评分组时刻的 causal
    ranked z 快照（与推理逐决策步同一变换）。

    返回 {transition: model_sha256}；模型 JSON + manifest 落 out_dir。
    """
    rows: List[dict] = []
    for p in query_log_paths:
        rows.extend(read_jsonl(p))
    meta = _validate_train_lineage(rows, target_fold, train_datasets, seed_ids,
                                   method, backbone)
    method = meta["methods"][0]
    backbone = meta["backbones"][0]
    samples = build_transition_samples(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proto = protocol_hash(configs_dir) if configs_dir else "unknown"
    models: Dict[int, NetFlipTransitionModel] = {}
    hashes: Dict[int, str] = {}
    upstream: Dict[str, str] = {}
    for t in (0, 1, 2):
        # 冷启动/晋级：transition t 用已冻结的 {0..t-1} 模型 + 后级 NA 占位
        # roll reachable 轨迹（t=0 时前缀为空 → 全体 NA 占位 → 仅首步 t0 组
        # 被 rank，等价「tier0 全体可达」的因果快照）
        roll_models = dict(models)
        for tt in range(t, 3):               # 后级 NA 占位（不升级但仍 rank 快照）
            na = NetFlipTransitionModel(tt, version=f"{version}-na-stub")
            na.status = ModelStatus.NA_RARE_EVENT
            roll_models[tt] = na
        reach, snapshots, _audits = roll_reachable(rows, roll_models, TIER_COSTS,
                                                   budget_mult)
        for s in samples:
            s["reachable"] = reach.get(s["qid"], [])
        m = train_transition_model(samples, t, version=f"{version}-t{t}",
                                   reachable_only=True, snapshots=snapshots)
        models[t] = m
        mp = out_dir / f"transition_{t}.json"
        hashes[t] = m.save(str(mp))
        man = model_manifest(
            m, hashes[t], target_fold, train_datasets,
            episode_ids=sorted({s["episode_id"] for s in samples}),
            seed_ids=seed_ids, policy_version=version,
            upstream_hashes=dict(upstream), samples=samples,
            protocol_sha256=proto, method=method, backbone=backbone)
        man["log_datasets"] = meta["datasets"]      # 实际日志 dataset（血缘核查）
        man["budget_mult"] = float(budget_mult)
        man["n_query_log_rows"] = len(rows)
        assert_no_leak(man)
        atomic_write_json(out_dir / f"transition_{t}.manifest.json", man)
        upstream[str(t)] = hashes[t]
    atomic_write_json(out_dir / "models_index.json",
                      {"hashes": {str(k): v for k, v in hashes.items()},
                       "version": version, "protocol_sha256": proto,
                       "method": method, "backbone": backbone,
                       "target_fold": target_fold,
                       "train_datasets": sorted(train_datasets)})
    return hashes


# ---------------------------------------------------------------- eval

def replay_accuracy(rows: Sequence[dict], assign: Dict[str, int]) -> float:
    """按最终 tier 重放精度：聚合该 query 前 TIER_VIEWS[tier] 个视图的
    均值后验 → argmax == true_label（true_label 缺失的 query 不计入）。

    视图完整性由 validate_query_log 在入口保证（fail closed），此处不再
    静默跳过缺视图 query。"""
    by_q: Dict[str, Dict[int, dict]] = {}
    for r in rows:
        by_q.setdefault(r["qid"], {})[r["view"]] = r
    n = hit = 0
    for qid, bv in by_q.items():
        tier = assign.get(qid, 0)
        V = TIER_VIEWS[tier]
        if not all(v in bv for v in range(V)):
            raise RuntimeError(f"qid={qid} 缺 tier{tier} 所需视图——fail closed")
        y = bv[0]["true_label"]
        if y is None:
            continue
        C = len(bv[0]["posterior"])
        mean_p = [sum(bv[v]["posterior"][c] for v in range(V)) / V
                  for c in range(C)]
        hit += int(max(range(C), key=lambda c: mean_p[c]) == y)
        n += 1
    return hit / n if n else 0.0


def _hist_of(assign: Dict[str, int]) -> Dict[int, int]:
    h: Dict[int, int] = {}
    for t in assign.values():
        h[t] = h.get(t, 0) + 1
    return h


def _tie_hash(qid: str, seed: int) -> int:
    """hard-first 难度并列决胜（二十轮 P1-6）：标签无关、与 qid 字典序/插入序
    无关的确定性 hash。"""
    return int(hashlib.md5(f"{qid}|tie|{seed}".encode("utf-8")).hexdigest()[:12],
               16)


def _load_verified_models(model_dir: Path, meta: dict,
                          configs_dir: Optional[Path],
                          expect_index_sha256: Optional[str] = None):
    """eval 模型与血缘逐项验证（R3-P0-8，全部 fail closed；全部通过才允许
    读取 query 做分配）。返回 (index, models, manifests)。"""
    model_dir = Path(model_dir)
    index_p = model_dir / "models_index.json"
    if expect_index_sha256 is not None and \
            file_sha256(index_p) != expect_index_sha256:
        raise RuntimeError(
            f"models_index.json hash 与 job 锁定值不符——fail closed（R3-P0-8）")
    index = json.loads(index_p.read_text(encoding="utf-8"))
    proto_now = protocol_hash(configs_dir) if configs_dir else None
    if configs_dir is not None and proto_now == "unknown":
        raise RuntimeError(f"协议 md 缺失于 {configs_dir}——fail closed")
    for key in ("datasets", "methods", "backbones"):
        if len(meta[key]) != 1:
            raise RuntimeError(
                f"eval 日志 {key} 不单一: {meta[key]}——fail closed")
    models: Dict[int, NetFlipTransitionModel] = {}
    manifests: Dict[int, dict] = {}
    for t in (0, 1, 2):                            # 3 模型齐 + hash 校验（P0-2）
        h = index["hashes"].get(str(t))
        if h is None:
            raise RuntimeError(f"models_index 缺 transition{t}——fail closed")
        models[t] = NetFlipTransitionModel.load(
            str(model_dir / f"transition_{t}.json"), expect_sha256=h)
        man_p = model_dir / f"transition_{t}.manifest.json"
        if not man_p.exists():
            raise RuntimeError(f"缺 {man_p.name}——血缘不可验证，fail closed")
        man = json.loads(man_p.read_text(encoding="utf-8"))
        manifests[t] = man
        # 逐项血缘验证
        if man.get("transition") != t or man.get("model_sha256") != h:
            raise RuntimeError(f"transition{t} manifest 与模型 hash 不符——fail closed")
        if man.get("protocol_sha256") != index.get("protocol_sha256"):
            raise RuntimeError(f"transition{t} manifest/index 协议 hash 不符")
        if proto_now is not None and \
                man.get("protocol_sha256") != proto_now:
            raise RuntimeError(
                f"transition{t} manifest 协议 hash {man.get('protocol_sha256')} "
                f"!= 当前 v3.5 重算值 {proto_now}——fail closed")
        for k, vals in (("method", meta["methods"]), ("backbone", meta["backbones"])):
            if man.get(k) is not None and man[k] != vals[0]:
                raise RuntimeError(
                    f"transition{t} manifest {k}={man[k]} != 日志 {vals[0]}"
                    f"——格子不符，fail closed")
        tf = man.get("target_fold")
        if tf is not None and tf != meta["datasets"][0]:
            raise RuntimeError(
                f"transition{t} manifest target_fold={tf} != 日志 dataset "
                f"{meta['datasets'][0]}——cross-fit 格子不符，fail closed")
        assert_no_leak(man)                        # 真实调用（显式 raise 版）
        expect_up = {str(tt): index["hashes"][str(tt)] for tt in range(t)}
        if man.get("upstream_model_hashes", {}) != expect_up:
            raise RuntimeError(
                f"transition{t} upstream hash 链不符——fail closed")
        if man.get("status") != models[t].status:
            raise RuntimeError(f"transition{t} status 与模型文件不符")
        if int(man.get("n_train", -1)) != models[t].n_train:
            raise RuntimeError(f"transition{t} n_train 与模型文件不符")
    return index, models, manifests


def run_eval(query_log_path: Path, model_dir: Path, budget_mult: float,
             out_path: Path, random_seed: int = 0,
             configs_dir: Optional[Path] = None,
             expect_index_sha256: Optional[str] = None,
             n_random_perms: int = 20) -> dict:
    """eval 模式：模型 manifest/hash/血缘逐项验证（fail closed，R3-P0-8）→
    **逐 episode 独立**动态分配（R3-P0-1/2：episode 内 rank 池 +
    budget_mult × 基档 × 本 episode query 数预算）→ 同直方图逐 episode
    hard-first / random_matched（20 次固定排列，报排列方差）对照 →
    trace + 全量 score audit + assignments + 逐 transition 信号诊断 +
    matched-cost oracle gap 全量落盘（R3-P0-11）。

    产物：`<out>` 结果 JSON、`<out>.trace.jsonl`（升档轨迹）、
    `<out>.audit.jsonl`（全部 query×transition attempts，含未升档）、
    `<out>.assignments.jsonl`（完整 final assignment + 逐 tier 预测/标签/成本）。
    """
    import random as _random

    from ..vov.diagnostics import (oracle_matched_gap, per_tier_predictions,
                                   reachable_vs_all, transition_diagnostics)

    rows = read_jsonl(query_log_path)
    meta = validate_query_log(rows)                 # P1-4：坏日志 fail closed
    index, models, _manifests = _load_verified_models(
        model_dir, meta, configs_dir, expect_index_sha256)

    by_ep: Dict[str, List[dict]] = {}
    for r in rows:
        by_ep.setdefault(r["episode_id"], []).append(r)

    assign: Dict[str, int] = {}
    trace_all: List[dict] = []
    audit_all: List[dict] = []
    audit_counts: Dict[str, int] = {}
    per_ep_stats: Dict[str, dict] = {}
    ep_costs: Dict[str, dict] = {}
    for ep in sorted(by_ep):
        feats: Dict[str, Dict[int, dict]] = {}
        for r in by_ep[ep]:
            feats.setdefault(r["qid"], {})[r["view"]] = {
                "posterior": r["posterior"], "nn_dist": r["nn_dist"]}
        store = AuditedFeatureStore(feats)
        states = [QueryState(qid=q) for q in sorted(feats)]
        init = make_init_fn(store)
        for s in states:
            init(s)
        # 每 episode 独立预算（R3-P0-2）：budget_mult × 基档 × 本 episode query 数
        total_budget = budget_mult * TIER_COSTS[0] * len(states)
        alloc = DynamicVoVAllocator(TIER_COSTS)
        trace = alloc.allocate(states,
                               lambda ss: score_all(ss, models, TIER_COSTS),
                               make_observe_fn(store), total_budget)
        for rec in trace:
            rec["episode_id"] = ep
            rec["budget_mult"] = float(budget_mult)
        for a in alloc.last_audit["attempts"]:
            a["episode_id"] = ep
            a["budget_mult"] = float(budget_mult)
        for k, v in alloc.last_audit["counts"].items():
            audit_counts[k] = audit_counts.get(k, 0) + v
        trace_all.extend(trace)
        audit_all.extend(alloc.last_audit["attempts"])
        per_ep_stats[ep] = dict(alloc.last_stats)
        ep_assign = {s.qid: s.current_tier for s in states}
        assign.update(ep_assign)
        ep_costs[ep] = {"n_query": len(states),
                        "cost_views": float(sum(TIER_COSTS[t]
                                                for t in ep_assign.values())),
                        "nominal_budget_views": float(total_budget)}
    acc_vov = replay_accuracy(rows, assign)

    # ---------------- 同直方图逐 episode 对照（v3.5 P0-4-5；二十轮 P1-5/6） ----------------
    diff: Dict[str, float] = {}
    view0 = {r["qid"]: r for r in rows if r["view"] == 0}
    for qid, r in view0.items():
        p = sorted(r["posterior"], reverse=True)
        diff[qid] = 1.0 - (p[0] - p[1] if len(p) > 1 else p[0])   # margin 小 = 难
    hard_assign: Dict[str, int] = {}
    rand_accs: List[float] = []
    for ep in sorted(by_ep):
        qids = sorted({r["qid"] for r in by_ep[ep]})
        hist = _hist_of({q: assign[q] for q in qids})
        hard_order = sorted(qids, key=lambda q: (-diff[q],
                                                 _tie_hash(q, random_seed)))
        hard_assign.update(matched_histogram_assign(hard_order, hist))
        # random_matched：每 episode 内 20 次固定排列（P1-5），报排列方差
        rng = _random.Random(f"vov-rm-{random_seed}-{ep}")
        for _ in range(n_random_perms):
            perm = list(qids)
            rng.shuffle(perm)
            rand_accs.append(replay_accuracy(
                by_ep[ep], matched_histogram_assign(perm, hist)))
    acc_hard = replay_accuracy(rows, hard_assign)
    acc_rand = (sum(rand_accs) / len(rand_accs)) if rand_accs else 0.0
    rand_std = (sum((a - acc_rand) ** 2 for a in rand_accs)
                / max(len(rand_accs) - 1, 1)) ** 0.5 if rand_accs else 0.0

    # ---------------- 逐 tier 预测 / flip / 诊断 / oracle（R3-P0-9/11） ----------------
    tier_preds = per_tier_predictions(rows)
    trans_diag = transition_diagnostics(rows, models)
    oracle = oracle_matched_gap(rows, assign, TIER_COSTS)
    rv_all = reachable_vs_all(rows, assign, diff)

    spent = sum(s["actual_cost"] for s in per_ep_stats.values())
    nominal = sum(s["nominal_budget"] for s in per_ep_stats.values())
    flip_counts = {str(t): c for t, c in
                   _transition_flip_counts(rows, tier_preds).items()}

    assignments = []
    for ep in sorted(by_ep):
        for r in sorted((r for r in by_ep[ep] if r["view"] == 0),
                        key=lambda r: r["qid"]):
            qid = r["qid"]
            tier = assign[qid]
            tp = tier_preds.get(qid, {})
            assignments.append({
                "qid": qid, "episode_id": ep, "final_tier": tier,
                "cost_views": float(TIER_COSTS[tier]),
                "true_label": r["true_label"],
                "pred_by_tier": {str(k): v for k, v in sorted(tp.items())},
                "correct_final": (None if r["true_label"] is None
                                  else bool(tp.get(tier) == r["true_label"]))})

    # 日志侧车 meta（dump 产物）：cache index hash 等指纹随结果落盘
    log_meta_p = Path(str(query_log_path) + ".meta.json")
    log_meta = (json.loads(log_meta_p.read_text(encoding="utf-8"))
                if log_meta_p.exists() else None)

    result = {"schema_version": RESULT_SCHEMA_VERSION,
              "acc_vov": acc_vov, "acc_hard_first": acc_hard,
              "acc_random_matched": acc_rand,
              "acc_random_matched_perm_std": rand_std,
              "delta_vs_hard": acc_vov - acc_hard,
              "delta_vs_random": acc_vov - acc_rand,
              "budget_mult": float(budget_mult),
              "budget_stats": {"nominal_budget": float(nominal),
                               "actual_cost": float(spent),
                               "budget_utilization": (float(spent / nominal)
                                                      if nominal > 0 else None),
                               "unused_budget": float(nominal - spent)},
              "per_episode_budget_stats": per_ep_stats,
              "per_episode_costs": ep_costs,
              "cost_unit": "views",          # P1-7：非真实 FLOPs/C0
              "score_audit_counts": audit_counts,
              "n_query": len(assign), "n_episode": len(by_ep),
              "tier_hist": {str(k): v for k, v in _hist_of(assign).items()},
              "dataset": meta["datasets"][0], "method": meta["methods"][0],
              "backbone": meta["backbones"][0], "seeds": meta["seeds"],
              "method_impl": (rows[0].get("method_impl")
                              if rows else None),
              "text_prior": (rows[0].get("text_prior") if rows else None),
              "flip_counts": flip_counts,
              "transition_diagnostics": trans_diag,
              "oracle": oracle,
              "reachable_vs_all": rv_all,
              "query_log_sha256": file_sha256(query_log_path),
              "log_meta": log_meta,
              "cache_index_sha256": (log_meta or {}).get("cache_index_sha256"),
              "model_hashes": index["hashes"],
              "models_index_sha256": file_sha256(
                  Path(model_dir) / "models_index.json"),
              "protocol_sha256": index.get("protocol_sha256",
                                           protocol_hash(configs_dir)
                                           if configs_dir else "unknown"),
              "git_commit": git_commit(), "env": env_fingerprint()}
    atomic_write_json(out_path, result)
    atomic_write_jsonl(Path(str(out_path) + ".trace.jsonl"), trace_all)
    atomic_write_jsonl(Path(str(out_path) + ".audit.jsonl"), audit_all)
    atomic_write_jsonl(Path(str(out_path) + ".assignments.jsonl"), assignments)
    return result


def _transition_flip_counts(rows: Sequence[dict],
                            tier_preds: Dict[str, Dict[int, int]]
                            ) -> Dict[int, Dict[str, int]]:
    """逐 transition 的 wrong→correct / correct→wrong / same 计数（R3-P0-11）。"""
    out = {t: {"w2c": 0, "c2w": 0, "same": 0, "n_labeled": 0} for t in (0, 1, 2)}
    by_q: Dict[str, dict] = {}
    for r in rows:
        if r["view"] == 0:
            by_q[r["qid"]] = r
    for qid, r in by_q.items():
        y = r["true_label"]
        if y is None:
            continue
        preds = tier_preds.get(qid, {})
        for t in (0, 1, 2):
            if t not in preds or t + 1 not in preds:
                continue
            a_ok = preds[t] == y
            b_ok = preds[t + 1] == y
            out[t]["n_labeled"] += 1
            if not a_ok and b_ok:
                out[t]["w2c"] += 1
            elif a_ok and not b_ok:
                out[t]["c2w"] += 1
            else:
                out[t]["same"] += 1
    return out


# ---------------------------------------------------------------- CLI（R3-P0-4）

def _cmd_dump(args) -> int:
    from ..data.splits import EpisodeGenerator
    from ..vov.datasets import load_dataset

    ds = load_dataset(args.dataset)               # canonical → split/scope 锁定
    gen = EpisodeGenerator(ds.labels, n_way=args.n_way, k_shot=args.shot,
                           q_per_cls=args.q_per_cls, seed=args.seed)
    episodes = gen.generate(args.episodes)
    provider = CachedViewProvider(args.dataset, args.backbone, args.method,
                                  cache_root=args.cache_root, res=args.res,
                                  dataset_obj=ds, device=args.device)
    fp = run_dump(provider, episodes, args.dataset, args.method, args.backbone,
                  args.seed, args.split, Path(args.out),
                  configs_dir=Path(args.configs) if args.configs else None)
    print(f"[run_vov dump] {fp['n_episodes']} episodes → {args.out} "
          f"({fp['n_rows']} rows, method_impl={fp['method_impl']}, "
          f"cache_scope={fp['cache_scope']})")
    return 0


def _cmd_train(args) -> int:
    hashes = run_train([Path(p) for p in args.logs], Path(args.out),
                       target_fold=args.target_fold,
                       train_datasets=args.train_datasets.split(","),
                       seed_ids=[int(s) for s in args.seeds.split(",")],
                       budget_mult=args.budget_mult, version=args.version,
                       method=args.method, backbone=args.backbone,
                       configs_dir=Path(args.configs) if args.configs else None)
    for t, h in sorted(hashes.items()):
        print(f"[run_vov train] transition{t}: sha256={h[:16]}…")
    return 0


def _cmd_eval(args) -> int:
    out = Path(args.out)
    if args.budgets:                              # 单 job 多预算重放（R3-P0-10）
        mults = [float(x) for x in args.budgets.split(",")]
        rows = []
        for b in mults:
            r = run_eval(Path(args.log), Path(args.models), b,
                         Path(f"{out}.B{b:g}.json"),
                         random_seed=args.random_seed,
                         configs_dir=Path(args.configs) if args.configs else None,
                         expect_index_sha256=args.models_index_sha256)
            rows.append({"budget_mult": b, "acc_vov": r["acc_vov"],
                         "acc_hard_first": r["acc_hard_first"],
                         "acc_random_matched": r["acc_random_matched"],
                         "delta_vs_hard": r["delta_vs_hard"],
                         "delta_vs_random": r["delta_vs_random"],
                         "oracle_gap": r["oracle"]["gap_matched_cost"]})
        atomic_write_json(out, {"schema_version": RESULT_SCHEMA_VERSION,
                                "dataset": r["dataset"], "method": r["method"],
                                "backbone": r["backbone"], "budget_rows": rows})
        print(f"[run_vov eval] {len(mults)} 预算重放完成 → {out}")
        return 0
    res = run_eval(Path(args.log), Path(args.models), args.budget_mult, out,
                   random_seed=args.random_seed,
                   configs_dir=Path(args.configs) if args.configs else None,
                   expect_index_sha256=args.models_index_sha256)
    print(f"[run_vov eval] acc_vov={res['acc_vov']:.4f} "
          f"Δhard={res['delta_vs_hard']:+.4f} Δrand={res['delta_vs_random']:+.4f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="run_vov",
        description="v3.5 value-of-view runner：dump/train/eval（二十轮热修版）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    dp = sub.add_parser("dump", help="dataset/cache → query log（真实 episode）")
    dp.add_argument("--dataset", required=True, help="canonical id（如 dtd；"
                    "禁止手工加 _test）")
    dp.add_argument("--method", required=True)
    dp.add_argument("--backbone", required=True)
    dp.add_argument("--seed", type=int, default=0)
    dp.add_argument("--episodes", type=int, default=200)
    dp.add_argument("--n-way", type=int, default=5)
    dp.add_argument("--shot", type=int, default=1)
    dp.add_argument("--q-per-cls", type=int, default=15)
    dp.add_argument("--split", default="test")
    dp.add_argument("--res", type=int, default=224)
    dp.add_argument("--cache-root", default=None)
    dp.add_argument("--device", default="cuda")
    dp.add_argument("--configs", default=None)
    dp.add_argument("--out", required=True)
    dp.set_defaults(fn=_cmd_dump)

    tp = sub.add_parser("train", help="query log → 3 个 transition 模型（血缘校验）")
    tp.add_argument("--logs", nargs="+", required=True)
    tp.add_argument("--out", required=True)
    tp.add_argument("--target-fold", default=None)
    tp.add_argument("--train-datasets", required=True)
    tp.add_argument("--method", required=True)
    tp.add_argument("--backbone", required=True)
    tp.add_argument("--seeds", default="0")
    tp.add_argument("--budget-mult", type=float, required=True,
                    help="每 query 预算乘率（episode 预算 = mult × 基档 × "
                         "本 episode query 数；不接受绝对值）")
    tp.add_argument("--version", required=True)
    tp.add_argument("--configs", default=None)
    tp.set_defaults(fn=_cmd_train)

    ep = sub.add_parser("eval", help="query log + 模型 → 逐 episode 分配评估")
    ep.add_argument("--log", required=True)
    ep.add_argument("--models", required=True)
    ep.add_argument("--out", required=True)
    ep.add_argument("--budget-mult", type=float, default=1.5)
    ep.add_argument("--budgets", default=None,
                    help="逗号列表（如 1.25,1.5,1.75,2.5,3,6）：单 job 内多预算"
                         "重放，逐预算输出 + 汇总行")
    ep.add_argument("--random-seed", type=int, default=0)
    ep.add_argument("--models-index-sha256", default=None)
    ep.add_argument("--configs", default=None)
    ep.set_defaults(fn=_cmd_eval)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
