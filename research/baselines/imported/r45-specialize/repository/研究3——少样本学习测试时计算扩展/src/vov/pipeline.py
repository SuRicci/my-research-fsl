# -*- coding: utf-8 -*-
"""v3.5 运行管线核心（十九轮 P0-4/P0-5/P0-6/P0-7/P0-9；二十轮 R3-P0 热修）：
query log schema、transition 样本与 reachable 轨迹、训练驱动、分析、status、
jobs 账本。scripts/vov_prepare.py / vov_train.py / vov_analyze.py / vov_status.py /
vov_jobs.py 均为本模块的薄壳。所有落盘一律原子写（tmp + replace）。

query log 行 schema（JSONL；dump 侧另附 method_impl/text_prior 两列）：
    {"qid", "episode_id", "view", "posterior": [...], "nn_dist": float,
     "pred": int, "true_label": int|null, "dataset", "method", "backbone",
     "seed", "split"}

二十轮运行热修（不改科学协议语义）：
- R3-P0-1/2：`roll_reachable` 按 episode_id 分组逐 episode 独立 rollout，
  预算口径改为 `budget_mult × 基档成本 × 该 episode query 数`（调用者不得
  再传全日志绝对预算）；
- R3-P0-12（方案1）：rollout 在每个决策步对当前 eligible 组做 rank（与推理
  完全同一变换），并快照每个 (qid, transition) **首次进入评分组**时刻的
  causal ranked z；训练样本用这些快照（见 `roll_reachable`/`train_transition_model`）；
- R3-P0-7/13：`assert_no_leak` 显式 raise（python -O 下生效）；`.done` 写入
  job canonical hash + result hash + 结果文件 hash，resume 重算全部；
- P1：`protocol_hash` 重算协议 md（不信任 .sha256.txt）；query log 严格
  schema 校验 `validate_query_log`（fail closed）。
"""
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..allocators.dynamic_vov import (TIER_VIEWS, AuditedFeatureStore,
                                      DynamicVoVAllocator, QueryState,
                                      make_init_fn, make_observe_fn)
from ..allocators.net_flip import (CLASSES, MIN_CLASS_COUNT, ModelStatus,
                                   NetFlipTransitionModel,
                                   average_rank_columns, score_all)

QUERY_LOG_SCHEMA = ["qid", "episode_id", "view", "posterior", "nn_dist",
                    "pred", "true_label", "dataset", "method", "backbone",
                    "seed", "split"]
# dump 侧附加列（可选读、必须写）：method_impl（tip_adapter / tip_adapter_cache_only /
# protonet…）与 text_prior（bool）——R3-P0-6 口径披露
QUERY_LOG_OPTIONAL = ["method_impl", "text_prior"]
N_VIEWS = 8                                # tier 集 {1,2,4,8} → 每 query 恰好 8 视图
RANK_SPEC = ("episode×transition×首次可行动作时刻 eligible 组 causal ranked 快照, "
             "x'=(r−1)/(n−1), n=1→0.5（R3-P0-12 方案1：训练/推理逐点同变换）")


# ---------------------------------------------------------------- 指纹与原子写

def env_fingerprint() -> dict:
    """环境指纹（v3.5 P1-2 + 二十轮 P1-3）：版本记录 + pip freeze hash +
    git commit/dirty + 协议外全部依赖，不依赖任何版本默认值。"""
    import numpy
    import sklearn
    fp = {"python": sys.version.split()[0], "platform": platform.platform(),
          "numpy": numpy.__version__, "sklearn": sklearn.__version__,
          "git_commit": git_commit(), "git_dirty": git_dirty(),
          "pip_freeze_sha256": pip_freeze_hash()}
    try:
        import torch
        fp["torch"] = torch.__version__
        fp["cuda"] = torch.version.cuda
        fp["cudnn"] = torch.backends.cudnn.version()
        fp["gpu"] = (torch.cuda.get_device_name(0)
                     if torch.cuda.is_available() else "cpu-only")
    except ImportError:
        fp["torch"] = None
    return fp


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """工作区是否有未提交改动（env lock / job schema 用）。"""
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                             capture_output=True, text=True,
                             timeout=10).stdout
        return bool(out.strip())
    except Exception:
        return True                      # 无法判定 → 按 dirty 处理（保守）


def pip_freeze() -> str:
    """当前解释器的 pip freeze 全文（排序、去空行）；失败返回空串。"""
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=60).stdout
        return "\n".join(sorted(l for l in out.splitlines() if l.strip()))
    except Exception:
        return ""


def pip_freeze_hash() -> str:
    txt = pip_freeze()
    return hashlib.sha256(txt.encode("utf-8")).hexdigest() if txt else "unavailable"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_hash(configs_dir: Path, name: str = "gonogo_protocol_v3_5") -> str:
    """协议指纹：**重算** `<name>.md` 内容 sha256（二十轮 P1-2：不信任
    .sha256.txt）。若旁挂的 .sha256.txt 存在且与重算值不符 → RuntimeError
    （fail closed）；md 缺失返回 "unknown"。"""
    md = Path(configs_dir) / f"{name}.md"
    if not md.exists():
        return "unknown"
    h = hashlib.sha256(md.read_bytes()).hexdigest()
    ref = Path(configs_dir) / f"{name}.sha256.txt"
    if ref.exists() and ref.read_text(encoding="utf-8").strip() != h:
        raise RuntimeError(
            f"协议文件 {md} 重算 sha256={h} 与 {ref} 记录不符——fail closed")
    return h


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> str:
    """严格 JSON（无 NaN）原子写；返回内容 sha256。"""
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False)
    atomic_write_text(path, blob)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def atomic_write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    for r in rows:
        json.dumps(r, allow_nan=False)               # 严格校验（提前失败）
    blob = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                   for r in rows)
    atomic_write_text(path, blob)


def read_jsonl(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- query log 严格 schema

def validate_query_log(rows: Sequence[dict], n_views: int = N_VIEWS) -> dict:
    """query log 严格 schema 校验（二十轮 P1-4，fail closed）。

    任一违规立即 ValueError（确认性评估不得少算/静默覆盖 query）：
    - 缺必需字段 / 类型错误；
    - view 越界、每 qid 视图缺/重复（必须恰好 0..n_views-1 各一次）；
    - posterior 非有限 / 不归一（容差 1e-3）/ 类别维度全日志不一致；
    - pred/true_label 越界（true_label 允许 null）；
    - (qid, view) 重复行。
    返回 {"n_query", "n_episode", "n_class", "datasets", "methods",
          "backbones", "seeds", "episodes"} 摘要（血缘核对用）。
    """
    n_class: Optional[int] = None
    seen_qv = set()
    views_by_q: Dict[str, set] = {}
    meta = {"datasets": set(), "methods": set(), "backbones": set(),
            "seeds": set(), "episodes": set()}
    for li, r in enumerate(rows):
        missing = [k for k in QUERY_LOG_SCHEMA if k not in r]
        if missing:
            raise ValueError(f"query log 第 {li} 行缺字段 {missing}——fail closed")
        if not isinstance(r["view"], int) or not (0 <= r["view"] < n_views):
            raise ValueError(f"第 {li} 行 view={r['view']} 越界 [0,{n_views})")
        key = (r["qid"], r["view"])
        if key in seen_qv:
            raise ValueError(f"(qid,view) 重复: {key}——fail closed")
        seen_qv.add(key)
        views_by_q.setdefault(r["qid"], set()).add(r["view"])
        p = r["posterior"]
        if (not isinstance(p, list) or not p
                or any(not isinstance(x, (int, float)) or not math.isfinite(x)
                       for x in p)):
            raise ValueError(f"第 {li} 行 posterior 非有限/为空: qid={r['qid']}")
        if any(x < 0 for x in p) or abs(sum(p) - 1.0) > 1e-3:
            raise ValueError(f"第 {li} 行 posterior 不归一（sum={sum(p)}）")
        if n_class is None:
            n_class = len(p)
        elif len(p) != n_class:
            raise ValueError(
                f"第 {li} 行 posterior 维度 {len(p)} != 日志口径 {n_class}")
        if not isinstance(r["pred"], int) or not (0 <= r["pred"] < n_class):
            raise ValueError(f"第 {li} 行 pred={r['pred']} 越界 [0,{n_class})")
        y = r["true_label"]
        if y is not None and (not isinstance(y, int) or not (0 <= y < n_class)):
            raise ValueError(f"第 {li} 行 true_label={y} 越界 [0,{n_class})")
        if not math.isfinite(float(r["nn_dist"])):
            raise ValueError(f"第 {li} 行 nn_dist 非有限")
        meta["datasets"].add(r["dataset"])
        meta["methods"].add(r["method"])
        meta["backbones"].add(r["backbone"])
        meta["seeds"].add(r["seed"])
        meta["episodes"].add(r["episode_id"])
    want = set(range(n_views))
    for qid, vs in views_by_q.items():
        if vs != want:
            raise ValueError(
                f"qid={qid} 视图集合 {sorted(vs)} != 0..{n_views - 1}（缺/重复视图）"
                f"——fail closed")
    return {"n_query": len(views_by_q), "n_episode": len(meta["episodes"]),
            "n_class": n_class or 0,
            "datasets": sorted(meta["datasets"]),
            "methods": sorted(meta["methods"]),
            "backbones": sorted(meta["backbones"]),
            "seeds": sorted(meta["seeds"]),
            "episodes": sorted(meta["episodes"])}


# ---------------------------------------------------------------- transition 样本

def _pred_correct(rows_by_view: Dict[int, dict], tier: int) -> Optional[bool]:
    """tier t 的均值后验预测是否正确；true_label 缺失 → None。"""
    V = TIER_VIEWS[tier]
    probs, y = [], None
    for v in range(V):
        r = rows_by_view.get(v)
        if r is None:
            return None
        probs.append(r["posterior"])
        y = r["true_label"]
    if y is None:
        return None
    C = len(probs[0])
    mean_p = [sum(p[c] for p in probs) / V for c in range(C)]
    pred = max(range(C), key=lambda c: mean_p[c])
    return bool(pred == y)


def build_transition_samples(rows: Sequence[dict]) -> List[dict]:
    """query log → 逐 (qid, transition) 样本（all-query 口径；reachable 标记由
    roll_reachable 填充）。

    返回行：{qid, episode_id, transition, z_raw, y, dataset, method, backbone,
    seed}；y ∈ {−1,0,+1}（净 flip：升档纠正 +1 / 破坏 −1 / 不变 0），
    当前档或下一档观测不全/true_label 缺失 → 该 transition 不出样本。
    z_raw 由受控固定导出（dynamic_vov.derive_z*）从 query log 观测计算。
    """
    from ..allocators.dynamic_vov import derive_z1, derive_z2, derive_z4

    by_q: Dict[str, Dict[int, dict]] = {}
    meta: Dict[str, dict] = {}
    for r in rows:
        by_q.setdefault(r["qid"], {})[r["view"]] = r
        meta[r["qid"]] = {k: r[k] for k in
                          ("episode_id", "dataset", "method", "backbone", "seed")}
    out = []
    for qid, bv in sorted(by_q.items()):
        obs = {v: {"posterior": bv[v]["posterior"], "nn_dist": bv[v]["nn_dist"]}
               for v in bv}
        z1 = derive_z1(obs[0]) if 0 in obs else None
        z2 = derive_z2(obs[0], obs[1]) if (0 in obs and 1 in obs) else None
        z4 = (derive_z4([obs[v] for v in range(4)])
              if all(v in obs for v in range(4)) else None)
        c0 = _pred_correct(bv, 0)
        c1 = _pred_correct(bv, 1)
        c2 = _pred_correct(bv, 2)
        # transition 0（1→2）：z1；transition 1（2→4）：z1+z2；
        # transition 2（4→8）：z1+z2+z4
        spec = [(0, z1, c0, c1), (1, (z1 + z2) if z1 and z2 else None, c1, c2)]
        c3 = _pred_correct(bv, 3)
        spec.append((2, (z1 + z2 + z4) if (z1 and z2 and z4) else None, c2, c3))
        for t, z, ca, cb in spec:
            if z is None or ca is None or cb is None:
                continue
            y = 1 if (not ca and cb) else (-1 if (ca and not cb) else 0)
            out.append({"qid": qid, "transition": t, "z_raw": z, "y": y,
                        "reachable": None, **meta[qid]})
    return out


def roll_reachable(rows: Sequence[dict], models: Dict[int, NetFlipTransitionModel],
                   tier_costs: Sequence[float], budget_mult: float
                   ) -> Tuple[Dict[str, List[int]], Dict[Tuple[str, int], List[float]],
                              List[dict]]:
    """reachable 轨迹（v3.4 P1-1/v3.5 P0-7；二十轮 R3-P0-1/2/12 热修）。

    - **逐 episode 独立 rollout**（R3-P0-1）：按 episode_id 分组，每个 episode
      各自构造 store/states/allocator，rank 池与预算池都是 episode 内的；
    - **预算单位**（R3-P0-2）：`budget_mult` 为每 query 乘率，episode 预算 =
      `budget_mult × tier_costs[0] × 该 episode query 数`；调用者不得传绝对值；
    - **causal ranked z 快照**（R3-P0-12 方案1）：每个决策步 score_all 对当前
      eligible 组重算 rank（与推理完全同一变换，经 `on_rank` 回调取出）；每个
      (qid, transition) 记录其**首次进入评分组**那一决策步的 ranked z。
      训练样本使用这些快照——训练/推理变换逐点相同，无需新协议版本。

    返回 (reachable, snapshots, audits)：
    - reachable[qid] = 至少被评分过一次的 transition 升序列表（因果可达口径：
      进入过对应该 transition 的 eligible 评分组；未进入者不得回填训练）；
    - snapshots[(qid, t)] = 首次可行动作时刻的 ranked z（list[float]）；
    - audits = 逐 episode 的 allocator.last_audit（审计聚合用）。
    """
    by_ep: Dict[str, List[dict]] = {}
    for r in rows:
        by_ep.setdefault(r["episode_id"], []).append(r)
    reachable: Dict[str, List[int]] = {}
    snapshots: Dict[Tuple[str, int], List[float]] = {}
    audits: List[dict] = []
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
        local: Dict[Tuple[str, int], List[float]] = {}

        def on_rank(transition: int, pairs: List[Tuple[str, List[float]]],
                    _fs=local) -> None:
            for qid, zr in pairs:
                _fs.setdefault((qid, transition), list(zr))   # 仅首见快照

        alloc = DynamicVoVAllocator(tier_costs)
        # 每 episode 独立预算：budget_mult × 基档 × 本 episode query 数
        total_budget = float(budget_mult) * float(tier_costs[0]) * len(states)
        alloc.allocate(states,
                       lambda ss: score_all(ss, models, tier_costs,
                                            on_rank=on_rank),
                       make_observe_fn(store), total_budget)
        for (qid, t), zr in local.items():
            snapshots[(qid, t)] = zr
        for s in states:
            reachable[s.qid] = sorted(t for (q, t) in local if q == s.qid)
        audits.append({"episode_id": ep, **(alloc.last_audit or {})})
    return reachable, snapshots, audits


# ---------------------------------------------------------------- 训练

def train_transition_model(samples: Sequence[dict], transition: int,
                           version: str, reachable_only: bool = True,
                           snapshots: Optional[Dict[Tuple[str, int],
                                                  List[float]]] = None
                           ) -> NetFlipTransitionModel:
    """训练一个 transition 模型（reachable_only 时只用 reachable 样本，
    **禁止回填**——不足即 NA）。

    特征口径（R3-P0-12 方案1）：
    - `snapshots` 给定时（正式路径）：样本特征 = rollout 记录的该 (qid,
      transition) 首次可行动作时刻 causal ranked z，reachability 即快照存在性
      ——与推理逐决策步 eligible 组 rank 是同一变换，不再二次 rank；
    - `snapshots=None`（仅单元测试/调试的遗留路径）：退回 episode×transition
      组内 ex-post average-rank（与推理不严格同口径，正式训练禁用）。
    """
    sel = [s for s in samples if s["transition"] == transition]
    if reachable_only:
        sel = [s for s in sel
               if s.get("reachable") and transition in s["reachable"]]
    Z, y = [], []
    if snapshots is not None:
        for s in sel:
            key = (s["qid"], transition)
            if key not in snapshots:
                # reachable 标记与快照不一致 = 管线 bug，fail closed
                raise RuntimeError(
                    f"reachable 样本 {key} 缺 causal ranked 快照——fail closed")
            Z.append(list(snapshots[key]))
            y.append(s["y"])
    else:
        # 遗留路径：episode 内分组 rank（同 episode 的该 transition 样本为一组）
        by_ep: Dict[str, List[dict]] = {}
        for s in sel:
            by_ep.setdefault(s["episode_id"], []).append(s)
        for ep, grp in sorted(by_ep.items()):
            ranked = average_rank_columns([s["z_raw"] for s in grp])
            for s, zr in zip(grp, ranked):
                Z.append(zr)
                y.append(s["y"])
    m = NetFlipTransitionModel(transition, version=version)
    if not Z:                                   # 无样本 → 缺类 NA
        m.status = ModelStatus.NA_MISSING_CLASS
        return m
    m.fit(Z, y)
    return m


def class_counts(samples: Sequence[dict], transition: int) -> Dict[str, int]:
    out = {str(c): 0 for c in CLASSES}
    for s in samples:
        if s["transition"] == transition:
            out[str(s["y"])] += 1
    return out


def model_manifest(model: NetFlipTransitionModel, model_sha256: str,
                   target_fold: Optional[str], train_datasets: Sequence[str],
                   episode_ids: Sequence[str], seed_ids: Sequence[int],
                   policy_version: str, upstream_hashes: Dict[str, str],
                   samples: Sequence[dict], protocol_sha256: str,
                   method: Optional[str] = None, backbone: Optional[str] = None,
                   n_train: Optional[int] = None,
                   configs_note: str = "") -> dict:
    """模型血缘 manifest（v3.5 P0-7 字段全集 + 二十轮 R3-P0-7：method/backbone）。"""
    counts = class_counts(samples, model.transition)
    return {"target_fold": target_fold,
            "train_datasets": sorted(train_datasets),
            "method": method,
            "backbone": backbone,
            "episode_ids": sorted(episode_ids),
            "seed_ids": sorted(seed_ids),
            "policy_version": policy_version,
            "upstream_model_hashes": dict(upstream_hashes),
            "transition": model.transition,
            "status": model.status,
            "n_train": (model.n_train if n_train is None else int(n_train)),
            "class_counts": counts,
            "na_reason": (None if model.status == ModelStatus.READY
                          else model.status),
            "min_class_count_threshold": MIN_CLASS_COUNT,
            "rank_spec": RANK_SPEC,
            "model_sha256": model_sha256,
            "protocol_sha256": protocol_sha256,
            "git_commit": git_commit(),
            "env": env_fingerprint(),
            "note": configs_note}


def assert_no_leak(manifest: dict) -> None:
    """血缘断言（二十轮 R3-P0-7）：显式 raise——`python -O` 下也必须生效。

    - target fold 不得出现在训练数据集列表（P0-7）；
    - 训练日志实际 dataset 集合（manifest["log_datasets"]，若记录）必须 ⊆
      声明的 train_datasets。
    """
    tf = manifest.get("target_fold")
    if tf is not None and tf in manifest.get("train_datasets", []):
        raise RuntimeError(
            f"目标 fold {tf} 泄漏进训练集 {manifest['train_datasets']}——fail closed")
    log_ds = manifest.get("log_datasets")
    if log_ds is not None:
        extra = set(log_ds) - set(manifest.get("train_datasets", []))
        if extra:
            raise RuntimeError(
                f"日志实际 dataset {sorted(extra)} 超出声明 train_datasets "
                f"{manifest['train_datasets']}——fail closed")


# ---------------------------------------------------------------- 分析

def hierarchical_bootstrap(paired: Dict[str, Sequence[float]],
                           n_boot: int = 10000, seed: int = 0
                           ) -> Dict[str, float]:
    """两级配对 bootstrap（fold → fold 内 run）：fold 间训练依赖下只作
    描述性不确定度（v3.4 P1-2 措辞纪律）。

    paired: {fold: [逐 run 差值]}；返回 mean 与 95% 百分位区间。
    """
    import numpy as np

    folds = sorted(paired)
    vals = [np.asarray(paired[f], dtype=np.float64) for f in folds]
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        fi = rng.randint(0, len(folds), len(folds))
        per = []
        for i in fi:
            v = vals[i]
            per.append(v[rng.randint(0, len(v), len(v))].mean())
        diffs.append(float(np.mean(per)))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = float(np.mean([v.mean() for v in vals]))
    return {"diff": point, "ci95": (float(lo), float(hi)),
            "n_fold": len(folds), "descriptive_only": True}


def matched_histogram_assign(order: List[str], hist: Dict[int, int]
                             ) -> Dict[str, int]:
    """同 tier 直方图重分配：按 order（升档优先级从高到低）填充档位。
    hist = {tier: count}（vov 实际直方图）；返回 {qid: tier}。"""
    out: Dict[str, int] = {}
    pos = 0
    for tier in sorted(hist, reverse=True):     # 高档先分给优先级最高的
        for _ in range(hist[tier]):
            if pos < len(order):
                out[order[pos]] = tier
                pos += 1
    return out


# ---------------------------------------------------------------- jobs 账本（P0-9/R3-P0-10/13）

# 锁全的任务 schema（R3-P0-10）：除标识外还锁定 episode 协议、tier/预算集、
# 方法超参、text-prior 状态、cache/model/env/code 指纹、command 与输出 schema 版本。
# 预算形式锁死为「一次 dump、单 job 内在全部预算重放、逐预算输出行」——
# job 携带完整 budget_mults 列表，不拆成单预算 job。
JOB_SCHEMA = ["job_id", "dataset", "cache_scope", "method", "method_impl",
              "text_prior", "backbone", "seed", "stage", "shot", "n_way",
              "q_per_cls", "n_episodes", "episode_split_sha256", "tier_views",
              "budget_mults", "primary_budget", "method_params",
              "protocol_sha256", "cache_index_sha256", "models_index_sha256",
              "model_dir", "env_lock_sha256", "code_commit", "code_dirty",
              "command", "output", "output_schema_version"]

BUDGET_MULTS = [1.25, 1.5, 1.75, 2.5, 3.0, 6.0]   # 主 B=1.5 + 五个副预算（锁定）
PRIMARY_BUDGET = 1.5
OUTPUT_SCHEMA_VERSION = "vov_result_v2"

# build_jobs 的默认协议段（可被 spec 覆盖；emit 时应显式给全）
DEFAULT_JOB_SPEC = {"shot": 1, "n_way": 5, "q_per_cls": 15, "n_episodes": 200,
                    "episode_split_sha256": None, "tier_views": [1, 2, 4, 8],
                    "budget_mults": list(BUDGET_MULTS),
                    "primary_budget": PRIMARY_BUDGET, "method_params": {},
                    "cache_index_sha256": None, "models_index_sha256": None,
                    "model_dir": None, "env_lock_sha256": None,
                    "code_commit": None, "code_dirty": None, "command": None,
                    "output_schema_version": OUTPUT_SCHEMA_VERSION}


def hash_job(j: dict) -> str:
    """完整 job 的 canonical sha256（除 job_id 外全字段；R3-P0-13）。"""
    base = {k: j[k] for k in j if k != "job_id"}
    return hashlib.sha256(json.dumps(base, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


def hash_payload(obj) -> str:
    """任意 JSON 可序列化对象的 canonical sha256（result 指纹用）。"""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False)
                          .encode("utf-8")).hexdigest()


def make_job_id(j: dict) -> str:
    return hash_job(j)[:16]


def build_jobs(datasets: Sequence[str], methods: Sequence[str],
               backbones: Sequence[str], seeds: Sequence[int], stage: str,
               protocol_sha256: str, out_dir: str,
               spec: Optional[dict] = None) -> List[dict]:
    """生成锁全的 job 列表。`spec` 覆盖 DEFAULT_JOB_SPEC 的协议段；
    dataset 若在 canonical 表内则自动填 cache_scope（禁止手工 _test，
    表外名字——仅合成测试——原样透传）。"""
    from .datasets import CANONICAL_DATASETS, cache_scope as _cs

    fields = dict(DEFAULT_JOB_SPEC)
    if spec:
        fields.update(spec)
    jobs = []
    for ds in datasets:
        scope = _cs(ds) if ds in CANONICAL_DATASETS else ds
        for me in methods:
            for bb in backbones:
                for sd in seeds:
                    j = {"dataset": ds, "cache_scope": scope, "method": me,
                         "backbone": bb, "seed": int(sd), "stage": stage,
                         "protocol_sha256": protocol_sha256,
                         "output": str(Path(out_dir)
                                       / f"{ds}__{me}__{bb}__s{sd}")}
                    j.update(fields)
                    # text_prior/method_impl 默认按 method×backbone 静态推断；
                    # emit --resolve-text-prior 时用真实类名覆写
                    j.setdefault("text_prior", None)
                    j.setdefault("method_impl", None)
                    j["job_id"] = make_job_id(j)
                    jobs.append(j)
    ids = [j["job_id"] for j in jobs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("job_id 冲突——jobs 生成失败")
    outs = [j["output"] for j in jobs]
    if len(outs) != len(set(outs)):
        raise RuntimeError("output 路径冲突——jobs 生成失败")
    return jobs


def write_jobs(path: Path, jobs: Sequence[dict]) -> None:
    """不可变 jobs.jsonl：已存在且内容不同即拒绝（防运行中改矩阵）。"""
    blob = "".join(json.dumps(j, ensure_ascii=False, sort_keys=True) + "\n"
                   for j in jobs)
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8") != blob:
        raise RuntimeError(f"jobs.jsonl 已存在且内容不同，拒绝覆盖: {path}")
    atomic_write_text(path, blob)


def _valid_done(done_path: Path, job: dict) -> bool:
    """resume 校验（R3-P0-13）：.done 必须 schema 完整，且重算全部 hash：

    - job_sha256 == 当前 job canonical hash（改了 job 任何字段即失效）；
    - result_sha256 == payload 内 result 的重算 hash（篡改 result 即失效）；
    - result.result_files 中每个结果文件存在且内容 sha256 一致。
    """
    try:
        d = json.loads(done_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not (d.get("job_id") == job["job_id"]
            and d.get("protocol_sha256") == job.get("protocol_sha256")
            and d.get("status") == "ok"):
        return False
    if d.get("job_sha256") != hash_job(job):
        return False
    result = d.get("result")
    if result is None or d.get("result_sha256") != hash_payload(result):
        return False
    for rf in (result.get("result_files") or []):
        p = Path(rf.get("path", ""))
        if not p.exists() or file_sha256(p) != rf.get("sha256"):
            return False
    return True


def run_jobs(jobs: Sequence[dict], workdir: Path,
             executor: Callable[[dict], dict], resume: bool = True) -> dict:
    """执行账本：.running → 原子 rename .done；失败存 stderr/exit code 且不
    覆盖旧失败日志；resume 只跳过通过**全链 hash 校验**的 .done。

    executor(job) -> 结果 dict（可 JSON 序列化；若含 "result_files":
    [{path, sha256}]，落账前逐文件重算校验，不符按失败处理）；抛异常即失败。
    .done 写入完整 job canonical hash、command、result hash（R3-P0-13）。
    返回 status 计数（含 invalid=因校验不过而重跑的旧 .done 数）。
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    done = failed = skipped = invalid = 0
    for j in jobs:
        done_p = workdir / f"{j['job_id']}.done"
        if resume and done_p.exists():
            if _valid_done(done_p, j):
                skipped += 1
                continue
            invalid += 1                        # 篡改/过期 → 重跑并覆盖
        run_p = workdir / f"{j['job_id']}.running"
        atomic_write_json(run_p, {"job_id": j["job_id"], "status": "running",
                                  "started": time.time()})
        try:
            result = executor(j)
            for rf in (result.get("result_files") or []):   # 落账前重算校验
                p = Path(rf["path"])
                if not p.exists() or file_sha256(p) != rf["sha256"]:
                    raise RuntimeError(
                        f"executor 结果文件 hash 不符: {p}——fail closed")
        except Exception as e:                     # noqa: BLE001 - 失败落账
            n_prev = len(list(workdir.glob(f"{j['job_id']}.failed.*.json")))
            fail_p = workdir / f"{j['job_id']}.failed.{n_prev + 1}.json"
            atomic_write_json(fail_p, {"job_id": j["job_id"], "status": "failed",
                                       "error": repr(e), "exit_code": 1})
            run_p.unlink(missing_ok=True)
            failed += 1
            continue
        payload = {"job_id": j["job_id"], "status": "ok",
                   "protocol_sha256": j.get("protocol_sha256"),
                   "job_sha256": hash_job(j),
                   "command": j.get("command"),
                   "result": result, "result_sha256": hash_payload(result),
                   "finished": time.time()}
        atomic_write_json(done_p.with_suffix(".done.tmp"), payload)
        os.replace(done_p.with_suffix(".done.tmp"), done_p)   # 原子 rename
        run_p.unlink(missing_ok=True)
        done += 1
    return {"done": done, "failed": failed, "skipped": skipped,
            "invalid_rerun": invalid}


def status_report(jobs: Sequence[dict], workdir: Path) -> dict:
    """随时可用（R3-P0-13）：expected/completed/failed/missing/duplicate +
    invalid_done（校验不过的 .done）/orphan_files（不属于任何 job 的账本或
    结果文件）/duplicate_outputs（多 job 同 output）。

    调用方（vov_status/preflight）遇 failed/missing/invalid/duplicate/orphan
    任一非零必须非零退出。
    """
    workdir = Path(workdir)
    expected = len(jobs)
    ids = [j["job_id"] for j in jobs]
    completed = failed = invalid = 0
    for j in jobs:
        p = workdir / f"{j['job_id']}.done"
        if p.exists():
            if _valid_done(p, j):
                completed += 1
            else:
                invalid += 1
        failed += len(list(workdir.glob(f"{j['job_id']}.failed.*.json")))
    missing = expected - completed
    duplicate = len(ids) - len(set(ids))
    outs = [j.get("output") for j in jobs]
    duplicate_outputs = len(outs) - len(set(outs))
    # 孤儿文件：workdir 内不属于任何 job_id 的账本/结果文件
    known = set(ids)
    orphans = []
    if workdir.is_dir():
        for f in sorted(workdir.iterdir()):
            stem = f.name.split(".")[0]
            if stem not in known:
                orphans.append(f.name)
        for j in jobs:                       # job 输出目录里的孤儿不动（归 job 管）
            pass
    return {"expected": expected, "completed": completed, "failed_logs": failed,
            "missing": missing, "duplicate_job_ids": duplicate,
            "invalid_done": invalid, "duplicate_outputs": duplicate_outputs,
            "orphan_files": orphans}
