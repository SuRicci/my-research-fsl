# -*- coding: utf-8 -*-
"""三分类净 flip transition 模型 + batch rank scorer（gonogo_protocol_v3_5，十九轮）。

每个 transition（1→2 / 2→4 / 4→8）一个 multinomial LogisticRegression，预测净
flip 结果 {−1, 0, +1}（c→w / 不变 / w→c），utility = P(+1|z) − P(−1|z)，
score = utility / Δcost。

v3.5 相对 v3.4 的改动（十九轮评审）：
- **P0-1 batch rank**：新增 `score_all(states, models, tier_costs)`——按
  transition 分组当前 eligible states，组内逐 raw z 分量同值平均秩
  （x′=(r−1)/(n−1)，n=1 时 0.5），再送模型；训练侧同口径（episode×transition
  ×reachable set）秩化，禁止全数据集统一 rank。模型级 `score()` 保留为 raw
  口径调试路径，正式分配必须走 `score_all`。
- **P0-2 fail-closed**：UNFITTED 调 score/predict_proba 立即 RuntimeError；
  `score_all` 遇缺 transition 模型立即 RuntimeError；仅显式 NA
  （稀有事件/缺类）允许 utility=0。
- **P0-3 source 改名**：hard_first_fallback → missing_feature_no_upgrade
  （v3.4 起 hard-first 已不参与动态分配，fallback=utility 0 不升档）。
- **P1-2 load 校验**：`load(path, expect_sha256=...)` 可强制校验内容 hash。
- multinomial 显式锁定：拟合后断言 `coef_.shape == (3, d)`（不依赖版本默认）。
"""
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from sklearn.linear_model import LogisticRegression

CLASSES = (-1, 0, 1)          # c→w / 不变 / w→c（列序锁定，加载断言）
MIN_CLASS_COUNT = 20          # 稀有事件阈值（训练折内）
# 各 transition 的特征维度与特征键（协议 v3.4 §4b：z1=3 / z1+z2=6 / z1+z2+z4=9）
TRANSITION_FEATURE_DIM = {0: 3, 1: 6, 2: 9}
TIER_FEATURE_KEYS = {0: ("z1",), 1: ("z1", "z2"), 2: ("z1", "z2", "z4")}

# score source 五态（v3.5 P0-3；前三个会进入 ScoreResult.source）
SRC_MODEL = "net_flip_model"
SRC_NA = "na_no_upgrade"
SRC_MISSING = "missing_feature_no_upgrade"


class ModelStatus:
    READY = "ready"
    NA_MISSING_CLASS = "na_missing_class"
    NA_RARE_EVENT = "na_rare_event"
    UNFITTED = "unfitted"


@dataclass
class ScoreResult:
    """结构化打分型返回值（禁止靠外层捕获异常做控制流）。"""
    value: float              # 进入候选列表的 score（fallback/NA 恒 0，不主动升档）
    source: str               # net_flip_model / na_no_upgrade / missing_feature_no_upgrade
    utility: float
    p_plus: Optional[float]
    p_minus: Optional[float]


def _softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def average_rank_columns(mat: Sequence[Sequence[float]]) -> List[List[float]]:
    """逐分量同值平均秩归一化：x′=(r−1)/(n−1) ∈ [0,1]；n=1 时全 0.5（v3.5 P0-1）。"""
    n = len(mat)
    if n == 0:
        return []
    d = len(mat[0])
    if n == 1:
        return [[0.5] * d]
    out = [[0.0] * d for _ in range(n)]
    for j in range(d):
        order = sorted(range(n), key=lambda i: mat[i][j])
        ranks = [0.0] * n
        i = 0
        while i < n:
            k = i
            while k + 1 < n and mat[order[k + 1]][j] == mat[order[i]][j]:
                k += 1
            avg = (i + k) / 2.0 + 1.0          # 秩次 1..n 的同值平均秩
            for t in range(i, k + 1):
                ranks[order[t]] = avg
            i = k + 1
        for i in range(n):
            out[i][j] = (ranks[i] - 1.0) / (n - 1.0)
    return out


class NetFlipTransitionModel:
    """单个 transition 的三分类净 flip 概率模型。"""

    def __init__(self, transition: int, version: str = "unversioned"):
        if transition not in TRANSITION_FEATURE_DIM:
            raise ValueError(f"未知 transition {transition}（合法：0/1/2）")
        self.transition = transition
        self.feature_dim = TRANSITION_FEATURE_DIM[transition]
        self.version = version
        self.status = ModelStatus.UNFITTED
        self._clf: Optional[LogisticRegression] = None
        self._coef: Optional[List[List[float]]] = None     # [3, d]，行序 = CLASSES
        self._intercept: Optional[List[float]] = None      # [3]
        self.class_prior: Optional[Dict[int, float]] = None  # 训练折经验先验（BSS 参考）
        self.n_train: int = 0                               # 训练样本数（结果指纹用）
        self.converged: Optional[bool] = None               # lbfgs 收敛标记（P1-1）

    # ------------------------------------------------------------ 构造

    @classmethod
    def from_params(cls, transition: int, coef: Sequence[Sequence[float]],
                    intercept: Sequence[float],
                    class_prior: Optional[Dict[int, float]] = None,
                    classes: Sequence[int] = CLASSES,
                    version: str = "unversioned") -> "NetFlipTransitionModel":
        """由参数构造 READY 模型（序列化加载/测试注入路径）。"""
        if tuple(classes) != CLASSES:
            raise AssertionError(
                f"classes 顺序 {tuple(classes)} != 锁定 {CLASSES}（v3.4 §4c）")
        m = cls(transition, version=version)
        coef = [[float(x) for x in row] for row in coef]
        intercept = [float(x) for x in intercept]
        if len(coef) != 3 or any(len(r) != m.feature_dim for r in coef):
            raise AssertionError(
                f"coef 形状 {[len(coef), len(coef[0]) if coef else 0]} != "
                f"[3, {m.feature_dim}]")
        if len(intercept) != 3:
            raise AssertionError("intercept 必须为 3 维")
        m._coef = coef
        m._intercept = intercept
        m.class_prior = (dict(class_prior) if class_prior is not None
                         else {c: 1.0 / 3 for c in CLASSES})
        m.status = ModelStatus.READY
        return m

    def fit(self, Z: Sequence[Sequence[float]], y: Sequence[int]) -> str:
        """拟合（原始类先验，禁止 class_weight；Z 必须已按 v3.5 rank 口径变换）。
        返回 status。

        求解器显式锁定（二十轮 P1-1）：solver='lbfgs' / penalty='l2' /
        class_weight=None / C=1.0 / max_iter=1000 / random_state=0；
        ConvergenceWarning 被捕获并记入 `self.converged`（随 payload 落盘，
        不静默放过也不中断——未收敛结果需在审计中可见）。
        """
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        counts = {c: sum(1 for t in y if t == c) for c in CLASSES}
        missing = [c for c in CLASSES if counts[c] == 0]
        if missing:
            self.status = ModelStatus.NA_MISSING_CLASS
            return self.status
        if min(counts.values()) < MIN_CLASS_COUNT:
            self.status = ModelStatus.NA_RARE_EVENT
            return self.status
        for row in Z:
            if len(row) != self.feature_dim:
                raise AssertionError(
                    f"训练特征维度 {len(row)} != {self.feature_dim}（泄漏防线）")
        n = len(y)
        self.class_prior = {c: counts[c] / n for c in CLASSES}
        self.n_train = n
        clf = LogisticRegression(solver="lbfgs", penalty="l2",
                                 class_weight=None, C=1.0, max_iter=1000,
                                 fit_intercept=True, random_state=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit([list(r) for r in Z], list(y))
        self.converged = not any(issubclass(w.category, ConvergenceWarning)
                                 for w in caught)
        # multinomial 显式锁定：三分类问题必须产出 3 行系数（不依赖版本默认）
        if tuple(int(c) for c in clf.classes_) != CLASSES:
            raise AssertionError(
                f"sklearn classes_ {tuple(clf.classes_)} != 锁定 {CLASSES}")
        if clf.coef_.shape != (3, self.feature_dim):
            raise AssertionError(
                f"coef_ 形状 {clf.coef_.shape} != (3, {self.feature_dim})——"
                f"multinomial 行为不符合锁定口径")
        self._clf = clf
        self._coef = [[float(x) for x in row] for row in clf.coef_]
        self._intercept = [float(x) for x in clf.intercept_]
        self.status = ModelStatus.READY
        return self.status

    # ------------------------------------------------------------ 推理

    def _require_ready(self) -> None:
        if self.status == ModelStatus.UNFITTED:
            raise RuntimeError(
                f"transition{self.transition} 模型 UNFITTED——fail closed "
                f"（v3.5 P0-2）：训练器漏跑/模型未加载时不允许静默打分")
        if self.status != ModelStatus.READY:
            raise AssertionError("NA 模型不得推理（score/score_all 已短路）")

    def predict_proba(self, z: Sequence[float]) -> List[float]:
        """手写 softmax(W z + b)，列序 = CLASSES（与 sklearn predict_proba 对拍）。"""
        self._require_ready()
        if len(z) != self.feature_dim:
            raise AssertionError(f"z 维度 {len(z)} != {self.feature_dim}")
        logits = [sum(w * x for w, x in zip(row, z)) + b
                  for row, b in zip(self._coef, self._intercept)]
        return _softmax(logits)

    def utility(self, z: Sequence[float]) -> float:
        p = self.predict_proba(z)
        return p[CLASSES.index(1)] - p[CLASSES.index(-1)]

    def feature_vector(self, state) -> Optional[List[float]]:
        """从 QueryState 拼该 transition 的 raw 特征向量。

        - state 携带该 tier 未获准特征键 → AssertionError（泄漏防线）；
        - 所需 z 分量缺失 → None（部署 fallback，不抛 KeyError）；
        - 分量在但长度错 → AssertionError（数据管线 bug）。
        """
        want = TIER_FEATURE_KEYS[self.transition]
        extra = set(state.features) - set(want)
        if extra:
            raise AssertionError(
                f"qid={state.qid} transition{self.transition} 携带未获准特征 "
                f"{sorted(extra)}（信息泄漏防线）")
        z: List[float] = []
        for k in want:
            if k not in state.features:
                return None
            z += list(state.features[k])
        if len(z) != self.feature_dim:
            raise AssertionError(
                f"qid={state.qid} transition{self.transition} 特征维度 {len(z)} "
                f"!= {self.feature_dim}")
        return z

    def score(self, state, delta_cost: float) -> ScoreResult:
        """raw 口径单 query 打分（调试/单例路径）。**正式分配必须走 score_all
        batch rank 口径**（v3.5 P0-1），本方法不做 episode 秩化。"""
        if self.status == ModelStatus.UNFITTED:
            raise RuntimeError(
                f"transition{self.transition} 模型 UNFITTED——fail closed（v3.5 P0-2）")
        if self.status != ModelStatus.READY:
            return ScoreResult(0.0, SRC_NA, 0.0, None, None)
        z = self.feature_vector(state)
        if z is None:
            return ScoreResult(0.0, SRC_MISSING, 0.0, None, None)
        u = self.utility(z)
        p = self.predict_proba(z)
        return ScoreResult(u / max(float(delta_cost), 1e-12), SRC_MODEL,
                           u, p[CLASSES.index(1)], p[CLASSES.index(-1)])

    # ------------------------------------------------------------ 评估（原始分布）

    def brier(self, Z: Sequence[Sequence[float]], y: Sequence[int]) -> float:
        """multiclass Brier = mean Σ_k (p_k − 1[y=c_k])²。"""
        tot = 0.0
        for z, t in zip(Z, y):
            p = self.predict_proba(z)
            tot += sum((p[k] - (1.0 if t == CLASSES[k] else 0.0)) ** 2
                       for k in range(3))
        return tot / len(y)

    def log_loss(self, Z: Sequence[Sequence[float]],
                 y: Sequence[int], eps: float = 1e-15) -> float:
        tot = 0.0
        for z, t in zip(Z, y):
            p = max(self.predict_proba(z)[CLASSES.index(t)], eps)
            tot -= math.log(p)
        return tot / len(y)

    def bss(self, Z: Sequence[Sequence[float]], y: Sequence[int]) -> float:
        """Brier skill score = 1 − Brier/Brier_prior，参考 = 训练折经验先验。"""
        if self.class_prior is None:
            raise AssertionError("无 class_prior（未 fit）——BSS 参考缺失")
        prior = [self.class_prior[c] for c in CLASSES]
        tot = 0.0
        for t in y:
            tot += sum((prior[k] - (1.0 if t == CLASSES[k] else 0.0)) ** 2
                       for k in range(3))
        brier_prior = tot / len(y)
        if brier_prior <= 0:
            return 0.0
        return 1.0 - self.brier(Z, y) / brier_prior

    # ------------------------------------------------------------ 序列化

    def to_payload(self) -> dict:
        return {"transition": self.transition, "version": self.version,
                "classes": list(CLASSES), "feature_dim": self.feature_dim,
                "status": self.status, "coef": self._coef,
                "intercept": self._intercept, "n_train": self.n_train,
                "converged": self.converged,
                "class_prior": ({str(c): self.class_prior[c] for c in CLASSES}
                                if self.class_prior is not None else None)}

    def save(self, path: str) -> str:
        """纯参数 JSON **原子写**落盘（tmp + replace，二十轮 P1-2），返回内容
        sha256（结果指纹用）。"""
        import os

        blob = json.dumps(self.to_payload(), sort_keys=True)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(blob)
        os.replace(tmp, path)
        return digest

    @classmethod
    def content_hash(cls, path: str) -> str:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    @classmethod
    def load(cls, path: str,
             expect_sha256: Optional[str] = None) -> "NetFlipTransitionModel":
        """加载；给 expect_sha256 时强制校验内容 hash（v3.5 P1-2/P0-2）。"""
        if expect_sha256 is not None:
            actual = cls.content_hash(path)
            if actual != expect_sha256:
                raise RuntimeError(
                    f"模型 {path} hash 不符：{actual} != 预期 {expect_sha256}"
                    f"——fail closed")
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("status") != ModelStatus.READY:
            m = cls(d["transition"], version=d.get("version", "unversioned"))
            m.status = d.get("status", ModelStatus.UNFITTED)
            m.n_train = d.get("n_train", 0)
            m.converged = d.get("converged")
            m.class_prior = ({int(k): v for k, v in d["class_prior"].items()}
                             if d.get("class_prior") else None)
            return m
        m = cls.from_params(
            d["transition"], d["coef"], d["intercept"],
            class_prior=({int(k): v for k, v in d["class_prior"].items()}
                         if d.get("class_prior") else None),
            classes=d.get("classes", CLASSES),
            version=d.get("version", "unversioned"))
        m.n_train = d.get("n_train", 0)
        m.converged = d.get("converged")
        return m


def score_all(states, models: Dict[int, NetFlipTransitionModel],
              tier_costs: Sequence[float], on_rank=None) -> Dict[str, ScoreResult]:
    """v3.5 P0-1 batch scorer：按 transition 分组 eligible states，组内逐分量
    同值平均秩，再送对应 NetFlip 模型。返回 {qid: ScoreResult}。

    fail-closed：任一当前 transition 缺模型 → RuntimeError；UNFITTED →
    RuntimeError；显式 NA → 该组全体 na_no_upgrade；特征缺失 →
    missing_feature_no_upgrade（utility=0，不主动升档）。

    `on_rank`（可选，R3-P0-12）：每次组内 rank 完成后回调
    `on_rank(transition, [(qid, ranked_z), ...])`——rollout 用它快照每个
    query 首次可行动作时刻的 causal ranked z（训练/推理逐点同变换）。
    """
    costs = [float(c) for c in tier_costs]
    groups: Dict[int, list] = {}
    for s in states:
        t = s.current_tier
        if t in TRANSITION_FEATURE_DIM:      # tier3（8 视图）无后续 transition
            groups.setdefault(t, []).append(s)
    out: Dict[str, ScoreResult] = {}
    for t, grp in groups.items():
        m = models.get(t)
        if m is None:
            raise RuntimeError(
                f"transition{t} 模型缺失——fail closed（v3.5 P0-2）：primary "
                f"实验启动必须断言 3 个 transition 模型全部存在")
        if m.status == ModelStatus.UNFITTED:
            raise RuntimeError(
                f"transition{t} 模型 UNFITTED——fail closed（v3.5 P0-2）")
        dc = costs[t + 1] - costs[t]
        raw = [m.feature_vector(s) for s in grp]   # None = 特征缺失
        present = [i for i, z in enumerate(raw) if z is not None]
        ranked: Dict[int, List[float]] = {}
        if present:
            mat = [raw[i] for i in present]
            for i, zr in zip(present, average_rank_columns(mat)):
                ranked[i] = zr
        # R3-P0-12：rank 变换是 eligible 组的性质、与模型是否 NA 无关——
        # 快照回调必须在 NA 短路**之前**触发（NA 占位 rollout 也要产出
        # 后级 transition 的首次可行动作时刻 ranked z，供训练使用）
        if on_rank is not None and ranked:
            on_rank(t, [(grp[i].qid, list(ranked[i])) for i in sorted(ranked)])
        if m.status != ModelStatus.READY:    # 显式 NA（稀有事件/缺类）
            for s in grp:
                out[s.qid] = ScoreResult(0.0, SRC_NA, 0.0, None, None)
            continue
        for i, s in enumerate(grp):
            if i not in ranked:
                out[s.qid] = ScoreResult(0.0, SRC_MISSING, 0.0, None, None)
                continue
            u = m.utility(ranked[i])
            p = m.predict_proba(ranked[i])
            out[s.qid] = ScoreResult(u / max(dc, 1e-12), SRC_MODEL, u,
                                     p[CLASSES.index(1)], p[CLASSES.index(-1)])
    return out
