# -*- coding: utf-8 -*-
"""P1 级免训练 baseline（实验计划 §4）：PT+MAP / APE / TDA / TransCLIP / MuSC。

全部跑在缓存特征上（纯特征空间运算），统一签名：

    fn(query_feats [Q,D], support_feats [S,D], support_labels [S], n_class,
       zs_logits=None [Q,C], text_feats=None [C,D], k=-1, rounds=0, **kw) -> logits [Q,C]

特征约定与 cache_based.py 一致：输入特征已 L2 归一化，返回 logits（infer_episode 再 softmax）。

预算轴语义（b=(V,k,T,r)；V/r 由 run 层视图/分辨率特征生效，此处仅 k/T）：
- pt_map:    T=rounds 为传导 Sinkhorn-OT 迭代数（0=归纳式 PT+MAP，>0 为完整 PT-MAP）；k 不用。
- ape:       k=检索宽度（top-k 亲和剪枝，同 tip_adapter）；通道精简比例由 feat_ratio 控制（非预算轴）。
- tda:       k=正/负缓存每类容量（-1 → 官方默认 pos=3/neg=2）；T 不用（官方为单遍流式，固定 1 遍）。
- transclip: T=rounds 为 BMM 外层迭代数（官方 10）；k=亲和图邻居数（-1 → 官方默认 3）。
- musc:      k=LN 动态子集大小（-1 → 默认 10）；T=rounds 为互评分传播轮数（0/1=单轮）。

文本先验：ape/tda/transclip/musc 需要 zs_logits（registry 标 requires_text=True，
miniImageNet 等无类名数据集由 run 层跳过）；ape 另需 text_feats（通道选择准则与
cache-text KL 用）；pt_map 纯视觉、不需要文本。

出处（公式逐项核对，见各函数 docstring）：
- PT+MAP: Hu et al., "Leveraging the Feature Distribution in Transfer-based Few-Shot
  Learning" (ICANN 2021, arXiv:2006.03806)；实现参考 sicara/easy-few-shot-learning。
- APE: Zhu et al., "Not All Features Matter: Enhancing Few-shot CLIP with Adaptive
  Prior Refinement" (ICCV 2023, arXiv:2304.01195)；参考官方 yangyangyang127/APE。
- TDA: Karmanov et al., "Efficient Test-Time Adaptation of Vision-Language Models"
  (CVPR 2024, arXiv:2403.18293)；参考官方 kdiAAA/TDA。
- TransCLIP: Zanella et al., "Boosting Vision-Language Models with Transduction"
  (NeurIPS 2024, arXiv:2406.01837)；参考官方 MaxZanella/transduction-for-vlms。
- MuSC: Li et al., "MuSC: Zero-Shot Industrial Anomaly Classification and Segmentation
  with Mutual Scoring of the Unlabeled Images" (ICLR 2024, arXiv:2401.16753)。
  注：MuSC 原为 AD 方法，此处为其互评分机制的分类适配版（详见函数 docstring）。
"""
import math
from typing import Optional

import torch
import torch.nn.functional as F


def _onehot(labels: torch.Tensor, n_class: int) -> torch.Tensor:
    return F.one_hot(labels.long(), n_class).float()


def _topk_prune(affinity: torch.Tensor, k: int) -> torch.Tensor:
    """top-k 之外的亲和置 -1（exp(-β(1-(-1)))=exp(-2β)≈0），与 tip_adapter 一致。"""
    if 0 < k < affinity.shape[1]:
        topv, topi = affinity.topk(k, dim=1)
        mask = torch.zeros_like(affinity).scatter_(1, topi, 1.0)
        affinity = affinity * mask - (1 - mask)
    return affinity


# ---------------------------------------------------------------------------
# PT+MAP（power transform + MAP 高斯后验；rounds>0 时为完整传导 PT-MAP）
# ---------------------------------------------------------------------------

def _power_transform(feats: torch.Tensor, ref_mean: torch.Tensor, beta: float) -> torch.Tensor:
    """PT-MAP 预处理：中心化 → 保号幂变换 x←sign(x)|x|^β → L2 归一化（论文 Eq.1）。

    中心参考为 support 均值（归纳式口径；传导文献亦可用 support∪query 均值，
    此处固定 support 均值以避免依赖 query 批大小，可复现）。
    """
    x = feats - ref_mean
    x = torch.sign(x) * x.abs().clamp_min(0).pow(beta)
    return F.normalize(x, dim=-1)


def _sinkhorn(cost: torch.Tensor, lam: float, epsilon: float = 1e-6,
              max_iter: int = 1000) -> torch.Tensor:
    """Sinkhorn-Knopp 最优传输（easyfsl PTMAP.compute_optimal_transport 口径）。

    返回 [Q, C] 传输计划：行和≈1，列和≈Q/C（类均衡约束）。
    """
    n_q, n_c = cost.shape
    factor = max(1, n_q // n_c)
    plan = torch.exp(-lam * cost)
    plan = plan / plan.sum()
    for _ in range(max_iter):
        row_sums = plan.sum(1)
        plan = plan * (1.0 / (row_sums + 1e-10)).unsqueeze(1)
        plan = plan * (factor / (plan.sum(0) + 1e-10)).unsqueeze(0)
        if torch.max(torch.abs(row_sums - plan.sum(1))) < epsilon:
            break
    return plan


def pt_map_logits(query_feats: torch.Tensor, support_feats: torch.Tensor,
                  support_labels: torch.Tensor, n_class: int,
                  zs_logits: Optional[torch.Tensor] = None,
                  text_feats: Optional[torch.Tensor] = None,
                  k: int = -1, rounds: int = 0,
                  power_beta: float = 0.5, map_kappa: float = 1.0, reg: float = 1e-4,
                  sinkhorn_lambda: float = 10.0, lr: float = 0.2, **_) -> torch.Tensor:
    """PT+MAP：power transform(β=0.5) + 类均值向全局均值收缩的 MAP 高斯后验分类器。

    归纳式（rounds=0）：
    1) 预处理：x ← sign(x)|x|^β（support 均值中心化后），L2 归一化；
    2) MAP 类均值：正态先验 μ_c ~ N(μ̄, Σ/κ) 的共轭后验
       μ_c = (Σ_{x∈c} x + κ·μ̄) / (n_c + κ)，μ̄ 为 support 全局均值；
    3) 共享协方差高斯判别（与 gda 同型）：logits = -0.5·(x-μ_c)ᵀΣ⁻¹(x-μ_c)。

    传导式（rounds>0，完整 PT-MAP，Hu et al. 2021；easyfsl 实现口径）：
    Sinkhorn OT 软分配（λ=sinkhorn_lambda，类均衡列约束）→ 原型按软分配加权更新
    （μ ← μ + lr·(加权均值-μ)，lr=0.2），迭代 rounds 轮；返回 log P_OT 为 logits。

    预算轴：T=rounds（0 归纳 / >0 传导迭代）；k 不使用（support 全部入模）。
    zs_logits/text_feats 不使用（纯视觉方法）。
    """
    ref = support_feats.mean(dim=0, keepdim=True)
    q = _power_transform(query_feats, ref, power_beta)
    s = _power_transform(support_feats, ref, power_beta)

    mu_bar = s.mean(dim=0)
    protos = torch.stack([
        (s[support_labels == c].sum(dim=0) + map_kappa * mu_bar)
        / ((support_labels == c).sum().float() + map_kappa)
        for c in range(n_class)
    ])

    if rounds <= 0:
        D = s.shape[1]
        centered = s - protos[support_labels]
        cov = centered.T @ centered / max(1, s.shape[0] - 1)
        cov = cov + reg * torch.eye(D, device=cov.device, dtype=cov.dtype)
        prec = torch.linalg.pinv(cov)
        xm = q.unsqueeze(1) - protos.unsqueeze(0)              # [Q, C, D]
        maha = torch.einsum("qcd,de,qce->qc", xm, prec, xm)
        return -0.5 * maha                                     # 类先验均匀，log π 为常数

    sup_1h = _onehot(support_labels, n_class)
    for _ in range(rounds):
        dist = torch.cdist(q, protos) ** 2
        plan = _sinkhorn(dist, sinkhorn_lambda)
        all_a = torch.cat([sup_1h, plan], dim=0)
        all_f = torch.cat([s, q], dim=0)
        new_protos = (all_a.T @ all_f) / all_a.sum(dim=0).unsqueeze(1).clamp_min(1e-12)
        protos = protos + lr * (new_protos - protos)
    plan = _sinkhorn(torch.cdist(q, protos) ** 2, sinkhorn_lambda)
    return (plan + 1e-12).log()


# ---------------------------------------------------------------------------
# APE（training-free）：通道先验精简 + 三边亲和（test image / cache / text）
# ---------------------------------------------------------------------------

def ape_logits(query_feats: torch.Tensor, support_feats: torch.Tensor,
               support_labels: torch.Tensor, n_class: int,
               zs_logits: Optional[torch.Tensor] = None,
               text_feats: Optional[torch.Tensor] = None,
               k: int = -1, rounds: int = 0,
               alpha: float = 3.0, beta: float = 1.0, gamma: float = 0.1,
               feat_ratio: float = 0.7, w_sim: float = 0.5, w_var: float = 0.5,
               **_) -> torch.Tensor:
    """APE 免训练版（Zhu et al. ICCV 2023；官方 yangyangyang127/APE main.py APE()）。

    1) 通道选择（prior refinement，论文 §3.1 + 官方 cal_criterion）：
       S_k = mean_{c1≠c2} f̄[c1,k]·f̄[c2,k]（f̄ 为每类 [文本原型; 类内 support] 的均值行），
       V_k = Var_c(text[c,k])（文本类间方差）；
       criterion_k = -w_sim·S_k + w_var·V_k，取 criterion 最大的 feat_ratio·D 个通道。
       （官方 RN50 配 feat_num 400–700/1024，此处按比例折算；w 官方逐集不同，取中位 [0.5,0.5]）
    2) 精简通道后各自重新 L2 归一化；R_fF = q'·s'ᵀ；
    3) cache-text 分歧修正（三边亲和）：key_p = softmax(s'·t'ᵀ)，
       R_FW = exp(γ·KL(onehot‖key_p)) = exp(-γ·log2 key_p[y])，soft_values = L ⊙ R_FW；
    4) logits = zs_logits + α · exp(-β(1-R_fF)) @ soft_values。
       注意 zs 项按官方用未精简的全通道 zero-shot logits（R_fW）。

    预算轴：k=top-k 检索宽度（剪枝 R_fF）；rounds 不使用（单步闭式）。
    """
    if zs_logits is None or text_feats is None:
        raise ValueError("ape 需要 zs_logits 与 text_feats（文本先验）")
    D = query_feats.shape[1]
    text = F.normalize(text_feats.float(), dim=-1)             # [C, D]

    # ---- 通道选择 ----
    bar = []                                                   # 每类均值行 f̄ [C, D]
    for c in range(n_class):
        rows = torch.cat([text[c:c + 1], support_feats[support_labels == c]], dim=0)
        bar.append(rows.mean(dim=0))
    bar = torch.stack(bar)                                     # [C, D]
    sum_all = bar.sum(dim=0)
    sim_k = (sum_all * sum_all - (bar * bar).sum(dim=0)) / (n_class * (n_class - 1))
    var_k = torch.var(text, dim=0)                             # 官方 torch.var 默认无偏
    criterion = -w_sim * sim_k + w_var * var_k
    feat_num = max(1, min(D, int(round(feat_ratio * D))))
    idx = torch.topk(criterion, k=feat_num).indices

    q_r = F.normalize(query_feats[:, idx], dim=-1)
    s_r = F.normalize(support_feats[:, idx], dim=-1)
    t_r = F.normalize(text[:, idx], dim=-1)                    # [C, feat_num]

    # ---- 三边亲和 ----
    r_ff = _topk_prune(q_r @ s_r.T, k)                         # [Q, N]
    key_p = F.softmax(s_r @ t_r.T, dim=1)                      # [N, C]（官方无温度）
    cache_div = -torch.log2(key_p[torch.arange(s_r.shape[0]), support_labels].clamp_min(1e-12))
    soft_values = _onehot(support_labels, n_class) * torch.exp(gamma * cache_div).unsqueeze(1)
    cache_logits = torch.exp(-beta * (1 - r_ff)) @ soft_values
    return zs_logits + alpha * cache_logits


# ---------------------------------------------------------------------------
# TDA：训练-free 动态适配器（正/负双缓存，流式伪标签）
# ---------------------------------------------------------------------------

def _update_queue(queue: dict, pred: int, item: tuple, capacity: int) -> None:
    """官方 update_cache 逐行口径：未满直接 append；满则新样本熵更低时替换队尾并重排序。

    item = (feat, entropy, payload)：正缓存 payload=预测类号，负缓存 payload=概率向量。
    """
    if pred in queue:
        if len(queue[pred]) < capacity:
            queue[pred].append(item)
        elif item[1] < queue[pred][-1][1]:
            queue[pred][-1] = item
            queue[pred].sort(key=lambda x: x[1])
    else:
        queue[pred] = [item]


def tda_logits(query_feats: torch.Tensor, support_feats: torch.Tensor,
               support_labels: torch.Tensor, n_class: int,
               zs_logits: Optional[torch.Tensor] = None,
               text_feats: Optional[torch.Tensor] = None,
               k: int = -1, rounds: int = 0,
               alpha_pos: float = 4.0, beta_pos: float = 8.0,
               alpha_neg: float = 0.117, beta_neg: float = 1.0,
               pos_cap: int = 3, neg_cap: int = 2,
               ent_low: float = 0.2, ent_high: float = 0.5,
               mask_low: float = 0.03, mask_high: float = 1.0, **_) -> torch.Tensor:
    """TDA（Karmanov et al. CVPR 2024；官方 kdiAAA/TDA tda_runner.py 口径）。

    流式处理 query（官方时序：先 update 缓存再推理，当前样本含自亲和=1 自增强）：
    - 正缓存：每类容量 pos_cap 的熵优先队列，存 (特征, 熵)，值 = argmax 伪标签 one-hot；
      在少样本 episodic 设定下以 support（真值，熵 0）初始化——等价 Tip-Adapter 静态缓存。
    - 负缓存：仅收 τ_l < H < τ_h 的中等不确定样本，存 (特征, 熵, 概率向量)；
      推理时值 = 𝟙[mask_low < p < mask_high]（官方 compute_cache_logits 的 neg_mask 口径），
      贡献为负：logits -= α_neg·exp(-β_neg(1-sim)) @ mask。
    - 熵按官方归一化：H = -(Σ p·ln p) / log2(C)。
    - logits = zs + α_pos·exp(-β_pos(1-sim)) @ onehot − α_neg·(同上负项)。
      默认超参 = 官方 eurosat.yaml（pos α=4/β=8/cap3；neg α=0.117/β=1/cap2；
      熵阈 [0.2,0.5]，掩码阈 [0.03,1.0]）。

    预算轴：k>0 时正/负缓存容量均设为 k（缓存规模档位）；T 不使用（官方单遍流式）。
    """
    if zs_logits is None:
        raise ValueError("tda 需要 zs_logits（文本先验）")
    if k > 0:
        pos_cap = neg_cap = k
    pos_queue: dict = {}
    for c in range(n_class):
        for f in support_feats[support_labels == c]:
            _update_queue(pos_queue, c, (f, 0.0, c), pos_cap)  # support 熵 0，必入队
    neg_queue: dict = {}
    log2c = math.log2(n_class)

    out = []
    for i in range(query_feats.shape[0]):
        q = query_feats[i]
        probs = F.softmax(zs_logits[i].float(), dim=0)
        ent = float(-(probs * probs.clamp_min(1e-12).log()).sum().item()) / log2c
        pred = int(probs.argmax().item())

        _update_queue(pos_queue, pred, (q, ent, pred), pos_cap)
        if ent_low < ent < ent_high:
            _update_queue(neg_queue, pred, (q, ent, probs), neg_cap)

        logits = zs_logits[i].float().clone()
        pos_items = [it for items in pos_queue.values() for it in items]
        if pos_items:
            keys = torch.stack([it[0] for it in pos_items])
            vals = _onehot(torch.tensor([it[2] for it in pos_items]), n_class)
            aff = q @ keys.T
            logits = logits + alpha_pos * (torch.exp(-beta_pos * (1 - aff)) @ vals)
        neg_items = [it for items in neg_queue.values() for it in items]
        if neg_items:
            keys = torch.stack([it[0] for it in neg_items])
            pmaps = torch.stack([it[2] for it in neg_items])
            masks = ((pmaps > mask_low) & (pmaps < mask_high)).float()
            aff = q @ keys.T
            logits = logits - alpha_neg * (torch.exp(-beta_neg * (1 - aff)) @ masks)
        out.append(logits)
    return torch.stack(out)


# ---------------------------------------------------------------------------
# TransCLIP-FS：GMM + Laplacian + 文本 KL 正则的 BMM 传导
# ---------------------------------------------------------------------------

def transclip_logits(query_feats: torch.Tensor, support_feats: torch.Tensor,
                     support_labels: torch.Tensor, n_class: int,
                     zs_logits: Optional[torch.Tensor] = None,
                     text_feats: Optional[torch.Tensor] = None,
                     k: int = -1, rounds: int = 10,
                     lam: float = 0.5, gamma: float = 0.01,
                     n_neighbors: int = 3, inner: int = 5, **_) -> torch.Tensor:
    """TransCLIP-FS（Zanella et al. NeurIPS 2024；官方 TransCLIP_solver 口径）。

    目标（论文 Eq.4）：GMM 聚类（共享对角协方差、类均衡）+ Laplacian 正则
    + KL_λ(z‖ŷ_text)（少样本 λ=0.5）+ support 交叉熵（权重 γ）。
    BMM 交替（官方常数原样保留：Laplacian 系数 50/(2·nn)、exp 内 /50、support 权 50γ/S）：
    - z 更新（内层 MM 迭代 inner=5）：
      z_i ∝ ŷ_i^λ ⊙ exp((log p_i + (50/(2·nn))·(Wᵀz + Wz))/50)，support 钉住 one-hot；
    - μ 更新：类加权均值（query 权 1/Q，support 权 50γ/S）后 L2 归一化；
    - σ² 更新：共享对角方差（同权归一）。
    亲和 w_ij = max(0, cos)（论文口径；官方代码未截断负值，此处按论文），
    每节点取 query 列上 top-n_neighbors（官方 3）。
    与官方差异（已在实现中修正并注明）：官方 z/W 的行序错位（z 为 [Q;S] 而 W 为 [S;Q]），
    此处统一为 [query; support]；官方 support 行 top-(n+1) 丢最近邻的 quirk 不保留；
    官方用 val 集选 γ ∈ {0.002,...,0.2}，episodic 无 val → 固定 γ=0.01（可配置）。

    预算轴：T=rounds 外层迭代数（官方 10；T=0 返回文本先验 log ŷ）；k=亲和图邻居数。
    """
    if zs_logits is None:
        raise ValueError("transclip 需要 zs_logits（文本先验）")
    if k > 0:
        n_neighbors = k
    n_q, D = query_feats.shape
    n_s = support_feats.shape[0]
    y_hat = F.softmax(zs_logits.float(), dim=1)                # [Q, C]
    if rounds <= 0 or n_q == 0:
        return (y_hat + 1e-12).log()

    sup_1h = _onehot(support_labels, n_class)
    z = y_hat.clone()
    mu = torch.stack([support_feats[support_labels == c].mean(dim=0)
                      for c in range(n_class)])
    mu = F.normalize(mu, dim=-1)                               # [C, D]
    var = torch.full((D,), 1.0 / D, device=query_feats.device,
                     dtype=query_feats.dtype)                  # 官方 std_init=1/d（名为 std 实为方差）

    # 亲和图：行 = [query; support]，列 = query（与 z_aug=[z; onehot] 行序一致）
    q = F.normalize(query_feats, dim=-1)
    s = F.normalize(support_feats, dim=-1)
    all_f = torch.cat([q, s], dim=0)                           # [Q+S, D]
    aff = all_f @ q.T                                          # [Q+S, Q]
    aff[:n_q].fill_diagonal_(float("-inf"))                    # query 行去自环
    nn = max(1, min(n_neighbors, n_q - 1 if n_q > 1 else 1))
    topv, topi = aff.topk(nn, dim=1)
    W = torch.zeros(n_q + n_s, n_q, device=query_feats.device, dtype=query_feats.dtype)
    W.scatter_(1, topi, topv.clamp_min(0.0))

    for _ in range(rounds):
        d2 = ((q.unsqueeze(1) - mu.unsqueeze(0)) ** 2) / var.clamp_min(1e-8)
        gmm_ll = -0.5 * d2.sum(dim=-1)                         # [Q, C]（det 项各类相同，归一化消去）
        z_aug = torch.cat([z, sup_1h], dim=0)                  # [Q+S, C]
        for _ in range(inner):
            neigh = W.T @ z_aug + (W @ z_aug[:n_q])[:n_q]      # [Q, C] 双向边聚合
            inter = gmm_ll + (50.0 / (2 * nn)) * neigh
            inter = inter - inter.max(dim=1, keepdim=True).values
            z_new = (y_hat ** lam) * torch.exp(inter / 50.0)
            z = z_new / z_new.sum(dim=1, keepdim=True).clamp_min(1e-12)
            z_aug = torch.cat([z, sup_1h], dim=0)
        # μ / σ² 更新
        wq = z / n_q
        ws = (50.0 * gamma / n_s) * sup_1h
        mu = (wq.T @ q + ws.T @ s) / (wq.sum(dim=0) + ws.sum(dim=0)).unsqueeze(1).clamp_min(1e-12)
        mu = F.normalize(mu, dim=-1)
        num = (torch.einsum("qc,qcd->d", wq, (q.unsqueeze(1) - mu.unsqueeze(0)) ** 2)
               + torch.einsum("sc,scd->d", ws, (s.unsqueeze(1) - mu.unsqueeze(0)) ** 2))
        var = (num / (wq.sum() + ws.sum())).clamp_min(1e-8)
    return (z + 1e-12).log()


# ---------------------------------------------------------------------------
# MuSC（分类适配版）：LN 动态子集 + 互评分可靠性 + 跨模态评分融合
# ---------------------------------------------------------------------------

def musc_logits(query_feats: torch.Tensor, support_feats: torch.Tensor,
                support_labels: torch.Tensor, n_class: int,
                zs_logits: Optional[torch.Tensor] = None,
                text_feats: Optional[torch.Tensor] = None,
                k: int = -1, rounds: int = 0,
                alpha: float = 2.0, beta: float = 5.5, ln_k: int = 10, **_) -> torch.Tensor:
    """MuSC 互评分机制的分类适配（Li et al. ICLR 2024，arXiv:2401.16753；官方 xrli-U/MuSC
    为零样本工业 AD 方法，本函数是其 MSM 思想在少样本分类的对应物，非逐行复现）：

    - LN 动态子集：每个节点（support∪query）取余弦 top-ln_k 邻居作为局部邻域
      （对应 MuSC 的局部邻域聚合/动态子集，k 轴 = 邻域大小）；
    - 互评分可靠性 rel_j = Σ_{i∈LN(j)} w_ji·⟨vote_i, vote_j⟩ / Σ w_ji，
      w_ji = exp(-β(1-sim_ji))：与邻域标签一致的样本是可靠评分者
      （对应 MuSC「正常样本互相证实」的互评分）；
      support 的 vote = 真值 one-hot，query 的 vote = 文本后验（跨模态评分）；
    - query 的互评分 mutual_c(q) = Σ_{j∈LN(q)} w·rel·vote_jc / Σ w·rel；
    - logits = zs_logits + α·mutual；rounds>1 时用当前 logits 软更新 query vote 再迭代
      （EM-lite 传播，T 轴）。α=2.0 为 logit 量纲的融合权重（可配置）。

    预算轴：k=LN 大小；T=rounds 互评分传播轮数（0/1=单轮）；V=多视图在 run 层聚合
    （对应 MuSC 的多尺度融合，缓存 CLS 特征下尺度≈视图）。
    """
    if zs_logits is None:
        raise ValueError("musc 需要 zs_logits（文本先验）")
    if k > 0:
        ln_k = k
    n_q = query_feats.shape[0]
    all_f = torch.cat([support_feats, query_feats], dim=0)     # [S+Q, D]
    n_all = all_f.shape[0]
    sup_1h = _onehot(support_labels, n_class)
    votes = torch.cat([sup_1h, F.softmax(zs_logits.float(), dim=1)], dim=0)

    sim = all_f @ all_f.T
    sim.fill_diagonal_(float("-inf"))
    kk = max(1, min(ln_k, n_all - 1))
    topv, topi = sim.topk(kk, dim=1)
    w = torch.exp(-beta * (1 - topv))                          # [N, kk]
    v_nbr = votes[topi]                                        # [N, kk, C]

    n_rounds = max(1, rounds)
    logits = zs_logits.float()
    for it in range(n_rounds):
        if it > 0:
            v_nbr = votes[topi]
        agree = (v_nbr * votes.unsqueeze(1)).sum(dim=-1)         # [N, kk] 邻居与节点标签一致度
        rel = ((w * agree).sum(dim=1) / w.sum(dim=1).clamp_min(1e-12)).clamp_min(1e-6)  # [N]
        wr = w * rel.unsqueeze(1)
        mutual = (wr.unsqueeze(-1) * v_nbr).sum(dim=1) / wr.sum(dim=1, keepdim=True).clamp_min(1e-12)
        logits = zs_logits.float() + alpha * mutual[support_feats.shape[0]:]  # query 行
        votes = torch.cat([sup_1h, F.softmax(logits, dim=1)], dim=0)
    return logits


# ---------------------------------------------------------------------------
# P1 注册表
# ---------------------------------------------------------------------------

P1_METHODS = {
    "pt_map": {"fn": pt_map_logits, "requires_text": False, "uses_T": True},
    "ape": {"fn": ape_logits, "requires_text": True, "uses_T": False},
    "tda": {"fn": tda_logits, "requires_text": True, "uses_T": False},
    "transclip": {"fn": transclip_logits, "requires_text": True, "uses_T": True},
    "musc": {"fn": musc_logits, "requires_text": True, "uses_T": True},
}

P1_DEFAULT = ["pt_map", "ape", "tda", "transclip", "musc"]


def run_p1_method(name: str, query_feats: torch.Tensor, support_feats: torch.Tensor,
                  support_labels: torch.Tensor, n_class: int,
                  zs_logits: Optional[torch.Tensor] = None,
                  text_feats: Optional[torch.Tensor] = None,
                  k: int = -1, rounds: int = 0, **kw) -> torch.Tensor:
    """统一分发入口（供 run/ 脚本按名字调用）。"""
    if name not in P1_METHODS:
        raise KeyError(f"未知 P1 方法: {name}（可选: {list(P1_METHODS)}）")
    spec = P1_METHODS[name]
    if spec["requires_text"] and zs_logits is None:
        raise ValueError(f"{name} 需要文本先验（zs_logits），当前数据集/骨干不提供")
    return spec["fn"](query_feats, support_feats, support_labels, n_class,
                      zs_logits=zs_logits, text_feats=text_feats,
                      k=k, rounds=rounds, **kw)


if __name__ == "__main__":
    # 单元验证：合成可分簇 + 文本先验 = 类名文本原型的弱版（用类心加噪模拟）
    torch.manual_seed(0)
    C, D, S, Q = 5, 64, 4, 75
    centers = F.normalize(torch.randn(C, D), dim=-1) * 3
    sup = F.normalize(centers.repeat_interleave(S, 0) + 0.3 * torch.randn(C * S, D), dim=-1)
    qry = F.normalize(centers.repeat_interleave(Q // C, 0) + 0.3 * torch.randn(Q, D), dim=-1)
    sup_y = torch.arange(C).repeat_interleave(S)
    qry_y = torch.arange(C).repeat_interleave(Q // C)
    # 模拟文本先验：文本原型 = 类心方向（强先验）
    text = F.normalize(centers + 0.1 * torch.randn(C, D), dim=-1)
    zs = 100.0 * (qry @ text.T)

    for name in P1_DEFAULT:
        logits = run_p1_method(name, qry, sup, sup_y, C, zs_logits=zs, text_feats=text,
                               rounds=3)
        assert logits.shape == (Q, C), f"{name} 形状错误"
        assert torch.isfinite(logits).all(), f"{name} 出现非有限值"
        acc = (logits.argmax(1) == qry_y).float().mean().item()
        assert acc > 0.85, f"{name} 在可分簇上精度异常: {acc}"
        print(f"[p1] {name}: acc={acc:.3f} (shape/finite OK)")
    # 复现性
    l1 = run_p1_method("transclip", qry, sup, sup_y, C, zs_logits=zs, rounds=3)
    l2 = run_p1_method("transclip", qry, sup, sup_y, C, zs_logits=zs, rounds=3)
    assert torch.equal(l1, l2), "transclip 不可复现"
    # pt_map 归纳式（T=0）与传导式（T>0）应不同
    p0 = run_p1_method("pt_map", qry, sup, sup_y, C, rounds=0)
    p5 = run_p1_method("pt_map", qry, sup, sup_y, C, rounds=5)
    assert not torch.allclose(p0, p5), "pt_map 的 T 轴不生效"
    print("[p1] 单元验证全部通过")
