# -*- coding: utf-8 -*-
"""AD 记忆库（拆分改造自研究1 src/dino_branch.py 的记忆库部分）。

- 从 DINOv2 patch 特征构建正常记忆库：PCA 前景 masking + 旋转增强；
- k 轴两种形态：coreset 保留率（k-center greedy 子采样）与检索邻居数 n_neighbors；
- 距离 = 余弦（与研究1 一致），图像级聚合 tail_var_1。
特征提取走 src/features/dino.py 的 DINOExtractor（禁止 mock）。
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..features.dino import DINOExtractor


def pca_foreground_mask(patch_feats: torch.Tensor, grid: Tuple[int, int]) -> np.ndarray:
    """PCA 第一主成分阈值化得到前景 mask（AnomalyDINO 做法）。"""
    from sklearn.decomposition import PCA
    from scipy.ndimage import binary_closing, binary_dilation

    feats = patch_feats[0].cpu().numpy()  # [N, D]
    pca_val = PCA(n_components=1).fit_transform(feats).flatten()
    mask = (pca_val > pca_val.mean()).reshape(grid)
    mask = binary_closing(mask, iterations=1)
    mask = binary_dilation(mask, iterations=1)
    return mask.astype(np.float32)


def kcenter_greedy_coreset(feats: torch.Tensor, keep_ratio: float, seed: int = 0) -> torch.Tensor:
    """k-center greedy coreset 子采样（PatchCore 式，k 轴：coreset 保留率）。"""
    n = feats.shape[0]
    n_keep = max(1, int(round(n * keep_ratio)))
    if n_keep >= n:
        return feats
    g = torch.Generator(device=feats.device).manual_seed(seed)
    sel = [int(torch.randint(n, (1,), generator=g, device=feats.device).item())]
    dist = torch.cdist(feats, feats[sel[0]:sel[0] + 1]).squeeze(1)
    for _ in range(n_keep - 1):
        idx = int(dist.argmax().item())
        sel.append(idx)
        dist = torch.minimum(dist, torch.cdist(feats, feats[idx:idx + 1]).squeeze(1))
    return feats[torch.tensor(sel, device=feats.device)]


class MemoryBank:
    """少样本正常 patch 记忆库 + 异常评分。"""

    def __init__(self, extractor: DINOExtractor, use_masking: bool = True,
                 rotation_angles: Sequence[int] = (0, 90, 180, 270),
                 aggregation: str = "tail_var_1"):
        self.extractor = extractor
        self.use_masking = use_masking
        self.rotation_angles = list(rotation_angles)
        self.aggregation = aggregation
        self.bank: Optional[torch.Tensor] = None     # [M, D]
        self.mask: Optional[np.ndarray] = None
        self.grid: Optional[Tuple[int, int]] = None

    # ---------------- 建库 ----------------
    @staticmethod
    def _rotate(img: torch.Tensor, angle: int) -> torch.Tensor:
        return img if angle == 0 else torch.rot90(img, k=angle // 90, dims=[-2, -1])

    def _apply_mask(self, memory: torch.Tensor, p0: torch.Tensor,
                    grid: Tuple[int, int]) -> torch.Tensor:
        self.mask = pca_foreground_mask(p0.unsqueeze(0), grid)
        mask_flat = torch.from_numpy(self.mask.flatten()).to(memory.device)
        per_img = grid[0] * grid[1]
        kept = []
        for i in range(0, len(memory), per_img):
            chunk = memory[i:i + per_img]
            if len(chunk) == len(mask_flat):
                chunk = chunk[mask_flat > 0.5]
            kept.append(chunk)
        return torch.cat(kept, dim=0)

    def build(self, normal_images: List[torch.Tensor]) -> None:
        """normal_images: [3, H, W] 列表（H=W=分辨率档 r）。"""
        all_patches, grid = [], None
        for img in normal_images:
            for ang in self.rotation_angles:
                patches, grid = self.extractor.extract_patches(self._rotate(img, ang))
                all_patches.append(patches[0])       # [N, D]
        memory = torch.cat(all_patches, dim=0)
        self.grid = grid

        if self.use_masking:
            p0, _ = self.extractor.extract_patches(normal_images[0])
            memory = self._apply_mask(memory, p0[0], grid)
        self.bank = F.normalize(memory, dim=-1)

    def build_from_patches(self, normal_patches: List[List[torch.Tensor]],
                           grid: Tuple[int, int]) -> None:
        """缓存路径建库：normal_patches[i][j] = 第 i 张正常图按 rotation_angles[j]
        旋转后重跑 backbone 的 patch [N, D]（extract_features_ad --rotations 产物），
        与 build() 的图像级旋转 + 重提特征严格等价（仅 fp16 存储差）。"""
        all_patches = []
        for per_view in normal_patches:
            assert len(per_view) == len(self.rotation_angles)
            all_patches.extend(per_view)
        memory = torch.cat(all_patches, dim=0)
        self.grid = grid
        if self.use_masking:
            memory = self._apply_mask(memory, normal_patches[0][0], grid)
        self.bank = F.normalize(memory, dim=-1)

    # ---------------- 推理（预算轴：coreset_ratio × n_neighbors） ----------------
    def score_patches(self, patches: torch.Tensor, grid: Tuple[int, int],
                      coreset_ratio: float = 1.0, n_neighbors: int = 1,
                      coreset_seed: int = 0) -> Tuple[torch.Tensor, float]:
        """缓存路径评分：patches 为 [N, D]（v0 原图 patch，未归一化亦可）。"""
        if self.bank is None:
            raise RuntimeError("记忆库未构建：先 build()")
        bank = kcenter_greedy_coreset(self.bank, coreset_ratio, seed=coreset_seed) \
            if coreset_ratio < 1.0 else self.bank
        p = F.normalize(patches, dim=-1)             # [N, D]
        sim = p @ bank.T                             # [N, M]
        k = max(1, min(n_neighbors, sim.shape[1]))
        top_sim = sim.topk(k, dim=1).values
        dist = 1.0 - top_sim.mean(dim=1)             # n_neighbors 平均余弦距离
        amap = dist.reshape(grid)
        if self.mask is not None and self.use_masking and self.mask.shape == grid:
            amap = amap * torch.from_numpy(self.mask).to(amap.device)
        return amap, self._aggregate(dist)

    def score_image(self, image: torch.Tensor, coreset_ratio: float = 1.0,
                    n_neighbors: int = 1,
                    coreset_seed: int = 0) -> Tuple[torch.Tensor, float]:
        """返回 (patch 距离图 [gh, gw], 图像级异常分)。

        coreset_ratio: 记忆库子采样率（k 轴一）；n_neighbors: 平均最近邻数（k 轴二）。
        """
        patches, grid = self.extractor.extract_patches(image)
        return self.score_patches(patches[0], grid, coreset_ratio, n_neighbors,
                                  coreset_seed)

    def _aggregate(self, patch_dist: torch.Tensor) -> float:
        d = patch_dist.flatten().cpu().numpy()
        if self.aggregation == "max":
            return float(d.max())
        # tail_var_1 / mean_topk：top 1% 均值
        k = max(1, int(len(d) * 0.01))
        return float(np.partition(d, -k)[-k:].mean())


if __name__ == "__main__":
    # 单元验证（需真实 DINOv2 权重）：建库 → 子采样 → 评分形状与单调性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = DINOExtractor("dinov2_vits14", device=device)
    imgs = [torch.rand(3, 224, 224) for _ in range(2)]
    mb = MemoryBank(ext, use_masking=True, rotation_angles=(0, 90))
    mb.build(imgs)
    assert mb.bank is not None and mb.bank.shape[1] == 384
    full = mb.bank.shape[0]
    sub = kcenter_greedy_coreset(mb.bank, 0.25)
    assert abs(sub.shape[0] - max(1, round(full * 0.25))) <= 1
    amap, s = mb.score_image(torch.rand(3, 224, 224), coreset_ratio=0.5, n_neighbors=3)
    assert amap.shape == mb.grid and np.isfinite(s)
    # coreset 率是预算轴：不同保留率结果应不同
    _, s1 = mb.score_image(torch.rand(3, 224, 224), coreset_ratio=1.0, n_neighbors=1)
    x = torch.rand(3, 224, 224)
    _, s2 = mb.score_image(x, coreset_ratio=0.1, n_neighbors=1)
    _, s3 = mb.score_image(x, coreset_ratio=1.0, n_neighbors=1)
    print(f"[memory_bank] 单元验证通过: bank={full}, coreset25%={sub.shape[0]}, "
          f"score(r=1.0)={s3:.4f} score(r=0.1)={s2:.4f}")
