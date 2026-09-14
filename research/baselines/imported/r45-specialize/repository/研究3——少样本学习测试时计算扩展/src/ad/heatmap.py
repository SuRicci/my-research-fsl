# -*- coding: utf-8 -*-
"""热图归一化/融合/上采样/聚合（近乎原样拷贝自研究1 src/fusion.py，仅改文件头）。"""
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Optional, List


def normalize_heatmap(heatmap: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize anomaly heatmap to [0, 1].

    Args:
        heatmap: [H, W] numpy array
        method: "minmax", "zscore", or "category"
    Returns:
        normalized: [H, W] numpy array in [0, 1]
    """
    if method == "minmax":
        h_min, h_max = heatmap.min(), heatmap.max()
        if h_max - h_min < 1e-8:
            return np.zeros_like(heatmap)
        return (heatmap - h_min) / (h_max - h_min)

    elif method == "zscore":
        mean, std = heatmap.mean(), heatmap.std()
        if std < 1e-8:
            return np.zeros_like(heatmap)
        z = (heatmap - mean) / std
        # Map to [0, 1] using sigmoid-like clipping
        return np.clip((z + 3) / 6, 0, 1)

    elif method == "category":
        # Category-wise: use robust scaling with percentiles
        p5, p95 = np.percentile(heatmap, [5, 95])
        if p95 - p5 < 1e-8:
            return np.zeros_like(heatmap)
        return np.clip((heatmap - p5) / (p95 - p5), 0, 1)

    else:
        return heatmap


def fuse_heatmaps(
    clip_heatmap: np.ndarray,
    dino_heatmap: np.ndarray,
    weight_clip: float = 0.5,
    adaptive: bool = False,
    clip_stats: Optional[dict] = None,
    dino_stats: Optional[dict] = None,
) -> np.ndarray:
    """
    Fuse CLIP and DINOv2 anomaly heatmaps.

    Args:
        clip_heatmap: [H, W] normalized CLIP anomaly map
        dino_heatmap: [H, W] normalized DINO anomaly map
        weight_clip: weight for CLIP branch
        adaptive: whether to adaptively adjust weight
        clip_stats: statistics of CLIP heatmap for adaptive weighting
        dino_stats: statistics of DINO heatmap for adaptive weighting
    Returns:
        fused: [H, W] fused anomaly map
    """
    if adaptive and clip_stats is not None and dino_stats is not None:
        weight_clip = compute_adaptive_weight(clip_stats, dino_stats)

    weight_dino = 1.0 - weight_clip
    fused = weight_clip * clip_heatmap + weight_dino * dino_heatmap
    return fused


def compute_adaptive_weight(clip_stats: dict, dino_stats: dict) -> float:
    """
    Compute adaptive fusion weight based on heatmap statistics.

    Heuristics:
    - If CLIP heatmap is very dispersed (high entropy), lower CLIP weight
    - If DINO heatmap has high noise (high std), lower DINO weight
    - If CLIP has clear peak, increase CLIP weight
    """
    clip_entropy = clip_stats.get("entropy", 0.5)
    dino_entropy = dino_stats.get("entropy", 0.5)
    clip_peakness = clip_stats.get("peakness", 0.5)
    dino_peakness = dino_stats.get("peakness", 0.5)

    # Higher entropy = more dispersed = less reliable
    clip_reliability = 1.0 - clip_entropy
    dino_reliability = 1.0 - dino_entropy

    # Higher peakness = clearer anomaly signal
    clip_signal = clip_peakness
    dino_signal = dino_peakness

    # Combined score
    clip_score = clip_reliability * clip_signal
    dino_score = dino_reliability * dino_signal

    total = clip_score + dino_score
    if total < 1e-8:
        return 0.5

    weight_clip = clip_score / total
    # Clamp to reasonable range
    weight_clip = np.clip(weight_clip, 0.2, 0.8)
    return weight_clip


def compute_heatmap_stats(heatmap: np.ndarray) -> dict:
    """Compute statistics of a heatmap for adaptive weighting."""
    h = heatmap.flatten()
    h_norm = (h - h.min()) / (h.max() - h.min() + 1e-8)

    # Entropy: measure of dispersion
    hist, _ = np.histogram(h_norm, bins=20, range=(0, 1))
    hist = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(hist * np.log(hist + 1e-8))
    entropy = entropy / np.log(20)  # Normalize to [0, 1]

    # Peakness: ratio of max to mean
    peakness = h_norm.max() / (h_norm.mean() + 1e-8)
    peakness = min(peakness / 5.0, 1.0)  # Normalize

    return {
        "entropy": float(entropy),
        "peakness": float(peakness),
        "mean": float(h.mean()),
        "std": float(h.std()),
        "max": float(h.max()),
    }


def upsample_heatmap(
    heatmap: np.ndarray,
    target_size: Tuple[int, int],
    sigma: float = 4.0,
) -> np.ndarray:
    """
    Upsample patch-level heatmap to target resolution with Gaussian smoothing.

    Args:
        heatmap: [H_patch, W_patch]
        target_size: (H, W)
        sigma: Gaussian smoothing sigma
    Returns:
        upsampled: [H, W]
    """
    # Convert to torch tensor for interpolation
    h_tensor = torch.from_numpy(heatmap).float().unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(
        h_tensor, size=target_size, mode="bilinear", align_corners=False
    )
    upsampled = upsampled.squeeze().numpy()

    # Gaussian smoothing
    if sigma > 0:
        upsampled = gaussian_filter(upsampled, sigma=sigma)

    return upsampled


def aggregate_image_score(
    heatmap: np.ndarray,
    method: str = "tail_var_1",
    topk_ratio: float = 0.01,
) -> float:
    """
    Aggregate pixel-level anomaly scores to image-level score.

    Args:
        heatmap: [H, W] anomaly map
        method: aggregation method
        topk_ratio: ratio for top-k methods
    Returns:
        image_score: scalar
    """
    h = heatmap.flatten()

    if method == "max":
        return float(h.max())

    elif method == "mean":
        return float(h.mean())

    elif method == "mean_topk" or method == "tail_var_1":
        k = max(1, int(len(h) * topk_ratio))
        topk = np.partition(h, -k)[-k:]
        return float(topk.mean())

    elif method == "adaptive_topk":
        # Adapt topk_ratio based on heatmap distribution
        std = h.std()
        mean = h.mean()
        cv = std / (mean + 1e-8)

        # Higher CV (more spread) -> larger topk
        # Lower CV (more concentrated) -> smaller topk
        adaptive_ratio = topk_ratio * (1.0 + cv)
        adaptive_ratio = np.clip(adaptive_ratio, 0.005, 0.05)

        k = max(1, int(len(h) * adaptive_ratio))
        topk = np.partition(h, -k)[-k:]
        return float(topk.mean())

    elif method == "entropy_guided":
        # Use entropy to guide aggregation
        h_norm = (h - h.min()) / (h.max() - h.min() + 1e-8)
        hist, _ = np.histogram(h_norm, bins=20, range=(0, 1))
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist * np.log(hist + 1e-8))

        # High entropy (dispersed) -> use max (anomaly might be subtle)
        # Low entropy (concentrated) -> use top-k (anomaly is localized)
        if entropy > 2.0:
            return float(h.max())
        else:
            k = max(1, int(len(h) * topk_ratio))
            topk = np.partition(h, -k)[-k:]
            return float(topk.mean())

    else:
        return float(h.max())


class HeatmapFusion:
    """
    Complete fusion pipeline: normalize, fuse, upsample, aggregate.
    """
    def __init__(
        self,
        normalization: str = "minmax",
        fusion_weight: float = 0.5,
        adaptive_weight: bool = False,
        image_aggregation: str = "tail_var_1",
        topk_ratio: float = 0.01,
        gaussian_sigma: float = 4.0,
    ):
        self.normalization = normalization
        self.fusion_weight = fusion_weight
        self.adaptive_weight = adaptive_weight
        self.image_aggregation = image_aggregation
        self.topk_ratio = topk_ratio
        self.gaussian_sigma = gaussian_sigma

    def process(
        self,
        clip_heatmap: np.ndarray,
        dino_heatmap: np.ndarray,
        target_size: Tuple[int, int] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Process and fuse heatmaps.

        Args:
            clip_heatmap: [H_patch, W_patch]
            dino_heatmap: [H_patch, W_patch]
            target_size: (H, W) for upsampling
        Returns:
            fused_heatmap: upsampled to target_size
            image_score: aggregated image-level score
        """
        # If heatmaps have different sizes, upsample both to target_size before fusion
        if clip_heatmap.shape != dino_heatmap.shape:
            if target_size is not None:
                clip_heatmap = upsample_heatmap(clip_heatmap, target_size, sigma=0)
                dino_heatmap = upsample_heatmap(dino_heatmap, target_size, sigma=0)
            else:
                # Use the larger size as common size
                h_max = max(clip_heatmap.shape[0], dino_heatmap.shape[0])
                w_max = max(clip_heatmap.shape[1], dino_heatmap.shape[1])
                clip_heatmap = upsample_heatmap(clip_heatmap, (h_max, w_max), sigma=0)
                dino_heatmap = upsample_heatmap(dino_heatmap, (h_max, w_max), sigma=0)

        # Normalize
        clip_norm = normalize_heatmap(clip_heatmap, self.normalization)
        dino_norm = normalize_heatmap(dino_heatmap, self.normalization)

        # Compute stats for adaptive weighting
        clip_stats = compute_heatmap_stats(clip_heatmap) if self.adaptive_weight else None
        dino_stats = compute_heatmap_stats(dino_heatmap) if self.adaptive_weight else None

        # Fuse
        fused = fuse_heatmaps(
            clip_norm, dino_norm,
            weight_clip=self.fusion_weight,
            adaptive=self.adaptive_weight,
            clip_stats=clip_stats,
            dino_stats=dino_stats,
        )

        # Upsample if target size given and not already done
        if target_size is not None and fused.shape != target_size:
            fused = upsample_heatmap(fused, target_size, self.gaussian_sigma)

        # Aggregate to image score
        image_score = aggregate_image_score(
            fused, self.image_aggregation, self.topk_ratio
        )

        return fused, image_score
