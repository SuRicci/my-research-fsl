# -*- coding: utf-8 -*-
"""AD 评测指标（近乎原样拷贝自研究1 src/evaluate.py + 新增 bootstrap CI）。
五指标：I/P-AUROC、AP、F1-max、PRO（PRO 计算 CPU 密集，预算扫描时按子采样阈值数降本）。"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.ndimage import label as ndi_label
from typing import List, Tuple, Optional


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, scores)


def compute_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Average Precision."""
    if len(np.unique(labels)) < 2:
        return 0.0
    return average_precision_score(labels, scores)


def compute_f1_max(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute F1-score at optimal threshold."""
    if len(np.unique(labels)) < 2:
        return 0.0
    best_f1 = 0.0
    thresholds = np.percentile(scores, np.linspace(0, 100, 200))
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


def compute_pro(masks: np.ndarray, anomaly_maps: np.ndarray, num_thres: int = 200) -> float:
    """
    Compute Per-Region Overlap (PRO) metric.

    Args:
        masks: list of [H, W] binary ground truth masks
        anomaly_maps: list of [H, W] anomaly score maps
        num_thres: number of thresholds
    Returns:
        PRO score
    """
    # Flatten masks and anomaly maps
    all_masks = []
    all_anomaly = []
    for mask, amap in zip(masks, anomaly_maps):
        all_masks.append(mask.flatten())
        all_anomaly.append(amap.flatten())

    all_masks = np.concatenate(all_masks)
    all_anomaly = np.concatenate(all_anomaly)

    # Compute per-region overlaps at different thresholds
    min_th = all_anomaly.min()
    max_th = all_anomaly.max()
    thresholds = np.linspace(min_th, max_th, num_thres)

    fprs = []
    pros = []

    for thresh in thresholds:
        binary_pred = (all_anomaly >= thresh).astype(np.uint8)

        # False positive rate
        fp_pixels = np.sum((binary_pred == 1) & (all_masks == 0))
        total_normal = np.sum(all_masks == 0)
        fpr = fp_pixels / (total_normal + 1e-8)

        # Per-region overlap
        # Label connected components in ground truth
        pro_sum = 0.0
        num_regions = 0

        # Process each image separately for region labeling
        start_idx = 0
        for mask in masks:
            flat_len = mask.size
            pred_slice = binary_pred[start_idx:start_idx + flat_len].reshape(mask.shape)
            mask_slice = mask
            start_idx += flat_len

            labeled, num_features = ndi_label(mask_slice)
            for i in range(1, num_features + 1):
                region_mask = (labeled == i)
                if region_mask.sum() == 0:
                    continue
                overlap = np.sum(pred_slice & region_mask) / region_mask.sum()
                pro_sum += overlap
                num_regions += 1

        pro = pro_sum / (num_regions + 1e-8)

        fprs.append(fpr)
        pros.append(pro)

    fprs = np.array(fprs)
    pros = np.array(pros)

    # Integrate PRO up to FPR = 0.3 (standard practice)
    max_fpr = 0.3
    valid = fprs <= max_fpr
    if valid.sum() == 0:
        return 0.0

    # Sort by FPR
    sort_idx = np.argsort(fprs[valid])
    fprs_sorted = fprs[valid][sort_idx]
    pros_sorted = pros[valid][sort_idx]

    aupro = np.trapz(pros_sorted, fprs_sorted) / max_fpr
    return float(aupro)


class MetricsCollector:
    """Collect and compute metrics across a dataset."""
    def __init__(self):
        self.image_scores = []
        self.image_labels = []
        self.pixel_maps = []
        self.pixel_masks = []

    def add(self, image_score: float, label: int,
            pixel_map: Optional[np.ndarray] = None,
            pixel_mask: Optional[np.ndarray] = None):
        self.image_scores.append(image_score)
        self.image_labels.append(label)
        if pixel_map is not None:
            self.pixel_maps.append(pixel_map)
        if pixel_mask is not None:
            self.pixel_masks.append(pixel_mask)

    def compute(self) -> dict:
        """Compute all metrics."""
        scores = np.array(self.image_scores)
        labels = np.array(self.image_labels)

        results = {
            "image_auroc": compute_auroc(scores, labels),
            "image_ap": compute_ap(scores, labels),
            "image_f1_max": compute_f1_max(scores, labels),
        }

        if len(self.pixel_maps) > 0 and len(self.pixel_masks) > 0:
            # Flatten pixel-level for AUROC
            all_maps = np.concatenate([m.flatten() for m in self.pixel_maps])
            all_masks = np.concatenate([m.flatten() for m in self.pixel_masks])

            results["pixel_auroc"] = compute_auroc(all_maps, all_masks)
            results["pixel_ap"] = compute_ap(all_maps, all_masks)
            results["pixel_f1_max"] = compute_f1_max(all_maps, all_masks)

            try:
                results["pixel_pro"] = compute_pro(self.pixel_masks, self.pixel_maps)
            except Exception as e:
                results["pixel_pro"] = 0.0

        return results


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray, metric_fn=compute_auroc,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    """成对 bootstrap 置信区间（默认对 AUROC）。"""
    rng = np.random.RandomState(seed)
    n = len(scores)
    stats = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        stats.append(metric_fn(scores[idx], labels[idx]))
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


if __name__ == "__main__":
    # 单元验证：可分合成分数应得 AUROC≈1、F1≈1，CI 合法
    rng = np.random.RandomState(0)
    labels = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([rng.rand(50) * 0.3, 0.7 + rng.rand(50) * 0.3])
    assert compute_auroc(scores, labels) > 0.99
    assert compute_ap(scores, labels) > 0.99
    assert compute_f1_max(scores, labels) > 0.99
    lo, hi = bootstrap_ci(scores, labels, n_boot=200)
    assert 0.5 <= lo <= hi <= 1.0
    # 退化：单类标签 → AUROC=0.5
    assert compute_auroc(scores, np.zeros(100, dtype=int)) == 0.5
    # MetricsCollector 纯图像级
    mc = MetricsCollector()
    for s, l in zip(scores, labels):
        mc.add(float(s), int(l))
    r = mc.compute()
    assert "image_auroc" in r and "pixel_auroc" not in r
    print(f"[metrics] 单元验证通过: AUROC={r['image_auroc']:.3f}, CI=({lo:.3f},{hi:.3f})")
