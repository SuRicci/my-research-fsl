"""Class-associated finite-view metrics with unchanged prediction-fusion heads."""
from pathlib import Path
import importlib.util
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / 'experiments/main/scatter-score-fusion-20260913'
spec = importlib.util.spec_from_file_location('retained_fusion', PARENT / 'study.py')
fusion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fusion)
base = fusion.m
NAMES = ['incumbent', 'class_associated', 'metric_ensemble', 'foreign_metrics',
         'scatter_cs', 'scatter_r2', 'mean_cs']

def factor(sv, gamma, class_index, rho=0.5):
    residual = (sv - sv.mean(-2, keepdim=True)).reshape(len(sv), -1, sv.shape[-1])
    flat = residual.flatten(0, 1)
    trace = flat.square().sum() / len(flat)
    if gamma == 0 or trace <= 1e-12:
        return flat[:0], flat.new_empty(0)
    weights = torch.full((len(sv), residual.shape[1]), (1-rho)/len(flat), dtype=sv.dtype)
    weights[class_index] += rho/residual.shape[1]
    weighted = flat * weights.flatten().sqrt()[:, None]
    _, singular, vectors = torch.linalg.svd(weighted, full_matrices=False)
    coefficients = (1 + gamma*sv.shape[-1]*singular.square()/trace).rsqrt()-1
    return vectors, coefficients

def fused_heads(s, q, g):
    raw, raw_ix = base.head(s, q, g, False, 0.1)
    centered, centered_ix = base.head(s, q, g, True, 0.1)
    return (raw+centered)/2, centered, raw, torch.stack([raw_ix, centered_ix])

def assemble(bank):
    classes = torch.arange(bank.shape[0])
    associated = bank[classes, :, classes].T
    return associated, bank.mean(0), (bank.sum(0)-associated)/(len(bank)-1)

@torch.no_grad()
def evaluate(sv, qv, gallery_means, gamma, rho=0.5):
    sv, qv = sv.double(), qv.double()
    sm, qm = sv.mean(-2), qv.mean(-2)
    _, (s, q), pooled = base.prepare(sv, qv, gamma)
    shared, banks, neighbors = {}, {g: [] for g in gallery_means}, {g: [] for g in gallery_means}
    for g, gm in gallery_means.items():
        shared[g] = fused_heads(s, q, base.old.metric.transform(gm, pooled))
    for c in range(len(sv)):
        f = factor(sv, gamma, c, rho)
        cs, cq = [base.old.metric.transform(x, f) for x in [sm, qm]]
        for g, gm in gallery_means.items():
            scores, _, _, ix = fused_heads(cs, cq, base.old.metric.transform(gm, f))
            banks[g].append(scores)
            neighbors[g].append(ix)
    outputs = {}
    for g, gm in gallery_means.items():
        bank = torch.stack(banks[g])
        inc, centered, raw, _ = shared[g]
        mean_cs, _ = base.head(base.n(sm), base.n(qm), base.n(gm), True, 0.1)
        outputs[g] = (torch.stack([inc, *assemble(bank), centered, raw, mean_cs]),
                      bank, torch.stack(neighbors[g]))
    return outputs
