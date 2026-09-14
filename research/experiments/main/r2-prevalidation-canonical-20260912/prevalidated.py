"""Exact intercept-aware ridge prevalidation; support-only model selection."""
import numpy as np
import torch
import torch.nn.functional as F

LAMBDAS = [1., .1, .01, .001]

def fit(S):
    A = S.flatten(1, 2).double()
    e, n, d = A.shape
    mean = A.mean(1, keepdim=True)
    X = A - mean
    Y = F.one_hot(torch.arange(5).repeat_interleave(5), 5).double()[None].expand(e, -1, -1)
    ym = Y.mean(1, keepdim=True)
    K = X @ X.transpose(1, 2)
    eye = torch.eye(n, dtype=torch.double)[None].expand(e, -1, -1)
    coefficients, loo = [], []
    for penalty in LAMBDAS:
        inverse = torch.linalg.solve(K + penalty * eye, eye)
        coefficient = inverse @ (Y - ym)
        prediction = K @ coefficient + ym
        diagonal = (K @ inverse).diagonal(dim1=-2, dim2=-1) + 1. / n
        assert float((1 - diagonal).min()) > 1e-10
        loo.append(Y - (Y - prediction) / (1 - diagonal)[:, :, None])
        coefficients.append(coefficient)
    Z = torch.stack(loo)
    true = (Z * Y[None]).sum(-1)
    def gradient(scale):
        return ((torch.softmax(Z * scale[..., None, None], dim=-1) * Z).sum(-1) - true).mean(-1)
    lower = torch.zeros(Z.shape[:2], dtype=torch.double)
    upper = torch.full_like(lower, 100.)
    at_zero, at_max = gradient(lower), gradient(upper)
    for _ in range(48):
        middle = (lower + upper) / 2
        negative = gradient(middle) < 0
        lower = torch.where(negative, middle, lower)
        upper = torch.where(negative, upper, middle)
    scale = (lower + upper) / 2
    scale = torch.where(at_zero >= 0, torch.zeros_like(scale), scale)
    scale = torch.where(at_max <= 0, torch.full_like(scale, 100.), scale)
    nll = (torch.logsumexp(Z * scale[..., None, None], dim=-1) - true * scale[..., None]).mean(-1)
    mse = (Z - Y[None]).square().mean((-1, -2))
    return {"mean": mean, "X": X, "Y": Y, "ym": ym, "coefficients": torch.stack(coefficients), "loo": Z,
            "scale": scale, "nll": nll, "mse": mse, "preval_index": nll.argmin(0), "press_index": mse.argmin(0)}

def predict(model, Q):
    Q = Q.double()
    all_scores = (Q[None] - model["mean"][None]) @ model["X"][None].transpose(-1, -2) @ model["coefficients"] + model["ym"][None]
    indices = torch.arange(len(Q))
    choice = model["preval_index"]
    return {"preval": all_scores[choice, indices] * model["scale"][choice, indices, None, None],
            "press": all_scores[model["press_index"], indices],
            "raw_fixed_check": all_scores[0]}
