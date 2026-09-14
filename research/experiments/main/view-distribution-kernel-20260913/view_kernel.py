"""Finite empirical augmentation kernels; no query-set fitting."""
import torch
import torch.nn.functional as F

def unit(x):
    return F.normalize(x.to(torch.float64), dim=-1)

def features(x, mode):
    x = unit(x)
    if mode == 'original_rbf':
        return x[:, :1]
    if mode in ('mean_rbf', 'linear_mean'):
        return unit(x.mean(1))[:, None]
    assert mode == 'distribution_rbf'
    return x

def raw_kernel(a, b, beta, linear=False):
    dots = (a.flatten(0, 1) @ b.flatten(0, 1).T).reshape(len(a), a.shape[1], len(b), b.shape[1])
    return (dots if linear else torch.exp(beta * (dots - 1))).mean((1, 3))

def self_kernel(a, beta, linear=False):
    dots = a @ a.transpose(-1, -2)
    return (dots if linear else torch.exp(beta * (dots - 1))).mean((1, 2))

def gram(s, q, mode, beta):
    s, q = features(s, mode), features(q, mode)
    linear = mode == 'linear_mean'
    sd, qd = self_kernel(s, beta, linear), self_kernel(q, beta, linear)
    k = raw_kernel(s, s, beta, linear) / torch.sqrt(sd[:, None] * sd[None, :])
    t = raw_kernel(q, s, beta, linear) / torch.sqrt(qd[:, None] * sd[None, :])
    return k, t

def ridge(k, t, labels, penalty):
    n = len(k)
    y = F.one_hot(labels, 5).to(k.dtype)
    centered = k - k.mean(0)[None] - k.mean(1)[:, None] + k.mean()
    test = t - t.mean(1)[:, None] - k.mean(0)[None] + k.mean()
    weights = torch.linalg.solve(centered + penalty * torch.eye(n, dtype=k.dtype), y - y.mean(0))
    return test @ weights + y.mean(0)

def scores(s, q, mode, beta, penalty):
    labels = torch.arange(5).repeat_interleave(s.shape[1])
    return ridge(*gram(s.flatten(0, 1), q, mode, beta), labels, penalty)

def grid(s, q, mode, betas, penalties):
    labels = torch.arange(5).repeat_interleave(s.shape[1])
    values = []
    for beta in ([0] if mode == 'linear_mean' else betas):
        k, t = gram(s.flatten(0, 1), q, mode, beta)
        values.extend(ridge(k, t, labels, penalty) for penalty in penalties)
    return torch.stack(values)
