"""Gallery-only transfer of pinned OSLO/OSEM; query data never enter fit."""
import torch
import torch.nn.functional as F


def fit(support, gallery, steps=2, lambda_s=.05, lambda_z=.1, use_inlier=True):
    # support: task, class, shot, feature. gallery: image, feature.
    batch, ways, shot, dim = support.shape
    flat = support.reshape(batch, ways * shot, dim)
    origin = (flat.sum(1, keepdim=True) + gallery.sum(0)[None, None]) / (ways * shot + len(gallery))
    s = F.normalize(flat - origin, dim=-1)
    g = F.normalize(gallery[None] - origin, dim=-1)
    initial = s.reshape(batch, ways, shot, dim).mean(2)
    proto = initial.clone()
    z = torch.full((batch, len(gallery), ways), 1 / ways, dtype=s.dtype)
    xi = torch.full((batch, len(gallery), 1), .5, dtype=s.dtype)
    for _ in range(steps):
        logits = g @ F.normalize(proto, dim=-1).transpose(1, 2)
        xi = ((z * logits / lambda_s).sum(-1, keepdim=True)).sigmoid()
        z = ((xi if use_inlier else 1.) * logits / lambda_z).softmax(-1)
        weight = (xi if use_inlier else 1.) * z
        proto = (initial * shot + weight.transpose(1, 2) @ g) / (shot + weight.sum(1)[..., None])
    mass = ((xi if use_inlier else torch.ones_like(xi)) * z).sum(1) if steps else torch.zeros(batch, ways, dtype=s.dtype)
    stats = {'effective_mass': mass, 'inlier_mean': xi.mean(1).squeeze(-1),
             'prototype_movement': (F.normalize(proto, dim=-1) - F.normalize(initial, dim=-1)).norm(dim=-1)}
    return (origin, proto), stats


def predict(state, query):
    origin, proto = state
    return F.normalize(query - origin, dim=-1) @ F.normalize(proto, dim=-1).transpose(1, 2)
