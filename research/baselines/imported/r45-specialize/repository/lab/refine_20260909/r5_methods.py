"""Inductive candidate mechanisms. No labels other than class-ordered supports."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '研究5——少样本支持集的检索增强与可靠性加权/studies/r2_20260909'))
from methods import representation, ridge_scores, evaluate_configs


def menu(shot):
    rows = []
    for w in [.25, .5, .75]:
        for center in [0., .5, 1.]:
            for lam in [.1, 1.]:
                rows.append(dict(family='center', w=w, center=center, lam=lam))
    for w in [.25, .5, .75]:
        for lam in [.1, 1.]:
            rows.append(dict(family='late', w=w, lam=lam))
    if shot == 1:
        for quantile in [.05, .25]:
            for mix in [.5, .75]:
                for lam in [.1, 1.]:
                    rows.append(dict(family='gate', w=.5, quantile=quantile, mix=mix, lam=lam))
        for mix in [.25, .5, .75]:
            rows.append(dict(family='constant', w=.5, mix=mix, lam=.1))
    for c in rows:
        c['name'] = '_'.join(f'{k}{v}' for k, v in c.items())
    return rows


def legacy_configs(shot):
    import json
    selection = json.loads((ROOT / '研究5——少样本支持集的检索增强与可靠性加权/studies/r2_20260909/selection.json').read_text())
    s = selection[str(shot)]
    return [{**s[k], 'name': 'r2_' + k} for k in ['overall', 'control']] + [
        dict(family='raw', w=.5, name='fused_raw'),
        dict(family='uniform', w=.5, r=64, mix=.5, name='uniform_matched')]


@torch.no_grad()
def gallery_reference(g, quantiles=(.05, .25)):
    # Deterministic unlabeled gallery self-neighbour reference; no support/query data.
    ix = torch.linspace(0, len(g) - 1, min(512, len(g)), device=g.device).long()
    values = []
    for st in range(0, len(ix), 32):
        ids = ix[st:st + 32]
        sim = g[ids] @ g.T
        sim[torch.arange(len(ids), device=g.device), ids] = -torch.inf
        values.append(sim.topk(min(16, len(g) - 1), dim=-1).values.mean(-1))
    ref = torch.cat(values)
    return {q: float(torch.quantile(ref, q)) for q in quantiles}


@torch.no_grad()
def candidates(sc, sd, xc, xd, gc, gd, configs, thresholds):
    # Supports are [E,C,K,D], query [E,Q,D]; no query batch statistics are used.
    e, c, k = sc.shape[:3]
    y = F.one_hot(torch.arange(c, device=sc.device).repeat_interleave(k), c).float()[None].expand(e, -1, -1)
    eyes = torch.eye(c, device=sc.device)[None].expand(e, -1, -1)
    out = {}; diagnostics = {}; late = {}
    for w in sorted(set(cfg['w'] for cfg in configs)):
        s = representation(sc, sd, w)
        x = representation(xc, xd, w)
        selected = [cfg for cfg in configs if cfg['w'] == w]
        P = F.normalize(s.mean(2), dim=-1)
        need_gallery = any(cfg['family'] in ['gate', 'constant'] for cfg in selected)
        if need_gallery:
            G = representation(gc, gd, w)
            values, ids = (P.reshape(-1, P.shape[-1]) @ G.T).topk(min(64, len(G)), dim=-1)
            ids = ids.reshape(e, c, -1)
            U = F.normalize(G[ids].mean(2), dim=-1)
            coverage = values[:, :16].mean(-1).reshape(e, c)
        for cfg in selected:
            fam = cfg['family']; name = cfg['name']; lam = cfg['lam']
            if fam == 'center':
                mu = s.flatten(1, 2).mean(1, keepdim=True) * cfg['center']
                a = F.normalize(s.flatten(1, 2) - mu, dim=-1)
                q = F.normalize(x - mu, dim=-1)
                v = ridge_scores(a, y, q, lam)
            elif fam == 'late':
                if lam not in late:
                    late[lam] = [ridge_scores(F.normalize(a, dim=-1).flatten(1, 2), y, F.normalize(b, dim=-1), lam)
                                 for a, b in [(sc, xc), (sd, xd)]]
                v = w * late[lam][0] + (1 - w) * late[lam][1]
            elif fam in ['gate', 'constant']:
                if fam == 'gate':
                    # Coverage gate defined relative to gallery self-similarity, not an absolute cosine.
                    gate = torch.sigmoid((coverage - thresholds[cfg['quantile']]) / .03)
                else:
                    gate = torch.ones_like(coverage)
                mix = cfg['mix'] * gate[:, :, None]
                a = F.normalize((1 - mix) * P + mix * U, dim=-1)
                v = ridge_scores(a, eyes, x, lam)
                diagnostics[name] = dict(mean_gate=float(gate.mean()), gate_by_class=gate.cpu().tolist())
            else:
                raise ValueError(fam)
            out[name] = v.cpu()
    return out, diagnostics


@torch.no_grad()
def baseline(sc, sd, xc, xd, gc, gd, shot):
    result = {}
    configs = legacy_configs(shot)
    for w in sorted(set(c['w'] for c in configs)):
        s, x, g = representation(sc, sd, w), representation(xc, xd, w), representation(gc, gd, w)
        _, scores, _ = evaluate_configs(s, x, g, [c for c in configs if c['w'] == w], return_scores=True)
        result.update(scores)
    return result
