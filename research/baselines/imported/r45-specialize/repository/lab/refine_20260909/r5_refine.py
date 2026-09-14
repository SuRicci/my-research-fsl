"""Develop on designated old classes, freeze, then paired full-scale retest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'lab/server_scale_20260909'))
import r5_scale as old
from r5_methods import menu, candidates, legacy_configs, evaluate_configs, gallery_reference, representation

BASE = Path('/data/liuhaoyu/r45-scale-20260909/r5')
OUT = Path('/data/liuhaoyu/r45-refine-20260909/r5')
DEV = old.CELLS[:4]


def hashes():
    paths = [Path(__file__), HERE / 'r5_methods.py', ROOT / 'lab/server_scale_20260909/r5_scale.py',
             ROOT / '研究5——少样本支持集的检索增强与可靠性加权/studies/r2_20260909/methods.py']
    return {str(p.relative_to(ROOT)): old.digest(p) for p in paths}


def pair(cell):
    qm, qfs = old.load(BASE, cell[0], 'test')
    gm, gfs = old.load(BASE, cell[1], 'train')
    identity = json.loads((BASE / 'results' / ('_'.join(cell) + '_identity.json')).read_text())
    qi, gi = identity['query_pool_ids'], identity['gallery_ids']
    assert not set(qm['rgb'][i] for i in qi) & set(gm['rgb'][i] for i in gi)
    return qm, qfs, [g[gi] for g in gfs], qi, identity


def dev_episodes(qm, qi, shot):
    labels = np.array(qm['labels'])
    pools = {int(c): np.array([i for i in qi if labels[i] == c]) for c in set(labels)}
    classes = np.array(sorted(c for c in pools if len(pools[c]) >= 20))
    order = np.random.RandomState(120909).permutation(classes)
    classes = order[:len(order) // 2]
    rng = np.random.RandomState(130090 + shot)
    cs = np.stack([rng.choice(classes, 5, replace=False) for _ in range(200)])
    ix = np.stack([np.stack([rng.choice(pools[int(c)], shot + 15, replace=False) for c in row]) for row in cs])
    return ix[:, :, :shot], ix[:, :, shot:], cs


def run_cell(cell, shot, phase, configs, device):
    qm, qfs, gfs, qi, identity = pair(cell)
    G = [g.to(device) for g in gfs]
    # Reproduce the original CPU representation transform and 24-episode GEMM shape.
    # Changing GEMM row counts can choose a different member of a near-tied top-k.
    legacy = legacy_configs(shot)
    legacy_reps = {w: (representation(*qfs, w), representation(*gfs, w).to(device)) for w in sorted(set(c['w'] for c in legacy))}
    thresholds = gallery_reference(representation(*G, .5)) if any(c['family'] == 'gate' for c in configs) else {}
    sources = [None] if phase == 'dev' else sorted((BASE / 'results').glob('_'.join(cell) + f'_k{shot}_s*.npz'))
    metrics = []
    for source in sources:
        if source is None:
            si, xi, cs = dev_episodes(qm, qi, shot); seed = 130090
        else:
            z = np.load(source, allow_pickle=False)
            si, xi, cs, seed = z['support_indices'], z['query_indices'], z['class_ids'], int(z['seed'])
            assert len(si) == 600
        target = OUT / phase / ('_'.join(cell) + f'_k{shot}_s{seed}.npz')
        meta = target.with_suffix('.json')
        if meta.exists():
            record = json.loads(meta.read_text()); assert old.digest(target) == record['sha256']
            metrics.append(record); continue
        started = time.time(); scores = {}; gates = {}
        for st in range(0, len(si), 24):
            ss, xx = si[st:st + 24], xi[st:st + 24].reshape(-1, 75)
            sc, sd = [q[ss].to(device) for q in qfs]
            xc, xd = [q[xx].to(device) for q in qfs]
            v, diag = candidates(sc, sd, xc, xd, *G, configs, thresholds)
            for w, (qf, gallery) in legacy_reps.items():
                _, base, _ = evaluate_configs(qf[ss].to(device), qf[xx].to(device), gallery,
                                              [c for c in legacy if c['w'] == w], return_scores=True)
                v.update(base)
            for n, a in v.items(): scores.setdefault(n, []).append(a.numpy())
            for n, d in diag.items(): gates.setdefault(n, []).extend(d['gate_by_class'])
        names = list(scores); score = np.stack([np.concatenate(scores[n]) for n in names])
        pred = score.argmax(-1).astype('int8'); y = np.repeat(np.arange(5), 15)
        flips = 0; error = 0.
        if source is not None:
            bn = list(z['names']).index('r2_overall'); n = names.index('r2_overall')
            error = float(np.abs(score[n] - z['scores'][bn]).max())
            flips = int((pred[n] != z['predictions'][bn]).sum())
            assert error < 2e-4, error
            if flips:
                sorted_scores = np.sort(z['scores'][bn], axis=-1)
                margins = sorted_scores[:, :, -1] - sorted_scores[:, :, -2]
                assert np.all(margins[pred[n] != z['predictions'][bn]] < 2e-4)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix('.tmp.npz')
        np.savez_compressed(temp, names=names, predictions=pred, scores=score, yq=y,
                            support_indices=si, query_indices=xi, class_ids=cs, seed=seed)
        os.replace(temp, target)
        record = dict(cell=cell, shot=shot, seed=seed, episodes=len(si), names=names,
                      accuracy={n: float((p == y).mean()) for n, p in zip(names, pred)},
                      sha256=old.digest(target), seconds=time.time() - started,
                      source_sha256=old.digest(source) if source else None,
                      gallery_reference_thresholds=thresholds,
                      mean_gates={n: float(np.mean(g)) for n, g in gates.items()},
                      baseline_max_score_error=error, baseline_near_tie_flips=flips,
                      gallery_identity=identity)
        old.dump(meta, record); metrics.append(record)
        print('COMPLETE', phase, target.stem, 'seconds', round(record['seconds'], 1), flush=True)
    return metrics


def develop(device):
    allrows = {}; selection = {}
    for shot in [1, 5]:
        configs = menu(shot); rows = []
        for cell in DEV:
            rows.extend(run_cell(cell, shot, 'dev', configs, device))
        names = rows[0]['names']
        values = {n: [r['accuracy'][n] for r in rows] for n in names}
        base = np.array(values['r2_overall'])
        def value(name):
            a = np.array(values[name]); return float(a.mean() - np.maximum(base - a - .005, 0).mean())
        cfg_by_name = {c['name']: c for c in configs}
        best = max(list(cfg_by_name) + ['r2_overall'], key=value)
        family_best = {fam: max([c for c in configs if c['family'] == fam], key=lambda c: value(c['name']))
                       for fam in sorted(set(c['family'] for c in configs))}
        selection[str(shot)] = dict(selected_name=best, selected=cfg_by_name.get(best),
                                   family_best=family_best, development_means={n: float(np.mean(a)) for n, a in values.items()},
                                   criterion='mean accuracy minus mean per-condition loss beyond 0.5pp versus R2',
                                   dev_cell_accuracy=values)
    old.dump(OUT / 'selection.json', dict(selection=selection, hashes=hashes(), frozen_at=time.time(),
                                        test_results_opened_during_selection=False))
    print('FROZEN', {s: a['selected_name'] for s, a in selection.items()}, flush=True)


def test(device):
    receipt = json.loads((OUT / 'selection.json').read_text()); assert receipt['hashes'] == hashes()
    for cell in old.CELLS:
        for shot in [1, 5]:
            sel = receipt['selection'][str(shot)]
            cc = list(sel['family_best'].values())
            if sel['selected'] is not None: cc.append(sel['selected'])
            cc = list({c['name']: c for c in cc}.values())
            run_cell(cell, shot, 'test', cc, device)
    summarize()


def summarize():
    receipt = json.loads((OUT / 'selection.json').read_text()); rows = []; deltas = {}
    for cell in old.CELLS:
        for shot in [1, 5]:
            packs = []
            for p in sorted((OUT / 'test').glob('_'.join(cell) + f'_k{shot}_s*.npz')):
                meta = json.loads(p.with_suffix('.json').read_text()); assert old.digest(p) == meta['sha256']
                z = np.load(p, allow_pickle=False); names = list(z['names'])
                packs.append(((z['predictions'] == z['yq']).mean(-1), meta))
            assert len(packs) == 5
            a = np.concatenate([p[0] for p in packs], axis=1)
            selected = receipt['selection'][str(shot)]['selected_name']
            d = a[names.index(selected)] - a[names.index('r2_overall')]
            rng = np.random.default_rng(130999)
            boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(2000)])
            key = '_'.join(cell) + f'_k{shot}'; deltas[key] = d
            rows.append(dict(cell=cell, shot=shot, episodes=a.shape[1], selected=selected,
                             accuracy=dict(zip(names, a.mean(1).tolist())), gain_pp=float(d.mean() * 100),
                             paired_ci95_pp=(np.quantile(boot, [.025, .975]) * 100).tolist(),
                             max_baseline_error=max(p[1]['baseline_max_score_error'] for p in packs),
                             baseline_near_tie_flips=sum(p[1]['baseline_near_tie_flips'] for p in packs)))
    aggregates = []
    for shot in [1, 5]:
        for scope, cells in [('old4', DEV), ('new6', old.CELLS[4:]), ('all10', old.CELLS)]:
            rng = np.random.default_rng(130999)
            bootstrap = []
            for _ in range(2000):
                # Same query dataset has same test episodes across galleries: resample jointly.
                ids = {q: np.concatenate([s * 600 + rng.integers(600, size=600) for s in range(5)]) for q, g in cells}
                bootstrap.append(np.mean([deltas['_'.join(c) + f'_k{shot}'][ids[c[0]]].mean() for c in cells]))
            d = float(np.mean([deltas['_'.join(c) + f'_k{shot}'].mean() for c in cells]))
            aggregates.append(dict(shot=shot, scope=scope, gain_pp=d * 100,
                                   paired_ci95_pp=(np.quantile(bootstrap, [.025, .975]) * 100).tolist()))
    old.dump(OUT / 'summary.json', dict(complete=True, episodes=60000, rows=rows, aggregates=aggregates,
             scope='Historically viewed fixed-image-pool paired retest; no independent confirmation claim',
             hashes=hashes(), selection_sha256=old.digest(OUT / 'selection.json')))
    print('ALL_COMPLETE', aggregates, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--phase', choices=['dev', 'test', 'summarize'], required=True)
    ap.add_argument('--device', default='cuda:0'); args = ap.parse_args()
    torch.set_num_threads(4); OUT.mkdir(parents=True, exist_ok=True)
    old.bind(OUT / 'protocol.json', dict(hashes=hashes(), development_cells=[list(c) for c in DEV],
              dev_episodes_per_cell=200, test_episodes=60000, original_results=str(BASE),
              candidate_menus={str(k): menu(k) for k in [1, 5]}, query_batch_fitting=False,
              parameter_selection='development classes only; fixed before test'))
    if args.phase == 'dev': develop(args.device)
    elif args.phase == 'test': test(args.device)
    else: summarize()


if __name__ == '__main__': main()
