"""Factorial bridge-domain and mapping study under a strict target old-gallery interface."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / '研究4——少桥接样本的跨编码器检索'))
sys.path.insert(0, str(ROOT))
from src.methods import unit
from src.lab_methods import FeatureMap, METHODS
from src.improved import kernel_residual_scores
from src.lab_metrics import evaluate_rank
from research_common.records import sha256

BASE = Path('/data/liuhaoyu/r45-scale-20260909/r4')
OUT = Path('/data/liuhaoyu/r45-refine-20260909/r4')
PAIRS = [('clip_b32', 'clip_b16'), ('clip_b16', 'dino_s14')]
POLICIES = ['cub_random', 'landmark_random', 'landmark_cover']
SEEDS = [51851, 51852, 51853, 51854, 51855]


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp.json'); temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(temp, p)


def read_dataset(name):
    mpath = BASE / name / 'manifest.json'; m = json.loads(mpath.read_text()); a = {}
    for encoder in ['clip_b32', 'clip_b16', 'dino_s14']:
        p = BASE / name / 'cache' / encoder / 'embeddings.npy'
        meta = json.loads(p.with_suffix('.json').read_text())
        assert meta['manifest_sha256'] == sha256(mpath) and meta['array_sha256'] == sha256(p)
        a[encoder] = np.load(p, mmap_mode='r')
    return m, a


def allowed_bridges(target, external):
    # Never select by new features or target relevance labels.
    rgbs = {r['rgb_sha256'] for r in target['records']}
    buckets = {}
    for r in target['records']:
        v = int(r['dhash64'], 16)
        for j in range(4): buckets.setdefault((j, (v >> (16 * j)) & 65535), set()).add(v)
    permitted = set(external['roles']['gallery']); allowed = []; excluded = []; seen = set()
    for i, r in enumerate(external['records']):
        if r['id'] not in permitted: continue
        v = int(r['dhash64'], 16)
        possible = set().union(*(buckets.get((j, (v >> (16 * j)) & 65535), set()) for j in range(4)))
        if r['rgb_sha256'] in rgbs or r['rgb_sha256'] in seen or any(bin(v ^ x).count('1') <= 2 for x in possible):
            excluded.append(r['id']); continue
        allowed.append(i); seen.add(r['rgb_sha256'])
    return np.array(allowed), excluded


def cover_select(candidates, gallery, m):
    # Greedy facility location on a deterministic old-gallery probe; only old vectors.
    ids = np.linspace(0, len(gallery) - 1, min(512, len(gallery)), dtype=int)
    sim = unit(gallery[ids]) @ unit(candidates).T
    current = np.full(len(ids), -1.); selected = []
    for _ in range(m):
        gains = np.maximum(sim - current[:, None], 0).mean(0)
        gains[selected] = -np.inf
        j = int(np.argmax(gains)); selected.append(j); current = np.maximum(current, sim[:, j])
    return np.array(selected)


def forward_fit(ao, an):
    mo, mn = ao.mean(0), an.mean(0)
    x, y = ao - mo, an - mn
    k = x @ x.T; scale = max(np.trace(k) / len(k), 1e-8); rows = []
    for coefficient in [.001, .01, .1, 1., 10.]:
        inv = np.linalg.inv(k + coefficient * scale * np.eye(len(k)))
        hat = k @ inv + np.ones_like(k) / len(k)
        loss = float(np.mean(((an - hat @ an) / np.maximum(1 - np.diag(hat)[:, None], 1e-8)) ** 2))
        rows.append((loss, coefficient))
    loss, coefficient = min(rows)
    t = x.T @ np.linalg.solve(k + coefficient * scale * np.eye(len(k)), y)
    return mo, mn, t, dict(coefficient=coefficient, bridge_loo_mse=loss)


def candidate_scores(go, qo, qn, ao, an):
    # Method-only entry: no paths, identities, labels or target new-gallery vectors.
    go, qo, qn, ao, an = map(unit, [go, qo, qn, ao, an])
    old = qo @ go.T; mu = go.mean(0)
    scores = {'old_centered': unit(qo - mu) @ unit(go - mu).T}
    scores['linear_raw'], _ = kernel_residual_scores(go, qo, qn, ao, an, 'linear', False)
    scores['rbf_raw'], _ = kernel_residual_scores(go, qo, qn, ao, an, 'rbf_0.5', False)
    mo, mn, t, diag = forward_fit(ao, an)
    pseudo = unit((go - mo) @ t + mn)
    new = qn @ pseudo.T
    scores['forward_cosine'] = new
    centered = new - new.mean(1, keepdims=True)
    matched = centered * old.std(1, keepdims=True) / np.maximum(centered.std(1, keepdims=True), 1e-10)
    for mix in [.25, .5, .75]:
        scores[f'forward_blend_{mix}'] = (1 - mix) * old + mix * matched
    diag.update(old_bridge_coverage_mean=float(np.max(go @ ao.T, axis=1).mean()),
                gallery_effective_rank=float(np.linalg.matrix_rank(ao - ao.mean(0))),
                method_new_gallery_rows=0, paired_bridge_rows=len(ao))
    return scores, diag


def measure(scores, manifest, qids):
    rank = np.argsort(-np.asarray(scores, dtype='float32'), axis=1, kind='stable')
    rows = []; ap = []
    for qid, rr in zip(qids, rank):
        metrics = {c: evaluate_rank(rr, t, 'revisited') for c, t in manifest['truth_evaluator_only'][qid].items()}
        rows.append(dict(query_id=qid, metrics=metrics, top100=rr[:100].tolist()))
        ap.append([metrics[c]['ap'] for c in ['medium', 'hard']])
    return rows, np.asarray(ap)


def run(name):
    manifest, arrays = read_dataset(name)
    other = 'rparis' if name == 'roxford' else 'roxford'
    external, external_arrays = read_dataset(other)
    allowed, excluded = allowed_bridges(manifest, external)
    assert len(allowed) >= 256
    dump(OUT / name / 'bridge_audit.json', dict(external_source=other, candidate_count=len(allowed),
          excluded_count=len(excluded), excluded_ids=excluded, exact_rgb_disjoint=True,
          dhash_distance_le_2_excluded=True, semantic_instance_overlap='No shared city claimed; no manual semantic audit'))
    lookup = {r['id']: i for i, r in enumerate(manifest['records'])}
    gi = np.array([lookup[x] for x in manifest['roles']['gallery']])
    qi = np.array([lookup[x] for x in manifest['roles']['query']])
    for old, new in PAIRS:
        go, qo, qn = arrays[old][gi], arrays[old][qi], arrays[new][qi]
        for m in [32, 64]:
            for seed in SEEDS:
                for policy in POLICIES:
                    folder = OUT / name / f'{old}__{new}__m{m}__s{seed}__{policy}'
                    if (folder / 'complete.json').exists(): continue
                    if policy == 'cub_random':
                        ids = list(np.random.default_rng(seed).permutation(manifest['roles']['bridge'])[:m])
                        ix = [lookup[i] for i in ids]
                        ao, an = arrays[old][ix], arrays[new][ix]
                        bridge_ids = ids
                    else:
                        pool = np.random.default_rng(seed).permutation(allowed)[:256]
                        chosen = np.arange(m) if policy == 'landmark_random' else cover_select(external_arrays[old][pool], go, m)
                        ix = pool[chosen]
                        ao, an = external_arrays[old][ix], external_arrays[new][ix]
                        bridge_ids = [other + ':' + external['records'][i]['id'] for i in ix]
                    assert len(ao) == len(an) == m
                    start = time.time(); scores, diagnostics = candidate_scores(go, qo, qn, ao, an)
                    # All previous baseline methods receive exactly the same new bridge subset.
                    for method in METHODS:
                        fm = FeatureMap(ao, an, method)
                        scores[method] = np.asarray(fm.query(qo, qn), dtype='float32') @ np.asarray(fm.gallery(go), dtype='float32').T
                    scores['oracle_new_new'] = qn @ arrays[new][gi].T
                    summary = {}; all_ap = []
                    for method, score in scores.items():
                        rows, ap = measure(score, manifest, manifest['roles']['query'])
                        prediction = folder / (method + '.predictions.json')
                        dump(prediction, rows)
                        summary[method] = dict(medium=float(np.nanmean(ap[:, 0])), hard=float(np.nanmean(ap[:, 1])),
                                               prediction_sha256=sha256(prediction))
                        all_ap.append(ap)
                    # Exact same CUB seed/roles: verify legacy scores retain previous rankings/metrics.
                    parity = 0.
                    if policy == 'cub_random':
                        original = BASE / name / 'evaluation' / f'{old}__{new}__m{m}__s{seed}' / 'v0_press.summary.json'
                        s = json.loads(original.read_text())
                        parity = max(abs(summary['v0_press'][c] - s['metrics'][c]['map']) for c in ['medium', 'hard'])
                        assert parity < 2e-5, parity
                    np.savez_compressed(folder / 'aps.npz', names=list(scores), ap=np.stack(all_ap), query_ids=manifest['roles']['query'])
                    dump(folder / 'complete.json', dict(dataset=name, pair=[old, new], budget=m, seed=seed, policy=policy,
                          bridge_ids=bridge_ids, summary=summary, diagnostics=diagnostics, seconds=time.time() - start,
                          aps_sha256=sha256(folder / 'aps.npz'), v0_replay_max_map_error=parity))
                    print('COMPLETE', name, folder.name, round(time.time() - start, 1), flush=True)


def select():
    rows = [json.loads(p.read_text()) for p in (OUT / 'roxford').glob('*/complete.json')]
    assert len(rows) == 60
    names = ['v0_press', 'linear_raw', 'rbf_raw', 'forward_cosine', 'forward_blend_0.25', 'forward_blend_0.5', 'forward_blend_0.75']
    values = {n: float(np.mean([r['summary'][n][c] - r['summary']['v0_press'][c] for r in rows for c in ['medium', 'hard']])) for n in names}
    selected = max(values, key=values.get)
    dump(OUT / 'selection.json', dict(selected=selected, oxford_macro_delta=values, frozen_at=time.time(),
          selection_data='Oxford only, all three bridge policies equally weighted', previously_viewed_benchmarks=True,
          source_sha256=sha256(Path(__file__))))
    print('FROZEN', selected, values, flush=True)


def summarize():
    selected = json.loads((OUT / 'selection.json').read_text())['selected']
    rows = []; groups = {}
    for dataset in ['roxford', 'rparis']:
        for p in (OUT / dataset).glob('*/complete.json'):
            r = json.loads(p.read_text()); assert sha256(p.parent / 'aps.npz') == r['aps_sha256']
            for method, info in r['summary'].items():
                assert sha256(p.parent / (method + '.predictions.json')) == info['prediction_sha256']
            groups.setdefault((dataset, *r['pair'], r['budget'], r['policy']), []).append(r)
    assert len(groups) == 24 and all(len(v) == 5 for v in groups.values())
    for key, data in sorted(groups.items()):
        for c in ['medium', 'hard']:
            means = {n: float(np.mean([d['summary'][n][c] for d in data])) for n in data[0]['summary']}
            controls = {n: a for n, a in means.items() if n in METHODS and n not in ['v0_press', 'global_press'] or n == 'old_centered'}
            best = max(controls, key=controls.get)
            rows.append(dict(dataset=key[0], pair=list(key[1:3]), budget=key[3], policy=key[4], condition=c,
                             means=means, primary=selected, gain_vs_v0_pp=100 * (means[selected] - means['v0_press']),
                             strongest_control=best, gain_vs_strongest_control_pp=100 * (means[selected] - controls[best])))
    dump(OUT / 'summary.json', dict(complete=True, protocol_cells=120, methods_per_cell=17,
          selected=selected, rows=rows, scope='Historically viewed benchmarks, Oxford selection then Paris frozen retest',
          limitation='Bridge-domain intervention gains and mathematical-method gains must be reported separately'))
    print('ALL_COMPLETE', selected, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(source_sha256=sha256(Path(__file__)), baseline=str(BASE), policies=POLICIES,
                    pairs=PAIRS, budgets=[32, 64], seeds=SEEDS, development='roxford', retest='rparis',
                    target_new_gallery_to_method=False, reused_features=True, additional_encoder_calls=0)
    path = OUT / 'protocol.json'
    if path.exists(): assert json.loads(path.read_text()) == json.loads(json.dumps(protocol))
    else: dump(path, protocol)
    run('roxford')
    if not (OUT / 'selection.json').exists(): select()
    assert json.loads((OUT / 'selection.json').read_text())['source_sha256'] == sha256(Path(__file__))
    run('rparis'); summarize()


if __name__ == '__main__': main()
