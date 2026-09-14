"""Evidence-driven addendum: preserve V0 residual but improve the old-side geometry."""
from pathlib import Path
import json
import time
import numpy as np
from r4_refine import OUT as FIRST, read_dataset, candidate_scores, FeatureMap, unit, measure, dump, sha256

OUT = FIRST.parent / 'r4_geometry'
ALPHAS = [.5, 1., 2.]


def corrected(go, qo, qn, ao, an):
    fm = FeatureMap(ao, an, 'v0_press')
    old = unit(qo) @ unit(go).T
    v0 = fm.query(qo, qn) @ fm.gallery(go).T
    mu = unit(go).mean(0)
    centered = unit(unit(qo) - mu) @ unit(unit(go) - mu).T
    residual = v0 - old
    residual -= residual.mean(1, keepdims=True)
    ratio = centered.std(1, keepdims=True) / np.maximum(old.std(1, keepdims=True), 1e-10)
    results = dict(old_old=old, old_centered=centered, v0_press=v0)
    results.update({f'centered_residual_{alpha}': centered + alpha * ratio * residual for alpha in ALPHAS})
    return results


def run(dataset):
    manifest, arrays = read_dataset(dataset)
    other = 'rparis' if dataset == 'roxford' else 'roxford'
    external, ext = read_dataset(other)
    lookup = {r['id']: i for i, r in enumerate(manifest['records'])}
    elookup = {other + ':' + r['id']: i for i, r in enumerate(external['records'])}
    gi = [lookup[i] for i in manifest['roles']['gallery']]
    qi = [lookup[i] for i in manifest['roles']['query']]
    files = sorted((FIRST / dataset).glob('*/complete.json')); assert len(files) == 60
    for source in files:
        r = json.loads(source.read_text()); old, new = r['pair']
        dest = OUT / dataset / source.parent.name
        if (dest / 'complete.json').exists(): continue
        if r['policy'] == 'cub_random':
            ids = [lookup[i] for i in r['bridge_ids']]; ao, an = arrays[old][ids], arrays[new][ids]
        else:
            ids = [elookup[i] for i in r['bridge_ids']]; ao, an = ext[old][ids], ext[new][ids]
        scores = corrected(arrays[old][gi], arrays[old][qi], arrays[new][qi], ao, an)
        aps = []; summaries = {}
        for name, value in scores.items():
            predictions, ap = measure(value, manifest, manifest['roles']['query'])
            p = dest / (name + '.predictions.json'); dump(p, predictions); aps.append(ap)
            summaries[name] = dict(medium=float(np.nanmean(ap[:, 0])), hard=float(np.nanmean(ap[:, 1])),
                                   prediction_sha256=sha256(p))
        np.savez_compressed(dest / 'aps.npz', names=list(scores), ap=np.stack(aps), query_ids=manifest['roles']['query'])
        dump(dest / 'complete.json', dict(source_sha256=sha256(source), pair=r['pair'], dataset=dataset,
              budget=r['budget'], seed=r['seed'], policy=r['policy'], summary=summaries,
              aps_sha256=sha256(dest / 'aps.npz')))
        print('COMPLETE', dataset, dest.name, flush=True)


def select():
    rows = [json.loads(p.read_text()) for p in (OUT / 'roxford').glob('*/complete.json')]
    values = {n: float(np.mean([r['summary'][n][c] for r in rows for c in ['medium', 'hard']])) for n in rows[0]['summary']}
    selected = max(values, key=values.get)
    dump(OUT / 'selection.json', dict(selected=selected, oxford_macro=values, frozen_at=time.time(),
                                     source_sha256=sha256(Path(__file__))))
    print('FROZEN', selected, values, flush=True)


def summarize():
    selected = json.loads((OUT / 'selection.json').read_text())['selected']; groups = {}
    for dataset in ['roxford', 'rparis']:
        for p in (OUT / dataset).glob('*/complete.json'):
            r = json.loads(p.read_text()); assert sha256(p.parent / 'aps.npz') == r['aps_sha256']
            for n, v in r['summary'].items(): assert sha256(p.parent / (n + '.predictions.json')) == v['prediction_sha256']
            z = np.load(p.parent / 'aps.npz', allow_pickle=False)
            groups.setdefault((dataset, *r['pair'], r['budget'], r['policy']), []).append((r, z))
    assert len(groups) == 24 and all(len(g) == 5 for g in groups.values())
    rows = []
    for key, group in sorted(groups.items()):
        names = list(group[0][1]['names']); stack = np.stack([z['ap'] for _, z in group])
        for cidx, c in enumerate(['medium', 'hard']):
            means = {n: float(np.nanmean(stack[:, i, :, cidx])) for i, n in enumerate(names)}
            deltas = {}; rng = np.random.default_rng(51999)
            for base in ['v0_press', 'old_centered']:
                delta = stack[:, names.index(selected), :, cidx] - stack[:, names.index(base), :, cidx]
                # Same query identity is paired across bridge seeds. Two sources of variation.
                bootstrap = [np.nanmean(delta[rng.integers(5, size=5)][:, rng.integers(70, size=70)]) for _ in range(2000)]
                deltas[base] = dict(gain_pp=float(np.nanmean(delta) * 100),
                       paired_query_seed_ci95_pp=(np.quantile(bootstrap, [.025, .975]) * 100).tolist())
            rows.append(dict(dataset=key[0], pair=list(key[1:3]), budget=key[3], policy=key[4], condition=c,
                             means=means, selected=selected, comparisons=deltas))
    dump(OUT / 'summary.json', dict(complete=True, cells=120, selected=selected, rows=rows,
          selection_sha256=sha256(OUT / 'selection.json'),
          interval_note='Query-image and bridge-seed paired bootstrap; queries of the same landmark can be dependent; no landmark-group confirmation claim'))
    print('ALL_COMPLETE', selected, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(source_sha256=sha256(Path(__file__)), alphas=ALPHAS, justification='New Oxford old-geometry evidence',
                    development='roxford', frozen_retest='rparis', baseline=str(FIRST))
    p = OUT / 'protocol.json'
    if p.exists(): assert json.loads(p.read_text()) == protocol
    else: dump(p, protocol)
    run('roxford')
    if not (OUT / 'selection.json').exists(): select()
    assert json.loads((OUT / 'selection.json').read_text())['source_sha256'] == sha256(Path(__file__))
    run('rparis'); summarize()


if __name__ == '__main__': main()
