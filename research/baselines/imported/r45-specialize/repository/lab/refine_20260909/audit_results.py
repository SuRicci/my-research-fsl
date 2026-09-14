"""Read-only audit and paired secondary analysis after both selections were frozen."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

BASE = Path('/data/liuhaoyu/r45-refine-20260909')
ROOT = Path('/data/liuhaoyu/individual-research')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def r4_audit():
    packs = {}; counts = {}; replay_error = 0.
    for phase, methods in [('r4', 17), ('r4_geometry', 6)]:
        d = BASE / phase
        assert (d / 'exit_code').read_text().strip() == '0'
        assert read(d / 'summary.json')['complete']
        protocol = read(d / 'protocol.json'); receipt = read(d / 'selection.json')
        source = ROOT / 'lab/refine_20260909' / (phase.replace('r4', 'r4_refine', 1) + '.py' if phase == 'r4' else phase + '.py')
        assert sha(source) == protocol['source_sha256'] == receipt['source_sha256']
        paths = sorted(d.glob('*/*/complete.json')); assert len(paths) == 120
        total_predictions = 0
        for p in paths:
            r = read(p); assert sha(p.parent / 'aps.npz') == r['aps_sha256']
            with np.load(p.parent / 'aps.npz', allow_pickle=False) as z:
                names, ap, qids = list(z['names']), z['ap'], z['query_ids']
            assert ap.shape == (methods, 70, 2)
            assert np.all((ap[np.isfinite(ap)] >= 0) & (ap[np.isfinite(ap)] <= 1))
            assert len(set(qids)) == 70
            for i, name in enumerate(names):
                pred = p.parent / (name + '.predictions.json')
                assert sha(pred) == r['summary'][name]['prediction_sha256']
                rows = read(pred); assert [x['query_id'] for x in rows] == list(qids)
                derived = np.array([[x['metrics'][c]['ap'] for c in ['medium', 'hard']] for x in rows])
                np.testing.assert_allclose(derived, ap[i], atol=0, rtol=0, equal_nan=True)
                for ci, condition in enumerate(['medium', 'hard']):
                    np.testing.assert_allclose(np.nanmean(ap[i, :, ci]), r['summary'][name][condition], atol=1e-12)
                total_predictions += 1
            key = (r['dataset'], *r['pair'], r['budget'], r['policy'], r['seed'])
            packs[phase, key] = dict(names=names, ap=ap, qids=qids)
            if phase == 'r4_geometry':
                original = BASE / 'r4' / p.relative_to(d)
                assert sha(original) == r['source_sha256']
                a = packs['r4', key]
                for n in ['v0_press', 'old_old', 'old_centered']:
                    diff = np.nanmean(ap[names.index(n)], axis=0) - np.nanmean(a['ap'][a['names'].index(n)], axis=0)
                    replay_error = max(replay_error, float(np.max(np.abs(diff))))
            else:
                assert len(set(r['bridge_ids'])) == r['budget']
        counts[phase] = dict(cells=len(paths), prediction_files=total_predictions,
                             query_records=70 * total_predictions, source_sha256=sha(source))
    selected = read(BASE / 'r4_geometry/selection.json')['selected']
    rows = []
    keys = sorted(set(k[:-1] for phase, k in packs if phase == 'r4_geometry'))
    for key in keys:
        pp = [packs['r4_geometry', (*key, seed)] for seed in range(51851, 51856)]
        names = pp[0]['names']; stack = np.stack([p['ap'] for p in pp])
        old_cub = [packs['r4', (*key[:-1], 'cub_random', seed)] for seed in range(51851, 51856)]
        same = [packs['r4', (*key, seed)] for seed in range(51851, 51856)]
        for ci, condition in enumerate(['medium', 'hard']):
            means = {n: float(np.nanmean(stack[:, i, :, ci])) for i, n in enumerate(names)}
            cub = float(np.nanmean([p['ap'][p['names'].index('v0_press'), :, ci] for p in old_cub]))
            baselines = ['old_old', 'old_centered', 'relative_centered', 'procrustes', 'ridge',
                         'wip_restricted', 'local_std', 'linear_std', 'v0_press', 'global_press']
            control = {n: float(np.nanmean([p['ap'][p['names'].index(n), :, ci] for p in same])) for n in baselines}
            strongest = max(control, key=control.get)
            oracle = float(np.nanmean([p['ap'][p['names'].index('oracle_new_new'), :, ci] for p in same]))
            rows.append(dict(dataset=key[0], pair=list(key[1:3]), budget=key[3], policy=key[4], condition=condition,
                selected=selected, means=means, original_cub_v0=cub, strongest_baseline=strongest,
                strongest_baseline_map=control[strongest], gain_vs_strongest_pp=100 * (means[selected]-control[strongest]),
                oracle_map=oracle, total_gain_vs_original_cub_v0_pp=100*(means[selected]-cub),
                oracle_gain_retention=(means[selected]-means['old_old'])/(oracle-means['old_old']) if oracle>means['old_old'] else None))
    return dict(counts=counts, shared_baseline_map_max_difference=replay_error, selected=selected, rows=rows)


def paired_interval(delta, seed=130999):
    rng = np.random.default_rng(seed)
    bootstrap = [delta[np.concatenate([s*600+rng.integers(600,size=600) for s in range(5)])].mean() for _ in range(2000)]
    return (100*np.quantile(bootstrap,[.025,.975])).tolist()


def r5_audit():
    d = BASE / 'r5'; assert (d / 'exit_code').read_text().strip() == '0'
    summary, receipt, protocol = [read(d/n) for n in ['summary.json','selection.json','protocol.json']]
    assert summary['complete'] and sha(d/'selection.json') == summary['selection_sha256']
    assert summary['hashes'] == receipt['hashes'] == protocol['hashes']
    for path, digest in summary['hashes'].items(): assert sha(ROOT / path) == digest
    files = sorted((d/'test').glob('*.npz')); assert len(files) == 100
    grouped = {}; episodes = 0; max_error = 0.; flips = 0
    for p in files:
        meta = read(p.with_suffix('.json')); assert sha(p) == meta['sha256']
        source = Path(protocol['original_results']) / 'results' / p.name
        assert sha(source) == meta['source_sha256']
        with np.load(p, allow_pickle=False) as z, np.load(source, allow_pickle=False) as orig:
            for n in ['support_indices','query_indices','class_ids','seed']:
                np.testing.assert_array_equal(z[n],orig[n])
            names = list(z['names']); acc=(z['predictions']==z['yq']).mean(-1)
            np.testing.assert_array_equal(z['predictions'][names.index('r2_overall')],
                                          orig['predictions'][list(orig['names']).index('r2_overall')])
            assert acc.shape[1] == 600
        for n, a in zip(names, acc): np.testing.assert_allclose(a.mean(),meta['accuracy'][n],atol=1e-12)
        grouped.setdefault((tuple(meta['cell']),meta['shot']),[]).append((meta['seed'],names,acc))
        episodes += 600; max_error=max(max_error,meta['baseline_max_score_error']); flips+=meta['baseline_near_tie_flips']
    rows=[]; deltas={}
    for (cell,shot), data in sorted(grouped.items()):
        assert len(data)==5
        data.sort(); names=data[0][1]; assert all(p[1]==names for p in data)
        acc=np.concatenate([p[2] for p in data],axis=1); families=receipt['selection'][str(shot)]['family_best']
        for fam,cfg in families.items():
            name=cfg['name']; delta=acc[names.index(name)]-acc[names.index('r2_overall')]
            deltas[cell,shot,fam]=delta
            rows.append(dict(cell=list(cell),shot=shot,family=fam,configuration=name,
                accuracy=float(acc[names.index(name)].mean()),baseline=float(acc[names.index('r2_overall')].mean()),
                gain_pp=float(delta.mean()*100),paired_fixed_pool_ci95_pp=paired_interval(delta)))
    aggregates=[]
    old4=[tuple(c) for c in protocol['development_cells']]
    all10=list(dict.fromkeys(tuple(r['cell']) for r in summary['rows']))
    for shot in [1,5]:
        for scope,cells in [('old4',old4),('new6',[c for c in all10 if c not in old4]),('all10',all10)]:
            for fam in receipt['selection'][str(shot)]['family_best']:
                rng=np.random.default_rng(130999); boot=[]
                for _ in range(2000):
                    indices={q:np.concatenate([s*600+rng.integers(600,size=600) for s in range(5)]) for q in sorted({c[0] for c in cells})}
                    boot.append(np.mean([deltas[c,shot,fam][indices[c[0]]].mean() for c in cells]))
                aggregates.append(dict(shot=shot,scope=scope,family=fam,
                    gain_pp=float(np.mean([deltas[c,shot,fam].mean() for c in cells])*100),
                    paired_fixed_pool_ci95_pp=(100*np.quantile(boot,[.025,.975])).tolist()))
    return dict(files=len(files),episodes=episodes,max_baseline_score_error=max_error,baseline_prediction_flips=flips,
                rows=rows,aggregates=aggregates,interpretation='Secondary fixed-family results, not post-test primary promotion; fixed image pools and repeated classes')


if __name__=='__main__':
    result=dict(completed_at=time.time(),r4=r4_audit(),r5=r5_audit())
    result['all_checks_passed']=True
    result['analysis_source_sha256']=sha(Path(__file__))
    (BASE/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('AUDIT_COMPLETE',result['r4']['counts'],result['r5']['episodes'],flush=True)
