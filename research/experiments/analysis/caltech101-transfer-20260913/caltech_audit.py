"""Independent dense ridge/scatter reference; no target outcomes for preflight."""
import argparse, json, time
import numpy as np
import torch
import caltech_eval as e
P = json.loads((e.HERE/'audit_plan.json').read_text())

def norm(x):
    return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def ridge(x, q, y=None, weight=1., lam=.1):
    if y is None: y = np.eye(len(x))
    a, b = x-x.mean(0), y-y.mean(0)
    w = np.linalg.solve(weight*a.T@a+lam*np.eye(x.shape[-1]), weight*a.T@b)
    return (q-x.mean(0))@w+y.mean(0)

def r2(s, q, g):
    p = norm(s.mean(1))
    ix = np.argsort(-(p@g.T), axis=-1, kind='stable')[:, :64]
    x = norm(.5*p+.5*norm(g[ix].mean(1)))
    return ridge(x, q)

def independent(sv, qv, gv, gamma, eta):
    s, q, g = [x.numpy().astype(np.float64) for x in [sv, qv, gv]]
    sm, qm, gm = [norm(x.mean(-2)) for x in [s, q, g]]
    mu = sm.reshape(-1, sm.shape[-1]).mean(0)
    result = {'original_r2': r2(s[:, :, 0], q[:, 0], g[:, 0]),
        'mean_r2': r2(sm, qm, gm), 'mean_CS_l2': r2(norm(sm-mu), norm(qm-mu), norm(gm-mu)),
        'score_ensemble_r2': np.mean([r2(s[:, :, v], q[:, v], g[:, v]) for v in range(6)], axis=0),
        'augmented_support_ridge': ridge(s.reshape(-1, s.shape[-1]), qm, np.repeat(np.eye(5), 6, axis=0), 1/6)}
    d = (s-s.mean(-2, keepdims=True)).reshape(-1, s.shape[-1])
    cov = d.T@d/len(d)
    ev, u = np.linalg.eigh(np.eye(len(cov))+gamma*len(cov)*cov/np.trace(cov))
    transform = (u/np.sqrt(ev))@u.T
    s, q, g, v = norm(s.mean(-2)@transform), norm(q.mean(-2)@transform), norm(g.mean(-2)@transform), norm(q@transform)
    mu = s.reshape(-1, s.shape[-1]).mean(0)
    parents, candidates = [], []
    for sh, qh, gh, vh in [(s, q, g, v), (norm(s-mu), norm(q-mu), norm(g-mu), norm(v-mu))]:
        proto = norm(sh.mean(1))
        ix = np.argsort(-(proto@gh.T), axis=-1, kind='stable')[:, :64]
        x = norm(.5*proto+.5*norm(gh[ix].mean(1)))
        parent = ridge(x, qh)
        parents.append(parent)
        if eta == 0: candidates.append(parent); continue
        a = x-x.mean(0)
        b = np.eye(5)-np.ones((5, 5))/5
        values = []
        for point, views in zip(qh, vh):
            r = views-views.mean(0)
            w = np.linalg.solve(a.T@a+.1*np.eye(x.shape[-1])+eta*r.T@r/len(r), a.T@b)
            values.append((point-x.mean(0))@w+np.ones(5)/5)
        candidates.append(np.array(values))
    result.update(scatter_r2=parents[0], scatter_blend=np.mean(parents, 0), query_consistency=np.mean(candidates, 0))
    return result

def compare(actual, expected, label):
    records = []
    for name, want in expected.items():
        got = np.asarray(actual[name])
        error = float(np.max(abs(got-want)))
        tol = P['float64_stack_tolerance'] if name in ['scatter_r2', 'scatter_blend', 'query_consistency'] else P['float32_control_tolerance']
        assert error < tol, (label, name, error)
        assert np.array_equal(got.argmax(-1), want.argmax(-1)), ('argmax', label, name, error)
        records.append({'label': label, 'method': name, 'max_error': error})
    return records

def preflight():
    start = time.time()
    e.guard()
    e.check_lock(e.HERE/'design_lock.json')
    e.check_lock(e.z.HERE/'code_lock.json')
    torch.manual_seed(260913960)
    s, q, g = [e.metric.norm(torch.randn(*shape)) for shape in [(5, 1, 6, 32), (7, 6, 32), (90, 6, 32)]]
    records = []
    controls = e.control_scores(s, q, g)
    for source, cfg in e.CFG['source_configurations'].items():
        got = e.stack_scores(s, q, g, **cfg)
        records += compare({k: v.numpy() for k, v in {**controls, **got}.items()}, independent(s, q, g, **cfg), 'synthetic/'+source)
        order = torch.tensor([4, 0, 6, 3, 2, 1, 5])
        perm = e.stack_scores(s, q[order], g, **cfg)
        parts = [e.stack_scores(s, part, g, **cfg) for part in q.split(3)]
        cp = torch.tensor([3, 0, 4, 1, 2])
        classes = e.stack_scores(s[cp], q, g, **cfg)
        for name, score in got.items():
            assert torch.allclose(score[order], perm[name], atol=1e-10, rtol=0)
            assert torch.allclose(score, torch.cat([v[name] for v in parts]), atol=1e-10, rtol=0)
            assert torch.allclose(score[:, cp], classes[name], atol=1e-10, rtol=0)
        if cfg['eta'] == 0: assert torch.equal(got['scatter_blend'], got['query_consistency'])
    data = e.z.m.old.load()
    parity, parity_errors = 0, []
    for source, target in [('dtd', 'eurosat'), ('eurosat', 'dtd')]:
        old = np.load(e.z.HERE/'outputs'/(target+'_'+target+'.npz'))
        prior = np.load(e.ROOT/'experiments/main/representation-scatter-20260913/outputs/cells'/f'{source}_to_{target}_{target}_k1.npz')
        assert all(np.array_equal(old[k], prior[k]) for k in e.KEYS)
        cfg = e.CFG['source_configurations'][source]
        for i in P['source_tasks']:
            sv = data[target]['query'][old['support_indices'][i]]
            qv = data[target]['query'][old['query_indices'][i].reshape(-1)]
            gv = data[target]['gallery']
            controls = e.control_scores(sv, qv, gv)
            for name, scores in controls.items():
                expected = prior['scores'][prior['names'].tolist().index(name), i]
                error = float(abs(scores.numpy()-expected).max())
                assert error < 2e-6 and np.array_equal(scores.argmax(-1), expected.argmax(-1)), (name, error)
                parity_errors.append(error); parity += 1
            stack = e.stack_scores(sv, qv, gv, **cfg)
            for name, oi in [('scatter_blend', 0), ('query_consistency', 1)]:
                error = float(abs(stack[name].numpy()-old['scores'][oi, i]).max())
                assert error < 1e-10 and np.array_equal(stack[name].argmax(-1), old['predictions'][oi, i])
                parity_errors.append(error); parity += 1
            qi = P['target_query_indices_for_direct_primal']
            values = {**controls, **stack}
            records += compare({k: v[qi].numpy() for k, v in values.items()}, independent(sv, qv[qi], gv, **cfg), f'source/{target}/{i}')
    result = {'status': 'passed', 'target_features_or_classification_observed': False,
        'source_score_parity_checks': parity, 'source_max_score_error': max(parity_errors),
        'independent_score_checks': len(records), 'max_independent_score_error': max(x['max_error'] for x in records),
        'invariances': ['query_order', 'query_partition', 'class_permutation', 'eta_zero_parent'],
        'checks': records, 'torch': torch.__version__, 'numpy': np.__version__, 'seconds': time.time()-start}
    e.dump(e.OUT/'preflight.json', result)
    e.dump(e.HERE/'evaluation_code_lock.json', e.sources())
    print('PREFLIGHT_PASS', json.dumps({k: v for k, v in result.items() if k != 'checks'}), flush=True)

def independent_interval(delta, seeds):
    rng = np.random.RandomState(e.CFG['statistics']['seed'])
    size = e.CFG['statistics']['replicates']
    resampled = np.zeros(size)
    for seed in np.unique(seeds):
        d = delta[seeds == seed]
        idx = rng.randint(len(d), size=(size, len(d)))
        counts = np.zeros((size, len(d)), dtype=np.int32)
        np.add.at(counts, (np.arange(size)[:, None], idx), 1)
        resampled += counts@d/len(d)/len(np.unique(seeds))
    return np.r_[delta.mean()*100, np.percentile(resampled, [2.5, 97.5])*100]

def audit():
    start = time.time()
    e.check_lock(e.HERE/'evaluation_code_lock.json')
    target, galleries, tasks = e.load_data()
    result = json.loads((e.OUT/'analysis.json').read_text())
    banks, records, intervals = {}, [], []
    primary = e.NAMES.index(e.CFG['primary_method'])
    for source, cfg in e.CFG['source_configurations'].items():
        accs = []
        for g, gv in galleries.items():
            key = source+'_'+g
            f = np.load(e.OUT/(key+'.npz'))
            assert f['names'].tolist() == e.NAMES and np.array_equal(f['yq'], e.Y)
            assert f['scores'].shape == (10, 500, 75, 5) and np.isfinite(f['scores']).all()
            assert f['code_lock_sha256'].item() == e.sha(e.HERE/'evaluation_code_lock.json')
            assert all(np.array_equal(f[k], tasks[k]) for k in e.KEYS)
            assert np.array_equal(f['gallery_indices'], json.loads((e.HERE/'galleries.json').read_text())[g]['indices'])
            assert np.array_equal(f['scores'].argmax(-1), f['predictions'])
            acc = (f['predictions'] == e.Y).mean(-1)
            assert np.array_equal(acc, f['accuracy'])
            assert np.max(abs(acc.mean(-1)*100-np.array(list(result['cells'][key]['accuracy_pct'].values())))) < 1e-10
            for k, name in enumerate(e.NAMES):
                if k == primary: continue
                got = independent_interval(acc[primary]-acc[k], f['seeds'])
                want = result['cells'][key]['comparisons'][name]
                intervals.append(float(abs(got-np.r_[want['delta_pp'], want['ci95_pp']]).max()))
                rs = result['cells'][key]['repairs_spoils'][name]
                assert rs['repairs']-rs['spoils'] == int((f['predictions'][primary] == e.Y).sum()-(f['predictions'][k] == e.Y).sum())
            if cfg['eta'] == 0: assert np.array_equal(f['scores'][e.NAMES.index('scatter_blend')], f['scores'][primary])
            for i in P['target_tasks']:
                e.guard()
                qi = P['target_query_indices_for_direct_primal']
                s = target[tasks['support_indices'][i]]
                q = target[tasks['query_indices'][i].reshape(-1)[qi]]
                actual = {name: f['scores'][k, i, qi] for k, name in enumerate(e.NAMES)}
                records += compare(actual, independent(s, q, gv, **cfg), f'{key}/{i}')
            banks[key] = f['scores']
            accs.append(acc)
        acc = np.mean(accs, 0)
        item = result['source_configurations'][source]
        assert np.max(abs(acc.mean(-1)*100-np.array(list(item['accuracy_pct'].values())))) < 1e-10
        for k, name in enumerate(e.NAMES):
            if k == primary: continue
            got = independent_interval(acc[primary]-acc[k], tasks['seeds'])
            want = item['comparisons'][name]
            intervals.append(float(abs(got-np.r_[want['delta_pp'], want['ci95_pp']]).max()))
        tests = {name: item['comparisons'][name]['delta_pp'] >= .5 and item['comparisons'][name]['ci95_pp'][0] > 0 for name in e.CFG['gates']['comparators']}
        tests['all_gallery_cells'] = all(result['cells'][source+'_'+g]['comparisons'][name]['ci95_pp'][0] >= -.5 for g in galleries for name in e.CFG['gates']['comparators'])
        assert item['gate'] == {'tests': tests, 'passed': all(tests.values())}
    for g in galleries:
        for k, name in enumerate(e.NAMES[:7]): assert np.array_equal(banks['dtd_'+g][k], banks['eurosat_'+g][k]), name
    assert max(intervals) < 1e-10
    assert result['qualified'] == all(v['gate']['passed'] for v in result['source_configurations'].values())
    e.check_lock(e.HERE/'design_lock.json')
    audit_result = {'status': 'passed', 'complete_banks': len(banks), 'unique_tasks': 500,
        'prediction_entries': sum(x.shape[0]*x.shape[1]*x.shape[2] for x in banks.values()),
        'independent_dense_score_checks': len(records), 'max_score_error': max(x['max_error'] for x in records),
        'independent_intervals': len(intervals), 'max_interval_error_pp': max(intervals),
        'fixed_gate_qualified': result['qualified'], 'seconds': time.time()-start, 'checks': records,
        'scope': P['reference'], 'limitations': P['not_proven']}
    e.dump(e.OUT/'independent_audit.json', audit_result)
    print('AUDIT_PASS', json.dumps({k: v for k, v in audit_result.items() if k != 'checks'}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=['preflight', 'audit'], required=True)
    args = p.parse_args(); torch.set_num_threads(1)
    try:
        preflight() if args.phase == 'preflight' else audit()
    except Exception as exc:
        e.dump(e.OUT/(args.phase+'_failure.json'), {'status': 'failed', 'error': repr(exc), 'phase': args.phase})
        raise
