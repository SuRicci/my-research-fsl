"""Prospectively fixed Caltech transfer; inference functions receive features only."""
from pathlib import Path
import datetime, hashlib, json, os, shutil, sys, time, warnings
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'experiments/main/query-consistency-20260913'))
import consistency_model as z
metric = z.m.old.metric
CFG = json.loads((HERE/'protocol.json').read_text())
NAMES = CFG['methods']
OUT = HERE/'outputs'
OUT.mkdir(exist_ok=True)
KEYS = ['support_indices', 'query_indices', 'class_ids', 'seeds']
Y = np.repeat(np.arange(5), 15)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*2**20), b''): h.update(block)
    return h.hexdigest()

def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)

def guard():
    assert shutil.disk_usage(HERE).free/2**30 >= CFG['resources']['free_floor_gib']
    assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(CFG['resources']['deadline'])
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) < CFG['resources']['output_limit_mib']*2**20

def check_lock(path):
    lock = json.loads(path.read_text())
    for p, h in lock.get('sha256', lock).items(): assert sha(p) == h, p

def sources():
    paths = {HERE/'caltech_eval.py', HERE/'caltech_audit.py', HERE/'audit_plan.json'}
    for module in list(sys.modules.values()):
        p = getattr(module, '__file__', None)
        if p and Path(p).suffix == '.py' and ROOT in Path(p).resolve().parents: paths.add(Path(p).resolve())
    return {str(p): sha(p) for p in sorted(paths)}

@torch.no_grad()
def stack_scores(sv, qv, gv, gamma, eta):
    parts = [z.head(*p, [eta], [0]) for p in z.packs(sv, qv, gv.double().mean(-2), gamma)]
    return {'scatter_r2': parts[0]['incumbent'],
            'scatter_blend': .5*(parts[0]['incumbent']+parts[1]['incumbent']),
            'query_consistency': .5*(parts[0]['consistency'][0]+parts[1]['consistency'][0])}

@torch.no_grad()
def control_scores(sv, qv, gv):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = metric.controls(sv, qv, gv)
    bad = [str(w.message) for w in caught if 'ConvergenceWarning' in w.category.__name__]
    assert not bad, bad
    return {k: v for k, v in result.items() if k in NAMES}

def load_data():
    check_lock(HERE/'design_lock.json')
    mf = json.loads((HERE/'assets/manifest.json').read_text())
    assert mf['status'] == 'completed'
    assert mf['design_sha256'] == sha(HERE/'design_lock.json')
    check_lock(HERE/'encoding_code_lock.json')
    blocks = []
    for b in CFG['features']['backbones']:
        entry = mf['files'][b]
        assert sha(entry['path']) == entry['sha256']
        f = torch.load(entry['path'], weights_only=True)
        assert f['ids'].tolist() == list(range(mf['image_count']))
        assert list(f['features'].shape) == entry['shape'] and f['features'].dtype == torch.float16
        assert torch.isfinite(f['features']).all()
        assert float((f['features'].float().norm(dim=-1)-1).abs().max()) < .002
        blocks.append(f['features'].float())
    target = metric.ref.representation(*blocks, .5)
    data = z.m.old.load()
    rows = json.loads((HERE/'image_order.json').read_text())['rows']
    ident = json.loads(Path(CFG['identity_path']).read_text())
    galleries = json.loads((HERE/'galleries.json').read_text())
    tasks = dict(np.load(HERE/'tasks.npz'))
    assert len(rows) == len(target) == 8677 and np.array_equal(tasks['yq'], Y)
    assert tasks['support_indices'].shape == (500, 5, 1) and tasks['query_indices'].shape == (500, 5, 15)
    assert np.array_equal(np.unique(tasks['seeds']), CFG['eval_seeds'])
    assert all((tasks['seeds'] == s).sum() == 100 for s in CFG['eval_seeds'])
    classes = sorted({r['class_name'] for r in rows})
    labels = np.array([classes.index(r['class_name']) for r in rows])
    query_rgb = {r['rgb'] for r in ident['query']}
    for i, (s, q) in enumerate(zip(tasks['support_indices'], tasks['query_indices'])):
        assert len(set(s.ravel()) | set(q.ravel())) == 80
        assert np.array_equal(labels[s], tasks['class_ids'][i, :, None])
        assert np.array_equal(labels[q], np.repeat(tasks['class_ids'][i, :, None], 15, axis=1))
        assert all(rows[j]['rgb'] in query_rgb for j in np.r_[s.ravel(), q.ravel()])
    gs = {}
    for g, entry in galleries.items():
        gi = entry['indices']
        rgb = [rows[j]['rgb'] for j in gi] if g == 'caltech101' else [data[g]['ident']['gallery_rgb'][j] for j in gi]
        assert len(gi) == len(set(gi)) == 1024 and rgb == entry['rgb']
        assert not query_rgb & set(rgb) and not entry['selection_uses_labels']
        gs[g] = target[gi] if g == 'caltech101' else data[g]['gallery'][gi]
    provenance = {'status': 'passed', 'target_manifest_sha256': sha(HERE/'assets/manifest.json'),
        'design_sha256': sha(HERE/'design_lock.json'), 'tasks': 500, 'images': len(rows),
        'gallery_sizes': {g: len(v) for g, v in gs.items()}, 'exact_rgb_overlap': 0,
        'feature_shape': list(target.shape), 'labels_excluded_from_inference': True}
    dump(OUT/'feature_validation.json', provenance)
    return target, gs, tasks

def interval(delta, seeds):
    rng = np.random.RandomState(CFG['statistics']['seed'])
    samples = np.zeros(CFG['statistics']['replicates'])
    for seed in np.unique(seeds):
        d = delta[seeds == seed]
        samples += d[rng.randint(len(d), size=(len(samples), len(d)))].mean(1)/len(np.unique(seeds))
    return {'delta_pp': float(delta.mean()*100), 'ci95_pp': (np.quantile(samples, [.025, .975])*100).tolist()}

def summarize():
    cells, configurations = {}, {}
    j = NAMES.index(CFG['primary_method'])
    for source in CFG['source_configurations']:
        banks = []
        for g in CFG['gallery_conditions']:
            f = np.load(OUT/(source+'_'+g+'.npz'))
            a = (f['predictions'] == Y).mean(-1)
            banks.append(a)
            contrasts = {name: interval(a[j]-a[k], f['seeds']) for k, name in enumerate(NAMES) if k != j}
            cells[source+'_'+g] = {'accuracy_pct': dict(zip(NAMES, (a.mean(-1)*100).tolist())),
                'comparisons': contrasts,
                'repairs_spoils': {name: {'repairs': int(((f['predictions'][j] == Y) & (f['predictions'][k] != Y)).sum()),
                    'spoils': int(((f['predictions'][j] != Y) & (f['predictions'][k] == Y)).sum())} for k, name in enumerate(NAMES) if k != j}}
        a = np.mean(banks, axis=0)
        contrasts = {name: interval(a[j]-a[k], f['seeds']) for k, name in enumerate(NAMES) if k != j}
        tests = {name: contrasts[name]['delta_pp'] >= .5 and contrasts[name]['ci95_pp'][0] > 0 for name in CFG['gates']['comparators']}
        tests['all_gallery_cells'] = all(cells[source+'_'+g]['comparisons'][name]['ci95_pp'][0] >= -.5
            for g in CFG['gallery_conditions'] for name in CFG['gates']['comparators'])
        configurations[source] = {'parameters': CFG['source_configurations'][source],
            'accuracy_pct': dict(zip(NAMES, (a.mean(-1)*100).tolist())), 'comparisons': contrasts,
            'gate': {'passed': all(tests.values()), 'tests': tests}}
    return {'cells': cells, 'source_configurations': configurations,
        'qualified': all(c['gate']['passed'] for c in configurations.values()),
        'scope': CFG['tier'], 'interval_boundary': CFG['statistics']['interpretation'],
        'source_configurations_are_independent_replications': False, 'independent_audit': 'pending'}

def main():
    start = time.time()
    torch.set_num_threads(CFG['resources']['threads'])
    guard()
    check_lock(HERE/'evaluation_code_lock.json')
    assert sources() == json.loads((HERE/'evaluation_code_lock.json').read_text())
    assert json.loads((OUT/'preflight.json').read_text())['status'] == 'passed'
    target, galleries, tasks = load_data()
    lock = sha(HERE/'evaluation_code_lock.json')
    dump(OUT/'run_manifest.json', {'argv': [sys.executable, *sys.argv], 'code_lock_sha256': lock,
        'design_lock_sha256': sha(HERE/'design_lock.json'), 'torch': torch.__version__, 'numpy': np.__version__,
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'protocol': CFG})
    for g, gv in galleries.items():
        paths = {source: OUT/(source+'_'+g+'.npz') for source in CFG['source_configurations']}
        if all(p.exists() for p in paths.values()):
            for p in paths.values():
                f = np.load(p)
                assert f['code_lock_sha256'].item() == lock and all(np.array_equal(f[k], tasks[k]) for k in KEYS)
            print('REUSE_GALLERY', g, flush=True)
            continue
        banks = {source: [] for source in paths}
        for i in range(len(tasks['seeds'])):
            sv = target[tasks['support_indices'][i]]
            qv = target[tasks['query_indices'][i].reshape(-1)]
            controls = control_scores(sv, qv, gv)
            for source, cfg in CFG['source_configurations'].items():
                sc = {**controls, **stack_scores(sv, qv, gv, **cfg)}
                values = np.stack([sc[n].numpy() for n in NAMES])
                assert np.isfinite(values).all()
                banks[source].append(values)
            if i % 50 == 0:
                guard()
                p = {'gallery': g, 'tasks_done': i+1, 'elapsed_seconds': time.time()-start}
                dump(OUT/'progress.json', p)
                print('PROGRESS', json.dumps(p), flush=True)
        for source, values in banks.items():
            sc = np.stack(values, axis=1)
            pred = sc.argmax(-1).astype(np.uint8)
            np.savez_compressed(paths[source], scores=sc, predictions=pred, accuracy=(pred == Y).mean(-1),
                names=NAMES, yq=Y, gallery_indices=json.loads((HERE/'galleries.json').read_text())[g]['indices'],
                code_lock_sha256=lock, **{k: tasks[k] for k in KEYS})
        print('GALLERY_COMPLETE', g, flush=True)
    check_lock(HERE/'evaluation_code_lock.json')
    result = summarize()
    dump(OUT/'analysis.json', result)
    dump(OUT/'complete.json', {'status': 'completed', 'audit_status': 'pending', 'task_conditions': 3000,
        'unique_tasks': 500, 'methods': len(NAMES), 'seconds': time.time()-start})
    print('COMPLETE', json.dumps({s: v['accuracy_pct'] for s, v in result['source_configurations'].items()}), flush=True)

if __name__ == '__main__': main()
