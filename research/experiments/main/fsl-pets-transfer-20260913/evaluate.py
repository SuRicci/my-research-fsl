"""Fixed Pets comparison; baseline and iLPC phases have separate durable outputs."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, hashlib, json, os, sys, time, warnings
import multiprocessing as mp
from datetime import datetime, timezone
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
import reference_eval as ref
import geometry_reference as geo
from baseline_methods import representation
import ilpcz
HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'protocol.json').read_text())
ROOT = Path('/Users/decoqwq/DeepScientist/quests/012')
OUT = HERE / 'outputs'
CACHE = OUT / 'cache'
QDATA = None
GDATA = {}

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2)); os.replace(temp, path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def guard():
    import shutil
    assert shutil.disk_usage(HERE).free >= 10 * 2**30
    assert datetime.now(timezone.utc) < datetime.fromisoformat(CFG['resources']['deadline_utc'])

def load_data():
    assert (OUT / 'assets_complete.json').exists(), 'Feature acquisition not complete'
    guard()
    for name, row in json.loads((HERE / 'source_copies.json').read_text()).items():
        assert sha(HERE / name) == row['sha256']
    root = HERE / 'assets'; ident = json.loads((root / 'identities.json').read_text())
    manifest = json.loads((root / 'feature_manifest.json').read_text())
    assert sha(root / 'identities.json') == manifest['identity_sha256']
    assert sha(HERE / 'protocol.json') == manifest['protocol_sha256']
    data = {}
    for side in ['query', 'gallery']:
        blocks = []
        for b in ['clip_vitb16', 'dinov2_vits14']:
            row = manifest['files'][b + '_' + side]; p = Path(row['path'])
            assert sha(p) == row['sha256']
            pack = torch.load(p, weights_only=True)
            assert pack['ids'].tolist() == list(range(len(ident[side])))
            blocks.append(pack['features'].float())
        data[side] = representation(*blocks, .5)
    old = ROOT / 'baselines/local/r2-canonical/assets'
    old_manifest = json.loads((old / 'manifest.json').read_text())
    old_ident = json.loads((old / 'dtd_identities.json').read_text()); blocks = []
    assert not set(row['rgb'] for row in ident['query']) & set(old_ident['gallery_rgb'])
    for b in ['clip_vitb16', 'dinov2_vits14']:
        path = old / ('dtd_' + b + '_gallery.pt')
        assert sha(path) == old_manifest['datasets']['dtd']['backbones'][b + '_gallery']['sha256']
        pack = torch.load(path, weights_only=True)
        assert pack['ids'].tolist() == old_ident['gallery_ids']
        blocks.append(pack['features'].float())
    data['dtd'] = representation(*blocks, .5)
    assert len(data['gallery']) == len(data['dtd']) == CFG['gallery_size']
    return data, ident, manifest

def task_arrays(ident, shot):
    ref.CFG = {'eval': {'seeds': CFG['seeds'], 'episodes_per_seed': CFG['episodes_per_seed']}}
    return ref.tasks({'query_labels': [row['label'] for row in ident['query']]}, 'pets', shot, 'eval')

def save_tasks(ident):
    for shot in CFG['shots']:
        si, qi, cs, seeds = task_arrays(ident, shot)
        path = OUT / ('tasks_k%d.npz' % shot)
        if path.exists():
            with np.load(path) as saved:
                for name, arr in [('support_indices', si), ('query_indices', qi), ('class_ids', cs), ('seeds', seeds)]:
                    assert np.array_equal(saved[name], arr)
        else:
            np.savez_compressed(path, support_indices=si, query_indices=qi, class_ids=cs, seeds=seeds)

def logistic(S, Q, C):
    ys = np.repeat(np.arange(5), len(S) // 5)
    model = LogisticRegression(C=C, multi_class='multinomial', solver='lbfgs', max_iter=2000, tol=1e-8, random_state=26091275)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); model.fit(np.asarray(S, dtype=np.float64), ys)
    assert not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model.decision_function(np.asarray(Q, dtype=np.float64))

def baseline(data, ident):
    out = OUT / 'baseline'; out.mkdir(exist_ok=True); summary = {}
    for shot in CFG['shots']:
        si, qi, cs, seeds = task_arrays(ident, shot); support = data['query'][si]
        query = data['query'][qi.reshape(len(qi), 75)]; pure = {}
        for C in [1, 10]:
            pure['support_logistic_C%d' % C] = np.stack([logistic(s.flatten(0, 1).numpy(), q.numpy(), C) for s, q in zip(support, query)])
        for gallery in CFG['gallery_conditions']:
            guard(); G = data['gallery' if gallery == 'pets' else 'dtd']; scores = {'r2': [], 'CS_l2': []}; t = time.time()
            for start in range(0, len(si), 16):
                S = support[start:start+16]; Q = query[start:start+16]
                scores['r2'].append(ref.r2_scores(S, Q, G).numpy())
                scores['CS_l2'].append(geo.head(geo.prepare(S, Q, G, 'support'), .1).numpy())
            scores = {name: np.concatenate(values) for name, values in scores.items()}; scores.update(pure)
            names = list(scores); values = np.stack([scores[name] for name in names]); yq = np.repeat(np.arange(5), 15)
            assert np.isfinite(values).all()
            acc = (values.argmax(-1) == yq).mean(-1)
            cell = 'pets_' + gallery + '_k%d' % shot
            np.savez_compressed(out / (cell + '.npz'), scores=values, accuracy=acc, names=names, predictions=values.argmax(-1),
                support_indices=si, query_indices=qi, class_ids=cs, seeds=seeds, yq=yq)
            summary[cell] = {name: 100 * float(acc[i].mean()) for i, name in enumerate(names)}
            dump(out / 'summary.json', summary)
            print('BASELINE_CELL', cell, summary[cell], 'seconds', round(time.time()-t, 2), flush=True)
    dump(out / 'complete.json', {'status': 'success', 'cells': len(summary)})

def initialize_worker():
    torch.set_num_threads(1)

def ilpc_worker(job):
    global QDATA
    shot, gallery, index, signature = job
    output = OUT / 'ilpc_tasks' / ('pets_' + gallery + '_k%d' % shot) / ('%04d.npz' % index)
    if output.exists():
        with np.load(output) as p: assert str(p['signature']) == signature
        return str(output)
    guard()
    if QDATA is None: QDATA = np.load(CACHE / 'query.npy', mmap_mode='r')
    if gallery not in GDATA:
        with np.load(CACHE / (gallery + '.npz')) as a:
            GDATA[gallery] = (a['features'], {k: a[k] for k in ['unit_features', 'indices', 'similarities']})
    G, cache = GDATA[gallery]
    with np.load(OUT / ('tasks_k%d.npz' % shot)) as tasks:
        si = tasks['support_indices'][index]; qi = tasks['query_indices'][index]
    S = QDATA[si.reshape(-1)]; Q = QDATA[qi.reshape(-1)]; ys = np.repeat(np.arange(5), shot); t = time.time()
    spec = CFG['ilpcz']
    model, selected, labels, stats = ilpcz.fit(S, ys, G, cache, k=spec['k'], alpha=spec['alpha'], best=spec['selection_per_class'])
    values = model.decision_function(Q); stats['wall_seconds'] = time.time()-t
    assert np.isfinite(values).all() and stats['convergence_warnings'] == 0
    output.parent.mkdir(exist_ok=True, parents=True); temp = output.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, signature=signature, scores=values, predictions=values.argmax(-1),
            accuracy=float((values.argmax(-1) == np.repeat(np.arange(5), 15)).mean()),
            support_indices=si, query_indices=qi, selected_indices=selected, pseudo_labels=labels, stats_json=json.dumps(stats))
    os.replace(temp, output)
    return str(output)

def ilpc(data, signature):
    assert (OUT / 'baseline/confirmed_comparator.json').exists(), 'Pets baseline must be audited and confirmed before candidate phase'
    CACHE.mkdir(exist_ok=True); np.save(CACHE / 'query.npy', data['query'].numpy().astype(np.float64))
    for gallery in CFG['gallery_conditions']:
        G = data['gallery' if gallery == 'pets' else 'dtd'].numpy().astype(np.float64)
        path = CACHE / (gallery + '.npz')
        if path.exists():
            with np.load(path) as a: assert np.array_equal(a['features'], G)
        else: np.savez(path, features=G, **ilpcz.gallery_cache(G))
    n = len(CFG['seeds']) * CFG['episodes_per_seed']
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=initialize_worker) as pool:
        for shot in CFG['shots']:
            for gallery in CFG['gallery_conditions']:
                jobs = [(shot, gallery, index, signature) for index in range(n)]; started = time.time()
                for count, path in enumerate(pool.map(ilpc_worker, jobs, chunksize=1), 1):
                    if count % 50 == 0: print('ILPC_PROGRESS', shot, gallery, count, n, round(time.time()-started, 2), flush=True)
    dump(OUT / 'ilpc_complete.json', {'status': 'success', 'tasks': n * len(CFG['shots']) * len(CFG['gallery_conditions']), 'signature': signature})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['baseline', 'ilpc'], required=True); args = parser.parse_args()
    start = time.time(); data, ident, features = load_data(); save_tasks(ident)
    payload = {'config': CFG, 'feature_manifest': features, 'source_copies': json.loads((HERE / 'source_copies.json').read_text()), 'adapter_sha256': sha(Path(__file__))}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    dump(OUT / ('run_manifest_' + args.phase + '.json'), {'signature': signature, **payload, 'command': [sys.executable, *sys.argv], 'started_at': datetime.now(timezone.utc).isoformat()})
    if args.phase == 'baseline': baseline(data, ident)
    else: ilpc(data, signature)
    print('PHASE_COMPLETE', args.phase, round(time.time()-start, 2), flush=True)
