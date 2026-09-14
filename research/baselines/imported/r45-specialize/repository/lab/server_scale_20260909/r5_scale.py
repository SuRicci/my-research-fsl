"""Frozen R2 scale test. Outputs and fresh features belong to a separate run root."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
E12 = ROOT / '研究5——少样本支持集的检索增强与可靠性加权'
R2 = E12 / 'studies/r2_20260909'
sys.path.insert(0, str(E12))
sys.path.insert(0, str(R2))
from e12 import r3bridge as bridge
from e12.optimized import fit_optimized, prepare_gallery
from methods import representation, evaluate_configs

CELLS = [('cifar_fs', 'cifar100'), ('dtd', 'dtd'),
         ('miniimagenet', 'miniimagenet'), ('dtd', 'miniimagenet'),
         ('cub200', 'cub200'), ('eurosat', 'eurosat'),
         ('cub200', 'miniimagenet'), ('eurosat', 'miniimagenet'),
         ('flowers102', 'flowers102'), ('flowers102', 'miniimagenet')]
SEEDS = [129901, 129902, 129903, 129904, 129905]
BACKBONES = ['clip_vitb16', 'dinov2_vits14']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def bind(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Resume input changed: ' + str(path))
    else:
        dump(path, value)


def identity(ds, i):
    with ds.get_image(i) as im:
        a = np.asarray(im.convert('RGB'))
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def manifest(out, name, split):
    ds = bridge.load_cls_dataset(name, split)
    expected = {'cifar100': {'train': 50000}, 'cifar_fs': {'test': 2000},
                'miniimagenet': {'train': 38400, 'test': 12000},
                'dtd': {'train': 1880, 'test': 1880},
                'cub200': {'train': 5994, 'test': 5794},
                'eurosat': {'train': 21600, 'test': 5400},
                'flowers102': {'train': 1020, 'test': 6149}}
    assert len(ds) == expected[name][split], (name, split, len(ds))
    if name == 'cifar_fs':
        assert ds.meta['split_source'] == 'bertinetto'
    folder = out / 'assets' / (name + '_' + split)
    folder.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        hashes = list(pool.map(lambda i: identity(ds, i), range(len(ds))))
    m = {'name': name, 'split': split, 'rows': len(ds), 'ids': list(range(len(ds))),
         'labels': ds.labels.tolist(), 'classnames': ds.classnames,
         'rgb': hashes, 'dataset_meta': ds.meta}
    bind(folder / 'manifest.json', m)
    print('IDENTITY', name, split, len(ds), 'unique', len(set(hashes)), flush=True)
    return ds, folder, m


def encode(out, name, split, batch):
    ds, folder, m = manifest(out, name, split)
    for backbone in BACKBONES:
        p = folder / (backbone + '.npy')
        receipt = p.with_suffix('.json')
        state = json.loads(receipt.read_text()) if receipt.exists() else {'rows': 0}
        if state.get('complete'):
            assert digest(p) == state['sha256']
            continue
        ext = bridge.get_extractor(backbone, 'cuda:0')
        assert all(not param.requires_grad for param in ext.model.parameters())
        tf = bridge.center_transform(backbone)
        a = np.load(p, mmap_mode='r+') if p.exists() else None
        started = time.time()
        for st in range(state['rows'], len(ds), batch):
            indices = list(range(st, min(st + batch, len(ds))))
            def transformed(i):
                with ds.get_image(i) as im:
                    return tf(im)
            with ThreadPoolExecutor(4) as pool:
                x = torch.stack(list(pool.map(transformed, indices)))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                f = bridge.encode(backbone, x, 'cuda:0')
            # Preserve the R2 fp16 cache roundtrip and normalisation convention.
            f = F.normalize(f.half().float(), dim=-1).numpy()
            assert np.isfinite(f).all()
            if a is None:
                a = np.lib.format.open_memmap(p, mode='w+', dtype='float32', shape=(len(ds), f.shape[1]))
            a[st:st + len(f)] = f
            a.flush()
            dump(receipt, {'rows': st + len(f), 'complete': False,
                           'manifest_sha256': digest(folder / 'manifest.json')})
            if st % (batch * 10) == 0:
                print('ENCODE', name, split, backbone, st + len(f), '/', len(ds), flush=True)
        dump(receipt, {'rows': len(ds), 'complete': True, 'sha256': digest(p),
                       'manifest_sha256': digest(folder / 'manifest.json'),
                       'seconds_this_invocation': time.time() - started,
                       'encoder_image_forwards_total': len(ds), 'gradient_updates': 0,
                       'preprocess': 'R2 r3bridge center_transform: short edge 224, bicubic, center crop 224; autocast fp16'})
        del a
        bridge._extractor.pop(backbone)
        del ext
        torch.cuda.empty_cache()


def load(out, name, split):
    folder = out / 'assets' / (name + '_' + split)
    m = json.loads((folder / 'manifest.json').read_text())
    arrays = []
    for b in BACKBONES:
        p = folder / (b + '.npy')
        meta = json.loads(p.with_suffix('.json').read_text())
        assert meta['complete'] and meta['manifest_sha256'] == digest(folder / 'manifest.json')
        assert digest(p) == meta['sha256']
        arrays.append(torch.from_numpy(np.load(p)))
    return m, arrays


def configs(shot):
    selected = json.loads((R2 / 'selection.json').read_text())[str(shot)]
    cc = [{**selected[k], 'name': 'r2_' + k} for k in ['overall', 'control', 'candidate', 'single_dino']]
    cc += [{'name': 'fused_raw', 'family': 'raw', 'w': .5},
           {'name': 'fused_uniform_matched', 'family': 'uniform', 'w': .5, 'r': 64, 'mix': .5},
           {'name': 'dino_raw', 'family': 'raw', 'w': 0.}]
    old = json.loads((E12 / 'studies/r1_20260909/e12/selection.json').read_text())
    for b, w in [('dinov2_vits14', 0.), ('clip_vitb16', 1.)]:
        c = old[b + '_' + str(shot)]['uniform_mix']
        cc.append({'name': 'r1_' + b, 'family': 'uniform', 'w': w, 'r': c['r'], 'mix': c['mix']})
    return cc


def evaluate(out, count, diagnostic=False):
    outdir = out / ('diagnostic' if diagnostic else 'results')
    outdir.mkdir(exist_ok=True)
    cells = CELLS[:1] if diagnostic else CELLS
    for cell in cells:
        qm, qfs = load(out, cell[0], 'test')
        gm, gfs = load(out, cell[1], 'train')
        seen = set(); qi = []
        for i, h in enumerate(qm['rgb']):
            if h not in seen:
                qi.append(i); seen.add(h)
        banned = set(seen); gi = []
        for i, h in enumerate(gm['rgb']):
            if h not in seen:
                gi.append(i); seen.add(h)
        assert not banned.intersection(gm['rgb'][i] for i in gi)
        labels = np.asarray(qm['labels'])
        pools = {int(c): np.array([i for i in qi if labels[i] == c]) for c in sorted(set(labels))}
        # Keep old conditions on the R2 held-out class half; never select new parameters.
        classes = np.array(sorted(c for c, pool in pools.items() if len(pool) >= 20))
        if cell[0] in ['cifar_fs', 'dtd', 'miniimagenet']:
            order = np.random.RandomState(120909).permutation(classes)
            classes = order[len(order) // 2:]
        assert len(classes) >= 5
        dump(outdir / ('_'.join(cell) + '_identity.json'),
             {'gallery_ids': gi, 'query_pool_ids': qi, 'gallery_rows': len(gi),
              'classes': classes.tolist(), 'rgb_overlap_after_exclusion': 0,
              'query_manifest': digest(out / 'assets' / (cell[0] + '_test') / 'manifest.json'),
              'gallery_manifest': digest(out / 'assets' / (cell[1] + '_train') / 'manifest.json')})
        for shot in [1, 5]:
            cc = configs(shot)
            for seed in SEEDS[:1] if diagnostic else SEEDS:
                stem = '_'.join(cell) + f'_k{shot}_s{seed}'
                dest = outdir / (stem + '.npz')
                if dest.with_suffix('.json').exists():
                    assert digest(dest) == json.loads(dest.with_suffix('.json').read_text())['sha256']
                    continue
                rng = np.random.RandomState(seed + shot)
                cs = np.stack([rng.choice(classes, 5, replace=False) for _ in range(count)])
                ix = np.stack([np.stack([rng.choice(pools[int(c)], shot + 15, replace=False) for c in row]) for row in cs])
                si, xi = ix[:, :, :shot], ix[:, :, shot:]
                predictions = {}; scores = {}; timings = {}; start = time.time()
                for w in sorted(set(c['w'] for c in cc)):
                    qf = representation(*qfs, w)
                    G = representation(gfs[0][gi], gfs[1][gi], w).cuda()
                    selected = [c for c in cc if c['w'] == w]
                    pp = {c['name']: [] for c in selected}; ss = {c['name']: [] for c in selected}
                    for st in range(0, count, 24):
                        S = qf[si[st:st + 24]].cuda()
                        X = qf[xi[st:st + 24].reshape(-1, 75)].cuda()
                        p, s, t = evaluate_configs(S, X, G, selected, return_scores=True)
                        for n in p:
                            pp[n].append(p[n]); ss[n].append(s[n])
                        for n, seconds in t.items(): timings[n] = timings.get(n, 0) + seconds
                    predictions.update({n: torch.cat(v).numpy().astype('int8') for n, v in pp.items()})
                    scores.update({n: torch.cat(v).numpy() for n, v in ss.items()})
                    del G, S, X
                # Check the accelerated evaluator against the public frozen API on real episodes.
                pg = prepare_gallery(gfs[0][gi].cuda(), gfs[1][gi].cuda()) if shot == 1 else None
                max_error = 0.
                for e in range(min(2, count)):
                    s = si[e].ravel(); x = xi[e].ravel()
                    fitted = fit_optimized(qfs[0][s].cuda(), qfs[1][s].cuda(),
                                           torch.arange(5, device='cuda').repeat_interleave(shot), prepared_gallery=pg)
                    actual = fitted.scores(qfs[0][x].cuda(), qfs[1][x].cuda()).cpu().numpy()
                    max_error = max(max_error, float(np.max(np.abs(actual - scores['r2_overall'][e]))))
                    assert np.array_equal(actual.argmax(-1), predictions['r2_overall'][e])
                assert max_error < 2e-4, max_error
                y = np.repeat(np.arange(5), 15)
                names = list(predictions)
                tmp = dest.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, names=names, predictions=np.stack([predictions[n] for n in names]),
                                    scores=np.stack([scores[n] for n in names]), yq=y,
                                    support_indices=si, query_indices=xi, class_ids=cs, seed=seed)
                os.replace(tmp, dest)
                dump(dest.with_suffix('.json'), {'complete': True, 'sha256': digest(dest),
                      'cell': cell, 'shot': shot, 'seed': seed, 'episodes': count,
                      'gallery_n': len(gi), 'seconds': time.time() - start, 'timings': timings,
                      'api_max_score_error': max_error,
                      'accuracy': {n: float((p == y).mean()) for n, p in predictions.items()}})
                print('EVALUATED', stem, count, 'episodes', flush=True)
                del pg
    if not diagnostic:
        summarize(out)


def summarize(out):
    files = sorted((out / 'results').glob('*.npz'))
    grouped = {}
    for p in files:
        meta = json.loads(p.with_suffix('.json').read_text())
        z = np.load(p, allow_pickle=False)
        acc = (z['predictions'] == z['yq']).mean(-1)
        key = '_'.join(meta['cell']) + '_k' + str(meta['shot'])
        grouped.setdefault(key, []).append((list(z['names']), acc, meta))
    rows = []
    for key, packs in grouped.items():
        names = packs[0][0]
        assert all(p[0] == names for p in packs)
        a = np.concatenate([p[1] for p in packs], axis=1)
        comparisons = {}
        for n in ['r1_dinov2_vits14', 'r2_control', 'fused_raw', 'fused_uniform_matched']:
            delta = a[names.index('r2_overall')] - a[names.index(n)]
            rng = np.random.default_rng(129999)
            boot = np.array([delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(2000)])
            comparisons[n] = {'gain_pp': float(delta.mean() * 100),
                              'paired_episode_ci95_pp': (np.quantile(boot, [.025, .975]) * 100).tolist(),
                              'seed_gain_pp': [float((p[1][names.index('r2_overall')] - p[1][names.index(n)]).mean() * 100) for p in packs]}
        rows.append({'cell': key, 'episodes': a.shape[1], 'seeds': len(packs),
                     'accuracy': dict(zip(names, a.mean(1).tolist())), 'comparisons': comparisons})
    dump(out / 'summary.json', {'complete': len(files) == 100, 'completed_seed_cells': len(files),
         'expected_seed_cells': 100, 'rows': rows,
         'scope': 'Frozen-parameter full-gallery scale retest. CIs are conditional on fixed source-image pools; no new independent-data claim from episode resampling.'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--phase', choices=['prepare', 'evaluate', 'diagnostic', 'summarize'], required=True)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--episodes-per-seed', type=int, default=600)
    args = ap.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    sources = [Path(__file__), R2 / 'methods.py', R2 / 'selection.json', E12 / 'e12/optimized.py',
               E12 / 'e12/r3bridge.py', ROOT / '研究3——少样本学习测试时计算扩展/src/features/clip.py',
               ROOT / '研究3——少样本学习测试时计算扩展/src/features/dino.py',
               ROOT / '研究3——少样本学习测试时计算扩展/src/data/cls_datasets.py']
    protocol = {'campaign': 'R5-E12-full-gallery-20260909', 'cells': CELLS, 'shots': [1, 5],
                'seeds': SEEDS, 'episodes_per_seed': args.episodes_per_seed,
                'configs': {str(k): configs(k) for k in [1, 5]},
                'hashes': {str(p.relative_to(ROOT)): digest(p) for p in sources},
                'galleries': 'complete train splits; RGB duplicates and all query-pool identities removed',
                'selection': 'R2 frozen; no tuning',
                'weights_sha256': {str(p): digest(p) for p in [bridge.DINOV2_VITS14_PATH, Path.home() / '.cache/torch/hub/checkpoints/ViT-B-16.pt']},
                'versions': {'torch': torch.__version__, 'numpy': np.__version__}}
    # JSON roundtrip makes tuple/list types identical on resume.
    bind(out / 'protocol.json', json.loads(json.dumps(protocol)))
    dump(out / 'runtime.json', {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                               'pid': os.getpid(), 'phase': args.phase, 'started': time.time()})
    if args.phase == 'prepare':
        needed = sorted(set((q, 'test') for q, g in CELLS) | set((g, 'train') for q, g in CELLS))
        for name, split in needed:
            encode(out, name, split, args.batch_size)
        dump(out / 'preparation_complete.json', {'complete': True, 'datasets': needed})
    elif args.phase == 'summarize': summarize(out)
    else: evaluate(out, 4 if args.phase == 'diagnostic' else args.episodes_per_seed, args.phase == 'diagnostic')


if __name__ == '__main__':
    main()
