"""Validate frozen Pets assets and baseline outputs without selecting parameters."""
from pathlib import Path
import argparse, hashlib, json, platform, sys
import numpy as np
import torch
import sklearn
import evaluate as ev

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
CFG = ev.CFG

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 2**20), b''):
            h.update(block)
    return h.hexdigest()

def asset_audit():
    data, ident, manifest = ev.load_data()
    assert len(manifest['files']) == 4
    assert len(set(r['rgb'] for r in ident['query'])) == len(ident['query'])
    assert len(set(r['rgb'] for r in ident['gallery'])) == CFG['gallery_size']
    assert not set(r['rgb'] for r in ident['query']) & set(r['rgb'] for r in ident['gallery'])
    assert ident['known_prior_rgb_overlap'] == 0
    for name, value in data.items():
        assert value.shape[1] == 896 and torch.isfinite(value).all()
        assert float((value.norm(dim=1) - 1).abs().max()) < 1e-5
    for name, expected in CFG['sources_sha256'].items():
        assert sha(name) == expected, name
    weights = [Path('/Users/decoqwq/.cache/clip/ViT-B-16.pt'), Path('/Users/decoqwq/.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m/snapshots/4610ca143709d58a633b6397a74412c2c3842454/model.safetensors')]
    report = {'status': 'passed', 'feature_manifest_sha256': sha(HERE / 'assets/feature_manifest.json'),
              'protocol_sha256': sha(HERE / 'protocol.json'), 'audit_sha256': sha(__file__),
              'query_unique': len(ident['query']), 'gallery_selected': len(ident['gallery']),
              'class_counts': ident['class_counts'], 'exact_rgb_overlap': 0,
              'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                              'sklearn': sklearn.__version__, 'platform': platform.platform(),
                              'mps_available': torch.backends.mps.is_available()},
              'model_weights_sha256': {str(p): sha(p) for p in weights},
              'limitations': ['Exact RGB identity check does not exclude near duplicates.',
                             'New evaluation domain is not proven absent from encoder pretraining.']}
    ev.dump(OUT / 'asset_validation.json', report)
    print('ASSET_AUDIT_PASSED', len(ident['query']), len(ident['gallery']), flush=True)
    return data, ident

def baseline_audit():
    assert json.loads((OUT / 'asset_validation.json').read_text())['status'] == 'passed'
    data, ident, _ = ev.load_data()
    assert json.loads((OUT / 'baseline/complete.json').read_text())['cells'] == 4
    manifest = json.loads((OUT / 'run_manifest_baseline.json').read_text())
    assert manifest['adapter_sha256'] == sha(HERE / 'evaluate.py')
    assert manifest['config'] == CFG
    labels = np.array([r['label'] for r in ident['query']])
    records = {}; hashes = {}; stored = {}; replay = {}
    expected_names = ['r2', 'CS_l2', 'support_logistic_C1', 'support_logistic_C10']
    expected_n = len(CFG['seeds']) * CFG['episodes_per_seed']
    for shot in CFG['shots']:
        with np.load(OUT / ('tasks_k%d.npz' % shot)) as t:
            si, qi, cs, seeds = [t[k] for k in ['support_indices', 'query_indices', 'class_ids', 'seeds']]
        assert si.shape == (expected_n, 5, shot) and qi.shape == (expected_n, 5, 15)
        assert np.array_equal(seeds, np.repeat(CFG['seeds'], CFG['episodes_per_seed']))
        assert (labels[si] == cs[:, :, None]).all() and (labels[qi] == cs[:, :, None]).all()
        for s, q, c in zip(si, qi, cs):
            assert len(set(c)) == 5
            assert len(set(np.concatenate([s.ravel(), q.ravel()]))) == 5 * (shot + 15)
        for gallery in CFG['gallery_conditions']:
            cell = 'pets_' + gallery + '_k%d' % shot
            path = OUT / 'baseline' / (cell + '.npz'); hashes[cell] = sha(path)
            with np.load(path) as a:
                assert a['names'].tolist() == expected_names
                for k, v in [('support_indices', si), ('query_indices', qi), ('class_ids', cs), ('seeds', seeds)]:
                    assert np.array_equal(a[k], v)
                scores = a['scores']; pred = scores.argmax(-1)
                assert scores.shape == (4, expected_n, 75, 5) and np.isfinite(scores).all()
                assert np.array_equal(pred, a['predictions'])
                assert np.array_equal(a['yq'], np.repeat(np.arange(5), 15))
                acc = (pred == np.repeat(np.arange(5), 15)).mean(-1)
                assert np.array_equal(acc, a['accuracy'])
            stored[cell] = scores
            records[cell] = {n: float(acc[i].mean() * 100) for i, n in enumerate(expected_names)}
            # Spot replay all three representative positions; accuracy is checked for every task above.
            idx = [0, expected_n // 2, expected_n - 1]
            S = data['query'][si[idx]]; Q = data['query'][qi[idx].reshape(3, 75)]
            G = data['gallery' if gallery == 'pets' else 'dtd']
            computed = ev.ref.r2_scores(S, Q, G).numpy()
            err = float(np.max(abs(computed - scores[0, idx])))
            assert err < 2e-5
            replay[cell] = err
        a, b = [stored['pets_' + g + '_k%d' % shot] for g in CFG['gallery_conditions']]
        assert np.array_equal(a[2:], b[2:])
        if shot == 5:
            assert np.array_equal(a, b), 'All baseline five-shot methods are support-only'
    assert len(list((OUT / 'baseline').glob('pets_*_k*.npz'))) == 4
    summary = json.loads((OUT / 'baseline/summary.json').read_text())
    assert records == summary
    variants = {}
    for name in expected_names:
        metrics = {'pets_' + ('matched' if g == 'pets' else 'mismatched') + '_%dshot_accuracy' % k: records['pets_' + g + '_k%d' % k][name]
                   for k in CFG['shots'] for g in CFG['gallery_conditions']}
        metrics['pets_macro_1shot_accuracy'] = float(np.mean([metrics['pets_matched_1shot_accuracy'], metrics['pets_mismatched_1shot_accuracy']]))
        assert set(metrics) == set(CFG['new_metric_ids'])
        variants[name] = {'metrics_summary': metrics, 'role': 'frozen_local_comparator' if name == 'r2' else 'same_permission_control'}
    report = {'status': 'passed', 'cells': 4, 'tasks_per_cell': expected_n, 'variants': variants,
              'output_sha256': hashes, 'protocol_sha256': sha(HERE / 'protocol.json'),
              'run_manifest_sha256': sha(OUT / 'run_manifest_baseline.json'), 'audit_sha256': sha(__file__),
              'r2_spot_replay_max_abs_error': replay, 'full_task_and_prediction_check': True,
              'known_deviations': ['Local frozen R2 transferred to Pets; not a SWAT paper reproduction.',
                                   'CS_l2 five-shot is a fixed prespecified extension.',
                                   'Repeated tasks quantify conditional fixed-pool variation.']}
    ev.dump(OUT / 'baseline/validation.json', report)
    ev.dump(OUT / 'baseline/metrics_summary.json', variants['r2']['metrics_summary'])
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['assets', 'baseline'], required=True)
    args = parser.parse_args()
    if args.phase == 'assets': asset_audit()
    else: baseline_audit()
