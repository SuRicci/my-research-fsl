"""Known-data source replay and query-independence checks before Pets predictions."""
from pathlib import Path
import inspect, json
import numpy as np
import torch
import evaluate as ev
from baseline_methods import representation
HERE = Path(__file__).resolve().parent
root = ev.ROOT / 'baselines/local/r2-canonical/assets'
blocks = {}
for side in ['query', 'gallery']:
    packs = [torch.load(root / ('dtd_' + b + '_' + side + '.pt'), weights_only=True)['features'].float() for b in ['clip_vitb16', 'dinov2_vits14']]
    blocks[side] = representation(*packs, .5)
ident = json.loads((root / 'dtd_identities.json').read_text())
# Reuse the original stored evaluation draws, not new target-domain tasks.
old = HERE.parent / 'r2-fusion-stage-canonical-20260912/outputs'
paths = list(old.rglob('*.npz'))
match = None
for path in paths:
    if 'dtd_dtd_k1' in path.name:
        match = path; break
assert match is not None, 'Original source replay file missing'
with np.load(match) as p:
    si, qi = p['support_indices'][:2], p['query_indices'][:2]
    names = list(p['names']); original = p['scores'][:, :2]
S = blocks['query'][si]; Q = blocks['query'][qi.reshape(2, 75)]; G = blocks['gallery']
a = ev.ref.r2_scores(S, Q, G)
b = ev.geo.head(ev.geo.prepare(S, Q, G, 'support'), .1)
error_r2 = float(np.max(np.abs(a.numpy() - original[names.index('r2')])) )
error_cs = float(np.max(np.abs(b.numpy() - original[names.index('cs_l2_fixed')])) )
assert error_r2 < 1e-5 and error_cs < 1e-5
checks = {}
for name, scorer in [('r2', lambda x: ev.ref.r2_scores(S, x, G)), ('CS_l2', lambda x: ev.geo.head(ev.geo.prepare(S, x, G, 'support'), .1))]:
    full = scorer(Q); single = scorer(Q[:, :1]); reordered = scorer(Q.flip(1)).flip(1)
    error = max(float((full[:, :1] - single).abs().max()), float((full-reordered).abs().max()))
    assert error < 1e-5
    checks[name] = error
assert 'Q' not in inspect.signature(ev.ilpcz.fit).parameters
# Exact same C10 settings as ilpcz.fit; single/permuted predictions remain inductive.
s, q = S[0].flatten(0, 1).numpy(), Q[0].numpy()
a = ev.logistic(s, q, 10); b = ev.logistic(s, q[:1], 10)
assert np.max(np.abs(a[:1]-b)) < 1e-8
report = {'status': 'passed', 'source_task_path': str(match), 'source_task_count': 2,
          'r2_max_error': error_r2, 'CS_l2_max_error': error_cs,
          'query_batch_independence_max_error': checks, 'ilpc_fit_excludes_query_argument': True,
          'pets_predictions_computed': False, 'scope': 'source replay + interface/numerical invariants; not complete Pets metric validation'}
ev.dump(HERE / 'outputs/adapter_validation.json', report)
print(json.dumps(report, indent=2), flush=True)
