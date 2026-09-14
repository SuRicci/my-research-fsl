"""Descriptive audit of saved Pets outcomes; labels used only after evaluation."""
from pathlib import Path
import hashlib, json
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / 'main/fsl-pets-transfer-20260913'
OUT = SRC / 'outputs'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
a = json.loads((OUT / 'analysis.json').read_text())
v = json.loads((OUT / 'validation.json').read_text())
f = json.loads((SRC / 'assets/feature_manifest.json').read_text())
assert sha(SRC / 'assets/identities.json') == f['identity_sha256']
ident = json.loads((SRC / 'assets/identities.json').read_text())
gallery_labels = np.array([r['label'] for r in ident['gallery']])
report = {'parent_run_artifact_id': 'run-cb50d2d5', 'intervention': 'none; no refitting or predictions',
          'cells': {}, 'source_sha256': {str(p): sha(p) for p in [OUT / 'analysis.json', OUT / 'validation.json', SRC / 'assets/identities.json']},
          'boundary': 'Repeated task occurrences, not independent images. Labels are retrospective diagnostics only. DTD class ids are never compared numerically to Pets ids. No causal mediation or newly calibrated selector is established.'}
for cell, summary in a['cells'].items():
    k = summary['shot']
    with np.load(OUT / ('tasks_k%d.npz' % k)) as tasks:
        classes = tasks['class_ids']
    files = sorted((OUT / 'ilpc_tasks' / cell).glob('*.npz'))
    assert len(files) == 500
    totals = np.zeros(4, dtype=np.int64)
    per_task = []
    for index, p in enumerate(files):
        assert sha(p) == v['task_output_sha256'][str(p.relative_to(OUT))]
        with np.load(p) as task:
            selected = task['selected_indices'][5*k:] - 5*k
            pseudo = task['pseudo_labels'][5*k:]
            assert (selected >= 0).all() and (selected < 1876).all()
            assert len(selected) == len(pseudo)
            if summary['gallery'] == 'pets':
                truth = gallery_labels[selected]
                relevant = np.isin(truth, classes[index])
                correct = truth == classes[index][pseudo]
                assert not (correct & ~relevant).any()
                counts = np.array([len(selected), relevant.sum(), correct.sum(), (~relevant).sum()])
            else:
                # Every gallery image belongs to the distinct DTD domain, not a Pets breed.
                counts = np.array([len(selected), 0, 0, len(selected)])
            totals += counts; per_task.append(counts.tolist())
    assert totals[0] == totals[1] + totals[3]
    breed_deltas = {name: np.array([r['ilpc_delta_pp'][name] for r in summary['per_breed'].values()]) for name in ['r2','CS_l2','support_logistic_C10']}
    report['cells'][cell] = {'breed_count': len(summary['per_breed']),
        'breed_loss_counts': {name: int((values < -1e-10).sum()) for name, values in breed_deltas.items()},
        'selected_gallery_occurrences': int(totals[0]),
        'in_episode_class_fraction': float(totals[1]/totals[0]),
        'correct_pseudo_label_fraction_all_selected': float(totals[2]/totals[0]),
        'correct_fraction_given_in_episode': float(totals[2]/totals[1]) if totals[1] else None,
        'outside_episode_class_fraction': float(totals[3]/totals[0]),
        'mean_selected_per_task': float(totals[0]/500),
        'mean_task_wall_seconds': summary['ilpc_cost']['wall_seconds']['mean'],
        'max_linear_residual': summary['ilpc_cost']['max_linear_residual']['max'],
        'counts_per_task_total_relevant_correct_outside': per_task}
    assert abs(totals[0]/500-summary['ilpc_cost']['gallery_selected']['mean'])<1e-10
(HERE / 'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
lines=['# Pets failure coverage', '', report['boundary'], '',
       '| Condition | Breeds losing to R2 | Mean selected gallery images | Outside episode classes | Correct among all selected | Max linear residual |',
       '|---|---:|---:|---:|---:|---:|']
for cell,r in report['cells'].items():
    lines.append('| %s | %d/37 | %.1f | %.2f%% | %.2f%% | %.2e |' % (cell,r['breed_loss_counts']['r2'],r['mean_selected_per_task'],100*r['outside_episode_class_fraction'],100*r['correct_pseudo_label_fraction_all_selected'],r['max_linear_residual']))
lines += ['', 'The external-domain gallery is retained extensively despite broad query losses. Small linear-system residuals and exact classification-head replays make a numerical-solver explanation less plausible; they do not prove which modeling assumption causes the failure. Matched-gallery contamination is also substantial, so domain matching alone does not establish episode relevance or superiority over the existing retrieval control.', '',
          'No new gate was fitted. Next: challenge the whole-gallery, all-pseudolabel assumption and compare structurally distinct information/objective routes using prior literature and an untouched qualification plan.']
(HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({c:{k:v for k,v in r.items() if not k.startswith('counts_per_task')} for c,r in report['cells'].items()},indent=2))
