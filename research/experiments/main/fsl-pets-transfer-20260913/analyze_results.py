"""Audit every frozen Pets task and compute precommitted paired comparisons."""
from pathlib import Path
import json
import numpy as np
import evaluate as ev

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
CFG = ev.CFG
N = len(CFG['seeds']) * CFG['episodes_per_seed']
NAMES = ['r2', 'CS_l2', 'support_logistic_C1', 'support_logistic_C10', 'ilpcz_local']


def paired_draws(values, indices):
    blocks = values.reshape(len(CFG['seeds']), CFG['episodes_per_seed'], -1)
    return blocks[np.arange(len(CFG['seeds']))[None, :, None], indices].mean((1, 2)) * 100


def main():
    data, ident, _ = ev.load_data()
    manifest = json.loads((OUT / 'run_manifest_ilpc.json').read_text())
    baseline_manifest = json.loads((OUT / 'run_manifest_baseline.json').read_text())
    complete = json.loads((OUT / 'ilpc_complete.json').read_text())
    assert complete['status'] == 'success' and complete['tasks'] == 4 * N
    assert complete['signature'] == manifest['signature'] == baseline_manifest['signature']
    assert manifest['adapter_sha256'] == ev.sha(HERE / 'evaluate.py')
    assert manifest['config'] == CFG
    confirmation = json.loads((OUT / 'baseline/confirmed_comparator.json').read_text())
    assert ev.sha(Path(confirmation['metric_contract_path'])) == confirmation['metric_contract_sha256']
    assert ev.sha(OUT / 'baseline/validation.json') == confirmation['validation_sha256']
    validated = json.loads((OUT / 'baseline/validation.json').read_text())
    assert validated['status'] == 'passed'
    cells = {}; cell_draws = {}; accuracies = {}; hashes = {}; replay = {}
    rng = np.random.default_rng(CFG['bootstrap']['seed'])
    for shot in CFG['shots']:
        with np.load(OUT / ('tasks_k%d.npz' % shot)) as tasks:
            si, qi, cs, seeds = [tasks[k] for k in ['support_indices', 'query_indices', 'class_ids', 'seeds']]
        for actual, expected in zip([si, qi, cs, seeds], ev.task_arrays(ident, shot)):
            assert np.array_equal(actual, expected)
        indices = rng.integers(CFG['episodes_per_seed'], size=(CFG['bootstrap']['replicates'], len(CFG['seeds']), CFG['episodes_per_seed']))
        # A constant paired gain must survive resampling exactly; task pairing must be preserved.
        check = paired_draws(np.tile([0.25, 0.5], (N, 1)), indices)
        assert np.allclose(check[:, 1] - check[:, 0], 25)
        for gallery in CFG['gallery_conditions']:
            cell = 'pets_' + gallery + '_k%d' % shot
            basepath = OUT / 'baseline' / (cell + '.npz')
            assert ev.sha(basepath) == validated['output_sha256'][cell]
            with np.load(basepath) as base:
                assert base['names'].tolist() == NAMES[:-1]
                for key, expected in [('support_indices', si), ('query_indices', qi), ('class_ids', cs), ('seeds', seeds)]:
                    assert np.array_equal(base[key], expected)
                baseline_correct = (base['scores'].argmax(-1) == np.repeat(np.arange(5), 15))
                baseline_acc = baseline_correct.mean(-1)
                baseline_breed_acc = baseline_correct.reshape(4, N, 5, 15).mean(-1)
                assert np.array_equal(baseline_acc, base['accuracy'])
            files = sorted((OUT / 'ilpc_tasks' / cell).glob('*.npz'))
            assert [p.stem for p in files] == ['%04d' % i for i in range(N)]
            acc = []; breed_acc = []; stats = []; cell_replay = {}
            G = data['gallery' if gallery == 'pets' else 'dtd'].numpy().astype(np.float64)
            for index, path in enumerate(files):
                hashes[str(path.relative_to(OUT))] = ev.sha(path)
                with np.load(path) as task:
                    assert str(task['signature']) == manifest['signature']
                    assert np.array_equal(task['support_indices'], si[index])
                    assert np.array_equal(task['query_indices'], qi[index])
                    scores = task['scores']; pred = scores.argmax(-1)
                    assert scores.shape == (75, 5) and np.isfinite(scores).all()
                    assert np.array_equal(pred, task['predictions'])
                    accuracy = float((pred == np.repeat(np.arange(5), 15)).mean())
                    assert accuracy == float(task['accuracy'])
                    selected = task['selected_indices']; labels = task['pseudo_labels']
                    assert selected.ndim == labels.ndim == 1 and len(selected) == len(labels)
                    assert len(np.unique(selected)) == len(selected)
                    assert np.all((selected >= 0) & (selected < 5 * shot + len(G)))
                    assert np.all((labels >= 0) & (labels < 5))
                    assert np.array_equal(selected[:5 * shot], np.arange(5 * shot))
                    assert np.array_equal(labels[:5 * shot], np.repeat(np.arange(5), shot))
                    assert len(set(np.bincount(labels, minlength=5))) == 1
                    info = json.loads(str(task['stats_json']))
                    assert info['convergence_warnings'] == 0 and info['max_linear_residual'] < 1e-8
                    assert info['selected_total'] == len(selected)
                    assert info['gallery_selected'] == len(selected) - 5 * shot
                    assert len(info['trace']) == info['rounds']
                    assert info['stop'] in ['all_selected', 'one_predicted_class_exhausted']
                    if index in [0, N // 2, N - 1]:
                        S = data['query'][si[index].ravel()].numpy().astype(np.float64)
                        Q = data['query'][qi[index].ravel()].numpy().astype(np.float64)
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(C=10, multi_class='multinomial', solver='lbfgs', max_iter=2000, tol=1e-8, random_state=26091275)
                        model.fit(np.concatenate([S, G])[selected], labels)
                        error = float(abs(model.decision_function(Q) - scores).max())
                        assert error < 1e-8
                        assert abs(model.decision_function(Q[:1]) - scores[:1]).max() < 1e-8
                        cell_replay[str(index)] = error
                    acc.append(accuracy); breed_acc.append((pred == np.repeat(np.arange(5), 15)).reshape(5, 15).mean(-1)); stats.append(info)
            values = np.column_stack([baseline_acc.T, acc]); accuracies[cell] = values
            per_breed_values = np.concatenate([baseline_breed_acc, np.asarray(breed_acc)[None]], axis=0)
            per_breed = {}
            for breed in np.unique(cs):
                mask = cs == breed; means = per_breed_values[:, mask].mean(1) * 100
                per_breed[str(int(breed))] = {'episode_occurrences': int(mask.sum()),
                    'query_evaluation_occurrences': int(mask.sum() * 15),
                    'accuracy_pct': dict(zip(NAMES, means.tolist())),
                    'ilpc_delta_pp': dict(zip(NAMES[:-1], (means[-1] - means[:-1]).tolist()))}
            draws = paired_draws(values, indices); cell_draws[cell] = draws
            comparisons = {}
            for j, name in enumerate(NAMES[:-1]):
                diff = (values[:, -1] - values[:, j]) * 100
                comparisons[name] = {'delta_pp': float(diff.mean()),
                    'paired_ci95_pp': np.quantile(draws[:, -1] - draws[:, j], [.025, .975]).tolist(),
                    'seed_deltas_pp': diff.reshape(len(CFG['seeds']), -1).mean(1).tolist(),
                    'wins_ties_losses': [int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())]}
            cells[cell] = {'shot': shot, 'gallery': gallery, 'task_count': N,
                'accuracy_pct': dict(zip(NAMES, (values.mean(0) * 100).tolist())),
                'comparisons': comparisons, 'per_breed': per_breed,
                'per_seed_accuracy_pct': {str(seed): dict(zip(NAMES, (values.reshape(len(CFG['seeds']), -1, len(NAMES))[i].mean(0) * 100).tolist())) for i, seed in enumerate(CFG['seeds'])},
                'ilpc_cost': {key: {'mean': float(np.mean([s[key] for s in stats])), 'max': float(np.max([s[key] for s in stats]))}
                             for key in ['wall_seconds', 'gallery_selected', 'rounds', 'max_linear_residual', 'factor_nnz']}}
            replay[cell] = cell_replay
            print('AUDITED_CELL', cell, cells[cell]['accuracy_pct'], flush=True)
    metrics = {'pets_' + ('matched' if g == 'pets' else 'mismatched') + '_%dshot_accuracy' % k:
               cells['pets_' + g + '_k%d' % k]['accuracy_pct']['ilpcz_local']
               for k in CFG['shots'] for g in CFG['gallery_conditions']}
    metrics['pets_macro_1shot_accuracy'] = (metrics['pets_matched_1shot_accuracy'] + metrics['pets_mismatched_1shot_accuracy']) / 2
    assert set(metrics) == set(CFG['new_metric_ids']) and np.isfinite(list(metrics.values())).all()
    macro = np.mean([accuracies['pets_' + g + '_k1'] for g in CFG['gallery_conditions']], axis=0)
    macro_draws = np.mean([cell_draws['pets_' + g + '_k1'] for g in CFG['gallery_conditions']], axis=0)
    macro_comparisons = {name: {'delta_pp': float((macro[:, -1] - macro[:, j]).mean() * 100),
                         'paired_ci95_pp': np.quantile(macro_draws[:, -1] - macro_draws[:, j], [.025, .975]).tolist()}
                         for j, name in enumerate(NAMES[:-1])}
    strong = ['r2', 'CS_l2', 'support_logistic_C10']
    matched = cells['pets_pets_k1']['comparisons']
    matched_gate = all(matched[n]['delta_pp'] >= CFG['thresholds']['matched_gain_pp'] and
                       matched[n]['paired_ci95_pp'][0] > CFG['thresholds']['paired_ci_lower'] for n in strong)
    mismatch_losses = [{'cell': cell, 'control': name, 'delta_pp': cells[cell]['comparisons'][name]['delta_pp']}
                       for cell in ['pets_dtd_k1', 'pets_dtd_k5'] for name in strong
                       if cells[cell]['comparisons'][name]['delta_pp'] < -CFG['thresholds']['max_mismatch_loss_pp_for_robust_label']]
    report = {'metrics_summary': metrics, 'cells': cells, 'macro_1shot_comparisons': macro_comparisons,
              'matched_primary_gate_passed': matched_gate, 'mismatch_mean_loss_guard_violations': mismatch_losses,
              'bootstrap': CFG['bootstrap'], 'scope_id': CFG['scope_id'], 'baseline_id': confirmation['baseline_id'],
              'interpretation_boundary': 'Prespecified matched one-shot gate; all four cells and controls reported. Confidence intervals condition on fixed pools and seeds; repeated images across tasks are not independent new datasets. A mean loss guard is not an equivalence test.'}
    validation = {'status': 'passed', 'cells': 4, 'tasks': 4 * N, 'signature': manifest['signature'],
                  'full_prediction_and_identity_check': True, 'selected_head_spot_replay': replay,
                  'source_sha256': ev.sha(HERE / 'analyze_results.py'), 'task_output_sha256': hashes}
    ev.dump(OUT / 'analysis.json', report); ev.dump(OUT / 'validation.json', validation)
    np.savez_compressed(OUT / 'paired_accuracy.npz', names=NAMES, **accuracies)
    lines = ['# Frozen Pets transfer qualification', '', report['interpretation_boundary'], '',
             '| Condition | R2 | CS_l2 | Support C1 | Support C10 | iLPC |', '|---|---:|---:|---:|---:|---:|']
    for cell, row in cells.items():
        lines.append('| ' + cell + ' | ' + ' | '.join('%.4f' % row['accuracy_pct'][n] for n in NAMES) + ' |')
    lines += ['', 'Matched primary gate passed: ' + str(matched_gate), '',
              '| Condition | Comparator | Paired difference, pp | 95% interval, pp |', '|---|---|---:|---|']
    for cell, row in cells.items():
        for name, delta in row['comparisons'].items():
            lines.append('| %s | %s | %+.4f | [%+.4f, %+.4f] |' % (cell, name, delta['delta_pp'], *delta['paired_ci95_pp']))
    lines += ['', 'No target parameter fitting or outcome-based method selection. iLPC is a source-audited local formula implementation; this evaluation does not establish algorithmic novelty.']
    (HERE / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('ANALYSIS_COMPLETE', json.dumps({'metrics_summary': metrics, 'matched_gate': matched_gate, 'mismatch_loss_guard': mismatch_losses}), flush=True)


if __name__ == '__main__':
    main()
