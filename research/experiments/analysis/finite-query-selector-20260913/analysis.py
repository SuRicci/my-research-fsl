"""Finite-family diagnostic bounds from immutable predictions; not a selector."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, shutil, sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PARENT = ROOT / 'experiments/main/scatter-score-fusion-20260913'
OUT = HERE / 'outputs'
OUT.mkdir(exist_ok=True)
CFG = json.loads((HERE / 'protocol.json').read_text())
spec = importlib.util.spec_from_file_location('paired_summary', ROOT / 'experiments/main/scatter-centering-stack-20260913/summarize.py')
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert shutil.disk_usage(HERE).free / 2**30 >= 10
    cells, vectors, banks, hashes = {}, {}, {}, {}
    old = json.loads((PARENT / 'outputs/analysis.json').read_text())
    names = CFG['experts']
    for target in CFG['domains']:
        bank = []
        for gallery in CFG['domains']:
            p = PARENT / 'outputs' / (target + '_' + gallery + '.npz')
            hashes[str(p.relative_to(ROOT))] = digest(p)
            with np.load(p, allow_pickle=False) as z:
                assert z['predictions'].shape == (4, 500, 75)
                assert z['names'].tolist() == names + ['mean_cs_reference']
                acc = (z['predictions'] == z['yq']).mean(-1)
                assert np.array_equal(acc, z['accuracy'])
                for j, name in enumerate(z['names'].tolist()):
                    assert abs(acc[j].mean()*100-old['cells'][target+'_'+gallery]['accuracy_pct'][name]) < 1e-10
                current = {k:z[k].copy() for k in ['support_indices', 'query_indices', 'class_ids', 'seeds']}
                if bank:
                    assert all(np.array_equal(current[k], bank[0][k]) for k in current)
                pred = z['predictions'][:3]
                correct = pred == z['yq']
                chosen = correct[2]
                oracle = correct.any(0)
                gains = oracle & ~chosen
                disagreement = (pred != pred[2]).any(0)
                vulnerable = chosen & (~correct).any(0)
                accuracy = np.vstack([acc[:3], oracle.mean(-1), acc[:3].max(0)])
                delta = gains.mean(-1)
                cell = {
                    'accuracy_pct': dict(zip(names+['query_oracle','task_oracle'], (accuracy.mean(-1)*100).tolist())),
                    'query_oracle_minus_blend': stats.interval(delta, z['seeds']),
                    'task_oracle_minus_blend': stats.interval(accuracy[4]-acc[2], z['seeds']),
                    'query_occurrences': int(chosen.size),
                    'disagreement_count': int(disagreement.sum()),
                    'correctable_blend_errors': int(gains.sum()),
                    'vulnerable_blend_correct': int(vulnerable.sum()),
                    'all_experts_wrong_count': int((~oracle).sum()),
                    'blend_errors': int((~chosen).sum()),
                    'raw_only_rescues': int((~chosen & correct[1] & ~correct[0]).sum()),
                    'stack_only_rescues': int((~chosen & correct[0] & ~correct[1]).sum()),
                    'both_rescue': int((~chosen & correct[0] & correct[1]).sum()),
                    'sampled_score_task_count': len(z['sample_indices']),
                }
                assert gains.sum() + vulnerable.sum() <= disagreement.sum()
                assert np.all(accuracy[3] >= accuracy[4])
                key = target+'_'+gallery
                cells[key] = cell
                current.update(accuracy=accuracy, delta=delta, predictions=pred, labels=z['yq'].copy())
                bank.append(current)
                vectors[key+'_delta'] = delta
                vectors[key+'_seeds'] = z['seeds']
        banks[target] = bank
    domains, pooled_delta, pooled_groups = {}, [], []
    for i, target in enumerate(CFG['domains']):
        b = banks[target]
        acc = np.mean([x['accuracy'] for x in b], axis=0)
        delta = np.mean([x['delta'] for x in b], axis=0)
        seeds = b[0]['seeds']
        domains[target] = {'accuracy_pct': dict(zip(names+['query_oracle','task_oracle'],(acc.mean(-1)*100).tolist())),
                           'query_oracle_minus_blend': stats.interval(delta, seeds),
                           'per_seed_headroom_pp': {str(s):float(delta[seeds==s].mean()*100) for s in np.unique(seeds)}}
        pooled_delta.append(delta)
        pooled_groups.append(seeds+i*100000000)
    pooled = stats.interval(np.concatenate(pooled_delta), np.concatenate(pooled_groups))
    result = {
        'protocol': CFG, 'numpy': np.__version__, 'python':sys.executable,
        'input_hashes': hashes, 'script_sha256':digest(Path(__file__)),
        'cells':cells, 'domains':domains,
        'pooled_accuracy_pct': {n:float(np.mean([d['accuracy_pct'][n] for d in domains.values()])) for n in names+['query_oracle','task_oracle']},
        'pooled_query_oracle_minus_blend':pooled,
        'necessary_headroom_gate': pooled['delta_pp'] >= CFG['headroom_gate_pp'],
        'warning':'Oracles read target query labels. Finite three-expert ceiling only; not trainable accuracy, continuous mixture bound, independent target evidence or promotion.',
        'validation':{'all_four_cell_parent_means_match':True,'all_galleries_pair_identical_tasks':True,'query_oracle_dominates_task_oracle':True},
        'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    np.savez_compressed(OUT/'paired_vectors.npz', **vectors)
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['pooled_accuracy_pct','pooled_query_oracle_minus_blend','necessary_headroom_gate','domains']},indent=2))

if __name__ == '__main__':
    main()
