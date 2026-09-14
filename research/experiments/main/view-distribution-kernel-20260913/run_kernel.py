"""One frozen source-only study, reusing paired historical control outputs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import torch
import view_kernel as vk

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'representation-scatter-20260913'
sys.path.insert(0, str(PARENT))
import evaluate as prior
CFG = json.loads((HERE / 'protocol.json').read_text())
OUT = HERE / 'outputs'
MODES = CFG['new_methods']

def dump(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + '\n')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def params(mode):
    return [(b, l) for b in ([0] if mode == 'linear_mean' else CFG['betas']) for l in CFG['penalties']]

def sources():
    paths = list(HERE.glob('*.py')) + [HERE/'protocol.json', PARENT/'protocol.json', PARENT/'evaluate.py',
        PARENT/'metric.py', PARENT/'extract_views.py', PARENT/'evaluation_lock.json',
        PARENT/'assets/feature_manifest.json', Path(prior.sampler.__file__),
        Path(prior.metric.ref.__file__), Path(prior.metric.geo.__file__),
        Path(prior.metric.ref.ridge_scores.__code__.co_filename)]
    paths += sorted((PARENT/'outputs/cells').glob('*.npz'))
    return {str(p): sha(p) for p in paths}

def task_data(data, ds, shot, role):
    return prior.sampler.tasks(data[ds]['ident'], shot, role)

def select(data):
    choices = {}
    for ds in CFG['source_domains']:
        choices[ds] = {}
        for shot in CFG['shots']:
            t = task_data(data, ds, shot, 'selection')
            y = np.repeat(np.arange(5), t['query_indices'].shape[-1])
            values = {m: [] for m in MODES}
            for i in range(len(t['seeds'])):
                prior.guard()
                s = data[ds]['query'][t['support_indices'][i]]
                q = data[ds]['query'][t['query_indices'][i].reshape(-1)]
                for mode in MODES:
                    values[mode].append(vk.grid(s, q, mode, CFG['betas'], CFG['penalties']).numpy())
            choices[ds][str(shot)] = {}
            for mode in MODES:
                sc = np.stack(values[mode])
                counts = (sc.argmax(-1) == y).sum((0, 2))
                index = int(counts.argmax())
                choices[ds][str(shot)][mode] = dict(beta=params(mode)[index][0], penalty=params(mode)[index][1],
                    index=index, integer_correct=counts.tolist())
                np.savez_compressed(OUT/f'{ds}_k{shot}_{mode}_selection.npz', scores=sc, yq=y, **t)
            dump('selection_progress.json', choices)
            print('SELECTION_COMPLETE', ds, shot, choices[ds][str(shot)], flush=True)
    dump('selection_lock.json', choices)
    return choices

def evaluate(data, choices):
    for source, target in [('dtd','eurosat'), ('eurosat','dtd')]:
        for shot in CFG['shots']:
            t = task_data(data, target, shot, 'eval')
            y = np.repeat(np.arange(5), 15)
            values = []
            for i in range(len(t['seeds'])):
                prior.guard()
                s = data[target]['query'][t['support_indices'][i]]
                q = data[target]['query'][t['query_indices'][i].reshape(-1)]
                values.append(np.stack([vk.scores(s, q, mode, choices[source][str(shot)][mode]['beta'],
                    choices[source][str(shot)][mode]['penalty']).numpy() for mode in MODES]))
                if i % 100 == 0:
                    print('EVALUATION', source, target, shot, i, flush=True)
            new = np.stack(values, 1)
            assert np.isfinite(new).all()
            for gallery in [target, source]:
                name = f'{source}_to_{target}_{gallery}_k{shot}'
                old = np.load(PARENT/'outputs/cells'/(name+'.npz'))
                for key, value in t.items():
                    assert np.array_equal(value, old[key]), (name, key)
                assert np.array_equal(old['yq'], y)
                assert np.array_equal(old['predictions'], old['scores'].argmax(-1))
                assert np.array_equal(old['accuracy'], (old['predictions']==y).mean(-1))
                names = old['names'].tolist() + MODES
                sc = np.concatenate([old['scores'], new])
                pred = sc.argmax(-1)
                np.savez_compressed(OUT/'cells'/(name+'.npz'), scores=sc, predictions=pred,
                    accuracy=(pred==y).mean(-1), yq=y, names=names, **t)
                print('CELL_COMPLETE', name, dict(zip(names, ((pred==y).mean((1,2))*100).tolist())), flush=True)

def analyze():
    cells, directions = {}, {}
    passed = True
    for source, target in [('dtd','eurosat'), ('eurosat','dtd')]:
        pair = []
        for shot in CFG['shots']:
            for gallery in [target, source]:
                name = f'{source}_to_{target}_{gallery}_k{shot}'
                z = np.load(OUT/'cells'/(name+'.npz'))
                names = z['names'].tolist()
                a = z['accuracy']
                c = a[names.index('distribution_rbf')]
                comparisons = {m:prior.interval(c-a[j],z['seeds']) for j,m in enumerate(names) if m!='distribution_rbf'}
                passed &= all(v['ci95_pp'][0]>=-.5 for v in comparisons.values())
                cells[name] = dict(accuracy_pct=dict(zip(names,(a.mean(1)*100).tolist())), comparisons=comparisons)
                if shot == 1:
                    pair.append(a)
        z = np.load(OUT/'cells'/f'{source}_to_{target}_{target}_k1.npz')
        names = z['names'].tolist()
        a = np.mean(pair,axis=0)
        comparisons = {m:prior.interval(a[names.index('distribution_rbf')]-a[j],z['seeds'])
            for j,m in enumerate(names) if m!='distribution_rbf'}
        passed &= all(v['delta_pp']>=.5 and v['ci95_pp'][0]>0 for v in comparisons.values())
        directions[f'{source}_to_{target}'] = comparisons
    result = dict(metric_gate_passed=bool(passed), cells=cells, directions=directions,
        scope='auxiliary/development, reused exposed fixed sources; no new Pets/Caltech result',
        unique_eval_episodes=2000, gallery_task_conditions=4000, methods=14,
        original_scatter_strict_audit='failed1/88; preserved, not repaired',
        promotion_ready=False, promotion_note='Requires independent new output audit and resolution of inherited comparator risk if metric gate passes')
    dump('analysis.json', result)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['check','run'], required=True)
    args = ap.parse_args()
    torch.set_num_threads(6)
    OUT.mkdir(exist_ok=True)
    (OUT/'cells').mkdir(exist_ok=True)
    prior.guard()
    if args.phase == 'check':
        import verify_kernel
        data = prior.load()
        result = verify_kernel.precheck(data)
        dump('validation.json', result)
        dump('environment.json', dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads()))
        (HERE/'lock.json').write_text(json.dumps(sources(),indent=2)+'\n')
        print('VALIDATION',json.dumps(result),flush=True)
        return
    assert sources() == json.loads((HERE/'lock.json').read_text())
    assert json.loads((OUT/'validation.json').read_text())['status']=='passed'
    assert not (OUT/'selection_lock.json').exists(), 'Do not overwrite a previous real run'
    start = time.time()
    dump('manifest.json', dict(command=[sys.executable,*sys.argv], source_hashes=sources(), config=CFG))
    data = prior.load()
    with torch.no_grad():
        choices = select(data)
        selection_sha = sha(OUT/'selection_lock.json')
        evaluate(data,choices)
        assert sha(OUT/'selection_lock.json') == selection_sha
        result = analyze()
    assert sources() == json.loads((HERE/'lock.json').read_text())
    dump('complete.json',dict(status='computed',metric_gate_passed=result['metric_gate_passed'],
        elapsed_seconds=time.time()-start, selection_sha256=selection_sha, output_audit_pending=True))
    print('COMPUTE_COMPLETE', result['metric_gate_passed'], time.time()-start, flush=True)

if __name__ == '__main__':
    main()
