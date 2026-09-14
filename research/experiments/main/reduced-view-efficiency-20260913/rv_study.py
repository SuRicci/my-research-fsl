"""Frozen subset selection and paired source evaluation; cached features only."""
from pathlib import Path
import argparse, hashlib, json, sys, time, shutil
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'representation-scatter-20260913'
sys.path.insert(0, str(PARENT))
import evaluate as prior
CFG = json.loads((HERE/'protocol.json').read_text())
OUT = HERE/'outputs'
LINEAR = HERE.parent/'view-distribution-kernel-20260913/outputs/selection_lock.json'
PENALTIES = json.loads(LINEAR.read_text())
HEADS = CFG['classifiers']

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')

def sha(path):
    return prior.sha(path)

def guard():
    assert shutil.disk_usage(HERE).free >= CFG['resources']['free_gib_min']*2**30, 'disk floor'
    assert datetime.now(timezone.utc) < datetime.fromisoformat(CFG['resources']['deadline_utc'])
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < CFG['resources']['new_gib_max']*2**30

def unit(x):
    return F.normalize(x.to(torch.float64), dim=-1)

def aggregate(data, subset):
    return {ds:{side:unit(unit(v[side])[...,subset,:].mean(-2)) for side in ['query','gallery']} for ds,v in data.items()}

def ridge(x, q, y, penalty):
    a = x-x.mean(0)
    return (q-x.mean(0)) @ a.T @ torch.linalg.solve(a@a.T+penalty*torch.eye(len(x),dtype=x.dtype), y-y.mean(0)) + y.mean(0)

def predict(s, q, g, penalty):
    shot = s.shape[1]
    y = F.one_hot(torch.arange(5).repeat_interleave(shot),5).double()
    flat = s.flatten(0,1)
    out = [None,None,ridge(flat,q,y,penalty)]
    neighbors = []
    for h in range(2):
        x, z, gallery = flat, q, g
        if h == 1:
            mu = flat.mean(0)
            x,z = unit(flat-mu),unit(q-mu)
            if shot == 1: gallery = unit(g-mu)
        if shot == 1:
            proto = unit(x.reshape(5,shot,-1).mean(1))
            ids = torch.argsort(-(proto@gallery.T), dim=-1, stable=True)[:,:64]
            train = unit(.5*proto+.5*unit(gallery[ids].mean(1)))
            out[h] = ridge(train,z,torch.eye(5,dtype=s.dtype),.1)
            neighbors.append(ids)
        else:
            out[h] = ridge(x,z,y,1. if h==0 else .1)
    return torch.stack(out), (torch.stack(neighbors) if neighbors else torch.empty(0,dtype=torch.int64))

def tasks(data, ds, shot, phase):
    return prior.sampler.tasks(data[ds]['ident'],shot,phase)

def gallery_indices(data, ds, task):
    labels = np.array(data[ds]['ident']['gallery_labels'])
    result = []
    for i,cs in enumerate(task['class_ids']):
        parts = []
        for excluded in [False,True]:
            pool = np.flatnonzero(~np.isin(labels,cs)) if excluded else np.arange(len(labels))
            rng = np.random.RandomState(26091381+10000*excluded+i)
            parts.append(rng.choice(pool,1024,replace=False))
        result.append(parts)
    return np.asarray(result)

WEIGHTS = {}
def weights(seeds):
    key = tuple(seeds.tolist())
    if key not in WEIGHTS:
        rng = np.random.RandomState(CFG['bootstrap_seed'])
        w = np.zeros((CFG['bootstrap_replicates'],len(seeds)))
        groups = np.unique(seeds)
        for seed in groups:
            ix = np.flatnonzero(seeds==seed)
            draws = rng.randint(len(ix),size=(len(w),len(ix)))
            w[:,ix] = np.stack([np.bincount(row,minlength=len(ix)) for row in draws])/(len(ix)*len(groups))
        WEIGHTS[key] = w
    return WEIGHTS[key]

def interval(delta,seeds):
    return {'delta_pp':float(delta.mean()*100),'ci95_pp':(np.quantile(weights(seeds)@delta,[.025,.975])*100).tolist()}

def sources():
    paths = list(HERE.glob('*.py'))+[HERE/'protocol.json',LINEAR,PARENT/'evaluate.py',PARENT/'metric.py',
        PARENT/'extract_views.py',PARENT/'protocol.json',PARENT/'evaluation_contract.json',
        PARENT/'assets/feature_manifest.json',Path(prior.sampler.__file__)]
    return {str(p):sha(p) for p in paths}

def run_cell(data, reps, ds, source, shot, phase, condition, subset_index):
    t = tasks(data,ds,shot,phase)
    gi = gallery_indices(data,ds,t) if phase=='selection' else None
    y = np.repeat(np.arange(5),t['query_indices'].shape[-1])
    audits = CFG['selection_audit_indices'] if phase=='selection' else CFG['audit_indices']
    predictions,scores,neighbors = [],[],[]
    fixed_g = reps[condition]['gallery'] if phase=='eval' else None
    for i in range(len(t['seeds'])):
        if i%100 == 0: guard()
        s = reps[ds]['query'][t['support_indices'][i]]
        q = reps[ds]['query'][t['query_indices'][i].reshape(-1)]
        g = reps[ds]['gallery'][gi[i,condition]] if phase=='selection' else fixed_g
        sc,nb = predict(s,q,g,PENALTIES[source][str(shot)]['linear_mean']['penalty'])
        assert torch.isfinite(sc).all()
        predictions.append(sc.argmax(-1).numpy().astype(np.uint8))
        if i in audits: scores.append(sc.numpy()); neighbors.append(nb.numpy())
    pred = np.stack(predictions,1)
    name = f'{phase}_{ds}_k{shot}_{condition}_v{subset_index:02d}'
    np.savez_compressed(OUT/'cells'/(name+'.npz'),predictions=pred,
        audit_scores=np.stack(scores),audit_neighbors=np.stack(neighbors),audit_indices=audits)
    return (pred==y).mean(-1)

def select(data):
    accuracy = {}
    for j,subset in enumerate(CFG['subsets']):
        reps = aggregate(data,subset)
        for ds in CFG['source_domains']:
            for shot in CFG['shots']:
                t = tasks(data,ds,shot,'selection')
                if j==0:
                    np.savez_compressed(OUT/f'selection_{ds}_k{shot}_tasks.npz',gallery_indices=gallery_indices(data,ds,t),**t)
                for c in [0,1]:
                    key=f'{ds}_k{shot}_{c}'
                    accuracy.setdefault(key,[]).append(run_cell(data,reps,ds,ds,shot,'selection',c,j))
        print('SOURCE_SUBSET_COMPLETE',j,subset,flush=True)
    choice = {}
    for ds in CFG['source_domains']:
        rows=[]
        for j,subset in enumerate(CFG['subsets']):
            checks=[]
            for shot in CFG['shots']:
                seeds=tasks(data,ds,shot,'selection')['seeds']
                for c in [0,1]:
                    bank=accuracy[f'{ds}_k{shot}_{c}']
                    for h in range(len(HEADS)):
                        checks.append(interval(bank[j][h]-bank[-1][h],seeds))
            rows.append(dict(index=j,subset=subset,feasible=all(v['ci95_pp'][0]>=-.5 for v in checks),
                worst_delta_pp=min(v['delta_pp'] for v in checks),macro_delta_pp=float(np.mean([v['delta_pp'] for v in checks])),comparisons=checks))
        reduced=[r for r in rows[:-1] if r['feasible']]
        selected=min(reduced,key=lambda r:(len(r['subset']),-r['worst_delta_pp'],-r['macro_delta_pp'],r['subset'])) if reduced else rows[-1]
        choice[ds]={'selected_index':selected['index'],'subset':selected['subset'],'fallback':not bool(reduced),'rows':rows}
    dump(OUT/'selection_lock.json',choice)
    print('SOURCE_CHOICES_FROZEN',{ds:v['subset'] for ds,v in choice.items()},flush=True)
    return choice

def evaluate(data):
    for j,subset in enumerate(CFG['subsets']):
        reps=aggregate(data,subset)
        for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
            for shot in CFG['shots']:
                t=tasks(data,target,shot,'eval')
                if j==0:
                    np.savez_compressed(OUT/f'eval_{target}_k{shot}_tasks.npz',**t)
                for gallery in [target,source]:
                    assert not set(data[target]['ident']['query_rgb']) & set(data[gallery]['ident']['gallery_rgb'])
                    run_cell(data,reps,target,source,shot,'eval',gallery,j)
        print('EVALUATION_SUBSET_COMPLETE',j,subset,flush=True)

def analyze(choices):
    cells={}
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for shot in CFG['shots']:
            t=np.load(OUT/f'eval_{target}_k{shot}_tasks.npz')
            y=np.repeat(np.arange(5),t['query_indices'].shape[-1])
            for gallery in [target,source]:
                banks=[np.load(OUT/'cells'/f'eval_{target}_k{shot}_{gallery}_v{j:02d}.npz')['predictions'] for j in range(17)]
                acc=np.stack([(p==y).mean(-1) for p in banks])
                j=choices[source]['selected_index']
                rows={}
                for h,head in enumerate(HEADS):
                    rows[head]={'selected_accuracy_pct':float(acc[j,h].mean()*100),
                        'fullsix_accuracy_pct':float(acc[-1,h].mean()*100),
                        'selected_vs_fullsix':interval(acc[j,h]-acc[-1,h],t['seeds']),
                        'frontier':[dict(subset=CFG['subsets'][i],accuracy_pct=float(acc[i,h].mean()*100),
                            **interval(acc[i,h]-acc[-1,h],t['seeds'])) for i in range(17)]}
                cells[f'{source}_to_{target}_k{shot}_{gallery}']=rows
    passed=all(len(v['subset'])<=3 for v in choices.values()) and all(v['selected_vs_fullsix']['ci95_pp'][0]>=-.5 for r in cells.values() for v in r.values())
    result={'metric_gate_passed':passed,'selected_subsets':{ds:v['subset'] for ds,v in choices.items()},
        'cells':cells,'unique_eval_episodes':2000,'gallery_conditions':8,'classifier_count':3,'subset_count':17,
        'scope':'exposed development sources; no Pets/Caltech; frontier descriptive only',
        'uncertainty':'Paired stratified bootstrap conditional on fixed pools; duplicate support-only/gallery conditions not independent.'}
    dump(OUT/'analysis.json',result)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','run'],required=True);args=ap.parse_args()
    torch.set_num_threads(6);OUT.mkdir(exist_ok=True);(OUT/'cells').mkdir(exist_ok=True);guard()
    data=prior.load()
    if args.phase=='check':
        sys.path.insert(0,str(HERE))
        import rv_verify
        result=rv_verify.precheck(data)
        dump(OUT/'precheck.json',result);dump(HERE/'lock.json',sources())
        dump(OUT/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'threads':6})
        print('PRECHECK',result,flush=True);return
    assert sources()==json.loads((HERE/'lock.json').read_text())
    assert json.loads((OUT/'precheck.json').read_text())['status']=='passed'
    assert not (OUT/'selection_lock.json').exists(),'Never overwrite measured run'
    start=time.time()
    dump(OUT/'manifest.json',{'command':[sys.executable,*sys.argv],'sources':sources(),'protocol':CFG})
    with torch.no_grad():
        choice=select(data);lock=sha(OUT/'selection_lock.json')
        evaluate(data);result=analyze(choice)
    assert sha(OUT/'selection_lock.json')==lock
    assert sources()==json.loads((HERE/'lock.json').read_text())
    dump(OUT/'complete.json',{'status':'computed','elapsed_seconds':time.time()-start,
        'metric_gate_passed':result['metric_gate_passed'],'selection_sha256':lock,'independent_audit':'required'})
    print('COMPUTE_COMPLETE',result['metric_gate_passed'],time.time()-start,flush=True)

if __name__=='__main__':main()
