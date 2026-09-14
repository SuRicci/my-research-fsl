from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
OLD=ROOT/'experiments/main/representation-scatter-20260913/outputs/cells'
NEW=ROOT/'experiments/main/query-consistency-20260913/outputs'
CONTROLS=['original_r2','mean_r2','score_ensemble_r2','mean_CS_l2','augmented_support_ridge']
NAMES=CONTROLS+['incumbent','consistency']; cells={}; domains={}; hashes={}; checked=0
for target,source in [('dtd','eurosat'),('eurosat','dtd')]:
    aa=[]
    for gallery in ['dtd','eurosat']:
        paths=[OLD/f'{source}_to_{target}_{gallery}_k1.npz',NEW/f'{target}_{gallery}.npz']
        old,new=[np.load(p) for p in paths]
        for p in paths:hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        for k in ['support_indices','query_indices','class_ids','seeds','yq']:assert np.array_equal(old[k],new[k]),(target,gallery,k)
        for z in [old,new]:
            assert np.array_equal(z['scores'].argmax(-1),z['predictions'])
            assert np.allclose((z['predictions']==z['yq']).mean(-1),z['accuracy'],rtol=0,atol=1e-14)
        acc=np.stack([(old if n in CONTROLS else new)['accuracy'][(old if n in CONTROLS else new)['names'].tolist().index(n)] for n in NAMES])
        aa.append(acc);checked+=len(new['seeds']);cells[f'{target}_{gallery}']={'accuracy_pct':dict(zip(NAMES,(100*acc.mean(-1)).tolist())),'gain_vs_mean_r2_pp':float(100*(acc[-1]-acc[1]).mean())}
    domains[target]=(np.mean(aa,axis=0),new['seeds'])
def summary(acc,groups):
    rng=np.random.default_rng(26091359);result={'accuracy_pct':dict(zip(NAMES,(100*acc.mean(-1)).tolist())),'consistency_comparisons':{}}
    # Same resampling weights across methods; gallery conditions are already averaged per task.
    weights=np.zeros((5000,acc.shape[-1]),dtype=np.int16)
    for g in np.unique(groups):
        ix=np.flatnonzero(groups==g);weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=5000)
    for j,name in enumerate(NAMES[:-1]):
        delta=100*(acc[-1]-acc[j]);draws=weights@delta/len(delta)
        result['consistency_comparisons'][name]={'delta_pp':float(delta.mean()),'ci95_pp':np.quantile(draws,[.025,.975]).tolist()}
    return result
result={'status':'passed','task_condition_joins':checked,'distinct_task_episodes':1000,'cells':cells,'domains':{d:summary(*v) for d,v in domains.items()},'overall':summary(np.concatenate([v[0] for v in domains.values()],axis=1),np.concatenate([v[1]+i*100000000 for i,v in enumerate(domains.values())])),'input_sha256':hashes,'scope':'Post-hoc exposed development; conditional paired fixed-pool intervals; original R2 has smaller encoding budget. No independent-data claim; earlier source numerical tie failure and incremental gates remain failed.'}
(HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','task_condition_joins','domains','overall']},indent=2))
