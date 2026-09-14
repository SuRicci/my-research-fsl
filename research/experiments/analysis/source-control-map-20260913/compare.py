"""Reaggregate existing exposed-development predictions; no fitting or selection."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
NAMES=['original_r2','mean_r2','mean_CS_l2','score_ensemble_r2','augmented_support_ridge','mean_logistic_C1','mean_logistic_C10','scatter_blend','query_consistency']
KEYS=['support_indices','query_indices','class_ids','seeds','yq']
cells={};allacc=[];hashes={};supportonly={};domainacc={}
for source,target in [('eurosat','dtd'),('dtd','eurosat')]:
    arrays=[]
    for gallery in ['dtd','eurosat']:
        paths=[ROOT/'experiments/main/representation-scatter-20260913/outputs/cells'/f'{source}_to_{target}_{gallery}_k1.npz',ROOT/'experiments/main/query-consistency-20260913/outputs'/f'{target}_{gallery}.npz']
        old,new=[np.load(p) for p in paths]
        assert all(np.array_equal(old[k],new[k]) for k in KEYS)
        for p,z in zip(paths,[old,new]):
            hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
            assert np.isfinite(z['scores']).all() and np.array_equal(z['scores'].argmax(-1),z['predictions'])
            assert np.array_equal((z['predictions']==z['yq']).mean(-1),z['accuracy'])
        preds=np.stack([old['predictions'][old['names'].tolist().index(n)] for n in NAMES[:-2]]+[new['predictions'][new['names'].tolist().index(n)] for n in ['incumbent','consistency']])
        hit=preds==new['yq'];acc=hit.mean(-1);arrays.append(acc)
        if target in supportonly:assert np.array_equal(supportonly[target],preds[4:7])
        else:supportonly[target]=preds[4:7].copy()
        comparisons={n:{'delta_pp':float((acc[-1]-acc[k]).mean()*100),'candidate_only_correct':int((hit[-1]&~hit[k]).sum()),'control_only_correct':int((hit[k]&~hit[-1]).sum()),'both_wrong':int((~hit[k]&~hit[-1]).sum()),'disagreement_predictions':int((preds[-1]!=preds[k]).sum())} for k,n in enumerate(NAMES[:-1])}
        cells[f'{target}_{gallery}']={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'comparisons':comparisons,'task_conditions':len(new['seeds'])}
    mean=np.mean(arrays,axis=0);allacc.append(mean);domainacc[target]=dict(zip(NAMES,(mean.mean(-1)*100).tolist()))
overall=dict(zip(NAMES,(np.concatenate(allacc,axis=1).mean(-1)*100).tolist()))
assert abs(overall['query_consistency']-77.374)<1e-8 and abs(overall['scatter_blend']-77.334)<1e-8
result={'status':'complete','scope':'Repeatedly exposed DTD/EuroSAT development only, source-configuration transfer as originally recorded. Saved-prediction identity and arithmetic audit; does not recertify original numerical source gates. No new fitting, model selection or Caltech use.','methods':NAMES,'overall_accuracy_pct':overall,'domain_accuracy_pct':domainacc,'cells':cells,'input_sha256':hashes,'bank_count':len(hashes),'task_condition_joins':sum(c['task_conditions'] for c in cells.values()),'distinct_episodes':1000,'gallery_independent_controls_equal':True}
(HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','overall_accuracy_pct','domain_accuracy_pct','bank_count','task_condition_joins','distinct_episodes']},indent=2))
