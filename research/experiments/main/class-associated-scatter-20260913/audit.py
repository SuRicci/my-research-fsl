"""Independent output and scalar bootstrap audit; does not import study.py."""
from pathlib import Path
import json, hashlib, math
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
CFG=json.loads((HERE/'protocol.json').read_text())
result=json.loads((OUT/'analysis.json').read_text())
locks=json.loads((HERE/'code_lock.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in locks.items())
vectors={}; checks=0; transitions={}
for target in CFG['domains']:
    deltas=[]; accuracies=[]
    for g in CFG['domains']:
        z=np.load(OUT/(target+'_'+g+'.npz'))
        sc=z['scores'];pred=sc.argmax(-1);truth=z['yq']
        accuracy=(pred==truth).mean(-1)
        assert np.array_equal(pred,z['predictions']) and np.array_equal(accuracy,z['accuracy'])
        assert sc.shape==(7,500,75,5)
        assert z['metric_scores'].shape==(5,500,75,5)
        assert set(np.unique(pred)).issubset(set(range(5)))
        names=z['names'].tolist()
        parent=np.load(HERE.parent/'scatter-score-fusion-20260913/outputs'/(target+'_'+g+'.npz'))
        for j,k in [(0,2),(4,0),(5,1),(6,3)]:
            assert np.array_equal(pred[j],parent['predictions'][k]);checks+=500
        for key in ['class_ids','support_indices','query_indices','seeds']:
            assert np.array_equal(z[key],parent[key])
        reported=result['cells'][target+'_'+g]
        for j,name in enumerate(names):
            assert abs(reported['accuracy_pct'][name]-np.mean(accuracy[j])*100)<1e-10
        gain=int(((pred[1]==truth)&(pred[0]!=truth)).sum())
        loss=int(((pred[1]!=truth)&(pred[0]==truth)).sum())
        assert gain==reported['corrected_queries'] and loss==reported['harmed_queries']
        assert abs((gain-loss)/37500*100-reported['comparisons']['incumbent']['delta_pp'])<1e-10
        transitions[target+'_'+g]={'corrected':gain,'harmed':loss}
        deltas.append([accuracy[1]-accuracy[j] for j in [0,2,3]])
        accuracies.append(accuracy)
    for j,ref in enumerate(['incumbent','metric_ensemble','foreign_metrics']):
        vectors[target+'/'+ref]=(np.mean([x[j] for x in deltas],axis=0),z['seeds'])
    assert all(abs(result['domains'][target]['accuracy_pct'][n]-np.mean(accuracies,axis=0)[j].mean()*100)<1e-10 for j,n in enumerate(names))
audited={}
for ref in ['incumbent','metric_ensemble','foreign_metrics']:
    delta=np.concatenate([vectors[d+'/'+ref][0] for d in CFG['domains']])
    groups=np.concatenate([vectors[d+'/'+ref][1]+i*100000000 for i,d in enumerate(CFG['domains'])])
    rng=np.random.RandomState(CFG['bootstrap']['seed'])
    boot=np.zeros(CFG['bootstrap']['replicates'])
    group_ids=np.unique(groups)
    for group in group_ids:
        vals=delta[groups==group]
        indices=rng.randint(len(vals),size=(len(boot),len(vals)))
        for b,row in enumerate(indices):
            boot[b]+=math.fsum(float(vals[k]) for k in row)/len(vals)/len(group_ids)
    lower,upper=np.percentile(boot,[2.5,97.5])*100
    expected=result['pooled'][ref]
    assert abs(expected['delta_pp']-delta.mean()*100)<1e-10
    assert np.max(np.abs(np.array(expected['ci95_pp'])-[lower,upper]))<1e-10
    audited[ref]={'delta_pp':float(delta.mean()*100),'ci95_pp':[float(lower),float(upper)]}
assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())<CFG['resources']['max_outputs_mib']*2**20
report={'status':'passed','parent_task_prediction_checks':checks,
        'scalar_bootstrap_comparisons':audited,'query_transitions':transitions,
        'code_lock_verified':True,'scope':'all task arrays and scores; scalar bootstrap for three primary pooled comparisons; descriptive fixed pools'}
(OUT/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('INDEPENDENT_AUDIT_PASS',json.dumps(report),flush=True)
