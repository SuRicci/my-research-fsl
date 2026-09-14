"""Independent recorded-result audit: dense algebra, full tasks, bootstrap weights."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
import model as m
from verify import dense
from summarize import collect,verdict
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'
CFG=json.loads((HERE/'protocol.json').read_text())

def independent_interval(delta,groups):
    rng=np.random.RandomState(CFG['bootstrap']['seed']);bs=np.zeros(CFG['bootstrap']['replicates']);values=sorted(set(groups.tolist()))
    for group in values:
        x=delta[groups==group];idx=rng.randint(len(x),size=(len(bs),len(x)))
        # Count bootstrap multiplicities, then take an empirical weighted mean.
        weights=np.zeros((len(bs),len(x)),dtype=np.int16)
        np.add.at(weights,(np.arange(len(bs))[:,None],idx),1)
        bs+=weights@x/len(x)/len(values)
    return np.quantile(bs,[.025,.975])*100

def main():
    start=time.time();torch.set_num_threads(1)
    lock=json.loads((HERE/'code_lock.json').read_text());assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in lock.items())
    saved=json.loads((OUT/'analysis.json').read_text());data=m.old.load();checks=[];history={};count=0
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for shot in [1,5]:
            task=m.old.sampler.tasks(data[target]['ident'],shot,'eval');previous=None
            for gallery in [target,source]:
                name=f'{source}_to_{target}_{gallery}_k{shot}';z=np.load(OUT/(name+'.npz'));old=np.load(m.old.OUT/'cells'/(name+'.npz'))
                assert len(z['seeds'])==500 and z['predictions'].shape==(6,500,75)
                for key in task:assert np.array_equal(task[key],z[key]) and np.array_equal(z[key],old[key])
                assert z['names'].tolist()==m.NAMES
                assert np.array_equal(z['sample_indices'],CFG['audit_task_indices'])
                assert np.array_equal(z['yq'],np.repeat(np.arange(5),15))
                assert np.array_equal(z['accuracy'],(z['predictions']==z['yq']).mean(-1))
                for mname in m.NAMES:
                    a=100*(z['predictions'][m.NAMES.index(mname)]==z['yq']).mean()
                    assert abs(a-saved['cells'][name]['accuracy_pct'][mname])<1e-10
                gamma=CFG['gamma_by_source'][source][str(shot)];assert z['chosen_gamma']==gamma
                gm=data[gallery]['gallery'].double().mean(-2)
                for pos,i in enumerate(CFG['audit_task_indices']):
                    s=data[target]['query'][z['support_indices'][i]];q=data[target]['query'][z['query_indices'][i].reshape(-1)]
                    for j,method in enumerate(m.NAMES):
                        lam=.1 if shot==1 or method.endswith(('cs','lam01')) else 1.
                        expected,ix=dense(s,q,gm,gamma if method.startswith('scatter') else 0,method.endswith('cs'),lam)
                        error=float(abs(z['sample_scores'][j,pos]-expected).max())
                        assert error<2e-5 and np.array_equal(expected.argmax(-1),z['predictions'][j,i]),(name,i,method,error)
                        assert np.array_equal(ix,z['sample_neighbors'][j,pos]),(name,i,method,'neighbor')
                        checks.append({'cell':name,'task':i,'method':method,'max_error':error})
                h={}
                for method,prior in [('mean_r2','mean_r2'),('mean_cs','mean_CS_l2'),('scatter_r2','scatter_r2')]:
                    a=z['predictions'][m.NAMES.index(method)];b=old['predictions'][old['names'].tolist().index(prior)]
                    h[method]={'prediction_changes':int((a!=b).sum()),'accuracy_delta_pp':float(100*((a==z['yq']).mean()-(b==z['yq']).mean()))}
                history[name]=h;count+=500
                if shot==5:
                    if previous is not None:assert np.array_equal(previous,z['predictions'])
                    previous=z['predictions']
                print('AUDITED',name,flush=True)
    cells,domains,pooled,vectors=collect();assert cells==saved['cells'] and domains==saved['domains'] and pooled==saved['pooled'];assert verdict(cells,domains,pooled)==saved['gate']
    max_ci=0
    for key,(delta,groups) in vectors.items():
        level,rest=key.split('/',1)
        reported=pooled[rest] if level=='pooled' else (cells if level=='cell' else domains)[rest.rsplit('/',1)[0]]['comparisons'][rest.rsplit('/',1)[1]]
        assert abs(delta.sum()/len(delta)*100-reported['delta_pp'])<1e-10
        err=float(abs(independent_interval(delta,groups)-reported['ci95_pp']).max());assert err<1e-10,(key,err);max_ci=max(max_ci,err)
    result={'status':'passed','task_conditions':count,'independent_score_checks':len(checks),'max_score_error':max(v['max_error'] for v in checks),'exact_sampled_neighbors_and_predictions':True,'statistic_checks':len(vectors),'independent_bootstrap_max_error_pp':max_ci,'historical_precision_comparison':history,'checks':checks,'elapsed_seconds':time.time()-start,'scope':'alltask/aggregate and fixed240dense scorechecks; no claim of everyquery score reconstructed'}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2),flush=True)
if __name__=='__main__':main()
