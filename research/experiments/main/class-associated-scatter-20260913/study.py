"""Frozen auxiliary comparison of class association, ensemble and foreign metrics."""
from pathlib import Path
import json, sys, time, datetime, hashlib, shutil
import numpy as np
import torch
import class_metric as m
HERE = Path(__file__).resolve().parent
OUT = HERE/'outputs'
CFG = json.loads((HERE/'protocol.json').read_text())

def dump(p,x):
    p.write_text(json.dumps(x,indent=2)+'\n')

def guard():
    assert shutil.disk_usage(HERE).free/2**30 >= CFG['resources']['free_gib_min']
    assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(CFG['resources']['deadline'])

def interval(delta, groups):
    rng = np.random.RandomState(CFG['bootstrap']['seed'])
    group_ids = np.unique(groups)
    values = np.zeros(CFG['bootstrap']['replicates'])
    for group in group_ids:
        x=delta[groups==group]
        values += x[rng.randint(len(x),size=(len(values),len(x)))].mean(1)/len(group_ids)
    return {'delta_pp':float(delta.mean()*100),'ci95_pp':(np.quantile(values,[.025,.975])*100).tolist()}

def summarize():
    cells, domains, vectors = {}, {}, {}
    for target in CFG['domains']:
        pieces=[]
        for g in CFG['domains']:
            z=np.load(OUT/(target+'_'+g+'.npz'))
            sc=z['scores']; pred=sc.argmax(-1);acc=(pred==z['yq']).mean(-1)
            assert np.array_equal(pred,z['predictions']) and np.array_equal(acc,z['accuracy'])
            assert np.isfinite(sc).all()
            bank=z['metric_scores']
            independent=np.stack([bank[c,:,:,c] for c in range(5)],-1)
            assert np.allclose(sc[1],independent,atol=1e-12,rtol=0)
            assert np.allclose(sc[2],bank.mean(0),atol=1e-12,rtol=0)
            foreign=np.stack([np.delete(bank[:,:,:,c],c,axis=0).mean(0) for c in range(5)],-1)
            assert np.allclose(sc[3],foreign,atol=1e-12,rtol=0)
            inc_correct=pred[0]==z['yq']; new_correct=pred[1]==z['yq']
            comparisons={m.NAMES[j]:interval(acc[1]-acc[j],z['seeds']) for j in [0,2,3,4,5,6]}
            cells[target+'_'+g]={'accuracy_pct':dict(zip(m.NAMES,(acc.mean(-1)*100).tolist())),
                'comparisons':comparisons,'corrected_queries':int((~inc_correct&new_correct).sum()),
                'harmed_queries':int((inc_correct&~new_correct).sum()),
                'mean_class_scatter_trace_ratios':z['trace_ratios'].mean(0).tolist(),
                'mean_associated_score_sum':float(sc[1].sum(-1).mean())}
            pieces.append(acc);seeds=z['seeds']
        acc=np.mean(pieces,0)
        comparisons={m.NAMES[j]:interval(acc[1]-acc[j],seeds) for j in [0,2,3,4,5,6]}
        domains[target]={'accuracy_pct':dict(zip(m.NAMES,(acc.mean(-1)*100).tolist())),'comparisons':comparisons}
        for j in [0,2,3,4,5,6]:vectors[target+'/'+m.NAMES[j]]=(acc[1]-acc[j],seeds)
    pooled={}
    for ref in [m.NAMES[j] for j in [0,2,3,4,5,6]]:
        delta=np.concatenate([vectors[d+'/'+ref][0] for d in CFG['domains']])
        groups=np.concatenate([vectors[d+'/'+ref][1]+i*100000000 for i,d in enumerate(CFG['domains'])])
        pooled[ref]=interval(delta,groups)
    tests={'increment':pooled['incumbent']['delta_pp']>=CFG['gate']['gain_vs_incumbent_pp'],
        'incumbent_ci':pooled['incumbent']['ci95_pp'][0]>0,
        'class_association':all(pooled[x]['ci95_pp'][0]>0 for x in CFG['gate']['class_association_comparators']),
        'domains':all(d['comparisons']['incumbent']['delta_pp']>=0 for d in domains.values()),
        'cells':all(c['comparisons']['incumbent']['ci95_pp'][0]>=-.5 for c in cells.values())}
    result={'scope':CFG['scope'],'cells':cells,'domains':domains,'pooled':pooled,'gate':tests,'promotion':all(tests.values()),
            'dev_macro_1shot_accuracy':float(np.mean([d['accuracy_pct']['class_associated'] for d in domains.values()]))}
    dump(OUT/'analysis.json',result)
    return result

def main():
    start=time.time();torch.set_num_threads(CFG['resources']['threads']);guard()
    assert json.loads((OUT/'numeric_validation.json').read_text())['status']=='passed'
    locks=json.loads((HERE/'code_lock.json').read_text())
    assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in locks.items())
    data=m.base.old.load();gm={d:data[d]['gallery'].double().mean(-2) for d in CFG['domains']}
    dump(OUT/'manifest.json',{'argv':[sys.executable,*sys.argv],'protocol':CFG,'code_hashes':locks,
        'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'torch':torch.__version__,'numpy':np.__version__})
    parent_checks=0
    for target in CFG['domains']:
        tasks=m.base.old.sampler.tasks(data[target]['ident'],1,'eval')
        old={g:np.load(m.PARENT/'outputs'/(target+'_'+g+'.npz')) for g in CFG['domains']}
        for g,z in old.items():
            for k in tasks:assert np.array_equal(tasks[k],z[k])
            assert not set(data[target]['ident']['query_rgb'])&set(data[g]['ident']['gallery_rgb'])
        scores={g:[] for g in gm};banks={g:[] for g in gm};sample_ix={g:[] for g in gm};ratios=[]
        for i in range(len(tasks['seeds'])):
            guard()
            sv=data[target]['query'][tasks['support_indices'][i]]
            qv=data[target]['query'][tasks['query_indices'][i].reshape(-1)]
            result=m.evaluate(sv,qv,gm,CFG['gamma_by_target'][target],CFG['shrinkage'])
            residual=sv.double()-sv.double().mean(-2,keepdim=True)
            trace=residual.square().sum(-1).mean((1,2))
            ratios.append((trace/trace.mean()).numpy())
            for g,(sc,bank,ix) in result.items():
                assert torch.isfinite(sc).all()
                pred=sc.argmax(-1).numpy()
                for j,parent_j in [(0,2),(4,0),(5,1),(6,3)]:
                    assert np.array_equal(pred[j],old[g]['predictions'][parent_j,i]),(target,g,i,j)
                    parent_checks+=1
                scores[g].append(sc.numpy());banks[g].append(bank.numpy())
                if i in CFG['audit_indices']:sample_ix[g].append(ix.numpy())
            if i%50==0:
                progress={'target':target,'tasks_done':i+1,'elapsed_seconds':time.time()-start}
                dump(OUT/'progress.json',progress);print('PROGRESS',json.dumps(progress),flush=True)
        y=np.repeat(np.arange(5),15)
        for g in gm:
            sc=np.stack(scores[g],1);pred=sc.argmax(-1).astype(np.uint8)
            np.savez_compressed(OUT/(target+'_'+g+'.npz'),scores=sc,predictions=pred,accuracy=(pred==y).mean(-1),
                metric_scores=np.stack(banks[g],1),sample_neighbors=np.stack(sample_ix[g],1),
                sample_indices=CFG['audit_indices'],trace_ratios=np.stack(ratios),names=m.NAMES,yq=y,**tasks)
            print('CELL_COMPLETE',target,g,flush=True)
    result=summarize()
    dump(OUT/'validation.json',{'status':'passed','parent_prediction_checks':parent_checks,
        'full_score_aggregation_checks':True,'numeric_validation':'numeric_validation.json',
        'method_scope':'all task arrays, every prediction, independent aggregation and same paired interval recipe'})
    elapsed=time.time()-start
    dump(OUT/'complete.json',{'status':'completed','elapsed_seconds':elapsed,'task_conditions':2000,
        'promotion':result['promotion'],'output_bytes':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())})
    print('COMPLETE',json.dumps({'elapsed_seconds':elapsed,'accuracy':result['dev_macro_1shot_accuracy'],'pooled':result['pooled'],'gate':result['gate']}),flush=True)
if __name__ == '__main__':
    main()
