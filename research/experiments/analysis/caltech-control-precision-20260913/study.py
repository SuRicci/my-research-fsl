"""Fixed float64 control repair; immutable historical comparison banks."""
from pathlib import Path
import datetime, hashlib, json, shutil, sys, time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; OUT=HERE/'outputs'
CFG=json.loads((HERE/'protocol.json').read_text())
sys.path.insert(0,str(ROOT/CFG['parent']))
import caltech_audit as ref
e=ref.e

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,x): (OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def guard():
    assert shutil.disk_usage(HERE).free>=CFG['min_free_gib']*2**30
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline'])
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())<CFG['output_limit_mib']*2**20
def norm(x): return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-12)

def ensemble(s,q,g):
    values=[]
    for v in range(CFG['views']):
        proto=norm(s[:,:,v].mean(1))
        ix=torch.argsort(proto@g[:,v].T,dim=-1,descending=True,stable=True)[:,:CFG['k']]
        x=norm(.5*proto+.5*norm(g[:,v][ix].mean(1)))
        mu=x.mean(0); xc=x-mu; y=torch.eye(5,dtype=torch.float64); yc=y-y.mean(0)
        coef=torch.linalg.solve(xc@xc.T+CFG['ridge']*torch.eye(5,dtype=torch.float64),yc)
        values.append((q[:,v]-mu)@xc.T@coef+y.mean(0))
    return torch.stack(values).mean(0).numpy()

def reference(s,q,g):
    s,q,g=[x.numpy() for x in [s,q,g]]
    return np.mean([ref.r2(s[:,:,v],q[:,v],g[:,v]) for v in range(CFG['views'])],axis=0)

def compare(s,q,g):
    got=ensemble(s,q,g); want=reference(s,q,g); err=float(abs(got-want).max())
    assert err<CFG['original_tolerance'] and np.array_equal(got.argmax(-1),want.argmax(-1)),err
    return {'max_error':err,'passed_original_tolerance':True,'passed_strict_diagnostic':err<CFG['diagnostic_tolerance'],'argmax_equal':True}

def interval(delta,seeds):
    rng=np.random.RandomState(CFG['bootstrap_seed']); val=np.zeros(CFG['bootstrap_replicates'])
    for group in np.unique(seeds):
        d=delta[seeds==group]
        val+=d[rng.randint(len(d),size=(len(val),len(d)))].mean(1)/len(np.unique(seeds))
    return {'delta_pp':float(delta.mean()*100),'ci95_pp':(np.percentile(val,[2.5,97.5])*100).tolist()}

def summarize():
    result={'cells':{},'sources':{},'repair_vs_old':{},'scope':CFG['scope']}; vectors={}; paired={}
    for gallery in ['caltech101','dtd','eurosat']:
        b=np.load(OUT/(gallery+'.npz')); sc=b['scores']; pr=b['predictions']; y=b['yq']
        assert sc.shape==(500,75,5) and np.isfinite(sc).all()
        assert np.array_equal(sc.argmax(-1),pr)
        acc=(pr==y).mean(-1); old=(b['old_predictions']==y).mean(-1)
        assert np.array_equal(acc,b['accuracy'])
        comparisons={'repair_minus_old':interval(acc-old,b['seeds'])}
        for source in ['dtd','eurosat']:
            primary=(b[source+'_primary_predictions']==y).mean(-1)
            comparisons[source+'_primary_minus_repair']=interval(primary-acc,b['seeds'])
            paired.setdefault(source,[]).append(primary-acc)
        result['cells'][gallery]={'accuracy_pct':float(acc.mean()*100),'old_accuracy_pct':float(old.mean()*100),'prediction_changes':int((pr!=b['old_predictions']).sum()),'comparisons':comparisons}
        vectors[gallery]=acc-old
    result['repair_vs_old']=interval(np.mean(list(vectors.values()),0),b['seeds'])
    for source,deltas in paired.items(): result['sources'][source]=interval(np.mean(deltas,0),b['seeds'])
    result['prediction_changes']=sum(x['prediction_changes'] for x in result['cells'].values())
    return result

def main():
    guard();torch.set_num_threads(1);start=time.time()
    e.check_lock(e.HERE/'design_lock.json');e.check_lock(e.HERE/'evaluation_code_lock.json')
    original_failure_sha=sha(e.OUT/'audit_failure.json')
    code={str(p):sha(p) for p in [HERE/'study.py',HERE/'verify.py',HERE/'protocol.json',e.HERE/'caltech_eval.py',e.HERE/'caltech_audit.py']}
    dump('code_lock.json',code);assert not any(OUT.glob('*.npz'))
    rng=torch.Generator().manual_seed(260913105)
    s,q,g=[norm(torch.randn(*shape,generator=rng,dtype=torch.float64)) for shape in [(5,1,6,32),(7,6,32),(90,6,32)]]
    checks=[{'case':'synthetic',**compare(s,q,g)}]; actual=ensemble(s,q,g); cp=[2,0,4,1,3];qp=[3,0,6,1,2,5,4]
    assert np.allclose(ensemble(s[cp],q,g),actual[:,cp],atol=1e-10,rtol=0)
    assert np.allclose(ensemble(s,q[qp],g),actual[qp],atol=1e-10,rtol=0)
    assert np.allclose(np.concatenate([ensemble(s,q[i:i+1],g) for i in range(len(q))]),actual,atol=1e-10,rtol=0)
    original_dump=e.dump
    try:
        e.dump=lambda path,value: dump('feature_validation.json',value)
        target,galleries,tasks=e.load_data()
    finally:
        e.dump=original_dump
    outputs={};inputs={}
    for gallery,gv in galleries.items():
        gv=gv.double();values=[]
        for i in range(500):
            guard();s=target[tasks['support_indices'][i]].double();q=target[tasks['query_indices'][i].reshape(-1)].double()
            values.append(ensemble(s,q,gv))
            if i in CFG['audit_tasks']: checks.append({'case':gallery,'task':i,**compare(s,q[CFG['audit_queries']],gv)})
            if i%100==0:print('REPAIR',gallery,i,round(time.time()-start,2),flush=True)
        sc=np.stack(values);pr=sc.argmax(-1).astype(np.uint8);extra={}
        for source in ['dtd','eurosat']:
            path=e.OUT/(source+'_'+gallery+'.npz');inputs[str(path)]=sha(path);old=np.load(path)
            assert all(np.array_equal(old[k],tasks[k]) for k in e.KEYS)
            j=old['names'].tolist().index(CFG['method']);oldpr=old['predictions'][j]
            if source=='dtd': control=oldpr
            else: assert np.array_equal(control,oldpr)
            extra[source+'_primary_predictions']=old['predictions'][old['names'].tolist().index('query_consistency')]
        dest=OUT/(gallery+'.npz')
        np.savez_compressed(dest,scores=sc,predictions=pr,accuracy=(pr==e.Y).mean(-1),yq=e.Y,old_predictions=control,**extra,**{k:tasks[k] for k in e.KEYS})
        outputs[str(dest)]=sha(dest)
    analysis=summarize();dump('analysis.json',analysis)
    assert all(sha(p)==h for p,h in code.items()) and all(sha(p)==h for p,h in inputs.items())
    assert sha(e.OUT/'audit_failure.json')==original_failure_sha
    dump('audit.json',{'status':'passed','checks':checks,'max_error':max(x['max_error'] for x in checks),'input_hashes':inputs,'output_hashes':outputs,'historical_failure_sha256':original_failure_sha,'invariances':'class,query order andquery partition'})
    dump('complete.json',{'status':'completed','seconds':time.time()-start,'unique_tasks':500,'task_conditions':1500,'prediction_entries':112500,'torch':torch.__version__,'numpy':np.__version__,'new_output_mib':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())/2**20,'free_gib':shutil.disk_usage(HERE).free/2**30})
    print('COMPLETE',json.dumps(analysis),flush=True)
if __name__=='__main__':main()
