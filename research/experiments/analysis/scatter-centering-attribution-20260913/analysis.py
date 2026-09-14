"""Counterfactual neighbor-set x feature-geometry attribution, no tuning."""
from pathlib import Path
import sys,json,time,hashlib,shutil,datetime
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
PARENT=ROOT/'experiments/main/scatter-centering-stack-20260913'
sys.path.insert(0,str(PARENT));import model as m
sys.path.insert(0,str(PARENT));from summarize import interval
CFG=json.loads((HERE/'protocol.json').read_text());PCFG=json.loads((PARENT/'protocol.json').read_text())
OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)
NAMES=['raw_raw','raw_center','center_raw','center_center']

def dump(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
def norm(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)

def heads(s,q,g):
    mu=s.flatten(0,1).mean(0)
    packs=[(s,q,g),(m.n(s-mu),m.n(q-mu),m.n(g-mu))]
    indices=[torch.argsort(m.n(a.mean(1))@c.T,descending=True,stable=True)[:,:64] for a,b,c in packs]
    result=[]
    for f in [0,1]:
        a,b,c=packs[f];proto=m.n(a.mean(1))
        for r in [0,1]:
            x=m.n(.5*proto+.5*m.n(c[indices[r]].mean(1)));y=torch.eye(5,dtype=x.dtype)
            xc=x-x.mean(0);yc=y-y.mean(0)
            dual=torch.linalg.solve(xc@xc.T+.1*torch.eye(5,dtype=x.dtype),yc)
            result.append((b-x.mean(0))@xc.T@dual+y.mean(0))
    return torch.stack(result),indices

def independent(sv,qv,gm,gamma):
    s=sv.numpy().astype(float);q=qv.numpy().astype(float);g=gm.numpy().astype(float)
    d=(s-s.mean(-2,keepdims=True)).reshape(-1,s.shape[-1]);cov=d.T@d/len(d)
    ev,u=np.linalg.eigh(np.eye(len(cov))+gamma*len(cov)*cov/np.trace(cov));transform=(u/np.sqrt(ev))@u.T
    s=norm(s.mean(-2)@transform);q=norm(q.mean(-2)@transform);g=norm(g@transform);mu=s.reshape(-1,s.shape[-1]).mean(0)
    packs=[(s,q,g),(norm(s-mu),norm(q-mu),norm(g-mu))]
    indices=[np.argsort(-(norm(a.mean(1))@c.T),axis=-1,kind='stable')[:,:64] for a,b,c in packs]
    result=[]
    for f in [0,1]:
        a,b,c=packs[f];proto=norm(a.mean(1))
        for r in [0,1]:
            x=norm(.5*proto+.5*norm(c[indices[r]].mean(1)));y=np.eye(5);xc=x-x.mean(0)
            weights=np.linalg.solve(xc.T@xc+.1*np.eye(x.shape[-1]),xc.T@(y-y.mean(0)))
            result.append((b-x.mean(0))@weights+y.mean(0))
    return np.stack(result),indices

def summarize():
    cells={};domain={};vec={}
    contrasts={'retrieval_given_raw':(1,0),'geometry_given_raw_neighbors':(2,0),'retrieval_given_center':(3,2),'geometry_given_center_neighbors':(3,1),'total':(3,0)}
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        accs=[]
        for gallery in [target,source]:
            name=target+'_'+gallery;z=np.load(OUT/(name+'.npz'));a=(z['predictions']==z['yq']).mean(-1)
            assert np.array_equal(a,z['accuracy']);accs.append(a);seeds=z['seeds']
            co={k:interval(a[i]-a[j],seeds) for k,(i,j) in contrasts.items()}
            cells[name]={'accuracy_pct':dict(zip(NAMES,(a.mean(-1)*100).tolist())),'effects':co,'mean_neighbor_overlap':float(z['neighbor_overlap'].mean())}
        a=np.mean(accs,0);effects={}
        for k,(i,j) in contrasts.items():
            delta=a[i]-a[j];effects[k]=interval(delta,seeds);vec[target+'/'+k]=(delta,seeds)
        domain[target]={'accuracy_pct':dict(zip(NAMES,(a.mean(-1)*100).tolist())),'effects':effects}
    pooled={}
    for k in contrasts:
        delta=np.concatenate([vec[d+'/'+k][0] for d in ['dtd','eurosat']]);groups=np.concatenate([vec[d+'/'+k][1]+i*100000000 for i,d in enumerate(['dtd','eurosat'])])
        pooled[k]=interval(delta,groups)
    return {'cells':cells,'domains':domain,'pooled':pooled}

def main():
    start=time.time();torch.set_num_threads(CFG['resources']['cpu_threads'])
    for lockpath in [HERE/'code_lock.json',PARENT/'code_lock.json']:
        assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in json.loads(lockpath.read_text()).items())
    data=m.old.load();gm={d:data[d]['gallery'].double().mean(-2) for d in CFG['domains']};checks=[];parity=0
    dump(OUT/'manifest.json',{'argv':[sys.executable,*sys.argv],'protocol':CFG,'parent':'report-f208e67c','torch':torch.__version__,'numpy':np.__version__,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        t=m.old.sampler.tasks(data[target]['ident'],1,'eval');gamma=PCFG['gamma_by_source'][source]['1'];galleries=[target,source]
        old={g:np.load(PARENT/'outputs'/f'{source}_to_{target}_{g}_k1.npz') for g in galleries}
        preds={g:[] for g in galleries};samples={g:[] for g in galleries};overlap={g:[] for g in galleries}
        for g,z in old.items():
            for key in t:assert np.array_equal(t[key],z[key])
        for i in range(500):
            assert shutil.disk_usage(HERE).free/2**30>=10
            assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['resources']['deadline'])
            sv=data[target]['query'][t['support_indices'][i]];qv=data[target]['query'][t['query_indices'][i].reshape(-1)]
            _,(s,q),factor=m.prepare(sv,qv,gamma)
            for g in galleries:
                gal=m.old.metric.transform(gm[g],factor);sc,indices=heads(s,q,gal);pred=sc.argmax(-1).numpy().astype(np.uint8)
                for j,ref in [(0,'scatter_r2'),(3,'scatter_cs')]:
                    assert np.array_equal(pred[j],old[g]['predictions'][m.NAMES.index(ref),i]),(target,g,i,ref)
                    parity+=1
                preds[g].append(pred);overlap[g].append(np.mean([len(set(a.tolist())&set(b.tolist()))/64 for a,b in zip(*indices)]))
                if i in CFG['audit_indices']:
                    expected,ix=independent(sv,qv,gm[g],gamma)
                    assert all(np.array_equal(x.numpy(),y) for x,y in zip(indices,ix))
                    for j,name in enumerate(NAMES):
                        err=float(abs(sc[j].numpy()-expected[j]).max());assert err<CFG['score_tolerance'] and np.array_equal(pred[j],expected[j].argmax(-1)),(target,g,i,name,err)
                        checks.append({'domain':target,'gallery':g,'task':i,'method':name,'max_error':err})
                    samples[g].append(sc.numpy())
            if i%100==0:
                x={'domain':target,'tasks_done':i+1,'elapsed_seconds':time.time()-start};dump(OUT/'progress.json',x);print('PROGRESS',json.dumps(x),flush=True)
        y=np.repeat(np.arange(5),15)
        for g in galleries:
            pred=np.stack(preds[g],1);np.savez_compressed(OUT/(target+'_'+g+'.npz'),predictions=pred,accuracy=(pred==y).mean(-1),names=NAMES,yq=y,sample_scores=np.stack(samples[g],1),sample_indices=CFG['audit_indices'],neighbor_overlap=overlap[g],**t)
            print('CELL_COMPLETE',target,g,flush=True)
    result=summarize();dump(OUT/'analysis.json',result)
    dump(OUT/'validation.json',{'status':'passed','exact_parent_endpoint_task_checks':parity,'independent_score_checks':len(checks),'max_score_error':max(x['max_error'] for x in checks),'checks':checks,'scope':'fulltaskendpoints and80sampled independenthybrid scores; pairedinterval helper shared with already auditedparent'})
    dump(OUT/'complete.json',{'status':'completed','elapsed_seconds':time.time()-start,'task_conditions':2000,'methods':4,'output_bytes':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())})
    print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
