"""Fixed multi-depth readout; no new training or source/target selection."""
from pathlib import Path
import argparse, datetime, hashlib, importlib.util, json, os, shutil, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; OUT=HERE/'outputs'; ASSET=HERE/'assets'
CFG=json.loads((HERE/'protocol.json').read_text()); NAMES=CFG['methods']; Y=np.repeat(np.arange(5),15)
REF=HERE.parent/'query-consistency-20260913/outputs'
spec=importlib.util.spec_from_file_location('depth_retained',REF.parent/'consistency_model.py')
z=importlib.util.module_from_spec(spec);spec.loader.exec_module(z)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)
def guard(reserve=0):
    assert shutil.disk_usage(HERE).free>=CFG['min_free_gib']*2**30+reserve,'disk floor'
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline']),'deadline'
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<CFG['max_new_mib']*2**20,'output cap'
def duplicate(v):
    return torch.cat([v[...,:512],v[...,512:].repeat(*([1]*(v.ndim-1)),4)/2],-1)
def representation(v,early,kind):
    v=v.double()
    if kind=='final':return v
    if kind=='duplicate':return duplicate(v)
    e=F.normalize(torch.from_numpy(np.array(early)).double(),dim=-1)*v[...,512:].norm(dim=-1,keepdim=True)[...,None]
    if isinstance(kind,int):
        return v if kind==3 else torch.cat([v[...,:512],e[...,kind,:]],-1)
    assert kind=='concat'
    return torch.cat([v[...,:512],torch.cat([e.reshape(*v.shape[:-1],-1),v[...,512:]],-1)/2],-1)
def retained(s,q,g,gamma,eta):
    return z.evaluate(s,q,g,gamma*CFG['base_dimension']/s.shape[-1],[eta],[0.])
def load():
    data=z.m.old.load();manifest=json.loads((ASSET/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['images']==CFG['expected_images']
    maps={}
    for key,row in manifest['pools'].items():
        p=ASSET/(key+'.npy');assert sha(p)==row['sha256'];maps[key]=np.load(p,mmap_mode='r')
    return data,maps
def take(data,maps,ds,side,indices,kind):
    return representation(data[ds][side][indices],None if kind in ['final','duplicate'] else maps[ds+'_'+side][indices],kind)
def gallery(data,maps,ds,kind):
    return torch.cat([take(data,maps,ds,'gallery',slice(i,i+128),kind).mean(-2) for i in range(0,len(data[ds]['gallery']),128)])
def independent_ridge(x,q,weight,penalty):
    y=np.eye(5).repeat(len(x)//5,0);sw=weight.sum();xm=(x*weight[:,None]).sum(0)/sw;ym=(y*weight[:,None]).sum(0)/sw
    a=(x-xm)*np.sqrt(weight)[:,None];b=(y-ym)*np.sqrt(weight)[:,None]
    return (q-xm)@a.T@np.linalg.solve(a@a.T+penalty*np.eye(len(x)),b)+ym
def interval(delta,groups):
    rng=np.random.RandomState(CFG['bootstrap']['seed']);out=np.zeros(CFG['bootstrap']['replicates'])
    for seed in np.unique(groups):
        v=delta[groups==seed];out+=v[rng.randint(len(v),size=(len(out),len(v)))].sum(1)/len(delta)
    return {'delta_pp':float(delta.mean()*100),'ci95_pp':(np.percentile(out,[2.5,97.5])*100).tolist()}
def analyze():
    domains={};cells={};allacc=[];allgroups=[]
    for ds in CFG['domains']:
        aa=[]
        for gal in CFG['domains']:
            f=np.load(OUT/f'{ds}_{gal}.npz');a=(f['predictions']==Y).mean(-1);aa.append(a)
            cells[ds+'_'+gal]={'accuracy_pct':dict(zip(NAMES,(a.mean(-1)*100).tolist())),'vs_final':interval(a[0]-a[1],f['seeds'])}
        a=np.mean(aa,0);allacc.append(a);allgroups.append(f['seeds']+len(allgroups)*100000000)
        domains[ds]={'accuracy_pct':dict(zip(NAMES,(a.mean(-1)*100).tolist())),'comparisons':{n:interval(a[0]-a[NAMES.index(n)],f['seeds']) for n in CFG['gate_controls']}}
    a=np.concatenate(allacc,1);groups=np.concatenate(allgroups)
    pooled={n:interval(a[0]-a[NAMES.index(n)],groups) for n in CFG['gate_controls']}
    gate=all(x['delta_pp']>=.5 and x['ci95_pp'][0]>0 for x in pooled.values())
    gate &= all(x['delta_pp']>=0 for d in domains.values() for x in d['comparisons'].values())
    gate &= all(x['vs_final']['ci95_pp'][0]>=-.5 for x in cells.values())
    result={'status':'computed','gate_passed':bool(gate),'domains':domains,'cells':cells,'pooled':pooled,'unique_tasks':1000,'task_gallery_conditions':2000,'methods':NAMES,'scope':CFG['metric_boundary'],'audit_pending':True}
    dump(OUT/'analysis.json',result);return result
def immutable():
    paths=[HERE/f for f in ['protocol.json','study.py','audit.py','encode.py']]
    paths+=[ROOT/'baselines/local/r2-pets-transfer/json/metric_contract.json',ASSET/'manifest.json']
    paths+=[REF/f'{d}_{g}.npz' for d in CFG['domains'] for g in CFG['domains']]
    for m in list(sys.modules.values()):
        f=getattr(m,'__file__',None)
        if f and str(ROOT/'experiments') in str(f) and Path(f).suffix=='.py':paths.append(Path(f))
    return {str(p):sha(p) for p in sorted(set(paths))}
def run():
    guard();assert json.loads((OUT/'precheck.json').read_text())['status']=='passed'
    assert not (OUT/'run_manifest.json').exists(),'Adopt existing run'
    torch.set_num_threads(CFG['threads']);data,maps=load();lock=immutable();dump(OUT/'code_lock.json',lock)
    dump(OUT/'run_manifest.json',{'config':CFG,'command':[sys.executable,*sys.argv],'input_hashes':lock,'started':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    start=time.time()
    for target in CFG['domains']:
        source=next(d for d in CFG['domains'] if d!=target);gamma=CFG['source_gamma'][source];eta=CFG['source_eta'][source]
        for gal in CFG['domains']:
            bank=np.load(REF/f'{target}_{gal}.npz');g={k:gallery(data,maps,gal,k) for k in ['final','concat',0,1,2,3]}
            pred=[];audit=[];null=[]
            for i in range(CFG['eval_tasks']):
                guard();si=bank['support_indices'][i];qi=bank['query_indices'][i].reshape(-1);sv=data[target]['query'][si].double();qv=data[target]['query'][qi].double()
                base=retained(sv,qv,g['final'],gamma,eta)
                for key in ['incumbent','consistency']:
                    actual=base[key][0] if key=='consistency' else base[key]
                    expected=bank['scores'][bank['names'].tolist().index(key),i]
                    assert np.max(abs(actual.numpy()-expected))<1e-10
                s=take(data,maps,target,'query',si,'concat');q=take(data,maps,target,'query',qi,'concat');depth=retained(s,q,g['concat'],gamma,eta)
                perlayer=[retained(take(data,maps,target,'query',si,k),take(data,maps,target,'query',qi,k),g[k],gamma,eta)['consistency'][0] for k in range(4)]
                sm=F.normalize(s.mean(-2),dim=-1).reshape(5,-1).numpy();qm=F.normalize(q.mean(-2),dim=-1).numpy()
                logistic=[]
                for penalty in [1,10]:
                    clf=LogisticRegression(C=penalty,solver='lbfgs',multi_class='multinomial',max_iter=2000,tol=1e-8,random_state=26091275)
                    clf.fit(sm,np.arange(5));assert clf.n_iter_.max()<2000
                    logistic.append(clf.decision_function(qm))
                aug=independent_ridge(s.flatten(0,2).numpy(),qm,np.full(30,1/6),.1)
                scores=np.stack([depth['consistency'][0].numpy(),base['consistency'][0].numpy(),base['incumbent'].numpy(),torch.stack(perlayer).mean(0).numpy(),*logistic,aug,depth['incumbent'].numpy()])
                assert np.isfinite(scores).all();pred.append(scores.argmax(-1).astype(np.uint8))
                if i in CFG['audit_indices']:
                    dup=retained(duplicate(sv),duplicate(qv),duplicate(g['final']),gamma,eta)['consistency'][0]
                    err=float((dup-base['consistency'][0]).abs().max());assert err<1e-10
                    null.append(err);audit.append(scores)
                if i%50==0:print('EVAL',target,gal,i,'seconds',round(time.time()-start,1),flush=True)
            np.savez_compressed(OUT/f'{target}_{gal}.npz',predictions=np.stack(pred,1),audit_scores=np.stack(audit,1),names=np.array(NAMES),yq=Y,**{k:bank[k] for k in ['support_indices','query_indices','class_ids','seeds']})
            dump(OUT/f'{target}_{gal}_null.json',{'max_error':max(null),'samples':CFG['audit_indices']})
            print('CELL_COMPLETE',target,gal,flush=True)
    result=analyze();assert immutable()==lock
    dump(OUT/'complete.json',{'status':'computed','gate_passed':result['gate_passed'],'seconds':time.time()-start,'audit_pending':True})
    print('COMPUTE_COMPLETE',result['gate_passed'],flush=True)
if __name__=='__main__':run()
