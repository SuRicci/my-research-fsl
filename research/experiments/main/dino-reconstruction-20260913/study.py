"""Fixed normalized reconstruction transfer; source development only."""
from pathlib import Path
import datetime, hashlib, json, shutil, time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]; OUT=HERE/'outputs'
CFG=json.loads((HERE/'protocol.json').read_text()); NAMES=CFG['methods']; EPS=CFG['residual_epsilon']
ASSET=ROOT/CFG['source_asset']; RAW=ROOT/CFG['raw_bank']; PARENT=ROOT/CFG['parent_bank']
KEYS=['support_indices','query_indices','class_ids','seeds','yq']

def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def guard():
    assert shutil.disk_usage(HERE).free>=CFG['min_free_gib']*2**30
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline'])
def norm(x):
    size=x.norm(dim=-1,keepdim=True)
    return torch.where(size>EPS,x/size.clamp_min(EPS),torch.zeros_like(x))
def standard(x): return (x-x.mean(-1,keepdims=True))/np.maximum(x.std(-1,keepdims=True),EPS)
def shuffle(rgb):
    order=np.argsort(rgb); out=np.empty(len(rgb),dtype=int); out[order]=np.roll(order,1); return out

def parts(q,s):
    q=norm(torch.as_tensor(np.array(q),dtype=torch.float64))
    s=norm(torch.as_tensor(np.array(s),dtype=torch.float64)).flatten(1,2)
    rank1=norm(s.mean(1,keepdim=True)).expand_as(s)
    def reconstruct(support):
        lam=support.shape[1]/support.shape[2]+CFG['ridge_epsilon']
        gram=support@support.transpose(-1,-2)
        weights=torch.linalg.solve(gram+lam*torch.eye(gram.shape[-1],dtype=torch.float64),support)
        qflat=q.flatten(0,1)
        qhat=(qflat@support.transpose(-1,-2))@weights
        return (-(qhat-qflat).square().sum(-1).reshape(5,len(q),q.shape[1]).mean(-1)).T.numpy()
    return reconstruct(s),reconstruct(rank1)

def reference(q,s):
    def unit(x):
        n=np.linalg.norm(x,axis=-1,keepdims=True)
        return np.where(n>EPS,x/np.maximum(n,EPS),0.)
    q=unit(np.asarray(q,dtype=np.float64));s=unit(np.asarray(s,dtype=np.float64)).reshape(5,-1,q.shape[-1])
    rank1=np.repeat(unit(s.mean(1,keepdims=True)),s.shape[1],axis=1)
    def reconstruct(support):
        out=[]
        for rows in support:
            _,sigma,vt=np.linalg.svd(rows,full_matrices=False)
            lam=len(rows)/rows.shape[-1]+CFG['ridge_epsilon']
            shrink=sigma**2/(sigma**2+lam)
            qhat=((q@vt.T)*shrink)@vt
            out.append(-np.sum((q-qhat)**2,axis=-1).mean(-1))
        return np.array(out).T
    return reconstruct(s),reconstruct(rank1)

def scores(parent,raw,mean,part,rgb):
    reconstruction,rank1=part; p=standard(parent)
    return np.stack([parent,reconstruction,(p+standard(reconstruction))/2,raw,(p+standard(rank1))/2,(p+standard(reconstruction[:,shuffle(rgb)]))/2,mean])

def validate(q,s,parent,raw,mean,rgb):
    a=parts(q,s); b=reference(q,s); err=max(float(abs(x-y).max()) for x,y in zip(a,b)); assert err<1e-10,err
    x=scores(parent,raw,mean,a,rgb); y=scores(parent,raw,mean,b,rgb)
    assert np.array_equal(x.argmax(-1),y.argmax(-1))
    return {'matching_error':err,'fusion_error':float(abs(x-y).max())}

def synthetic():
    rng=np.random.RandomState(260913104);q=rng.randn(3,16,24);s=rng.randn(5,1,16,24)
    p=rng.randn(3,5);raw=rng.randn(3,5);mean=rng.randn(3,5);rgb=list('abcde')
    checks=[{'case':'random',**validate(q,s,p,raw,mean,rgb)}]
    for name,qq,ss in [('zero',np.zeros_like(q),np.zeros_like(s)),('constant',np.ones_like(q),np.ones_like(s)),('rank1',q,np.repeat(s[:,:,:1],16,axis=2))]:
        a=parts(qq,ss);b=reference(qq,ss);err=max(float(abs(x-y).max()) for x,y in zip(a,b));assert err<1e-10 and all(np.isfinite(x).all() for x in a)
        checks.append({'case':name,'reconstruction_error':err})
    a=scores(p,raw,mean,parts(q,s),rgb);cp=[2,0,4,1,3];qp=[2,0,1];pp=list(range(15,-1,-1))
    assert np.allclose(scores(p[:,cp],raw[:,cp],mean[:,cp],parts(q,s[cp]),[rgb[i] for i in cp]),a[:,:,cp],atol=1e-10,rtol=0)
    assert np.allclose(scores(p[qp],raw[qp],mean[qp],parts(q[qp],s),rgb),a[:,qp],atol=1e-10,rtol=0)
    assert np.allclose(scores(p,raw,mean,parts(q[:,pp],s[:,:,pp]),rgb),a,atol=1e-10,rtol=0)
    for i in range(len(q)):
        assert all(np.allclose(x[i:i+1],y,atol=1e-10,rtol=0) for x,y in zip(parts(q,s),parts(q[i:i+1],s)))
    return checks

def interval(delta,groups):
    rng=np.random.RandomState(CFG['bootstrap_seed']); values=np.zeros(CFG['bootstrap_replicates']); unique=np.unique(groups)
    for group in unique:
        d=delta[groups==group]; values+=d[rng.randint(len(d),size=(len(values),len(d)))].mean(1)/len(unique)
    return {'delta_pp':float(delta.mean()*100),'ci95_pp':(np.quantile(values,[.025,.975])*100).tolist()}

def summarize():
    cells={}; domains={}; vectors={}
    def comparisons(a,g): return {NAMES[j]:interval(a[2]-a[j],g) for j in CFG['comparison_indices']}
    for d in CFG['domains']:
        arrays=[]
        for g in CFG['domains']:
            b=np.load(OUT/(d+'_'+g+'.npz')); pr=b['predictions']; acc=(pr==b['yq']).mean(-1); assert np.array_equal(acc,b['accuracy']); arrays.append(acc)
            cells[d+'_'+g]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'comparisons':comparisons(acc,b['seeds']),'repairs':int(((pr[2]==b['yq'])&(pr[0]!=b['yq'])).sum()),'spoils':int(((pr[2]!=b['yq'])&(pr[0]==b['yq'])).sum())}
        acc=np.mean(arrays,0); domains[d]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'comparisons':comparisons(acc,b['seeds'])}; vectors[d]=(acc,b['seeds'])
    acc=np.concatenate([vectors[d][0] for d in CFG['domains']],1); groups=np.concatenate([vectors[d][1]+i*100000000 for i,d in enumerate(CFG['domains'])]); pooled=comparisons(acc,groups)
    tests={'increment':pooled['parent']['delta_pp']>=CFG['min_gain_pp'],'positive_ci':pooled['parent']['ci95_pp'][0]>0,'domains':all(v['comparisons']['parent']['delta_pp']>=0 for v in domains.values()),'cells':all(v['comparisons']['parent']['ci95_pp'][0]>=CFG['cell_ci_floor_pp'] for v in cells.values()),'controls':all(pooled[NAMES[j]]['ci95_pp'][0]>0 for j in CFG['comparison_indices'] if j!=0)}
    return {'scope':CFG['metric_deviation'],'cells':cells,'domains':domains,'pooled':pooled,'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'gate':{'passed':all(tests.values()),'tests':tests}}

def main():
    guard(); torch.set_num_threads(CFG['threads']); start=time.time(); checks=synthetic()
    manifest=json.loads((ASSET/'manifest.json').read_text()); assert manifest['status']=='completed'
    code={str(p):sha(p) for p in [HERE/'study.py',HERE/'protocol.json',HERE/'verify.py']}; dump('code_lock.json',code); assert not any(OUT.glob('*.npz')), 'Adopt completed outputs instead of overwriting'
    inputs={}; outputs={}; total=0
    for d in CFG['domains']:
        path=ASSET/(d+'_query.npy'); h=sha(path); assert h==manifest['pools'][d+'_query']['sha256']; inputs[str(path)]=h
        ident_path=ASSET/(d+'_query_identities.json'); ident=json.loads(ident_path.read_text()); inputs[str(ident_path)]=sha(ident_path); assert inputs[str(ident_path)]==manifest['pools'][d+'_query']['identity_sha256']
        f=np.load(path,mmap_mode='r'); base=np.load(RAW/(d+'_dtd.npz')); assert f.shape==(len(ident['ids']),16,384)
        assert base['support_indices'].shape==(500,5,1); part=[]
        for i in range(500):
            part.append(parts(f[base['query_indices'][i].reshape(-1)],f[base['support_indices'][i]]))
            if i%100==0: guard(); print('RECONSTRUCTING',d,i,round(time.time()-start,2),flush=True)
        for g in CFG['domains']:
            raw_path=RAW/(d+'_'+g+'.npz'); par_path=PARENT/(d+'_'+g+'.npz'); old=np.load(raw_path); parent=np.load(par_path)
            inputs[str(raw_path)]=sha(raw_path); inputs[str(par_path)]=sha(par_path)
            assert all(np.array_equal(old[k],base[k]) and np.array_equal(old[k],parent[k]) for k in KEYS)
            assert np.array_equal(old['scores'][0],parent['scores'][1]); assert old['names'].tolist()==['parent','patch','fusion','mean_fusion','shuffled_fusion']
            result=[]
            for i in range(500):
                si=old['support_indices'][i]; qi=old['query_indices'][i].reshape(-1); rgb=[ident['rgb'][int(j)] for j in si.reshape(-1)]
                args=(parent['scores'][1,i],old['scores'][2,i],old['scores'][3,i])
                result.append(scores(*args,part[i],rgb))
                if i in CFG['audit_indices']: checks.append({'case':d+'_'+g,'task':i,**validate(f[qi],f[si],*args,rgb)})
            sc=np.stack(result,1); pr=sc.argmax(-1).astype(np.uint8); assert np.isfinite(sc).all()
            for j,k in [(0,0),(3,2),(6,3)]: assert np.array_equal(sc[j],old['scores'][k]) and np.array_equal(pr[j],old['predictions'][k])
            dest=OUT/(d+'_'+g+'.npz'); np.savez_compressed(dest,scores=sc,predictions=pr,accuracy=(pr==old['yq']).mean(-1),names=NAMES,**{k:old[k] for k in KEYS}); outputs[str(dest)]=sha(dest); total+=pr.size
    analysis=summarize(); dump('analysis.json',analysis)
    assert all(sha(p)==h for p,h in code.items()); assert all(sha(p)==h for p,h in inputs.items())
    dump('audit.json',{'status':'passed','independent_cases':checks,'permutations_and_query_independence':'passed','checked_prediction_entries':total,'input_hashes':inputs,'output_hashes':outputs,'parent_raw_mean_exact':True})
    dump('complete.json',{'status':'completed','seconds':time.time()-start,'task_conditions':2000,'unique_episodes':1000,'independent_statistics':'pending','torch':torch.__version__,'numpy':np.__version__,'threads':CFG['threads'],'free_gib':shutil.disk_usage(HERE).free/2**30})
    print('COMPLETE',json.dumps(analysis),flush=True)
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','run'],required=True);args=ap.parse_args()
    if args.phase=='check':
        torch.set_num_threads(CFG['threads']);guard();checks=synthetic()
        for d in CFG['domains']:
            f=np.load(ASSET/(d+'_query.npy'),mmap_mode='r');b=np.load(RAW/(d+'_dtd.npz'));ident=json.loads((ASSET/(d+'_query_identities.json')).read_text())
            si=b['support_indices'][0];qi=b['query_indices'][0].reshape(-1);rgb=[ident['rgb'][int(j)] for j in si.ravel()]
            checks.append({'case':d,**validate(f[qi],f[si],b['scores'][0,0],b['scores'][2,0],b['scores'][3,0],rgb)})
        dump('preflight.json',{'status':'passed','checks':checks,'tolerance':1e-10,'classification_outcomes_used':False});print('PREFLIGHT_PASSED',json.dumps(checks))
    else: main()
