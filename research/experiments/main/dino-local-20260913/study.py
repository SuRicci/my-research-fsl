"""Frozen local matching qualification with independent scalar matching checks."""
from pathlib import Path
import datetime,hashlib,json,shutil,sys,time
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'outputs';ASSET=HERE/'assets';CFG=json.loads((HERE/'protocol.json').read_text());NAMES=CFG['methods'];PARENT=ROOT/CFG['parent_bank'];KEYS=['support_indices','query_indices','class_ids','seeds']

def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
    assert shutil.disk_usage(HERE).free>=CFG['min_free_gib']*2**30
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline'])
def standard(x):return (x-x.mean(-1,keepdims=True))/np.maximum(x.std(-1,keepdims=True),1e-12)
def shuffle(rgb):
    order=np.argsort(rgb);permutation=np.empty(len(rgb),dtype=int);permutation[order]=np.roll(order,1);return permutation
def matching(q,s,rgb):
    q=F.normalize(torch.as_tensor(np.array(q),dtype=torch.float64),dim=-1);s=F.normalize(torch.as_tensor(np.array(s),dtype=torch.float64),dim=-1).flatten(1,2)
    sim=q.unsqueeze(0)@s[:,None].transpose(-1,-2);local=sim.max(-1).values.mean(-1).T.numpy()
    mean=(F.normalize(q.mean(1),dim=-1)@F.normalize(s.mean(1),dim=-1).T).numpy()
    perm=shuffle(rgb);shuffled=local[:,perm]  # Protocol is one shot; permuting whole support images permutes class score columns.
    return local,mean,shuffled

def independent(q,s,rgb):
    q=np.asarray(q,dtype=np.float64);s=np.asarray(s,dtype=np.float64).reshape(5,-1,q.shape[-1]);q=q/np.linalg.norm(q,axis=-1,keepdims=True);s=s/np.linalg.norm(s,axis=-1,keepdims=True)
    score=np.zeros((len(q),5))
    for i in range(len(q)):
        for c in range(5):score[i,c]=sum(max(float(np.dot(v,t)) for t in s[c]) for v in q[i])/len(q[i])
    qm=q.mean(1);sm=s.mean(1);mean=(qm/np.linalg.norm(qm,axis=-1,keepdims=True))@(sm/np.linalg.norm(sm,axis=-1,keepdims=True)).T
    order=sorted(range(5),key=lambda k:rgb[k]);assigned=list(range(5))
    for i,k in enumerate(order):assigned[k]=order[(i-1)%5]
    return score,mean,score[:,assigned]

def scores(parent,parts):
    local,mean,shuffled=parts;p=standard(parent)
    return np.stack([parent,local,(p+standard(local))/2,(p+standard(mean))/2,(p+standard(shuffled))/2])
def validate(q,s,parent,rgb):
    a=matching(q,s,rgb);b=independent(q,s,rgb);err=max(float(abs(x-y).max()) for x,y in zip(a,b));assert err<1e-10
    x=scores(parent,a);y=scores(parent,b);assert np.array_equal(x.argmax(-1),y.argmax(-1));return {'local_score_error':err,'fusion_score_error':float(abs(x-y).max())}

def interval(d,groups):
    rng=np.random.RandomState(CFG['bootstrap_seed']);v=np.zeros(CFG['bootstrap_replicates']);unique=np.unique(groups)
    for g in unique:
        x=d[groups==g];v+=x[rng.randint(len(x),size=(len(v),len(x)))].mean(1)/len(unique)
    return {'delta_pp':float(d.mean()*100),'ci95_pp':(np.quantile(v,[.025,.975])*100).tolist()}
def summarize():
    cells={};domains={};vectors={}
    def comparisons(a,g):return {NAMES[j]:interval(a[2]-a[j],g) for j in [0,3,4]}
    for d in CFG['domains']:
        arrays=[]
        for g in CFG['domains']:
            b=np.load(OUT/(d+'_'+g+'.npz'));pr=b['predictions'];acc=(pr==b['yq']).mean(-1);assert np.array_equal(acc,b['accuracy']);arrays.append(acc)
            cells[d+'_'+g]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'comparisons':comparisons(acc,b['seeds']),'repairs':int(((pr[2]==b['yq'])&(pr[0]!=b['yq'])).sum()),'spoils':int(((pr[2]!=b['yq'])&(pr[0]==b['yq'])).sum())}
        acc=np.mean(arrays,0);domains[d]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'comparisons':comparisons(acc,b['seeds'])};vectors[d]=(acc,b['seeds'])
    acc=np.concatenate([vectors[d][0] for d in CFG['domains']],1);groups=np.concatenate([vectors[d][1]+i*100000000 for i,d in enumerate(CFG['domains'])]);pooled=comparisons(acc,groups)
    tests={'increment':pooled['parent']['delta_pp']>=CFG['min_gain_pp'],'positive_ci':pooled['parent']['ci95_pp'][0]>0,'domains':all(v['comparisons']['parent']['delta_pp']>=0 for v in domains.values()),'cells':all(v['comparisons']['parent']['ci95_pp'][0]>=CFG['cell_ci_floor_pp'] for v in cells.values()),'controls':all(pooled[k]['ci95_pp'][0]>0 for k in ['mean_fusion','shuffled_fusion'])}
    return {'scope':CFG['metric_deviation'],'cells':cells,'domains':domains,'pooled':pooled,'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'gate':{'passed':all(tests.values()),'tests':tests}}

def main():
    guard();torch.set_num_threads(CFG['threads']);start=time.time();manifest=json.loads((ASSET/'manifest.json').read_text());assert manifest['status']=='completed'
    code={str(p):sha(p) for p in [HERE/'study.py',HERE/'protocol.json',HERE/'verify_statistics.py']};dump('code_lock.json',code)
    saved={};checks=[];total=0;parents={}
    rng=np.random.RandomState(260913102);q=rng.randn(3,16,24);s=rng.randn(5,1,16,24);parent=rng.randn(3,5);rgb=[str(i) for i in range(5)];checks.append({'case':'synthetic',**validate(q,s,parent,rgb)})
    a=scores(parent,matching(q,s,rgb));cp=np.array([2,0,4,1,3]);qp=[2,0,1];pp=list(range(15,-1,-1))
    assert np.allclose(scores(parent[:,cp],matching(q,s[cp],[rgb[i] for i in cp])),a[:,:,cp],atol=1e-10,rtol=0)
    assert np.allclose(scores(parent[qp],matching(q[qp],s,rgb)),a[:,qp],atol=1e-10,rtol=0)
    assert np.allclose(scores(parent,matching(q[:,pp],s[:,:,pp],rgb)),a,atol=1e-10,rtol=0)
    for d in CFG['domains']:
        path=ASSET/(d+'_query.npy');assert sha(path)==manifest['pools'][d+'_query']['sha256'];f=np.load(path,mmap_mode='r');ident=json.loads((ASSET/(d+'_query_identities.json')).read_text());base=np.load(PARENT/(d+'_dtd.npz'))
        assert base['support_indices'].shape==(500,5,1);assert f.shape==(len(ident['ids']),16,384);parts=[]
        for i in range(500):
            si=base['support_indices'][i];qi=base['query_indices'][i].reshape(-1);rgb=[ident['rgb'][int(j)] for j in si.reshape(-1)];q=f[qi];s=f[si];parts.append(matching(q,s,rgb))
            if i in CFG['audit_indices']:checks.append({'case':d,'task':i,**validate(q,s,base['scores'][1,i],rgb)})
            if i%100==0:guard();print('MATCHING',d,i,round(time.time()-start,1),flush=True)
        for g in CFG['domains']:
            oldpath=PARENT/(d+'_'+g+'.npz');old=np.load(oldpath);parents[str(oldpath)]=sha(oldpath);assert all(np.array_equal(old[k],base[k]) for k in KEYS)
            
            if g!='dtd':
                for i in CFG['audit_indices']:
                    si=old['support_indices'][i];qi=old['query_indices'][i].reshape(-1);rgb=[ident['rgb'][int(j)] for j in si.reshape(-1)];checks.append({'case':d+'_'+g,'task':i,**validate(f[qi],f[si],old['scores'][1,i],rgb)})
            sc=np.stack([scores(old['scores'][1,i],parts[i]) for i in range(500)],1);assert np.array_equal(sc[0],old['scores'][1]);pr=sc.argmax(-1).astype(np.uint8);assert np.array_equal(pr[0],old['predictions'][1]);assert np.isfinite(sc).all();total+=pr.size
            np.savez_compressed(OUT/(d+'_'+g+'.npz'),scores=sc,predictions=pr,accuracy=(pr==old['yq']).mean(-1),yq=old['yq'],names=NAMES,**{k:old[k] for k in KEYS})
            saved[d+'_'+g]=sha(OUT/(d+'_'+g+'.npz'))
    analysis=summarize();dump('analysis.json',analysis)
    for name,h in saved.items():
        b=np.load(OUT/(name+'.npz'));assert np.array_equal(b['predictions'],b['scores'].argmax(-1));assert np.array_equal(b['accuracy'],(b['predictions']==b['yq']).mean(-1))
    assert all(sha(p)==h for p,h in code.items());dump('audit.json',{'status':'passed','independent_matching_cases':checks,'query_class_patch_permutation':'passed','checked_prediction_entries':total,'all_parent_scores_exact':True,'output_hashes':saved,'parent_hashes':parents})
    dump('complete.json',{'status':'completed','seconds':time.time()-start,'task_conditions':2000,'unique_episodes':1000,'feature_manifest_sha256':sha(ASSET/'manifest.json'),'independent_statistics':'pending'})
    print('COMPLETE',json.dumps(analysis),flush=True)
if __name__=='__main__':main()
