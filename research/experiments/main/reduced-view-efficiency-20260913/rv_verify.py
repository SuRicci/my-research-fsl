"""Independent NumPy score algebra and raw-prediction statistics validation."""
from pathlib import Path
import sys,json
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import rv_study as m

def norm(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def ridge(x,q,y,lam):
    # Center with projection matrix; independently form kernel/test centering.
    n=len(x);h=np.eye(n)-np.ones((n,n))/n
    k=x@x.T;t=q@x.T
    kt=(t-k.mean(0)[None])@h
    return kt@np.linalg.solve(h@k@h+lam*np.eye(n),h@y)+y.mean(0)

def reference(s,q,g,penalty):
    shot=s.shape[1];flat=s.reshape(-1,s.shape[-1]);y=np.eye(5)[np.repeat(np.arange(5),shot)]
    out=[];neighbors=[]
    for head in range(3):
        x,z,gallery=flat.copy(),q.copy(),g
        if head==1:
            mu=x.mean(0);x=norm(x-mu);z=norm(z-mu)
            if shot==1:gallery=norm(g-mu)
        if shot==1 and head<2:
            proto=norm(x.reshape(5,shot,-1).mean(1))
            sims=np.einsum('id,jd->ij',proto,gallery)
            ids=np.stack([np.lexsort((np.arange(len(gallery)),-row))[:64] for row in sims])
            x=norm(.5*proto+.5*norm(gallery[ids].mean(1)));neighbors.append(ids)
            yy=np.eye(5)
        else: yy=y
        lam=penalty if head==2 else (.1 if head==1 or shot==1 else 1.)
        out.append(ridge(x,z,yy,lam))
    return np.stack(out),np.stack(neighbors) if neighbors else np.empty((0,),dtype=np.int64)

def precheck(data):
    rng=np.random.RandomState(26091391);errors=[]
    for shot in [1,5]:
        s=norm(rng.randn(5,shot,23));q=norm(rng.randn(17,23));g=norm(rng.randn(96,23))
        # Exact duplicate features stress the declared index tie policy.
        g[64:]=g[:32]
        a,ids=m.predict(torch.tensor(s),torch.tensor(q),torch.tensor(g),.1)
        b,nb=reference(s,q,g,.1)
        errors.append(float(np.max(abs(a.numpy()-b))))
        assert np.array_equal(ids.numpy(),nb)
        # Query predictions do not depend on other queries or query ordering.
        one,_=m.predict(torch.tensor(s),torch.tensor(q[:1]),torch.tensor(g),.1)
        assert np.max(abs(one.numpy()-a[:,:1].numpy()))<1e-10
    for ds in m.CFG['source_domains']:
        for shot in [1,5]:
            task=m.tasks(data,ds,shot,'selection')
            for subset in [[0],list(range(6))]:
                reps=m.aggregate(data,subset);i=0;gi=m.gallery_indices(data,ds,task)[i,0]
                s=reps[ds]['query'][task['support_indices'][i]]
                q=reps[ds]['query'][task['query_indices'][i].reshape(-1)]
                g=reps[ds]['gallery'][gi]
                a,nb=m.predict(s,q,g,m.PENALTIES[ds][str(shot)]['linear_mean']['penalty'])
                b,ids=reference(s.numpy(),q.numpy(),g.numpy(),m.PENALTIES[ds][str(shot)]['linear_mean']['penalty'])
                errors.append(float(np.max(abs(a.numpy()-b))))
                assert np.array_equal(nb.numpy(),ids)
    assert max(errors)<m.CFG['score_atol']
    return {'status':'passed','score_cases':len(errors),'max_score_error':max(errors),'tie_policy':'exact duplicate retrieval indices match','query_independence':'passed'}

def direct_interval(delta,seeds):
    rng=np.random.RandomState(m.CFG['bootstrap_seed'])
    draws=np.zeros(m.CFG['bootstrap_replicates'])
    for seed in np.unique(seeds):
        d=delta[seeds==seed]
        draws+=d[rng.randint(len(d),size=(len(draws),len(d)))].mean(1)/len(np.unique(seeds))
    return np.r_[delta.mean()*100,np.quantile(draws,[.025,.975])*100]

def audit():
    data=m.prior.load();errors=[];checks=0;stat_error=0.;selection={}
    assert m.sources()==json.loads((m.HERE/'lock.json').read_text())
    for j,subset in enumerate(m.CFG['subsets']):
        reps=m.aggregate(data,subset)
        for ds in m.CFG['source_domains']:
            other=next(x for x in m.CFG['source_domains'] if x!=ds)
            for shot in [1,5]:
                for phase in ['selection','eval']:
                    source=ds if phase=='selection' else other
                    t=np.load(m.OUT/f'{phase}_{ds}_k{shot}_tasks.npz')
                    expected=m.tasks(data,ds,shot,phase)
                    for k,v in expected.items():assert np.array_equal(v,t[k])
                    conds=[0,1] if phase=='selection' else [ds,other]
                    for c in conds:
                        z=np.load(m.OUT/'cells'/f'{phase}_{ds}_k{shot}_{c}_v{j:02d}.npz')
                        y=np.repeat(np.arange(5),t['query_indices'].shape[-1])
                        if phase=='selection':
                            selection.setdefault((ds,shot,c),[]).append((z['predictions']==y).mean(-1))
                            assert np.array_equal(t['gallery_indices'],m.gallery_indices(data,ds,expected))
                        for a,i in enumerate(z['audit_indices']):
                            s=reps[ds]['query'][t['support_indices'][i]].numpy()
                            q=reps[ds]['query'][t['query_indices'][i].reshape(-1)].numpy()
                            g=(reps[ds]['gallery'][t['gallery_indices'][i,c]] if phase=='selection' else reps[c]['gallery']).numpy()
                            sc,nb=reference(s,q,g,m.PENALTIES[source][str(shot)]['linear_mean']['penalty'])
                            err=float(np.max(abs(sc-z['audit_scores'][a])));errors.append(err)
                            assert err<m.CFG['score_atol'],(phase,ds,shot,c,j,i,err)
                            assert np.array_equal(nb,z['audit_neighbors'][a])
                            assert np.array_equal(sc.argmax(-1),z['predictions'][:,i])
                            checks+=3
        print('AUDIT_SUBSET_COMPLETE',j,flush=True)
    choice=json.loads((m.OUT/'selection_lock.json').read_text())
    for ds in m.CFG['source_domains']:
        ranked=[]
        for j,subset in enumerate(m.CFG['subsets']):
            values=[]
            for shot in [1,5]:
                t=np.load(m.OUT/f'selection_{ds}_k{shot}_tasks.npz')
                for c in [0,1]:
                    bank=selection[(ds,shot,c)]
                    for h in range(3):values.append(direct_interval(bank[j][h]-bank[-1][h],t['seeds']))
            saved=choice[ds]['rows'][j]
            for v,r in zip(values,saved['comparisons']):
                stat_error=max(stat_error,float(np.max(abs(v-np.r_[r['delta_pp'],r['ci95_pp']]))))
            feasible=all(v[1]>=-.5 for v in values)
            assert feasible==saved['feasible']
            if feasible and len(subset)<=3:ranked.append((len(subset),-min(v[0] for v in values),-np.mean([v[0] for v in values]),subset,j))
        chosen=min(ranked)[-1] if ranked else 16
        assert chosen==choice[ds]['selected_index']
    analysis=json.loads((m.OUT/'analysis.json').read_text());statchecks=0
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for shot in [1,5]:
            t=np.load(m.OUT/f'eval_{target}_k{shot}_tasks.npz');y=np.repeat(np.arange(5),15)
            for c in [target,source]:
                full=np.load(m.OUT/'cells'/f'eval_{target}_k{shot}_{c}_v16.npz')['predictions'];full=(full==y).mean(-1)
                for j in range(17):
                    pred=np.load(m.OUT/'cells'/f'eval_{target}_k{shot}_{c}_v{j:02d}.npz')['predictions'];acc=(pred==y).mean(-1)
                    for h,head in enumerate(m.HEADS):
                        r=analysis['cells'][f'{source}_to_{target}_k{shot}_{c}'][head]['frontier'][j]
                        v=direct_interval(acc[h]-full[h],t['seeds'])
                        stat_error=max(stat_error,float(np.max(abs(v-np.r_[r['delta_pp'],r['ci95_pp']]))))
                        assert abs(r['accuracy_pct']-acc[h].mean()*100)<1e-10
                        statchecks+=1
    assert stat_error<m.CFG['stat_atol']
    result={'status':'passed','independent_classifier_task_scores':checks,'max_score_error':max(errors),
        'independent_frontier_statistics':statchecks,'max_stat_error_pp':stat_error,'source_choices_independently_recovered':True}
    m.dump(m.OUT/'independent_audit.json',result);print('AUDIT_COMPLETE',result,flush=True)

if __name__=='__main__':
    torch.set_num_threads(6);audit()
