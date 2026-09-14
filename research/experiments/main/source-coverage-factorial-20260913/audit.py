"""Audit every saved task and score; independently rebuild class influence in NumPy."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import torch
P=Path(__file__).resolve().parent;O=P/'outputs'
spec=importlib.util.spec_from_file_location('coverage_exp',P/'run.py');ex=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ex;spec.loader.exec_module(ex)
assert (O/'complete.json').exists()
locks=json.loads((P/'locked_sources.json').read_text());assert all(ex.old.sha(f)==h for f,h in locks.items())
data,_=ex.ref.assets();analysis=json.loads((O/'analysis.json').read_text());source_rows=[]


def labels(z,ident):
    yy=np.asarray(ident['query_labels'])
    for key in ['support_indices','query_indices']:
        assert np.array_equal(yy[z[key]],np.broadcast_to(z['class_ids'][...,None],z[key].shape))
    for s,q in zip(z['support_indices'],z['query_indices']):assert not set(s.flat)&set(q.flat)


def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def numpy_cache(S,G):
    proto=norm(S.mean(1));sim=proto@G.T;idx=np.argsort(-sim,axis=1,kind='stable')[:,:64];mu=norm(G[idx].mean(1));feat=[];owners=[]
    for lo,hi in [(0,512),(512,S.shape[-1])]:
        simpart=norm(S[:,:,lo:hi].mean(1))@norm(G[:,lo:hi]).T
        feat.append(np.take_along_axis(simpart,idx,axis=1));owners.append(simpart.argmax(0)[idx])
    margins=np.stack([sim[c]-np.delete(sim,c,axis=0).max(0) for c in range(5)])
    feat.append(np.take_along_axis(margins,idx,axis=1));feat.append(((owners[0]==np.arange(5)[:,None])&(owners[1]==np.arange(5)[:,None])).astype(float))
    return proto,mu,np.stack(feat,axis=-1).mean(1)


def numpy_score(cache,Q,m,globalized=False):
    proto,mu,f=cache;al=1/(1+np.exp(-(m['alpha_logit']+f@(5*np.tanh(m['coef'])))))
    if globalized:al[:]=al.mean()
    A=norm((1-al[:,None])*proto+al[:,None]*mu);center=A.mean(0);xc=A-center
    beta=np.linalg.solve(xc@xc.T+.1*np.eye(5),np.eye(5)-.2)
    return (Q-center)@xc.T@beta+.2,al


with torch.no_grad():
    for source in ['dtd','eurosat']:
        ident=data[source]['ident'];gy=np.asarray(ident['gallery_labels'])
        for shot in [1,5]:
            for rep in [0,1,2]:
                paths=[O/f'{source}_k{shot}_r{rep}_train_{c}.npz' for c in ['ordinary','mixed']]+[O/f'{source}_k{shot}_r{rep}_selection.npz']
                zz=[np.load(p) for p in paths]
                for z in zz:labels(z,ident);assert len(z['seeds'])==100
                for key in ['support_indices','query_indices','class_ids','seeds']:assert np.array_equal(zz[0][key],zz[1][key])
                ti=set(zz[0]['support_indices'].flat)|set(zz[0]['query_indices'].flat);vi=set(zz[2]['support_indices'].flat)|set(zz[2]['query_indices'].flat);assert not ti&vi
                for zi,z in enumerate(zz):
                    for i,(gi,cs) in enumerate(zip(z['gallery_indices'],z['class_ids'])):
                        assert len(gi)==1024 and len(set(gi))==1024
                        if zi>0 and i%2:assert not np.isin(gy[gi],cs).any()
                    assert np.array_equal(z['gallery_indices'][::2],zz[0]['gallery_indices'][::2]) if zi==1 else True
                model=json.loads((O/f'models_{source}_k{shot}_r{rep}.json').read_text());assert model['grid_winner']==max(model['grid_means'],key=model['grid_means'].get)
                z=zz[2];t={k:z[k] for k in ['support_indices','query_indices','class_ids','seeds']};ca,Q,gi=ex.source_cache(data[source],t,'selection',rep,'mixed');assert np.array_equal(gi,z['gallery_indices'])
                for arm,m in model['models'].items():
                    assert np.isfinite(m['coef']).all() and np.isfinite(m['alpha_logit'])
                    if arm.endswith('scalar'):assert np.array_equal(m['coef'],np.zeros(4))
                    best=max(model['history'][arm],key=lambda h:(h['selection_accuracy'],-h['selection_ce'],-h['epoch']));assert best['epoch']==m['epoch']
                    ce,acc=ex.quality(ca,Q,torch.tensor(m['coef']),torch.tensor(m['alpha_logit']))
                    assert abs(float(ce)-best['selection_ce'])<1e-6 and abs(float(acc)-best['selection_accuracy'])<1e-7
                source_rows.append(dict(source=source,shot=shot,replicate=rep,source_only_checkpoint_reconstruction=True))

rows=[];group={};max_error=0.
with torch.no_grad():
    for p in sorted((O/'cells').glob('*.npz')):
        source,other=p.stem.split('_to_');target,gallery,k=other.split('_');shot=int(k[1:]);z=np.load(p);names=z['names'].tolist();labels(z,data[target]['ident'])
        assert len(names)==15 and len(set(names))==15 and len(z['seeds'])==500
        assert np.array_equal(np.unique(z['seeds'],return_counts=True)[1],np.repeat(100,5))
        assert np.isfinite(z['scores']).all() and np.isfinite(z['alpha']).all()
        assert np.array_equal(z['predictions'],z['scores'].argmax(-1));assert np.array_equal(z['accuracy'],(z['predictions']==z['yq']).mean(-1))
        assert not set(data[target]['ident']['query_rgb'])&set(data[gallery]['ident']['gallery_rgb'])
        reuse=json.loads((O/(p.stem+'_reuse.json')).read_text());assert ex.old.sha(reuse['path'])==reuse['sha256'];base=np.load(reuse['path']);bn=base['names'].tolist()
        for key in ['support_indices','query_indices','class_ids','seeds']:assert np.array_equal(z[key],base[key])
        models=[json.loads((O/f'models_{source}_k{shot}_r{r}.json').read_text()) for r in range(3)]
        assert reuse['selection_winners']==[m['grid_winner'] for m in models]
        acc={arm:np.mean([z['accuracy'][names.index(f'r{r}_{arm}')] for r in range(3)],axis=0) for arm in ex.ARMS+['globalized']}
        for key in ['r2','cs_0.1','logistic_1','logistic_10']:acc[key]=base['accuracy'][bn.index(key)]
        acc['source_selected']=np.mean([base['accuracy'][bn.index(k)] for k in reuse['selection_winners']],axis=0)
        for key,a in acc.items():assert abs(a.mean()*100-analysis['cells'][p.stem]['accuracy_pct'][key])<1e-9
        errors=[]
        for i in [0,249,499]:
            S=data[target]['query'][z['support_indices'][i]].numpy();Q=data[target]['query'][z['query_indices'][i].flatten()].numpy();G=data[gallery]['gallery'].numpy();cache=numpy_cache(S,G)
            for rep,md in enumerate(models):
                for arm in ex.ARMS+['globalized']:
                    m=md['models']['mixed_adaptive' if arm=='globalized' else arm];sc,al=numpy_score(cache,Q,m,arm=='globalized');ni=names.index(f'r{rep}_{arm}')
                    err=float(np.max(np.abs(sc-z['scores'][ni,i])));errors.append(err);assert err<2e-5,(p.name,i,arm,err)
                    assert np.allclose(al,z['alpha'][ni,i],atol=2e-5)
                    assert np.array_equal(sc.argmax(-1),z['predictions'][ni,i])
                    split=np.concatenate([numpy_score(cache,q,m,arm=='globalized')[0] for q in np.array_split(Q,4)]);assert np.allclose(sc,split,atol=1e-10)
        max_error=max(max_error,max(errors));rows.append(dict(cell=p.stem,sha256=ex.old.sha(p),independent_reconstructed_scores=len(errors),max_numpy_error=max(errors)))
        if shot==1:group.setdefault(source,[]).append(acc)
for source,rr in group.items():
    for key,stats in analysis['directions'][source]['comparisons'].items():
        delta=sum(a['mixed_adaptive']-a[key] for a in rr)/2;assert abs(delta.mean()*100-stats['delta_pp'])<1e-9
    e=analysis['directions'][source]['effects'];aa={key:sum(a[key] for a in rr)/2 for key in ex.ARMS}
    dd=(aa['mixed_adaptive']-aa['ordinary_adaptive'])-(aa['mixed_scalar']-aa['ordinary_scalar']);assert abs(dd.mean()*100-e['interaction']['delta_pp'])<1e-9
assert len(rows)==8 and len(source_rows)==12
result=dict(status='passed',cells=8,task_conditions=4000,model_count=48,source_allocations_checked=12,source_gallery_conditions_checked=36,all_task_labels_checked=True,all_source_exclusions_checked=True,source_selection_statistics_recomputed=True,all_saved_score_prediction_accuracy_checked=True,control_hash_and_task_identity_reuse_checked=True,independent_numpy_tasks=24,independent_numpy_model_task_reconstructions=360,max_numpy_error=max_error,query_partition_invariant=True,rows=rows,source_rows=source_rows,limits=['Task intervals condition on fixed pools and three training allocations','Two exposed development domains; no fresh-domain result','Coverage manipulation is class exclusion, not natural domain shift'])
ex.old.dump(O/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['rows','source_rows']},indent=2))
