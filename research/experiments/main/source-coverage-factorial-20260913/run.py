"""Four-arm auxiliary source-coverage experiment; labels never enter inference."""
from pathlib import Path
import sys,json,argparse,time
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'utility-composition-qualification-20260913'))
import run as prev
from utility import prepare
old,ref=prev.old,prev.ref
CFG=json.loads((HERE/'protocol.json').read_text());OUT=HERE/'outputs'
old.CFG=CFG;old.HERE=HERE;torch.set_num_threads(CFG['resources']['cpu_threads'])
ARMS=['ordinary_scalar','ordinary_adaptive','mixed_scalar','mixed_adaptive']


def dump(name,data):old.dump(OUT/name,data)


def tasks(ident,shot,phase,rep=0):
    prev.CFG={**CFG,'train_seed':CFG['train_seed']+1000*rep,'selection_seed':CFG['selection_seed']+1000*rep}
    return prev.tasks(ident,shot,phase)


def compact(S,G):
    P,N,f,idx=prepare(S,G,ref.retrieve)
    return P,F.normalize(N.mean(2),dim=-1),f.mean(2),idx


def score(ca,Q,b,a,globalized=False):
    P,M,f=ca[:3];alpha=torch.sigmoid(a+f@(5*b.tanh()))[...,None]
    if globalized:alpha=alpha.mean(1,keepdim=True).expand_as(alpha)
    A=F.normalize((1-alpha)*P+alpha*M,dim=-1)
    Y=torch.eye(5,dtype=A.dtype)[None].expand(len(A),-1,-1)
    return ref.ridge_scores(A,Y,Q,.1),alpha


def source_cache(data,t,phase,rep,condition):
    gy=np.asarray(data['ident']['gallery_labels']);arrays=[[],[],[]];gallery=[]
    for i,cs in enumerate(t['class_ids']):
        rng=np.random.RandomState(CFG['source_gallery_seed']+100000*rep+10000*(phase=='selection')+i)
        pool=np.flatnonzero(~np.isin(gy,cs)) if condition=='mixed' and i%2 else np.arange(len(gy))
        gi=rng.choice(pool,CFG['source_gallery_size'],replace=False);gallery.append(gi)
        S=data['query'][t['support_indices'][i:i+1]];ca=compact(S,data['gallery'][gi])
        for j in range(3):arrays[j].append(ca[j])
    Q=data['query'][t['query_indices'].reshape(len(gallery),-1)]
    return tuple(torch.cat(x) for x in arrays),Q,np.stack(gallery)


def quality(ca,Q,b,a):
    sc,_=score(ca,Q,b,a);y=torch.arange(5).repeat_interleave(Q.shape[1]//5)
    return F.cross_entropy((10*sc).flatten(0,1),y.repeat(len(Q))),(sc.argmax(-1)==y).float().mean()


def learn(source,shot,rep,data):
    start=time.time();tr=tasks(data['ident'],shot,'train',rep);va=tasks(data['ident'],shot,'selection',rep)
    assert not set(np.r_[tr['support_indices'].flat,tr['query_indices'].flat]) & set(np.r_[va['support_indices'].flat,va['query_indices'].flat])
    vca,V,vg=source_cache(data,va,'selection',rep,'mixed');cache={}
    for cond in ['ordinary','mixed']:
        cache[cond]=source_cache(data,tr,'train',rep,cond)
        np.savez_compressed(OUT/f'{source}_k{shot}_r{rep}_train_{cond}.npz',**tr,gallery_indices=cache[cond][2])
    np.savez_compressed(OUT/f'{source}_k{shot}_r{rep}_selection.npz',**va,gallery_indices=vg)
    models={};history={}
    for arm in ARMS:
        cond,cap=arm.split('_');ca,Q,_=cache[cond]
        b=torch.zeros(4,requires_grad=cap=='adaptive');a=torch.tensor(0. if shot==1 else np.log(.1/.9),dtype=torch.float32,requires_grad=True)
        opt=torch.optim.Adam([a,b] if cap=='adaptive' else [a],lr=CFG['learning_rate']);best=None;rows=[]
        for epoch in range(1,CFG['epochs']+1):
            loss,_=quality(ca,Q,b,a);assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step()
            if epoch in CFG['checkpoints']:
                with torch.no_grad():
                    ce,acc=quality(vca,V,b,a);rank=(float(acc),-float(ce),-epoch)
                    rows.append(dict(epoch=epoch,train_ce=float(loss),selection_ce=float(ce),selection_accuracy=float(acc)))
                    if best is None or rank>best[0]:best=(rank,dict(coef=b.detach().tolist(),alpha_logit=float(a),epoch=epoch))
        models[arm]=best[1];history[arm]=rows
    means={}
    for i in range(len(vg)):
        S=data['query'][va['support_indices'][i:i+1]]
        for k,sc in old.controls(S,V[i:i+1],data['gallery'][vg[i]]).items():
            means.setdefault(k,[]).append(float((sc.argmax(-1)==np.repeat(np.arange(5),V.shape[1]//5)).mean()))
    means={k:float(np.mean(v)) for k,v in means.items()};win=max(means,key=means.get)
    result=dict(models=models,history=history,grid_means=means,grid_winner=win,seconds=time.time()-start)
    dump(f'models_{source}_k{shot}_r{rep}.json',result);print('TRAIN_COMPLETE',source,shot,rep,win,round(result['seconds'],2),flush=True)
    return result


def evaluate(source,target,gallery,shot,data,models):
    name=f'{source}_to_{target}_{gallery}_k{shot}';t=tasks(data[target]['ident'],shot,'eval');G=data[gallery]['gallery']
    assert not set(data[target]['ident']['query_rgb']) & set(data[gallery]['ident']['gallery_rgb'])
    prior=prev.OUT/'cells'/(name+'.npz');vals={};alphas={}
    with np.load(prior) as z:
        for key in t:assert np.array_equal(t[key],z[key])
        base_names=[k for k in z['names'].tolist() if k not in ['utility','scalar','bce','same_alpha']]
        base={k:z['scores'][z['names'].tolist().index(k)] for k in base_names}
    for i in range(0,len(t['seeds']),8):
        old.guard();S=data[target]['query'][t['support_indices'][i:i+8]];Q=data[target]['query'][t['query_indices'][i:i+8].reshape(-1,75)];ca=compact(S,G)
        with torch.no_grad():
            if i==0:
                rec=old.controls(S[:1],Q[:1],G)
                for k in base:assert np.allclose(rec[k],base[k][0:1],atol=1e-6)
            for rep,md in enumerate(models):
                for arm,m in md['models'].items():
                    key=f'r{rep}_{arm}';sc,al=score(ca,Q,torch.tensor(m['coef']),torch.tensor(m['alpha_logit']))
                    vals.setdefault(key,[]).append(sc.numpy());alphas.setdefault(key,[]).append(al[...,0].numpy())
                m=md['models']['mixed_adaptive'];sc,al=score(ca,Q,torch.tensor(m['coef']),torch.tensor(m['alpha_logit']),True)
                key=f'r{rep}_globalized';vals.setdefault(key,[]).append(sc.numpy());alphas.setdefault(key,[]).append(al[...,0].numpy())
    names=list(vals);sc=np.stack([np.concatenate(vals[k]) for k in names]);aa=np.stack([np.concatenate(alphas[k]) for k in names]);assert np.isfinite(sc).all()
    y=np.repeat(np.arange(5),15);pred=sc.argmax(-1);acc=(pred==y).mean(-1)
    np.savez_compressed(OUT/'cells'/(name+'.npz'),names=names,scores=sc,alpha=aa,predictions=pred,accuracy=acc,yq=y,**t)
    dump(name+'_reuse.json',dict(path=str(prior),sha256=old.sha(prior),base_names=base_names,selection_winners=[m['grid_winner'] for m in models],first_task_control_reconstruction='passed'))
    print('CELL_COMPLETE',name,{arm:round(float(np.mean([acc[names.index(f'r{r}_{arm}')].mean() for r in range(3)]))*100,4) for arm in ARMS},flush=True)


def analyze():
    cells={};group={};passed=True
    for p in sorted((OUT/'cells').glob('*.npz')):
        z=np.load(p);names=z['names'].tolist();re=json.loads((OUT/(p.stem+'_reuse.json')).read_text());base=np.load(re['path']);bn=base['names'].tolist();seeds=z['seeds']
        a={arm:np.mean([z['accuracy'][names.index(f'r{r}_{arm}')] for r in range(3)],axis=0) for arm in ARMS+['globalized']}
        for k in ['r2','cs_0.1','logistic_1','logistic_10']:a[k]=base['accuracy'][bn.index(k)]
        a['source_selected']=np.mean([base['accuracy'][bn.index(k)] for k in re['selection_winners']],axis=0)
        c=a['mixed_adaptive'];co={k:old.interval(c-v,seeds) for k,v in a.items() if k!='mixed_adaptive'}
        effects={'coverage_scalar':a['mixed_scalar']-a['ordinary_scalar'],'coverage_adaptive':c-a['ordinary_adaptive'],'capacity_ordinary':a['ordinary_adaptive']-a['ordinary_scalar'],'capacity_mixed':c-a['mixed_scalar'],'interaction':(c-a['ordinary_adaptive'])-(a['mixed_scalar']-a['ordinary_scalar'])}
        cells[p.stem]=dict(accuracy_pct={k:float(v.mean()*100) for k,v in a.items()},comparisons=co,effects={k:old.interval(v,seeds) for k,v in effects.items()},mean_alpha={arm:float(z['alpha'][[names.index(f'r{r}_{arm}') for r in range(3)]].mean()) for arm in ARMS})
        if any(v['ci95_pp'][0]<-.5 for k,v in co.items() if k!='globalized'):passed=False
        if p.stem.endswith('_k1'):
            reps=[z['accuracy'][names.index(f'r{r}_mixed_adaptive')]-z['accuracy'][names.index(f'r{r}_mixed_scalar')] for r in range(3)]
            group.setdefault(p.stem.split('_to_')[0],[]).append((a,seeds,effects,reps))
    directions={}
    for source,rows in group.items():
        assert len(rows)==2 and np.array_equal(rows[0][1],rows[1][1]);seeds=rows[0][1]
        co={k:old.interval(sum(a['mixed_adaptive']-a[k] for a,_,_,_ in rows)/2,seeds) for k in rows[0][0] if k!='mixed_adaptive'}
        effects={k:old.interval(sum(e[k] for _,_,e,_ in rows)/2,seeds) for k in rows[0][2]}
        reps=[old.interval(sum(rr[r] for _,_,_,rr in rows)/2,seeds) for r in range(3)]
        directions[source]=dict(comparisons=co,effects=effects,replicate_vs_mixed_scalar=reps)
        if any(v['delta_pp']<.5 or v['ci95_pp'][0]<=0 for k,v in co.items() if k!='globalized'):passed=False
        if co['globalized']['ci95_pp'][0]<=0 or any(v['delta_pp']<0 for v in reps):passed=False
    assert len(cells)==8 and len(directions)==2
    dump('analysis.json',dict(status='passed' if passed else 'refuted_source_gate',cell_count=8,training_models=48,training_allocations=3,task_conditions=4000,cells=cells,directions=directions,interval_scope='paired bootstrap over tasks, averaged across three source allocations; conditional on image pools, not random-domain CI'))
    print('VERDICT','passed' if passed else 'refuted_source_gate',directions,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','run'],required=True);args=ap.parse_args();OUT.mkdir(exist_ok=True);old.guard()
    sources=[Path(__file__),HERE/'verify.py',HERE/'protocol.json',Path(prev.__file__),Path(old.__file__),Path(ref.__file__),Path(prepare.__code__.co_filename),Path(ref.ridge_scores.__code__.co_filename),Path(old.geo.__file__)]
    hashes={str(p):old.sha(p) for p in sources}
    if args.phase=='check':
        from verify import check
        result=check(sys.modules[__name__]);dump('implementation_validation.json',result);old.dump(HERE/'locked_sources.json',hashes);print(result);return
    assert not (OUT/'complete.json').exists();assert hashes==json.loads((HERE/'locked_sources.json').read_text())
    assert json.loads((OUT/'implementation_validation.json').read_text())['status']=='passed'
    (OUT/'cells').mkdir(exist_ok=True);dump('run_manifest.json',dict(command=[sys.executable,*sys.argv],config=CFG,source_hashes=hashes,torch=torch.__version__,numpy=np.__version__,python=sys.version))
    data,manifest=ref.assets();dump('asset_manifest.json',manifest)
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for shot in [1,5]:
            models=[learn(source,shot,r,data[source]) for r in CFG['training_replicates']]
            for gallery in [target,source]:evaluate(source,target,gallery,shot,data,models)
    analyze();dump('complete.json',dict(status='success',source_hashes=hashes));print('RUN_COMPLETE',flush=True)


if __name__=='__main__':main()
