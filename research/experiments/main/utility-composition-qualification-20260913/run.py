"""Source-only utility qualification. No target outcomes select models."""
from pathlib import Path
import sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import utility
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'prior-calibration-qualification-20260913'))
import qualify as old
CFG = json.loads((HERE/'protocol.json').read_text())
OUT = HERE/'outputs'
old.CFG = CFG
old.HERE = HERE
ref = old.ref
ref.CFG = {'asset_root':CFG['asset_root']}
torch.set_num_threads(CFG['resources']['cpu_threads'])
y = torch.arange(5).repeat_interleave(15)


def dump(name, value): old.dump(OUT/name, value)


def tasks(ident, shot, phase):
    # Select disjoint source image halves for train/selection, matching prior split seed.
    labels=np.array(ident['query_labels']);rng=np.random.RandomState(CFG['image_split_seed']);pools={}
    for cls in np.unique(labels):
        ids=rng.permutation(np.flatnonzero(labels==cls));mid=len(ids)//2
        pools[int(cls)]=ids[:mid] if phase=='train' else ids[mid:] if phase=='selection' else ids
    seeds=[CFG['train_seed']] if phase=='train' else [CFG['selection_seed']] if phase=='selection' else CFG['eval_seeds']
    nq=13 if shot==5 and phase in ['train','selection'] else 15
    n=CFG['train_episodes'] if phase=='train' else CFG['selection_episodes'] if phase=='selection' else CFG['episodes_per_seed']
    ss=[];qq=[];cc=[];sd=[]
    for seed in seeds:
        rng=np.random.RandomState(seed+shot)
        for _ in range(n):
            cs=rng.choice(sorted(pools),5,replace=False)
            ix=np.stack([rng.choice(pools[int(c)],shot+nq,replace=False) for c in cs])
            ss.append(ix[:,:shot]);qq.append(ix[:,shot:]);cc.append(cs);sd.append(seed)
    return dict(support_indices=np.stack(ss),query_indices=np.stack(qq),class_ids=np.stack(cc),seeds=np.array(sd))


def batch(data, t, G, sl=slice(None)):
    S=data['query'][t['support_indices'][sl]]
    Q=data['query'][t['query_indices'][sl].reshape(-1,5*t['query_indices'].shape[-1])]
    return S,Q,utility.prepare(S,G,ref.retrieve)


def quality(cache,Q,c,a,uniform=False):
    sc=utility.scores(cache,Q,c,a,ref.ridge_scores,uniform)
    yy=torch.arange(5).repeat_interleave(Q.shape[1]//5)
    ce=F.cross_entropy((CFG['query_logit_scale']*sc).flatten(0,1),yy.repeat(len(Q)))
    acc=(sc.argmax(-1)==yy).float().mean()
    return ce,acc


def learn(source,shot,data):
    start=time.time();G=data['gallery'];tr=tasks(data['ident'],shot,'train');va=tasks(data['ident'],shot,'selection')
    assert not set(np.r_[tr['support_indices'].flatten(),tr['query_indices'].flatten()]) & set(np.r_[va['support_indices'].flatten(),va['query_indices'].flatten()])
    for phase,t in [('train',tr),('selection',va)]:np.savez_compressed(OUT/(f'{source}_k{shot}_{phase}_tasks.npz'),**t)
    _,Q,cache=batch(data,tr,G);_,V,vca=batch(data,va,G)
    # Source gallery labels may train BCE; they are not passed to inference.
    gy=np.asarray(data['ident']['gallery_labels'])
    target=torch.from_numpy((gy[cache[3].numpy()]==tr['class_ids'][:,:,None]).astype(np.float32))
    models={};hist={}
    for mode in ['utility','scalar','bce']:
        c=torch.zeros(4,requires_grad=True);a=torch.tensor(0. if shot==1 else np.log(.1/.9),dtype=torch.float32,requires_grad=True)
        b=torch.tensor(0.,requires_grad=True)
        if mode=='bce':
            opt=torch.optim.Adam([c,b],lr=CFG['learning_rate'])
            for epoch in range(CFG['epochs']):
                loss=F.binary_cross_entropy_with_logits(cache[2]@(5*c.tanh())+b,target)
                opt.zero_grad();loss.backward();opt.step()
            c=c.detach()
        opt=torch.optim.Adam([a] if mode in ['scalar','bce'] else [c,a],lr=CFG['learning_rate'])
        best=None;history=[]
        for epoch in range(1,CFG['epochs']+1):
            loss,_=quality(cache,Q,c,a,mode=='scalar');assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step()
            if epoch in CFG['checkpoints']:
                with torch.no_grad():
                    ce,acc=quality(vca,V,c,a,mode=='scalar');rank=(float(acc),-float(ce),-epoch)
                    history.append(dict(epoch=epoch,train_ce=float(loss),selection_ce=float(ce),selection_accuracy=float(acc)))
                    if best is None or rank>best[0]:best=(rank,dict(coef=c.detach().tolist(),alpha_logit=float(a),epoch=epoch))
        models[mode]=best[1];hist[mode]=history
        print('MODEL_FROZEN',source,shot,mode,models[mode],flush=True)
    # Give the fixed hyperparameter-grid controls the same source-selection tasks.
    accs={}
    for k in range(0,len(V),8):
        S=data['query'][va['support_indices'][k:k+8]]
        for key,sc in old.controls(S,V[k:k+8],G).items():accs.setdefault(key,[]).extend((sc.argmax(-1)==np.repeat(np.arange(5),V.shape[1]//5)).mean(-1).tolist())
    means={k:float(np.mean(v)) for k,v in accs.items()};winner=max(means,key=means.get)
    result=dict(models=models,history=hist,grid_winner=winner,grid_means=means,elapsed_seconds=time.time()-start)
    dump(f'models_{source}_k{shot}.json',result)
    return result


def evaluate(source,target,gallery,shot,data,model):
    name=f'{source}_to_{target}_{gallery}_k{shot}';t=tasks(data[target]['ident'],shot,'eval');G=data[gallery]['gallery']
    assert not set(data[target]['ident']['query_rgb']) & set(data[gallery]['ident']['gallery_rgb'])
    values={}
    with torch.no_grad():
        for i in range(0,len(t['seeds']),8):
            old.guard();S,Q,ca=batch(data[target],t,G,slice(i,i+8));v=old.controls(S,Q,G)
            for mode,m in model['models'].items():v[mode]=utility.scores(ca,Q,torch.tensor(m['coef']),torch.tensor(m['alpha_logit']),ref.ridge_scores,mode=='scalar').numpy()
            m=model['models']['utility'];v['same_alpha']=utility.scores(ca,Q,torch.tensor(m['coef']),torch.tensor(m['alpha_logit']),ref.ridge_scores,True).numpy()
            for k,sc in v.items():assert np.isfinite(sc).all();values.setdefault(k,[]).append(sc)
    names=list(values);sc=np.stack([np.concatenate(values[k]) for k in names]);pred=sc.argmax(-1);acc=(pred==y.numpy()).mean(-1)
    np.savez_compressed(OUT/'cells'/(name+'.npz'),names=names,scores=sc,predictions=pred,accuracy=acc,yq=y.numpy(),source_winner=model['grid_winner'],**t)
    print('CELL_COMPLETE',name,{k:round(float(acc[j].mean()*100),4) for j,k in enumerate(names) if k in ['utility','scalar','bce','r2','cs_0.1','same_alpha']},flush=True)


def analyze():
    cells={};directions={};ok=True;group={}
    for p in sorted((OUT/'cells').glob('*.npz')):
        with np.load(p) as z:
            assert z['scores'].shape[1]==500 and np.array_equal(z['predictions'],z['scores'].argmax(-1))
            assert np.array_equal(z['accuracy'],(z['predictions']==z['yq']).mean(-1))
            names=z['names'].tolist();a={k:z['accuracy'][j] for j,k in enumerate(names)};seeds=z['seeds'];win=str(z['source_winner'])
            co={k:old.interval(a['utility']-v,seeds) for k,v in a.items() if k!='utility'}
            cells[p.stem]=dict(accuracy_pct={k:float(v.mean()*100) for k,v in a.items()},comparisons=co,source_winner=win,sha256=old.sha(p))
            for k in ['scalar','bce','same_alpha','r2','cs_0.1','logistic_1','logistic_10',win]:
                if co[k]['ci95_pp'][0]<-.5:ok=False
            if p.stem.endswith('_k1'):group.setdefault(p.stem.split('_to_')[0],[]).append((a,seeds,win))
    for source,rows in group.items():
        assert len(rows)==2 and np.array_equal(rows[0][1],rows[1][1]);win=rows[0][2]
        directions[source]={}
        for k in ['scalar','bce','same_alpha','r2','cs_0.1','logistic_1','logistic_10',win]:
            d=sum(a['utility']-a[k] for a,_,_ in rows)/2;r=old.interval(d,rows[0][1]);directions[source][k]=r
            if r['delta_pp']<.5 or r['ci95_pp'][0]<=0:ok=False
    assert len(cells)==8 and len(directions)==2
    result=dict(status='passed' if ok else 'refuted_source_gate',cell_count=8,task_conditions=4000,cells=cells,directions=directions,scope='auxiliary exposed-development qualification; no Pets or Caltech outcomes')
    dump('analysis.json',result);print('VERDICT',result['status'],directions,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','run'],required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);old.guard()
    sources=[HERE/'run.py',HERE/'utility.py',HERE/'protocol.json',Path(old.__file__),Path(ref.__file__),Path(old.geo.__file__),Path(ref.ridge_scores.__code__.co_filename)]
    hashes={str(p):old.sha(p) for p in sources}
    if args.phase=='check':
        result=utility.validate(ref)
        for name in ['dtd','eurosat']:
            ident=json.loads((Path(CFG['asset_root'])/(name+'_identities.json')).read_text())
            for shot in [1,5]:
                for phase in ['train','selection','eval']:tasks(ident,shot,phase)
        result['all_source_target_task_construction']='passed'
        dump('implementation_validation.json',result);old.dump(HERE/'locked_sources.json',hashes);print(result);return
    assert hashes==json.loads((HERE/'locked_sources.json').read_text());assert not (OUT/'complete.json').exists()
    assert json.loads((OUT/'implementation_validation.json').read_text())['status']=='passed'
    (OUT/'cells').mkdir(exist_ok=True);dump('run_manifest.json',dict(command=[sys.executable,*sys.argv],config=CFG,source_hashes=hashes,torch=torch.__version__,numpy=np.__version__,python=sys.version))
    data,manifest=ref.assets();dump('asset_manifest.json',manifest)
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for shot in [1,5]:
            model_path=OUT/f'models_{source}_k{shot}.json'
            m=json.loads(model_path.read_text()) if args.resume and model_path.exists() else learn(source,shot,data[source])
            for gallery in [target,source]:
                if args.resume and (OUT/'cells'/f'{source}_to_{target}_{gallery}_k{shot}.npz').exists():continue
                evaluate(source,target,gallery,shot,data,m)
    analyze();dump('complete.json',dict(status='success',source_hashes=hashes));print('RUN_COMPLETE',flush=True)


if __name__=='__main__':main()
