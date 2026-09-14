import argparse,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from campaign_common import ROOT,HERE,OUT,read,dump,sha,validate_sources
from r5_candidates import run,support_cv
sys.path.insert(0,str(ROOT/'lab/refine_20260909'))
from r5_refine import pair,old,BASE
from r5_methods import representation,evaluate_configs,legacy_configs


def execute(method):
    validate_sources();torch.set_num_threads(4);device='cuda:0';cfg=read(HERE/'candidates.json')[method];shot=cfg['shot'];dest=OUT/method
    for cell in old.CELLS:
        qm,qfs,gfs,qi,identity=pair(cell)
        qf=representation(*qfs,.5)
        if method=='r5_support_cv':
            reps={w:representation(*qfs,w) for w in cfg['weights']};G=torch.empty(0,qf.shape[-1],device=device);gc=gd=None
        else:
            G=representation(*gfs,.5).to(device);gc,gd=[F.normalize(g,dim=-1).to(device) for g in gfs]
        configs=[next(c for c in legacy_configs(shot) if c['name']=='r2_overall'),dict(family='raw_ridge',w=.5,lam=cfg['lambda'],name='no_retrieval')]
        paths=sorted((BASE/'results').glob('_'.join(cell)+f'_k{shot}_s*.npz'));assert len(paths)==5
        for source in paths:
            folder=dest/'test';p=folder/source.name
            if p.with_suffix('.json').exists():assert sha(p)==read(p.with_suffix('.json'))['sha256'];continue
            z=np.load(source,allow_pickle=False);si,xi,cs,seed=[z[n] for n in ['support_indices','query_indices','class_ids','seed']]
            score=[];base=[];control=[];details={};started=time.time()
            torch.cuda.reset_peak_memory_stats()
            for st in range(0,len(si),24):
                ss=si[st:st+24];xx=xi[st:st+24].reshape(-1,75);S=qf[ss].to(device);X=qf[xx].to(device)
                _,v,_=evaluate_configs(S,X,G,configs,return_scores=True)
                base.append(v['r2_overall'].numpy());control.append(v['no_retrieval'].numpy())
                if method=='r5_support_cv':
                    out,diag=support_cv([reps[w][ss].to(device) for w in cfg['weights']],[reps[w][xx].to(device) for w in cfg['weights']],cfg)
                else:out,diag=run(S,X,G,qfs[0][ss].to(device),qfs[1][ss].to(device),gc,gd,method,cfg)
                assert torch.isfinite(out).all();score.append(out.cpu().numpy())
                for n,value in diag.items():
                    if isinstance(value,np.ndarray):details.setdefault(n,[]).append(value)
                    else:details[n]=value
            scores=np.stack([np.concatenate(score),np.concatenate(base),np.concatenate(control)]);pred=scores.argmax(-1).astype('int8')
            bidx=list(z['names']).index('r2_overall');error=float(np.abs(scores[1]-z['scores'][bidx]).max());assert error<2e-4,(method,cell,error)
            np.testing.assert_array_equal(pred[1],z['predictions'][bidx]);names=[method,'r2_overall','no_retrieval'];y=z['yq']
            accuracy=(pred==y).mean(-1);folder.mkdir(parents=True,exist_ok=True)
            extra={n:np.concatenate(v) for n,v in details.items() if isinstance(v,list)}
            np.savez_compressed(p,scores=scores,predictions=pred,accuracy=accuracy,names=names,yq=y,support_indices=si,query_indices=xi,class_ids=cs,seed=seed,**extra)
            summarydiag={n:np.mean(v,axis=0).tolist() for n,v in extra.items()}
            dump(p.with_suffix('.json'),dict(method=method,cell=cell,shot=shot,seed=int(seed),episodes=600,source=str(source),source_sha256=sha(source),sha256=sha(p),
                baseline_error=error,baseline_prediction_flips=0,accuracy=dict(zip(names,accuracy.mean(1).tolist())),diagnostics=summarydiag,
                seconds=time.time()-started,peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20,identity=identity,
                method_gallery_access=method!='r5_support_cv',note='Timing includes paired baselines, I/O and saved predictions; not isolated production latency'))
            print('COMPLETE',method,p.stem,'gain_pp',round(float((accuracy[0]-accuracy[1]).mean()*100),3),'seconds',round(time.time()-started,1),flush=True)
    summarize(method)

def summarize(method):
    rows=[];all_delta={}
    for cell in old.CELLS:
        paths=sorted((OUT/method/'test').glob('_'.join(cell)+'_k*_s*.npz'));assert len(paths)==5;packs=[]
        for p in paths:
            m=read(p.with_suffix('.json'));assert sha(p)==m['sha256'];packs.append(np.load(p,allow_pickle=False)['accuracy'])
        a=np.stack(packs);d=a[:,0]-a[:,1];all_delta[cell]=d;comparisons={}
        for i,name in [(1,'r2_overall'),(2,'no_retrieval')]:
            delta=a[:,0]-a[:,i];rng=np.random.default_rng(902193)
            boot=[np.mean([v[rng.integers(600,size=600)].mean() for v in delta]) for _ in range(1000)]
            ci=(100*np.quantile(boot,[.025,.975])).tolist();gain=float(delta.mean()*100);seedg=delta.mean(1)*100
            comparisons[name]=dict(gain_pp=gain,ci95_pp=ci,seed_gain_pp=seedg.tolist(),positive_seeds=int((seedg>0).sum()),
                clear_local_gain=bool(gain>=1 and ci[0]>0 and (seedg>0).sum()>=4))
        rows.append(dict(cell=cell,shot=m['shot'],episodes=3000,accuracy=float(a[:,0].mean()),baseline_accuracy=float(a[:,1].mean()),
            no_retrieval_accuracy=float(a[:,2].mean()),comparisons=comparisons))
    aggregates=[]
    for label,cells in [('old4',old.CELLS[:4]),('new6',old.CELLS[4:]),('all10',old.CELLS)]:
        rng=np.random.default_rng(902194);boot=[]
        for _ in range(1000):
            ids={q:np.stack([rng.integers(600,size=600) for s in range(5)]) for q in {c[0] for c in cells}}
            boot.append(np.mean([np.take_along_axis(all_delta[c],ids[c[0]],axis=1).mean() for c in cells]))
        aggregates.append(dict(scope=label,gain_pp=float(np.mean([all_delta[c].mean() for c in cells])*100),ci95_pp=(100*np.quantile(boot,[.025,.975])).tolist()))
    dump(OUT/method/'summary.json',dict(complete=True,method=method,episodes=30000,files=50,rows=rows,aggregates=aggregates,protocol_sha256=sha(OUT/'protocol.json'),
        scope='Historically viewed fixed image pools; paired seed-stratified episode intervals; same query-domain episodes across galleries resampled jointly; no multiplicity-adjusted or fresh-dataset confirmation'))
    print('METHOD_ALL_COMPLETE',method,aggregates,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('method');args=ap.parse_args();execute(args.method)
