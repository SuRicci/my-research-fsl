import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from campaign_common import ROOT,HERE,OUT,read,dump,sha
from run_r5 import old,BASE,representation
from r5_candidates import ridge_scores

def main():
    torch.set_num_threads(4);device='cuda:0';dest=OUT/'cv_uniform_control';dest.mkdir(parents=True,exist_ok=True)
    cfg=read(HERE/'candidates.json')['r5_support_cv'];rows=[];cache={}
    for cell in old.CELLS:
        if cell[0] not in cache:
            qm,qfs=old.load(BASE,cell[0],'test');cache[cell[0]]={w:representation(*qfs,w) for w in cfg['weights']}
        reps=cache[cell[0]];packs=[]
        for p in sorted((OUT/'r5_support_cv/test').glob('_'.join(cell)+'_k5_s*.npz')):
            source=read(p.with_suffix('.json'));assert sha(p)==source['sha256'];z=np.load(p,allow_pickle=False);target=dest/p.name;parts=[];t=time.time()
            for st in range(0,600,24):
                ss=z['support_indices'][st:st+24];xx=z['query_indices'][st:st+24].reshape(-1,75)
                Y=torch.nn.functional.one_hot(torch.arange(5,device=device).repeat_interleave(5),5).float()[None].expand(len(ss),-1,-1);v=[]
                for w in cfg['weights']:v.append(ridge_scores(reps[w][ss].to(device).flatten(1,2),Y,reps[w][xx].to(device),cfg['lambda']))
                score=.5*v[cfg['weights'].index(.5)]+.5*torch.stack(v).mean(0);parts.append(score.cpu().numpy())
            scores=np.concatenate(parts);pred=scores.argmax(-1).astype('int8');a=(pred==z['yq']).mean(-1);packs.append((z['accuracy'][0],a,z['accuracy'][1]))
            np.savez_compressed(target,scores=scores,predictions=pred,accuracy=a,yq=z['yq'])
            dump(target.with_suffix('.json'),dict(source=str(p),source_sha256=sha(p),sha256=sha(target),seconds=time.time()-t))
        assert len(packs)==5;a=np.array(packs);delta=a[:,0]-a[:,1];rng=np.random.default_rng(902198)
        boot=[np.mean([v[rng.integers(600,size=600)].mean() for v in delta]) for _ in range(1000)]
        rows.append(dict(cell=cell,cv_accuracy=float(a[:,0].mean()),uniform_accuracy=float(a[:,1].mean()),r2_accuracy=float(a[:,2].mean()),
            cv_vs_uniform_pp=float(delta.mean()*100),ci95_pp=(100*np.quantile(boot,[.025,.975])).tolist(),uniform_vs_r2_pp=float((a[:,1]-a[:,2]).mean()*100)))
        print('CV_CONTROL',rows[-1],flush=True)
    dump(dest/'summary.json',dict(complete=True,episodes=30000,files=50,rows=rows,analysis_source_sha256=sha(__file__),
        note='Predeclared conditional attribution control; same five predictor outputs with constant weights, no parameter fitting'))

if __name__=='__main__':main()
