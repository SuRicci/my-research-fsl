# -*- coding: utf-8 -*-
"""Fixed counterfactual ablations on saved episodes. No parameter selection."""
import json
import numpy as np
import torch
from assets_io import HERE,Assets,dump
from methods import evaluate_configs,representation
from analyze import bootstrap

def main():
    store=Assets();rows=[];out=HERE/'diagnostics';out.mkdir(exist_ok=True)
    for path in sorted((HERE/'outputs').glob('*.npz')):
        if '_dev.' in path.name:continue
        d=np.load(path);meta=json.loads(path.with_suffix('.json').read_text());q,g=store.pair(*meta['cell']);shot=meta['shot']
        lut={int(v):i for i,v in enumerate(q['ids'])}
        si=np.array([lut[int(i)] for i in d['support_indices'].ravel()]).reshape(d['support_indices'].shape)
        qi=np.array([lut[int(i)] for i in d['query_indices'].ravel()]).reshape(len(d['seed']),75)
        cfg=[{'family':'raw','w':0.,'name':'dino_raw'}, {'family':'raw','w':.5,'name':'fused_raw'},
             {'family':'raw_ridge','w':0.,'lam':.1 if shot==1 else 1.,'name':'dino_raw_ridge'},
             {'family':'raw_ridge','w':.5,'lam':.1 if shot==1 else 1.,'name':'fused_raw_ridge'}]
        if shot==1:
            cfg.extend([{'family':'uniform','w':.5,'r':64,'mix':.5,'name':'matched_fused_uniform'},
                        {'family':'retrieval_ridge','w':0.,'r':64,'mix':.5,'lam':.1,'name':'matched_dino_retrieval_ridge'}])
        else:cfg.append({'family':'uniform','w':.5,'r':16,'mix':.25,'name':'matched_fused_uniform'})
        preds={}
        for w in [0.,.5]:
            qf=representation(*q['features'],w);G=representation(*g['features'],w).cuda()
            pred,_,_=evaluate_configs(qf[si].cuda(),qf[qi].cuda(),G,[c for c in cfg if c['w']==w]);preds.update({n:p.numpy() for n,p in pred.items()})
        selection=json.loads((HERE/'selection.json').read_text())[str(shot)]['overall']['name']
        chosen=d['predictions'][d['names'].tolist().index(selection)]
        preds['frozen_overall']=chosen
        acc={n:(p==d['yq']).mean(-1) for n,p in preds.items()}
        row={'cell':meta['cell'],'phase':meta['phase'],'shot':shot,'accuracy':{n:float(v.mean()) for n,v in acc.items()}}
        for a,b in [('frozen_overall','fused_raw'),('frozen_overall','matched_fused_uniform'),('fused_raw','dino_raw'),('fused_raw_ridge','fused_raw')]:
            delta=acc[a]-acc[b];row[a+'_minus_'+b]={'mean':float(delta.mean()),'ci95':bootstrap(delta)}
        rows.append(row)
        np.savez_compressed(out/path.name,names=np.array(list(preds)),predictions=np.stack(list(preds.values())),yq=d['yq'],seed=d['seed'])
    dump(HERE/'ablation_summary.json',{'scope':'Post-freeze fixed ablations only, on the same already-evaluated episodes. No new independent evidence or policy reselection.','rows':rows})
    print('fixed ablation comparisons complete',len(rows),flush=True)

if __name__=='__main__':main()
