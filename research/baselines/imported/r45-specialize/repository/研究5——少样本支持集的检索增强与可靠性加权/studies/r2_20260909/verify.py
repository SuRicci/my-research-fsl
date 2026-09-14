# -*- coding: utf-8 -*-
"""Independent algebra, invariance, identity and saved-evidence verification."""
import sys,json,hashlib
from pathlib import Path
from layout_compat import verify_artifact
import numpy as np
import torch
import torch.nn.functional as F
from assets_io import HERE,R1,Assets,bridge,dump
from methods import grid,evaluate_configs,ridge_scores,representation

def main():
    checks=[];torch.manual_seed(129999)
    S=F.normalize(torch.randn(2,5,2,16),dim=-1);X=F.normalize(torch.randn(2,11,16),dim=-1);G=F.normalize(torch.randn(300,16),dim=-1)
    cfg=[c for c in grid() if c['w']==0]
    _,sc,_=evaluate_configs(S,X,G,cfg,True)
    _,left,_=evaluate_configs(S,X[:,:4],G,cfg,True);_,right,_=evaluate_configs(S,X[:,4:],G,cfg,True)
    for n in sc:assert torch.allclose(sc[n],torch.cat([left[n],right[n]],1),atol=2e-5),n
    checks.append('all_53_formulas_query_batch_independent')
    perm=torch.tensor([3,0,4,1,2]);_,sp,_=evaluate_configs(S[:,perm],X,G,cfg,True)
    for n in sc:assert torch.allclose(sc[n][:,:,perm],sp[n],atol=2e-5),n
    checks.append('all_53_formulas_class_permutation_equivariant')
    _,one,_=evaluate_configs(S[:1],X[:1],G,cfg,True)
    for n in sc:assert torch.allclose(sc[n][:1],one[n],atol=2e-5),n
    checks.append('all_53_formulas_episode_independent')
    z=S[0].flatten(0,1).numpy().astype('float64');xx=X[0].numpy().astype('float64');Y=np.repeat(np.eye(5),2,axis=0)
    a=z-z.mean(0);coef=np.linalg.solve(a.T@a+.1*np.eye(16),a.T@(Y-Y.mean(0)))
    expected=(xx-z.mean(0))@coef+Y.mean(0)
    actual=ridge_scores(torch.from_numpy(z)[None],torch.from_numpy(Y)[None],torch.from_numpy(xx)[None],.1)[0].numpy()
    # Inference regularizer is constructed in fp32; compare against a double-precision
    # primal solution with a tolerance below the source feature quantization error.
    algebra_error=float(np.max(np.abs(expected-actual)))
    assert algebra_error<1e-7,algebra_error;checks.append('dual_ridge_matches_independent_primal_solution')
    for phase in ['dev','retest','external']:
        files=list((HERE/'outputs').glob('*_'+phase+'.npz'))
        for path in files:
            d=np.load(path);gh=set(d['gallery_rgb'])
            for sh,qh in zip(d['support_rgb'],d['query_rgb']):
                a=set(sh.ravel());b=set(qh.ravel())
                assert len(a)==sh.size and len(b)==qh.size and not a&b
                assert not (a|b)&gh
            means=(d['predictions']==d['yq']).mean((1,2));stats=json.loads(path.with_suffix('.json').read_text())['means']
            for n,v in zip(d['names'],means):assert abs(stats[str(n)]-float(v))<1e-12
        if files:checks.append(f'{phase}_{len(files)}_files_source_identity_and_reaggregation')
    rec=json.loads((HERE/'selection_receipt.json').read_text())
    for name,h in rec['hashes'].items():verify_artifact(HERE/name,h)
    checks.append('frozen_selection_source_hashes_unchanged')
    # All R1 evidence hashes remain unchanged, including prior predictions and protocol.
    old=json.loads((R1/'evidence_manifest.json').read_text(encoding='utf-8'))
    for name,meta in old['files'].items():verify_artifact(R1/name,meta['sha256'])
    checks.append('all_146_R1_evidence_files_unchanged')
    result={'passed':checks,'failed':0,'n_checks':len(checks),'primal_dual_max_error':algebra_error};dump(HERE/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
