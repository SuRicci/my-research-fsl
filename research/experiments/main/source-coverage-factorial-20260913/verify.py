"""Numerical invariants for nested influence and source task construction."""
import numpy as np
import torch
import torch.nn.functional as F
import json
from pathlib import Path


def check(ex):
    torch.manual_seed(26091391)
    S=F.normalize(torch.randn(2,5,1,896,dtype=torch.double),dim=-1)
    G=F.normalize(torch.randn(80,896,dtype=torch.double),dim=-1)
    Q=F.normalize(torch.randn(2,15,896,dtype=torch.double),dim=-1)
    ca=ex.compact(S,G);b=torch.zeros(4,dtype=torch.double,requires_grad=True);a=torch.tensor(0.,dtype=torch.double,requires_grad=True)
    sc,_=ex.score(ca,Q,b,a);parity=float((sc-ex.ref.r2_scores(S.float(),Q.float(),G.float())).abs().max());assert parity<1e-6
    b=torch.tensor([.3,-.2,.1,.05],dtype=torch.double,requires_grad=True)
    loss,_=ex.quality(ca,Q,b,a);grad=torch.cat([torch.autograd.grad(loss,(b,a))[0],torch.autograd.grad(ex.quality(ca,Q,b,a)[0],a)[0][None]])
    fd=[];eps=1e-5
    with torch.no_grad():
        for j in range(5):
            vals=[]
            for sign in [1,-1]:
                bb=b.clone();aa=a.clone()
                if j<4:bb[j]+=sign*eps
                else:aa+=sign*eps
                vals.append(ex.quality(ca,Q,bb,aa)[0])
            fd.append((vals[0]-vals[1])/(2*eps))
        err=float((torch.stack(fd)-grad).abs().max());assert err<1e-6
        sc,_=ex.score(ca,Q,b,a);split=torch.cat([ex.score(ca,q,b,a)[0] for q in Q.split(4,1)],1);assert torch.allclose(sc,split,atol=1e-10)
        perm=torch.tensor([2,4,1,0,3]);sp=ex.score(ex.compact(S[:,perm],G),Q,b,a)[0];assert torch.allclose(sp,sc[...,perm],atol=1e-10)
        gp=torch.randperm(len(G));sg=ex.score(ex.compact(S,G[gp]),Q,b,a)[0];assert torch.allclose(sc,sg,atol=1e-10)
        _,al=ex.score(ca,Q,b,a,True);assert torch.equal(al,al[:,:1].expand_as(al))
    counts={}
    for domain in ['dtd','eurosat']:
        ident=json.loads((Path(ex.CFG['asset_root'])/(domain+'_identities.json')).read_text());gy=np.asarray(ident['gallery_labels']);qy=np.asarray(ident['query_labels'])
        assert not set(ident['gallery_rgb']) & set(ident['query_rgb'])
        minimum=len(gy)
        for shot in [1,5]:
            for rep in [0,1,2]:
                tr=ex.tasks(ident,shot,'train',rep);va=ex.tasks(ident,shot,'selection',rep)
                ti=set(np.r_[tr['support_indices'].flatten(),tr['query_indices'].flatten()]);vi=set(np.r_[va['support_indices'].flatten(),va['query_indices'].flatten()]);assert not ti & vi
                for t in [tr,va]:
                    assert np.array_equal(qy[t['support_indices']],np.broadcast_to(t['class_ids'][...,None],t['support_indices'].shape))
                    assert np.array_equal(qy[t['query_indices']],np.broadcast_to(t['class_ids'][...,None],t['query_indices'].shape))
                    for s,q,c in zip(t['support_indices'],t['query_indices'],t['class_ids']):
                        assert len(set(s.flatten())&set(q.flatten()))==0
                        available=int((~np.isin(gy,c)).sum());minimum=min(minimum,available);assert available>=1024
        counts[domain]={'min_excluded_gallery_available':minimum,'allocations':3,'shots':2,'source_train_and_selection_disjoint':True}
    return {'status':'passed','scalar_r2_score_max_error':parity,'finite_difference_max_error':err,'query_partition_invariant':True,'class_permutation_equivariant':True,'gallery_permutation_invariant':True,'all_source_task_feasibility':counts}
