"""Shared-information early/late ridge controls; no label or query-batch inputs."""
import math
import torch
import torch.nn.functional as F
from baseline_methods import representation,ridge_scores
import geometry_reference as geo

FAMILIES=['late_shared','early_shared','early_native','centered_native','late_support']
WEIGHTS=[.5,.25,.75,0.,1.]
LAMBDAS=[.1,1.,.01,.001]
SETTINGS=[(w,l) for w in WEIGHTS for l in LAMBDAS]

def key(f,w,l):return f+'_'+str(w)+'_'+str(l)
def split(x,d):return x[...,:d]*math.sqrt(2),x[...,d:]*math.sqrt(2)
def join(a,b,w):return torch.cat([math.sqrt(w)*a,math.sqrt(1-w)*b],dim=-1)
def fit(A,Q,l):
    y=torch.eye(5,device=A.device,dtype=A.dtype)[None].expand(len(A),-1,-1)
    return ridge_scores(A,y,Q,l)
def late(Ac,Ad,Qc,Qd,w,l):return w*fit(Ac,Qc,l)+(1-w)*fit(Ad,Qd,l)

@torch.no_grad()
def batch_scores(Sc,Sd,Qc,Qd,Gc,Gd,choices):
    S=representation(Sc,Sd,.5);Q=representation(Qc,Qd,.5);G=representation(Gc,Gd,.5)
    shared=geo.prepare(S,Q,G,'none');A=shared[3];dim=Sc.shape[-1]
    Ac,Ad=split(A,dim);Xc,Xd=split(Q,dim);Pc,Pd=split(shared[2],dim)
    out={'r2':fit(A,Q,.1)};view_cache={};native_cache={.5:shared};center_cache={}
    for family,settings in choices.items():
        for w,l in settings:
            if family in ['late_shared','late_support']:
                cache_key=(family,l)
                if cache_key not in view_cache:
                    U,V=(Ac,Ad) if family=='late_shared' else (Pc,Pd)
                    view_cache[cache_key]=(fit(U,Xc,l),fit(V,Xd,l))
                c,d=view_cache[cache_key];score=w*c+(1-w)*d
            elif family=='early_shared':score=fit(join(Ac,Ad,w),join(Xc,Xd,w),l)
            else:
                cache=native_cache if family=='early_native' else center_cache
                if w not in cache:
                    sw=representation(Sc,Sd,w);qw=representation(Qc,Qd,w);gw=representation(Gc,Gd,w)
                    cache[w]=geo.prepare(sw,qw,gw,'none' if family=='early_native' else 'support')
                score=geo.head(cache[w],l)
            out[key(family,w,l)]=score
    if .5 not in center_cache:center_cache[.5]=geo.prepare(S,Q,G,'support')
    out['cs_l2_fixed']=geo.head(center_cache[.5],.1)
    assert all(torch.isfinite(x).all() for x in out.values())
    return out,shared[4]
