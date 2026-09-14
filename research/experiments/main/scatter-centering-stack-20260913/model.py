"""Fixed float64 scatter then support-centering composition; no fitting on queries."""
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'representation-scatter-20260913'))
import evaluate as old
NAMES = ['mean_r2','mean_cs','scatter_r2','scatter_cs','mean_raw_lam01','scatter_raw_lam01']

def n(x):
    assert x.norm(dim=-1).min() > 1e-10
    return F.normalize(x, dim=-1)

def head(s, q, g, center, lam):
    k=s.shape[1]; c=len(s)
    if center:
        mu=s.flatten(0,1).mean(0)
        s=n(s-mu); q=n(q-mu)
        if k==1: g=n(g-mu)
    ix=torch.empty((c,0),dtype=torch.long)
    if k==1:
        proto=n(s.mean(1))
        ix=torch.argsort(proto@g.T,dim=-1,descending=True,stable=True)[:,:64]
        x=n(.5*proto+.5*n(g[ix].mean(1))); y=torch.eye(c,dtype=s.dtype)
    else:
        x=s.flatten(0,1); y=torch.eye(c,dtype=s.dtype).repeat_interleave(k,0)
    xm=x.mean(0); ym=y.mean(0); xc=x-xm; yc=y-ym
    dual=torch.linalg.solve(xc@xc.T+lam*torch.eye(len(x),dtype=s.dtype),yc)
    return (q-xm)@xc.T@dual+ym,ix

def prepare(sv,qv,gamma):
    sv=sv.double(); qv=qv.double()
    factor=old.metric.scatter_factor(sv,gamma)
    raw=(n(sv.mean(-2)),n(qv.mean(-2)))
    corrected=tuple(old.metric.transform(x.mean(-2),factor) for x in [sv,qv])
    return raw,corrected,factor

@torch.no_grad()
def packages(prepared, gallery_mean):
    raw,corrected,factor=prepared; shot=raw[0].shape[1]
    g=n(gallery_mean.double()) if shot==1 else None
    gs=old.metric.transform(gallery_mean.double(),factor) if shot==1 else None
    output={};neighbors={}
    for name,features,gal,center,lam in [
        ('mean_r2',raw,g,False,.1 if shot==1 else 1.),
        ('mean_cs',raw,g,True,.1),
        ('scatter_r2',corrected,gs,False,.1 if shot==1 else 1.),
        ('scatter_cs',corrected,gs,True,.1),
        ('mean_raw_lam01',raw,g,False,.1),
        ('scatter_raw_lam01',corrected,gs,False,.1)]:
        output[name],neighbors[name]=head(*features,gal,center,lam)
    return torch.stack([output[x] for x in NAMES]),torch.stack([neighbors[x] for x in NAMES])
