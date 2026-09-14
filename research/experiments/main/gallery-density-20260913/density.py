"""Fixed gallery-local background correction; no labels or query aggregation."""
from pathlib import Path
import hashlib,importlib.util
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('inherited_consistency',HERE.parent/'query-consistency-20260913/consistency_model.py')
z=importlib.util.module_from_spec(spec);spec.loader.exec_module(z)
NAMES=['parent','local','global','shuffled']

def identities(rgb):
    order=np.array(sorted(range(len(rgb)),key=lambda i:hashlib.sha256(('density-v1:'+rgb[i]).encode()).hexdigest()))
    shuffled=np.empty(len(rgb),dtype=np.int64);shuffled[order]=np.roll(order,len(order)//2)
    return order[:128],shuffled

def penalties(g,ri,perm):
    sim=g@g[ri].T
    valid=torch.arange(len(g))[:,None]!=torch.as_tensor(ri)[None]
    local=sim.masked_fill(~valid,-torch.inf).topk(10,dim=1).values.mean(1)
    glob=(sim*valid).sum(1)/valid.sum(1)
    return torch.stack([torch.zeros_like(local),local,glob,local[perm]])

@torch.no_grad()
def evaluate(sv,qv,gm,gamma,eta,ri,perm):
    _,(s,q),factor=z.m.prepare(sv,qv,gamma)
    g=z.m.old.metric.transform(gm.double(),factor)
    v=z.m.old.metric.transform(qv.double(),factor)
    mu=s.flatten(0,1).mean(0);scores=[];neighbors=[];bias=[]
    for sh,qh,gh,vh in [(s,q,g,v),(z.m.n(s-mu),z.m.n(q-mu),z.m.n(g-mu),z.m.n(v-mu))]:
        p=z.m.n(sh.mean(1));b=penalties(gh,ri,perm)
        ix=torch.argsort(p@gh.T-.5*b[:,None,:],dim=-1,descending=True,stable=True)[...,:64]
        xx=z.m.n(.5*p[None]+.5*z.m.n(gh[ix].mean(-2)))
        scores.append(torch.stack([z.head(x,qh,vh,[eta],[0.])['consistency'][0] for x in xx]))
        neighbors.append(ix);bias.append(b)
    return .5*(scores[0]+scores[1]),torch.stack(neighbors),torch.stack(bias)

def independent(sv,qv,gm,gamma,eta,ri,perm):
    # Separate NumPy SVD, dense neighbor scores, direct augmented-design ridge.
    def n(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
    sv=sv.numpy().astype(np.float64);qv=qv.numpy().astype(np.float64);g=gm.numpy().astype(np.float64)
    d=(sv-sv.mean(-2,keepdims=True)).reshape(-1,sv.shape[-1]);trace=(d*d).sum()/len(d)
    _,sigma,V=np.linalg.svd(d,full_matrices=False)
    c=1/np.sqrt(1+gamma*sv.shape[-1]*(sigma*sigma/len(d))/trace)-1 if trace>1e-12 else np.zeros(len(sigma))
    def transform(x):return n(x+((x@V.T)*c)@V)
    s=transform(sv.mean(-2));q=transform(qv.mean(-2));g=transform(g);v=transform(qv);mu=s.reshape(-1,s.shape[-1]).mean(0)
    result=[];neighbors=[];bias=[];eye=np.eye(5);B=eye-eye.mean(0)
    for sh,qh,gh,vh in [(s,q,g,v),(n(s-mu),n(q-mu),n(g-mu),n(v-mu))]:
        similarity=gh@gh[ri].T;mask=np.arange(len(gh))[:,None]!=ri[None];loc=[];glob=[]
        for row,ok in zip(similarity,mask):
            values=row[ok];loc.append(np.sort(values)[-10:].mean());glob.append(values.mean())
        loc=np.array(loc);pen=np.array([np.zeros(len(gh)),loc,np.array(glob),loc[perm]]);bias.append(pen)
        p=n(sh.mean(1));ix=np.argsort(-(p@gh.T-.5*pen[:,None,:]),axis=-1,kind='stable')[...,:64];neighbors.append(ix)
        xx=n(.5*p[None]+.5*n(gh[ix].mean(-2)));rows=[]
        for x in xx:
            xm=x.mean(0);A=x-xm;sc=[]
            for qi,vi in zip(qh,vh):
                R=(vi-vi.mean(0))/np.sqrt(len(vi));X=np.concatenate([A,np.sqrt(eta)*R]);Y=np.concatenate([B,np.zeros((len(vi),5))])
                W=np.linalg.solve(X@X.T+.1*np.eye(len(X)),Y)
                sc.append((qi-xm)@X.T@W+eye.mean(0))
            rows.append(sc)
        result.append(rows)
    return np.mean(result,0),np.array(neighbors),np.array(bias)
