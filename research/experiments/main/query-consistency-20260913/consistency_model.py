"""Per-query quadratic response consistency on a fixed two-head ridge blend."""
from pathlib import Path
import importlib.util
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('frozen_fusion',ROOT/'experiments/main/scatter-score-fusion-20260913/study.py')
fusion=importlib.util.module_from_spec(spec);spec.loader.exec_module(fusion)
a=fusion.a; m=fusion.m

def packs(sv,qv,gm,gamma):
    _,(s,q),factor=m.prepare(sv,qv,gamma)
    g=m.old.metric.transform(gm.double(),factor)
    v=m.old.metric.transform(qv.double(),factor)
    mu=s.flatten(0,1).mean(0)
    for sh,qh,gh,vh in [(s,q,g,v),(m.n(s-mu),m.n(q-mu),m.n(g-mu),m.n(v-mu))]:
        proto=m.n(sh.mean(1))
        ix=torch.argsort(proto@gh.T,descending=True,stable=True)[:,:64]
        x=m.n(.5*proto+.5*m.n(gh[ix].mean(1)))
        yield x,qh,vh

def head(x,q,v,etas,iso_etas):
    lam=.1; eye=torch.eye(len(x),dtype=x.dtype)
    xm=x.mean(0); A=x-xm; B=eye-eye.mean(0); qc=q-xm
    gram=A@A.T; F=torch.linalg.solve(gram+lam*eye,eye)
    W=A.T@F@B; parent=qc@W+eye.mean(0)
    R=(v-v.mean(1,keepdim=True))/v.shape[1]**.5
    # H^-1 R^T via the parent five-prototype dual, separately for each query.
    AR=A@R.transpose(-1,-2)
    U=(R.transpose(-1,-2)-A.T@F@AR)/lam
    small=R@U; rhs=R@W; left=(qc[:,None,:]@U)
    consistency=[]
    for eta in etas:
        correction=eta*left@torch.linalg.solve(torch.eye(v.shape[1],dtype=x.dtype)+eta*small,rhs)
        consistency.append(parent-correction.squeeze(1))
    isotropic=[]; trace=R.square().sum((1,2))/x.shape[-1]
    for eta in iso_etas:
        dual=torch.linalg.solve(gram[None]+(lam+eta*trace)[:,None,None]*eye,B.expand(len(q),-1,-1))
        isotropic.append(((qc@A.T)[:,None,:]@dual).squeeze(1)+eye.mean(0))
    return {'consistency':torch.stack(consistency),'isotropic':torch.stack(isotropic),
            'incumbent':parent,'query_view_mean':(v.mean(1)-xm)@W+eye.mean(0)}

@torch.no_grad()
def evaluate(sv,qv,gm,gamma,etas,iso_etas):
    heads=[head(*p,etas,iso_etas) for p in packs(sv,qv,gm,gamma)]
    return {k:.5*(heads[0][k]+heads[1][k]) for k in heads[0]}
