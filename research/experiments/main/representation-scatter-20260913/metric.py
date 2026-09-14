"""Finite-view scatter metric and fixed equal-view controls (CPU)."""
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'r2-geometry-canonical-20260912'))
import reference_eval as ref
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'r2-fusion-stage-canonical-20260912'))
import geometry_reference as geo

def norm(x):return F.normalize(x,dim=-1)

def scatter_factor(S,gamma):
    # S:class,shot,view,dimension. Never estimate from gallery/query.
    D=(S-S.mean(-2,keepdim=True)).reshape(-1,S.shape[-1])
    trace=D.square().sum()/len(D)
    if gamma==0 or trace<=1e-12:return D[:0],D.new_empty((0,))
    _,sv,Vh=torch.linalg.svd(D,full_matrices=False)
    eigen=sv.square()/len(D)
    coeff=(1+gamma*S.shape[-1]*eigen/trace).rsqrt()-1
    return Vh,coeff

def transform(X,factor):
    V,c=factor
    return norm(X+((X@V.T)*c)@V)

def scatter(S,Q,G,gamma):
    factor=scatter_factor(S,gamma)
    return ref.r2_scores(transform(S.mean(-2),factor)[None],transform(Q.mean(-2),factor)[None],transform(G.mean(-2),factor))[0]

def medoid(X):
    index=(X*X.mean(-2,keepdim=True)).sum(-1).argmax(-1)
    return X.gather(-2,index[...,None,None].expand(*index.shape,1,X.shape[-1])).squeeze(-2)

def controls(S,Q,G):
    s,q,g=norm(S.mean(-2)),norm(Q.mean(-2)),norm(G.mean(-2))
    out={'original_r2':ref.r2_scores(S[:,:,0][None],Q[:,0][None],G[:,0])[0],
         'mean_r2':ref.r2_scores(s[None],q[None],g)[0],
         'medoid_r2':ref.r2_scores(medoid(S)[None],medoid(Q)[None],medoid(G))[0],
         'score_ensemble_r2':torch.stack([ref.r2_scores(S[:,:,v][None],Q[:,v][None],G[:,v])[0] for v in range(S.shape[-2])]).mean(0)}
    out['original_CS_l2']=geo.head(geo.prepare(S[:,:,0][None],Q[:,0][None],G[:,0],'support'),.1)[0]
    out['mean_CS_l2']=geo.head(geo.prepare(s[None],q[None],g,'support'),.1)[0]
    shot=S.shape[1];X=S.flatten(0,2)[None]
    Y=F.one_hot(torch.arange(5).repeat_interleave(shot*6),5).to(S.dtype)[None]
    weights=torch.full((5*shot*6,),1/6,dtype=S.dtype)
    out['augmented_support_ridge']=ref.weighted_ridge(X,Y,q[None],weights,.1 if shot==1 else 1.)[0]
    for C in [1,10]:
        lr=LogisticRegression(C=C,solver='lbfgs',multi_class='multinomial',max_iter=2000,tol=1e-8,random_state=26091275)
        lr.fit(s.reshape(-1,s.shape[-1]).numpy(),np.repeat(np.arange(5),shot))
        out['mean_logistic_C'+str(C)]=torch.from_numpy(lr.decision_function(q.numpy())).to(S.dtype)
    return out
