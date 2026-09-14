# -*- coding: utf-8 -*-
"""Inductive frozen-feature methods; signatures contain no gallery/query labels."""
import math,time
from itertools import product
import torch
import torch.nn.functional as F

def representation(clip,dino,w):
    if w==0:return F.normalize(dino,dim=-1)
    if w==1:return F.normalize(clip,dim=-1)
    return F.normalize(torch.cat([math.sqrt(w)*F.normalize(clip,dim=-1),math.sqrt(1-w)*F.normalize(dino,dim=-1)],dim=-1),dim=-1)

def config_name(c):
    return '_'.join(str(c.get(k,'')) for k in ['family','w','r','mix','iterations','beta','lam'])

def grid():
    out=[]
    for w in [0.,.25,.5,1.]:
        cc=[{'family':'raw','w':w}]
        cc += [{'family':'raw_ridge','w':w,'lam':l} for l in [.001,.01,.1,1.]]
        cc += [{'family':'uniform','w':w,'r':r,'mix':m} for r,m in product([16,64,256],[0.,.25,.5,.75,1.])]
        cc += [{'family':'anchor_refine','w':w,'r':64,'mix':m,'iterations':it} for m,it in product([.25,.5,.75],[2,4])]
        cc += [{'family':'kernel_residual','w':w,'r':64,'mix':m,'beta':b} for b,m in product([5.,10.,20.],[.25,.5,.75])]
        cc += [{'family':'retrieval_ridge','w':w,'r':r,'mix':m,'lam':l} for r,m,l in product([16,64],[.25,.5,.75],[.01,.1,1.])]
        for c in cc:c['name']=config_name(c)
        out.extend(cc)
    return out

def ridge_scores(S,Y,X,lam):
    """Batched centered dual ridge with an unpenalized intercept."""
    sm=S.mean(1,keepdim=True);ym=Y.mean(1,keepdim=True)
    centered=S-sm
    K=centered@centered.transpose(1,2)
    coef=torch.linalg.solve(K+lam*torch.eye(S.shape[1],device=S.device)[None],Y-ym)
    return (X-sm)@centered.transpose(1,2)@coef+ym

def retrieve(P,G,r,chunk=24):
    top=[]
    for st in range(0,len(P),chunk):
        sim=P[st:st+chunk].reshape(-1,P.shape[-1])@G.T
        top.append(sim.topk(min(r,len(G)),dim=-1).indices.reshape(-1,5,min(r,len(G))))
    return torch.cat(top)

@torch.no_grad()
def evaluate_configs(S,X,G,configs,return_scores=False):
    """S [E,5,K,D], X [E,75,D], G [M,D]; no query sharing occurs."""
    p=F.normalize(S.mean(2),dim=-1);raw=X@p.transpose(1,2)
    e,k=S.shape[0],S.shape[2];results={};scores={};timing={}
    start=time.perf_counter()
    needs=[c.get('r',0) for c in configs]
    maxr=max(needs) if needs else 0
    idx=retrieve(p,G,maxr) if maxr else None
    means={r:F.normalize(G[idx[:,:,:r]].mean(2),dim=-1) for r in sorted(set(needs)-{0})}
    torch.cuda.synchronize() if S.is_cuda else None
    timing['shared_retrieval_seconds']=time.perf_counter()-start
    kernels={};refined={};rawridge={};Y=F.one_hot(torch.arange(5,device=S.device).repeat_interleave(k),5).float()[None].expand(e,-1,-1)
    for c in configs:
        t=time.perf_counter();fam=c['family']
        if fam=='raw':v=raw
        elif fam=='raw_ridge':
            if c['lam'] not in rawridge:rawridge[c['lam']]=ridge_scores(S.flatten(1,2),Y,X,c['lam'])
            v=rawridge[c['lam']]
        elif fam in ['uniform','retrieval_ridge']:
            aug=F.normalize((1-c['mix'])*p+c['mix']*means[c['r']],dim=-1)
            if fam=='uniform':v=X@aug.transpose(1,2)
            else:v=ridge_scores(aug,torch.eye(5,device=S.device)[None].expand(e,-1,-1),X,c['lam'])
        elif fam=='anchor_refine':
            key=c['mix']
            if key not in refined:
                steps=sorted({cc['iterations'] for cc in configs if cc['family']==fam and cc['mix']==key})
                current=p;refined[key]={}
                for it in range(1,max(steps)+1):
                    if it==1:mu=means[c['r']]
                    else:mu=F.normalize(G[retrieve(current,G,c['r'])].mean(2),dim=-1)
                    current=F.normalize((1-key)*p+key*mu,dim=-1)
                    if it in steps:refined[key][it]=current
            v=X@refined[key][c['iterations']].transpose(1,2)
        elif fam=='kernel_residual':
            if not kernels:
                betas=sorted({cc['beta'] for cc in configs if cc['family']==fam});outputs={b:[] for b in betas}
                for st in range(0,e,24):
                    cand=G[idx[st:st+24,:,:c['r']]]
                    sim=torch.einsum('eqd,ecrd->eqcr',X[st:st+24],cand)
                    for b in betas:outputs[b].append((torch.logsumexp(b*sim,dim=-1)-math.log(cand.shape[2]))/b)
                kernels={b:torch.cat(vv) for b,vv in outputs.items()}
            v=(1-c['mix'])*raw+c['mix']*kernels[c['beta']]
        else:raise ValueError(fam)
        results[c['name']]=v.argmax(-1).cpu()
        if return_scores:scores[c['name']]=v.cpu()
        timing[c['name']]=time.perf_counter()-t
    return results,scores,timing
