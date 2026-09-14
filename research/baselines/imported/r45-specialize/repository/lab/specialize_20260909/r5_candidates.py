"""Inductive candidate transformations; target labels never enter these functions."""
import sys
import torch
import torch.nn.functional as F
from campaign_common import ROOT
sys.path.insert(0,str(ROOT/'lab/refine_20260909'))
from r5_methods import representation,ridge_scores


def weighted_ridge(S,Y,X,weights,lam):
    w=weights[:,:,None];sw=w.sqrt();total=w.sum(1,keepdim=True)
    sm=(w*S).sum(1,keepdim=True)/total;ym=(w*Y).sum(1,keepdim=True)/total
    c=(S-sm)*sw;k=c@c.transpose(1,2)
    coef=torch.linalg.solve(k+lam*torch.eye(S.shape[1],device=S.device,dtype=S.dtype)[None],(Y-ym)*sw)
    return (X-sm)@c.transpose(1,2)@coef+ym

def geometric_median(x,steps=8):
    m=x.mean(2)
    for _ in range(steps):
        w=1/torch.linalg.vector_norm(x-m[:,:,None],dim=-1).clamp_min(1e-6)
        m=(x*w[:,:,:,None]).sum(2)/w.sum(2)[:,:,None]
    return F.normalize(m,dim=-1)

def support_cv(supports,queries,cfg):
    e,c,k,d=supports[0].shape;n=c*k;device=supports[0].device
    Y=F.one_hot(torch.arange(c,device=device).repeat_interleave(k),c).to(supports[0].dtype)[None].expand(e,-1,-1)
    risks=[];preds=[]
    for S,X in zip(supports,queries):
        a=S.flatten(1,2);z=a-a.mean(1,keepdim=True);K=z@z.transpose(1,2)
        inv=torch.linalg.inv(K+cfg['lambda']*torch.eye(n,device=device)[None])
        H=K@inv+torch.ones_like(K)/n;loo=Y-(Y-H@Y)/(1-H.diagonal(dim1=-2,dim2=-1)).clamp_min(1e-6)[:,:,None]
        risks.append(((loo-Y)**2).mean((1,2)));preds.append(ridge_scores(a,Y,X,cfg['lambda']))
    risks=torch.stack(risks,1);weights=torch.softmax(-risks/cfg['temperature'],dim=1)
    weights=(1-cfg['shrink_to_equal'])*weights;weights[:,cfg['weights'].index(.5)]+=cfg['shrink_to_equal']
    score=(torch.stack(preds,1)*weights[:,:,None,None]).sum(1)
    return score,{'fusion_weights':weights.cpu().numpy(),'support_loo_mse':risks.cpu().numpy()}

@torch.no_grad()
def run(S,X,G,sc,sd,gc,gd,method,cfg):
    e,c,k,d=S.shape;P=F.normalize(S.mean(2),dim=-1)
    ids=(P.reshape(-1,d)@G.T).topk(min(cfg['topk'],len(G)),dim=-1).indices.reshape(e,c,-1)
    N=G[ids];U=F.normalize(N.mean(2),dim=-1);info={}
    eye=torch.eye(c,device=S.device,dtype=S.dtype)[None].expand(e,-1,-1)
    if method=='r5_consensus':
        pc=F.normalize(sc.mean(2),dim=-1);pd=F.normalize(sd.mean(2),dim=-1)
        ic=(pc.flatten(0,1)@gc.T).topk(cfg['topk'],dim=-1).indices.reshape(e,c,-1)
        it=(pd.flatten(0,1)@gd.T).topk(cfg['topk'],dim=-1).indices.reshape(e,c,-1)
        mask=(ids[:,:,:,None]==ic[:,:,None,:]).any(-1)&(ids[:,:,:,None]==it[:,:,None,:]).any(-1)
        count=mask.sum(-1);v=(N*mask[:,:,:,None]).sum(2)/count.clamp_min(1)[:,:,None]
        U=torch.where((count>0)[:,:,None],F.normalize(v,dim=-1),P)
        info['retained_count']=count.cpu().numpy()
    elif method=='r5_median':U=geometric_median(N,cfg['iterations'])
    elif method=='r5_variance':
        v=((N-N.mean(2,keepdim=True))**2).mean((1,2));v=(1-cfg['shrink'])*v+cfg['shrink']*v.mean(1,keepdim=True)
        scale=v.clamp_min(1e-8).pow(-cfg['power']);scale=scale/scale.mean(1,keepdim=True)
        P=F.normalize(P*scale[:,None],dim=-1);X=F.normalize(X*scale[:,None],dim=-1)
        N=F.normalize(N*scale[:,None,None],dim=-1);U=F.normalize(N.mean(2),dim=-1)
        info['scale_max']=scale.max(1).values.cpu().numpy()
    elif method=='r5_distribution':
        ix=torch.linspace(0,N.shape[2]-1,cfg['kept'],device=S.device).long();n=N[:,:,ix];r=len(ix)
        a=torch.cat([S.flatten(1,2),n.flatten(1,2)],1)
        y=torch.cat([eye.repeat_interleave(k,dim=1),eye.repeat_interleave(r,dim=1)],1)
        w=torch.cat([torch.ones(e,c*k,device=S.device),torch.full((e,c*r),cfg['pseudo_total_weight_per_class']/r,device=S.device)],1)
        return weighted_ridge(a,y,X,w,cfg['lambda']),{'pseudo_count':c*r,'true_count':c*k,'pseudo_weight':cfg['pseudo_total_weight_per_class']}
    else:raise ValueError(method)
    aug=F.normalize((1-cfg['mix'])*P+cfg['mix']*U,dim=-1)
    return ridge_scores(aug,eye,X,cfg['lambda']),info
