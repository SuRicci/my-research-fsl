"""Method-only interfaces: no target new-gallery vectors or relevance labels."""
import sys
import numpy as np
import torch
from scipy.special import ndtri,softmax
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from campaign_common import ROOT
sys.path.insert(0,str(ROOT/'lab/refine_20260909'))
from r4_geometry import corrected
from r4_refine import unit,FeatureMap
from src.nextstage import residual_field,eigensystem,press_choices
from src.improved import kernel_matrices


def base_scores(go,qo,qn,ao,an):return corrected(go,qo,qn,ao,an)

def graph(go,cfg,device='cuda:0'):
    g=torch.tensor(unit(go),device=device);rows=[];cols=[];values=[]
    k=min(cfg['neighbors'],len(g)-1)
    for start in range(0,len(g),256):
        end=min(start+256,len(g));s=g[start:end]@g.T
        s[torch.arange(end-start,device=device),torch.arange(start,end,device=device)]=-torch.inf
        v,ix=s.topk(k,dim=1);v=torch.softmax(v/cfg['temperature'],dim=1)
        rows.append(np.repeat(np.arange(start,end),k));cols.append(ix.cpu().numpy().ravel());values.append(v.cpu().numpy().ravel())
    return csr_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(len(g),len(g)))

def prepare(go,qo,method,cfg,device='cuda:0'):
    if method=='r4_whiten':
        mu=unit(go).mean(0);x=(unit(go)-mu).astype('float64');q=(unit(qo)-mu).astype('float64')
        cov=x.T@x/max(len(x)-1,1);e,u=np.linalg.eigh(cov)
        e=np.maximum(e,0)+cfg['ridge_fraction']*max(np.trace(cov)/len(cov),1e-10)
        t=u*(e**(-cfg['power']))
        return {'scores':unit(q@t)@unit(x@t).T}
    if method=='r4_diffusion':return {'graph':graph(go,cfg,device)}
    return {}

def interpolate(delta,go,ao):
    ka,kg=kernel_matrices(go,ao,'rbf_0.5');e,u,scale=eigensystem(ka);regs,_=press_choices(delta,ka)
    return ((delta@u)/(e[None,:]+scale*regs[:,None]))@(kg@u).T

def run(go,qo,qn,ao,an,method,cfg,prepared=None):
    go,qo,qn,ao,an=map(unit,[go,qo,qn,ao,an]);prepared=prepared or {}
    base=base_scores(go,qo,qn,ao,an);old=base['old_old'];center=base['old_centered']
    ratio=center.std(1,keepdims=True)/np.maximum(old.std(1,keepdims=True),1e-10)
    info={'bridge_rows':len(ao),'new_gallery_rows':0,'query_batch_fitting':False}
    if method=='r4_whiten':
        info['bridge_rows_used']=0;return prepared['scores'],None,base,info
    if method=='r4_rank':
        olda=qo@ao.T;newa=qn@an.T
        zn=ndtri((rankdata(newa,axis=1,method='average')-.5)/len(an))
        zo=ndtri((rankdata(olda,axis=1,method='average')-.5)/len(ao))
        delta=(zn-zo)*olda.std(1,keepdims=True);r=interpolate(delta,go,ao)
    elif method=='r4_diffusion':
        h0=softmax((go@ao.T)/cfg['temperature'],axis=1);h=h0.copy()
        for _ in range(cfg['steps']):h=cfg['restart']*h0+(1-cfg['restart'])*(prepared['graph']@h)
        delta,_=residual_field(qo,qn,ao,an,'std');r=delta@h.T
        info['max_weight_row_sum_error']=float(np.abs(h.sum(1)-1).max())
    elif method=='r4_ensemble':
        order=np.lexsort(np.concatenate([ao,an],axis=1).T[::-1]);rng=np.random.default_rng(cfg['seed']);rows=[]
        for _ in range(cfg['members']):
            ix=order[rng.permutation(len(order))[:max(4,int(len(order)*cfg['fraction']))]]
            fm=FeatureMap(ao[ix],an[ix],'v0_press');rows.append(fm.query(qo,qn)@fm.gallery(go).T-old)
        v=np.stack(rows);mu=v.mean(0);var=v.var(0)
        factor=mu*mu/(mu*mu+var+1e-12);r=mu*factor
        info.update(mean_shrink=float(factor.mean()),members=cfg['members'],distinct_bridge_budget=len(ao))
    elif method=='r4_top100':
        rank=np.argsort(-center.astype('float32'),axis=1,kind='stable');k=min(cfg['topk'],rank.shape[1]);ix=rank[:,:k].copy()
        inside=np.take_along_axis(base['centered_residual_2.0'].astype('float32'),ix,axis=1)
        order=np.argsort(-inside,axis=1,kind='stable');rank[:,:k]=np.take_along_axis(ix,order,axis=1)
        info['topk']=k;return None,rank,base,info
    else:raise ValueError(method)
    r=r-r.mean(1,keepdims=True)
    return center+cfg['alpha']*ratio*r,None,base,info
