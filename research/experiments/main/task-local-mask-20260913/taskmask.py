"""SUR-style task-only block masks; query vectors never enter mask fitting."""
import torch
import torch.nn.functional as F

def n(x): return F.normalize(x,dim=-1)
def weight(alpha):
    z=alpha.sigmoid().square()
    return z[:,0]/z.sum(-1)
def support_loss(grams,w,k):
    g=w[:,None,None]*grams[0]+(1-w[:,None,None])*grams[1]
    b,t,_=g.shape;c=t//k
    sp=g.reshape(b,t,c,k).mean(-1)
    pp=sp.reshape(b,c,k,c).mean(2)
    sims=sp/pp.diagonal(dim1=-2,dim2=-1).clamp_min(1e-12).sqrt()[:,None,:]
    y=torch.arange(c).repeat_interleave(k).to(g.device)
    return -sims.log_softmax(-1)[:,torch.arange(t),y].mean(-1)
def masks(sc,sd,cfg):
    k=sc.shape[2];v=[n(s.flatten(1,2)).detach() for s in [sc,sd]]
    grams=[s@s.transpose(1,2) for s in v]
    alpha=torch.zeros(len(sc),2,dtype=sc.dtype,requires_grad=True)
    opt=torch.optim.Adadelta([alpha],lr=cfg['lr'],rho=cfg['rho'],eps=cfg['eps'])
    before=support_loss(grams,weight(alpha),k).detach()
    for _ in range(cfg['steps']):
        opt.zero_grad();loss=support_loss(grams,weight(alpha),k).sum();loss.backward();opt.step()
    w=weight(alpha).detach();after=support_loss(grams,w,k).detach()
    return w,before,after

def combine(a,b,w):
    shape=(len(w),)+tuple(1 for _ in range(a.ndim-1))
    return torch.cat([n(a)*w.sqrt().reshape(shape),n(b)*(1-w).sqrt().reshape(shape)],-1)

@torch.no_grad()
def scores(sc,sd,qc,qd,gc,gd,w,origin,ridge):
    s=combine(sc,sd,w);q=combine(qc,qd,w);b,c,k,d=s.shape
    g=combine(gc[None].expand(b,-1,-1),gd[None].expand(b,-1,-1),w) if k==1 else None
    if origin=='support':
        mean=s.flatten(1,2).mean(1,keepdim=True)
        s=n(s.flatten(1,2)-mean).reshape_as(s);q=n(q-mean)
        if g is not None:g=n(g-mean)
    proto=n(s.mean(2))
    if origin=='ncc': return q@proto.transpose(1,2)
    if k==1:
        idx=(proto@g.transpose(1,2)).topk(64,dim=-1).indices
        mu=n(g[torch.arange(b)[:,None,None],idx].mean(2))
        a=n(.5*proto+.5*mu);y=torch.eye(c)[None].expand(b,-1,-1)
    else:
        a=s.flatten(1,2);y=F.one_hot(torch.arange(c).repeat_interleave(k),c).float()[None].expand(b,-1,-1)
    return ridge(a,y,q,.1 if origin=='support' or k==1 else 1.)
