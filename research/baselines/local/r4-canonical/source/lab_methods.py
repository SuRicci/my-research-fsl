"""Reusable old-gallery feature maps for bounded-memory exact inner-product evaluation."""
import numpy as np
from .methods import unit,local_weights
from .improved import kernel_matrices
from .nextstage import residual_field,eigensystem,press_choices,REGS

METHODS=['old_old','relative_centered','procrustes','ridge','wip_restricted','local_std','linear_std','v0_press','global_press']

class FeatureMap:
    """Fit accepts only paired bridge embeddings. No paths, labels, or gallery-new values."""
    def __init__(self,ao,an,method):
        if method not in METHODS:raise ValueError(method)
        self.ao,self.an=unit(ao),unit(an);self.method=method
        self.do,self.dn=ao.shape[1],an.shape[1]
        if method in ['linear_std','v0_press','global_press']:
            self.kind='linear' if method=='linear_std' else 'rbf_0.5'
            self.ka,_=kernel_matrices(self.ao,self.ao,self.kind)
            self.e,self.u,self.scale=eigensystem(self.ka)
        if method in ['ridge','procrustes']:
            self.mo,self.mn=self.ao.mean(0),self.an.mean(0)
            x,y=self.an-self.mn,self.ao-self.mo
            if method=='ridge':
                k=x@x.T;n=len(x);scores=[]
                for reg in [.001,.01,.1,1.]:
                    inverse=np.linalg.inv(k+reg*np.eye(n));h=k@inverse+np.ones((n,n))/n
                    loss=np.mean(((self.ao-h@self.ao)/np.maximum(1-np.diag(h)[:,None],1e-10))**2)
                    scores.append((loss,reg))
                self.reg=min(scores)[1];self.transform=x.T@np.linalg.solve(k+self.reg*np.eye(n),y)
            else:
                self.d=max(self.do,self.dn);self.so=max(np.sqrt(np.mean(y*y)),1e-10);self.sn=max(np.sqrt(np.mean(x*x)),1e-10)
                u,_,vt=np.linalg.svd(np.pad(x/self.sn,((0,0),(0,self.d-self.dn))).T@np.pad(y/self.so,((0,0),(0,self.d-self.do))),full_matrices=False)
                self.transform=u@vt
        if method=='wip_restricted':
            self.wip=[]
            for a in [self.ao,self.an]:
                r=a@a.T;mean=r.mean(0);c=r-mean;cov=c.T@c/max(len(a)-1,1)
                e,u=np.linalg.eigh((cov+cov.T)/2);scale=max(np.trace(cov)/len(cov),1e-8)
                self.wip.append((mean,(u/np.sqrt(np.maximum(e,0)+.1*scale))@u.T))

    def gallery(self,old):
        old=unit(old);m=self.method
        if m in ['old_old','ridge']:return old
        if m=='procrustes':return unit(np.pad((old-self.mo)/self.so,((0,0),(0,self.d-self.do))))
        if m=='relative_centered':
            x=old@self.ao.T;return unit(x-x.mean(1,keepdims=True))
        if m=='wip_restricted':
            mean,t=self.wip[0];return unit((old@self.ao.T-mean)@t)
        if m=='local_std':w,_=local_weights(old,self.ao,16,.05);return np.concatenate([old,w],1)
        _,kg=kernel_matrices(old,self.ao,self.kind)
        return np.concatenate([old,kg@self.u],1)

    def query(self,old,new):
        old,new=unit(old),unit(new);m=self.method
        if m=='old_old':return old
        if m=='ridge':return unit((new-self.mn)@self.transform+self.mo)
        if m=='procrustes':return unit(np.pad((new-self.mn)/self.sn,((0,0),(0,self.d-self.dn)))@self.transform)
        if m=='relative_centered':
            x=new@self.an.T;return unit(x-x.mean(1,keepdims=True))
        if m=='wip_restricted':
            mean,t=self.wip[1];return unit((new@self.an.T-mean)@t)
        delta,_=residual_field(old,new,self.ao,self.an,'global' if m=='global_press' else 'std')
        if m=='local_std':return np.concatenate([old,delta],1)
        coeff,_=press_choices(delta,self.ka)
        return np.concatenate([old,(delta@self.u)/(self.e[None,:]+self.scale*coeff[:,None])],1)
