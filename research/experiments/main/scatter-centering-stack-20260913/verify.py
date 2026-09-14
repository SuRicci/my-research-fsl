"""Independent dense-eigen transform and primal ridge; actual-case regression."""
from pathlib import Path
import json,time
import numpy as np
import torch
import model as m
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)

def norm(x):
    return x/np.linalg.norm(x,axis=-1,keepdims=True)

def dense(sv,qv,gm,gamma,center,lam):
    s=sv.numpy().astype(float);q=qv.numpy().astype(float);g=gm.numpy().astype(float)
    d=(s-s.mean(-2,keepdims=True)).reshape(-1,s.shape[-1]);tr=np.sum(d*d)/len(d)
    mat=np.eye(s.shape[-1])
    if gamma and tr>1e-12:
        vals,vectors=np.linalg.eigh(mat+gamma*s.shape[-1]*(d.T@d/len(d))/tr)
        mat=(vectors/np.sqrt(vals))@vectors.T
    s=norm(s.mean(-2)@mat);q=norm(q.mean(-2)@mat);g=norm(g@mat)
    k=s.shape[1];c=len(s)
    if center:
        mu=s.reshape(-1,s.shape[-1]).mean(0);s=norm(s-mu);q=norm(q-mu);g=norm(g-mu)
    ix=np.empty((c,0),dtype=np.int64)
    if k==1:
        proto=norm(s.mean(1));ix=np.argsort(-(proto@g.T),axis=-1,kind='stable')[:,:64]
        x=norm(.5*proto+.5*norm(g[ix].mean(1)));y=np.eye(c)
    else:x=s.reshape(-1,s.shape[-1]);y=np.repeat(np.eye(c),k,axis=0)
    xm=x.mean(0);ym=y.mean(0);xc=x-xm
    weights=np.linalg.solve(xc.T@xc+lam*np.eye(x.shape[-1]),xc.T@(y-ym))
    return (q-xm)@weights+ym,ix

def check(s,q,gm,gamma,label):
    sc,ix=m.packages(m.prepare(s,q,gamma),gm);rows=[]
    for j,name in enumerate(m.NAMES):
        lam=.1 if s.shape[1]==1 or name.endswith(('cs','lam01')) else 1.
        expected,ind=dense(s,q,gm,gamma if name.startswith('scatter') else 0,name.endswith('cs'),lam)
        error=float(abs(sc[j].numpy()-expected).max())
        assert error<2e-5,(label,name,error)
        assert np.array_equal(sc[j].argmax(-1).numpy(),expected.argmax(-1)),(label,name,'prediction')
        assert np.array_equal(ix[j].numpy(),ind),(label,name,'neighbors')
        rows.append({'label':label,'method':name,'max_error':error,'neighbors_equal':True})
    return rows

def main():
    start=time.time();torch.set_num_threads(1);torch.manual_seed(26091392);rows=[]
    for k in [1,5]:
        s=m.n(torch.randn(5,k,6,24,dtype=torch.float64));q=m.n(torch.randn(17,6,24,dtype=torch.float64));g=m.n(torch.randn(80,6,24,dtype=torch.float64));gm=g.mean(-2)
        rows+=check(s,q,gm,1.,'synthetic_k'+str(k))
        sc,_=m.packages(m.prepare(s,q,1.),gm)
        split=torch.cat([m.packages(m.prepare(s,v,1.),gm)[0] for v in q.split(4)],1)
        assert torch.allclose(sc,split,atol=1e-10,rtol=0)
        perm=torch.tensor([3,0,4,1,2]); permuted=m.packages(m.prepare(s[perm],q,1.),gm)[0]
        assert torch.allclose(sc[:,:,perm],permuted,atol=1e-10,rtol=0)
        zero=m.packages(m.prepare(s,q,0.),gm)[0]
        assert torch.allclose(zero[0],zero[2],atol=1e-12) and torch.allclose(zero[1],zero[3],atol=1e-12)
        # Exact ties follow the canonical array identity rather than unstable topk.
        sim=torch.tensor([[1.,1.,0.]],dtype=torch.float64)
        assert torch.argsort(sim,descending=True,stable=True).tolist()==[[0,1,2]]
    data=m.old.load();z=np.load(m.old.OUT/'dtd_k1_selection.npz');i=99
    s=data['dtd']['query'][z['support_indices'][i]];q=data['dtd']['query'][z['query_indices'][i].reshape(-1)]
    gm=data['dtd']['gallery'][z['gallery_indices'][i,1]].double().mean(-2)
    rows+=check(s,q,gm,.1,'original_failure_dtd99')
    result={'status':'passed','checks':rows,'max_error':max(x['max_error'] for x in rows),'query_partition':True,'class_equivariance':True,'zero_gamma_identity':True,'stable_tie_order':True,'original_failure_repaired_for_new_float64_variant':True,'original_frozen_audit_status':'failed_unchanged','torch':torch.__version__,'numpy':np.__version__,'elapsed_seconds':time.time()-start}
    (OUT/'numeric_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
