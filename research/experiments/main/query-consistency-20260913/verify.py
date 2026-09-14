"""Independent primal checks and inference-boundary invariances."""
import json,time
import numpy as np
import torch
import consistency_model as z

HERE=z.HERE

def dense(x,q,v,eta,isotropic=False):
    x=x.numpy();q=q.numpy();v=v.numpy();y=np.eye(len(x));xm=x.mean(0);A=x-xm;B=y-y.mean(0)
    out=[]
    for point,views in zip(q,v):
        D=views-views.mean(0);C=D.T@D/len(D)
        if isotropic:C=np.eye(x.shape[-1])*np.trace(C)/x.shape[-1]
        W=np.linalg.solve(A.T@A+.1*np.eye(x.shape[-1])+eta*C,A.T@B)
        out.append((point-xm)@W+y.mean(0))
    return np.array(out)

def main():
    start=time.time();torch.set_num_threads(6);torch.manual_seed(26091391);rows=[]
    for dim in [13,896]:
        x=z.m.n(torch.randn(5,dim,dtype=torch.float64));v=z.m.n(torch.randn(3,6,dim,dtype=torch.float64));q=z.m.n(v.mean(1))
        out=z.head(x,q,v,[0,.1,1,10],[0,.1,1,10])
        for i,eta in enumerate([0,.1,1,10]):
            for family in ['consistency','isotropic']:
                expected=dense(x,q,v,eta,family=='isotropic');err=float(abs(expected-out[family][i].numpy()).max());assert err<1e-10
                rows.append({'dim':dim,'eta':eta,'family':family,'error':err})
        assert torch.equal(out['consistency'][0],out['incumbent'])
        repeated=q[:,None].expand(-1,6,-1);same=z.head(x,q,repeated,[10],[10]);assert torch.allclose(same['consistency'][0],same['incumbent'],atol=1e-12,rtol=0)
        assert (out['consistency'][-1]-out['incumbent']).abs().max()>1e-4
        assert (out['consistency'][-1]-out['query_view_mean']).abs().max()>1e-4
        perm=torch.tensor([3,0,4,1,2]);pp=z.head(x[perm],q,v,[1],[1]);assert torch.allclose(pp['consistency'][0],out['consistency'][2][:,perm],atol=1e-10,rtol=0)
        parts=[z.head(x,q[i:i+1],v[i:i+1],[1],[1])['consistency'][0] for i in range(3)]
        assert torch.allclose(torch.cat(parts),out['consistency'][2],atol=1e-10,rtol=0)
        vp=z.head(x,q,v[:,[4,0,5,3,1,2]],[1],[1]);assert torch.allclose(vp['consistency'][0],out['consistency'][2],atol=1e-10,rtol=0)
    sv=z.m.n(torch.randn(5,1,6,32,dtype=torch.float64));qv=z.m.n(torch.randn(7,6,32,dtype=torch.float64));gm=z.m.n(torch.randn(80,32,dtype=torch.float64))
    out=z.evaluate(sv,qv,gm,1,[0,1],[1]);_,(s,q),f=z.m.prepare(sv,qv,1);parent,_=z.a.heads(s,q,z.m.old.metric.transform(gm,f))
    assert torch.allclose(out['incumbent'],.5*(parent[0]+parent[3]),atol=1e-12,rtol=0)
    order=torch.tensor([3,1,6,0,2,5,4]);rev=z.evaluate(sv,qv[order],gm,1,[0,1],[1]);assert torch.allclose(rev['consistency'],out['consistency'][:,order],atol=1e-10,rtol=0)
    result={'status':'passed','checks':rows,'max_dense_error':max(x['error'] for x in rows),'invariances':['eta0','repeated_views','class_permutation','query_partition','query_order','view_permutation'],'distinct_from_parent_and_view_mean':True,'complete_parent_score_parity':True,'seconds':time.time()-start}
    (HERE/'outputs/numeric_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
