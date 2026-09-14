"""Independent dense NumPy checks and permission invariants, no new-view accuracy."""
from pathlib import Path
import numpy as np
import torch
import metric as m

def normalize(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def dense_metric(S,Q,G,gamma):
    s=S.numpy().astype(float);q=Q.numpy().astype(float);g=G.numpy().astype(float)
    d=(s-s.mean(-2,keepdims=True)).reshape(-1,s.shape[-1]);C=d.T@d/len(d)
    M=np.eye(s.shape[-1])+gamma*s.shape[-1]*C/np.trace(C)
    ev,U=np.linalg.eigh(M);T=(U/np.sqrt(ev))@U.T
    return [normalize(x.mean(-2)@T) for x in [s,q,g]],T

def numpy_r2(S,Q,G):
    k=S.shape[1]
    if k==1:
        P=normalize(S.mean(1));ix=np.argsort(-(P@G.T),axis=-1)[:,:64]
        X=normalize(.5*P+.5*normalize(G[ix].mean(1)));Y=np.eye(5);lam=.1
    else:X=S.reshape(-1,S.shape[-1]);Y=np.repeat(np.eye(5),k,axis=0);lam=1.
    xc=X-X.mean(0);yc=Y-Y.mean(0)
    W=np.linalg.solve(xc.T@xc+lam*np.eye(X.shape[1]),xc.T@yc)
    return (Q-X.mean(0))@W+Y.mean(0)

def verify(sampler):
    torch.manual_seed(26091391);errors=[];algebra=[]
    for shot in [1,5]:
        S=m.norm(torch.randn(5,shot,6,24));Q=m.norm(torch.randn(17,6,24));G=m.norm(torch.randn(80,6,24))
        for gamma in [.1,1.,10.]:
            factor=m.scatter_factor(S,gamma);dense,T=dense_metric(S,Q,G,gamma)
            tr=[m.transform(x.mean(-2),factor).numpy() for x in [S,Q,G]]
            algebra.append(max(float(np.max(abs(a-b))) for a,b in zip(tr,dense)))
            score=m.scatter(S,Q,G,gamma);ind=numpy_r2(*dense);errors.append(float(np.max(abs(score.numpy()-ind))))
            part=torch.cat([m.scatter(S,q,G,gamma) for q in Q.split(4)],0)
            assert torch.allclose(part,score,atol=2e-6)
            perm=torch.tensor([3,0,4,1,2]);assert torch.allclose(m.scatter(S[perm],Q,G,gamma),score[:,perm],atol=2e-6)
            assert torch.allclose(m.scatter(S,Q,G[torch.randperm(len(G))],gamma),score,atol=2e-6)
            assert torch.allclose(m.scatter(S.flip(-2),Q.flip(-2),G.flip(-2),gamma),score,atol=2e-6)
        controls=m.controls(S,Q,G)
        assert torch.allclose(m.scatter(S,Q,G,0),controls['mean_r2'],atol=1e-6)
        # Augmented-support ridge weights sum to the original number of examples.
        X=S.flatten(0,2).numpy().astype(float);Y=np.repeat(np.eye(5),shot*6,axis=0);xc=X-X.mean(0);yc=Y-Y.mean(0)
        lam=.1 if shot==1 else 1.;W=np.linalg.solve(xc.T@xc/6+lam*np.eye(24),xc.T@yc/6)
        independent=(m.norm(Q.mean(-2)).numpy()-X.mean(0))@W+Y.mean(0)
        errors.append(float(np.max(abs(controls['augmented_support_ridge'].numpy()-independent))))
        # Constant views have zero scatter and reduce every view aggregation to the same R2.
        s=S[:,:,0:1].expand(-1,-1,6,-1);q=Q[:,0:1].expand(-1,6,-1);g=G[:,0:1].expand(-1,6,-1)
        base=m.ref.r2_scores(s[:,:,0][None],q[:,0][None],g[:,0])[0]
        assert torch.allclose(m.scatter(s,q,g,10),base,atol=2e-6)
    assert max(errors)<5e-6 and max(algebra)<5e-6,(errors,algebra)
    # Full historical task arrays are compared; only already-exposed one-view predictions are reconstructed.
    data,_=m.ref.assets();historical=[];task_count=0
    prior=Path(__file__).resolve().parent.parent/'utility-composition-qualification-20260913/outputs/cells'
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for shot in [1,5]:
            selection=sampler.tasks(data[source]['ident'],shot,'selection')
            assert selection['query_indices'].shape[-1]==(13 if shot==5 else 15)
            t=sampler.tasks(data[target]['ident'],shot,'eval')
            for gallery in [target,source]:
                path=prior/(f'{source}_to_{target}_{gallery}_k{shot}.npz')
                with np.load(path) as z:
                    for key in t:assert np.array_equal(t[key],z[key])
                    task_count+=len(t['seeds']);idx=[0,249,499]
                    S=data[target]['query'][t['support_indices'][idx]];Q=data[target]['query'][t['query_indices'][idx].reshape(len(idx),75)];G=data[gallery]['gallery']
                    sc=m.ref.r2_scores(S,Q,G).numpy();old=z['scores'][z['names'].tolist().index('r2'),idx]
                    historical.append(float(abs(sc-old).max()))
                    cs=m.geo.head(m.geo.prepare(S,Q,G,'support'),.1).numpy();old_cs=z['scores'][z['names'].tolist().index('cs_0.1'),idx]
                    historical.append(float(abs(cs-old_cs).max()))
    assert max(historical)<2e-6
    return {'status':'passed','dense_metric_transform_max_error':max(algebra),'independent_numpy_prediction_max_error':max(errors),'historical_r2_max_error':max(historical),'full_task_conditions_verified':task_count,'historical_predictions_reconstructed':48,'query_partition_invariant':True,'class_permutation_equivariant':True,'gallery_permutation_invariant':True,'view_order_invariant':True,'zero_strength_mean_parity':True,'constant_views_r2_parity':True,'augmented_support_total_weight':'5*shot','new_view_accuracy_observed':False}
