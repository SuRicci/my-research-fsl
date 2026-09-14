"""Paper-defined iLPC-z local comparator, no query or gallery labels in adaptation.
Based on Lazarou et al., Pattern Recognition161(2025)111304, Eq1-4,18-20.
The published distractor entrypoint is missing from the audited source; see SOURCE_AUDIT.md.
"""
import time,warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
def unit(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def gallery_cache(G,maxk=20):
    V=unit(G);n=len(G);ii=[];vv=[]
    for start in range(0,n,128):
        stop=min(start+128,n);sim=V[start:stop]@V.T
        sim[np.arange(stop-start),np.arange(start,stop)]=-np.inf
        idx=np.argpartition(sim,-maxk,axis=1)[:,-maxk:]
        ii.append(idx);vv.append(np.take_along_axis(sim,idx,axis=1))
    return {"unit_features":V,"indices":np.concatenate(ii),"similarities":np.concatenate(vv)}
def graph(S,cache,k,alpha,direct=False):
    U=unit(S);V=cache["unit_features"];n=len(U)+len(V);ns=len(U)
    if direct:
        X=np.concatenate([U,V]);sim=X@X.T;np.fill_diagonal(sim,-np.inf)
        ids=np.argpartition(sim,-k,axis=1)[:,-k:];values=np.take_along_axis(sim,ids,axis=1)
    else:
        cross=U@V.T;support=np.concatenate([U@U.T,cross],axis=1)
        support[np.arange(ns),np.arange(ns)]=-np.inf
        si=np.argpartition(support,-k,axis=1)[:,-k:];sv=np.take_along_axis(support,si,axis=1)
        gi=np.concatenate([cache["indices"]+ns,np.broadcast_to(np.arange(ns),(len(V),ns))],axis=1)
        gv=np.concatenate([cache["similarities"],cross.T],axis=1)
        choose=np.argpartition(gv,-k,axis=1)[:,-k:]
        ids=np.concatenate([si,np.take_along_axis(gi,choose,axis=1)])
        values=np.concatenate([sv,np.take_along_axis(gv,choose,axis=1)])
    W=sp.csr_matrix((np.maximum(values,0).ravel()**3,(np.repeat(np.arange(n),k),ids.ravel())),shape=(n,n))
    W=W+W.T
    degree=np.asarray(W.sum(1)).ravel();scale=np.zeros_like(degree)
    scale[degree>0]=1/np.sqrt(degree[degree>0])
    D=sp.diags(scale);Wn=D@W@D
    return (sp.eye(n,format="csc")-alpha*Wn).tocsc()
def infer_labels(S,ys,G,cache,k=20,alpha=.8,best=3,direct=False,progress=None):
    start=time.perf_counter();A=graph(S,cache,k,alpha,direct=direct);graph_seconds=time.perf_counter()-start
    t=time.perf_counter();factor=splu(A,permc_spec="MMD_AT_PLUS_A");factor_seconds=time.perf_counter()-t
    n=A.shape[0];c=int(np.max(ys))+1;Y=np.zeros((n,c));Y[np.arange(len(S)),ys]=1
    labeled=np.zeros(n,dtype=bool);labeled[:len(S)]=True;order=list(range(len(S)));assigned=list(ys)
    max_residual=0.;rounds=0;trace=[];loop=time.perf_counter()
    while not labeled.all():
        Z=factor.solve(Y)
        residual=float(np.max(np.abs(A@Z-Y)));max_residual=max(max_residual,residual)
        assert np.isfinite(Z).all() and residual<1e-8
        remaining=np.flatnonzero(~labeled);pseudo=Z[remaining].argmax(1);counts=np.bincount(pseudo,minlength=c)
        take=min(best,int(counts.min()))
        if take==0:break
        picked=[];labels=[]
        for cls in range(c):
            pos=np.flatnonzero(pseudo==cls)
            rank=np.argsort(-Z[remaining[pos],cls],kind="stable")[:take]
            picked.extend(remaining[pos[rank]].tolist());labels.extend([cls]*take)
        Y[np.asarray(picked),np.asarray(labels)]=1;labeled[picked]=True
        order.extend(picked);assigned.extend(labels);rounds+=1
        trace.append([rounds,len(order),int(counts.min()),int(counts.max())])
        if progress and rounds%25==0:progress(rounds,len(order),time.perf_counter()-loop)
    stats={"graph_seconds":graph_seconds,"factor_seconds":factor_seconds,"loop_seconds":time.perf_counter()-loop,
           "rounds":rounds,"selected_total":len(order),"gallery_selected":len(order)-len(S),
           "max_linear_residual":max_residual,"factor_nnz":factor.L.nnz+factor.U.nnz,"trace":trace,
           "stop":"all_selected" if labeled.all() else "one_predicted_class_exhausted"}
    return np.asarray(order),np.asarray(assigned),stats
def fit(S,ys,G,cache,k=20,alpha=.8,best=3,progress=None):
    ids,labels,stats=infer_labels(S,ys,G,cache,k,alpha,best,progress=progress)
    X=np.concatenate([S,G]);t=time.perf_counter()
    clf=LogisticRegression(C=10,multi_class="multinomial",solver="lbfgs",max_iter=2000,tol=1e-8,random_state=26091275)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always");clf.fit(X[ids],labels)
    stats.update({"logistic_seconds":time.perf_counter()-t,"logistic_iterations":int(clf.n_iter_.max()),
                  "convergence_warnings":sum(issubclass(w.category,ConvergenceWarning) for w in caught)})
    return clf,ids,labels,stats
