"""Bounded check of weighted solver invariance, permissions and real-development runtime."""
import json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.linear_model import LogisticRegression
import evaluate as e
rng=np.random.RandomState(4172);X=rng.randn(15,20);y=np.arange(5).repeat(3)
w=np.tile([1,.5,.5],5);Q=rng.randn(9,20)
with threadpool_limits(1):
 a,n=e.logistic_one((X,y,w,Q,.1))
 counts=(w*2).astype(int)
 b,_=e.logistic_one((np.repeat(X,counts,axis=0),np.repeat(y,counts),np.ones(counts.sum()),Q,.2))
 one,_=e.logistic_one((X,y,w,Q[:1],.1))
 dup,_=e.logistic_one((X,y,w,np.concatenate([Q,Q]),.1))
 scale_error=float(np.max(np.abs(a-b)));query_error=float(np.max(np.abs(a[:1]-one)))
 assert scale_error<1e-5 and query_error<1e-10
 assert np.allclose(dup[:len(Q)],a,atol=1e-10)
 data,_=e.ref.assets();si,qi,_,_=e.ref.tasks(data["dtd"]["ident"],"dtd",1,"dev")
 S=data["dtd"]["query"][si[:4]];T=data["dtd"]["query"][qi[:4].reshape(-1,75)];G=data["dtd"]["gallery"]
 C=e.ref.candidates(S,G);t=time.time();v={}
 for method in ["distribution_logistic","mean_logistic","support_logistic"]:
  v[method]=e.scores(S,T,G,C,method,.001)
 elapsed=time.time()-t
 assert all(x.shape==(4,75,5) and np.isfinite(x).all() for x in v.values())
 res={"status":"passed","weighted_replication_error":scale_error,"query_independence_error":query_error,
      "real_dev_4episode_3head_seconds":elapsed,"max_iterations_used":max(e.NITERS),"sklearn":e.sklearn.__version__,
      "purpose":"solver invariance and runtime, not efficacy selection"}
 (e.HERE/"implementation_validation.json").write_text(json.dumps(res,indent=2));print(json.dumps(res))
