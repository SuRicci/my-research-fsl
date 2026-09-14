"""Validate original-R2 recovery, batching, and query independence of task centering."""
import json,time
import numpy as np
import torch
import torch.nn.functional as F
import evaluate as e
torch.manual_seed(5192)
S=F.normalize(torch.randn(4,5,1,32),dim=-1);Q=F.normalize(torch.randn(4,75,32),dim=-1);G=F.normalize(torch.randn(300,32),dim=-1)
base=e.ref.r2_scores(S,Q,G);zero=e.head(e.prepare(S,Q,G,"none"),.1)
parity=float((base-zero).abs().max());assert parity<1e-5
center=e.head(e.prepare(S,Q,G,"support"),.1)
single=torch.cat([e.head(e.prepare(S[i:i+1],Q[i:i+1],G,"support"),.1) for i in range(len(S))])
batch_error=float((single-center).abs().max());assert batch_error<1e-5
one=e.head(e.prepare(S,Q[:,:1],G,"support"),.1)
query_error=float((one-center[:,:1]).abs().max());assert query_error<1e-5
data,_=e.ref.assets();t=time.time()
for shot in [1,5]:
 si,qi,_,_=e.ref.tasks(data["dtd"]["ident"],"dtd",shot,"dev")
 A=data["dtd"]["query"][si[:4]];X=data["dtd"]["query"][qi[:4].reshape(-1,75)];H=data["dtd"]["gallery"]
 raw=e.head(e.prepare(A,X,H,"none"),.1 if shot==1 else 1.)
 expected=e.ref.r2_scores(A,X,H);assert torch.allclose(raw,expected,atol=1e-5)
 out=e.head(e.prepare(A,X,H,"support"),.1);assert torch.isfinite(out).all()
res={"status":"passed","zero_origin_reference_error":parity,"batch_error":batch_error,
     "query_independence_error":query_error,"real_dev_eight_episode_seconds":time.time()-t}
(e.HERE/"implementation_validation.json").write_text(json.dumps(res,indent=2));print(json.dumps(res))
