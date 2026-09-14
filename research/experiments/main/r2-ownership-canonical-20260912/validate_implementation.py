"""Independent scalar pooling, reference endpoint, permutation and query-independence checks."""
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from evaluate import allocation,scores_for,pool,HEADS
import reference_eval as ref
import geometry_reference as geo
torch.manual_seed(26091274)
S=F.normalize(torch.randn(3,5,1,32),dim=-1);Q=F.normalize(torch.randn(3,11,32),dim=-1);G=F.normalize(torch.randn(90,32),dim=-1)
ll={m:[.1] for m in HEADS}
v,info,ci=scores_for(S,Q,G,ll)
endpoint=(v["r2"]-ref.r2_scores(S,Q,G)).abs().max().item()
center_error=(v["centered_r2"]-geo.head(geo.prepare(S,Q,G,"support"),.1)).abs().max().item()
raw,_,_=allocation(S,Q,G);P=S.squeeze(2);independent=[]
for e in range(3):
 rows=[]
 for c in range(5):
  sims=G@P[e].T;idx=torch.topk(sims[:,c],64).indices
  kept=[int(i) for i in idx if int(sims[i].argmax())==c]
  expected=P[e,c] if not kept else F.normalize(.5*P[e,c]+.5*F.normalize(G[kept].mean(0),dim=0),dim=0)
  rows.append(expected)
 independent.append(torch.stack(rows))
pool_error=(raw["ownership"]-torch.stack(independent)).abs().max().item()
single=scores_for(S,Q[:,:1],G,ll)[0]
query_error=max((v[m][:,:1]-single[m]).abs().max().item() for m in v)
perm=torch.tensor([3,0,4,1,2]);pv=scores_for(S[:,perm],Q,G,ll)[0]
permutation_error=max((pv[m]-v[m][:,:,perm]).abs().max().item() for m in v)
fallback=pool(P,G[info["indices"]],torch.zeros_like(info["mask"]),"fixed")
fallback_error=(fallback-P).abs().max().item()
assert max(endpoint,center_error,pool_error,query_error,permutation_error,fallback_error)<2e-5
assert torch.equal(info["counts"],info["mask"].sum(-1))
result={"status":"passed","raw_r2_endpoint_error":endpoint,"centered_r2_endpoint_error":center_error,
        "independent_scalar_pool_error":pool_error,"query_subset_error":query_error,
        "class_permutation_error":permutation_error,"empty_pool_fallback_error":fallback_error,
        "synthetic_only":True,"all_finite":True}
Path(__file__).with_name("implementation_validation.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))
