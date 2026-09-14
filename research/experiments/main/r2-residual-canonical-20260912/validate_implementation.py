"""Validate attribution algebra, reference ridge equivalence, and query independence."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from evaluate import weighted_ridge,fit_pseudo,residual_scores,ridge_scores
torch.manual_seed(260912)
e,c,k,n,d=2,5,1,16,12
S=F.normalize(torch.randn(e,c,k,d),dim=-1)
G=F.normalize(torch.randn(e,c,n,d),dim=-1);Q=torch.randn(e,7,d)
P=S.mean(2,keepdim=True)+G-G.mean(2,keepdim=True)
assert (P.mean(2)-S.mean(2)).abs().max()<1e-6
W=torch.randn(e,d,c);bias=torch.randn(e,1,c);Y=torch.eye(c)[None,:,None].expand(e,-1,n,-1)
full=((P@W[:,None]+bias[:,None])-Y).square().sum((-1,-2))/n
mean=((S.mean(2)@W+bias)-torch.eye(c)[None]).square().sum(-1)
residual=(((G-G.mean(2,keepdim=True))@W[:,None]).square().sum((-1,-2))/n)
loss_error=(full-mean-residual).abs().max().item();assert loss_error<2e-5
xx=S.flatten(1,2);yy=torch.eye(c)[None].expand(e,-1,-1)
ref=ridge_scores(xx,yy,Q,.1);wr=weighted_ridge(xx,yy,Q,torch.ones(c),.1)
ridge_error=(ref-wr).abs().max().item();assert ridge_error<1e-6
v=residual_scores(S,Q,G,.1)
single=torch.cat([residual_scores(S,Q[:,i:i+1],G,.1) for i in range(Q.shape[1])],dim=1)
query_error=(single-v).abs().max().item();assert query_error<1e-5
# Each episode remains independent when other episodes are absent.
episode_error=max((residual_scores(S[i:i+1],Q[i:i+1],G[i:i+1],.1)-v[i:i+1]).abs().max().item() for i in range(e))
assert episode_error<1e-5
result={"status":"passed","loss_decomposition_max_error":loss_error,"uniform_weight_reference_ridge_error":ridge_error,"single_query_error":query_error,"single_episode_error":episode_error,"finite":bool(torch.isfinite(v).all())}
Path(__file__).with_name("implementation_validation.json").write_text(json.dumps(result,indent=2));print(json.dumps(result))
