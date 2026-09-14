"""Independent numerical identities plus a real canonical-task baseline check."""
import json,time
import numpy as np
import torch
import torch.nn.functional as F
import fusion
import evaluate as ev
from baseline_methods import ridge_scores
HERE=ev.HERE

def main():
    torch.manual_seed(101);errors={}
    A=torch.randn(2,5,11,dtype=torch.float64);Q=torch.randn(2,7,11,dtype=torch.float64);Y=torch.eye(5,dtype=torch.float64)[None].expand(2,-1,-1)
    Ac,Ad=fusion.split(A,6);Qc,Qd=fusion.split(Q,6)
    for l in fusion.LAMBDAS:
        Z=A-A.mean(1,keepdim=True);B=Y-Y.mean(1,keepdim=True)
        W=torch.linalg.solve(Z.transpose(1,2)@Z+l*torch.eye(11),Z.transpose(1,2)@B)
        primal=(Q-A.mean(1,keepdim=True))@W+Y.mean(1,keepdim=True)
        dual=ridge_scores(A,Y,Q,l);errors['primal_'+str(l)]=float((dual-primal).abs().max())
        for w in [0.,.5,1.]:
            early=fusion.fit(fusion.join(Ac,Ad,w),fusion.join(Qc,Qd,w),l)
            target=dual if w==.5 else fusion.late(Ac,Ad,Qc,Qd,w,l)
            errors['fusion_'+str(w)+'_'+str(l)]=float((early-target).abs().max())
    assert max(errors.values())<1e-9,errors
    data,manifest=ev.assets();name='dtd';si,qi,cs,seeds=ev.ref.tasks(data[name]['ident'],name,1,'dev')
    assert si.shape==(200,5,1) and qi.shape==(200,5,15)
    keep=[i for i,h in enumerate(data[name]['ident']['gallery_rgb']) if h not in set(data[name]['ident']['query_rgb'])]
    Sc,Sd=[x[si[:2]] for x in data[name]['query']];Qc,Qd=[x[qi[:2].reshape(2,75)] for x in data[name]['query']];Gc,Gd=[x[keep] for x in data[name]['gallery']]
    choices={f:[(.5,.1),(0.,.1),(1.,.1)] for f in fusion.FAMILIES}
    values,idx=fusion.batch_scores(Sc,Sd,Qc,Qd,Gc,Gd,choices)
    S=ev.ref.representation(Sc,Sd,.5);Q=ev.ref.representation(Qc,Qd,.5);G=ev.ref.representation(Gc,Gd,.5)
    original=ev.ref.r2_scores(S,Q,G)
    errors['canonical_r2']=float((original-values['r2']).abs().max())
    errors['canonical_early']=float((original-values[fusion.key('early_shared',.5,.1)]).abs().max())
    subset,_=fusion.batch_scores(Sc,Sd,Qc[:,[0,7]],Qd[:,[0,7]],Gc,Gd,choices)
    errors['query_independence']=max(float((v[:,[0,7]]-subset[m]).abs().max()) for m,v in values.items())
    for w in [0.,1.]:errors['canonical_endpoint_'+str(w)]=float((values[fusion.key('late_shared',w,.1)]-values[fusion.key('early_shared',w,.1)]).abs().max())
    assert max(errors.values())<1e-5,errors
    for a,b in zip(si,qi):assert not set(a.ravel())&set(b.ravel())
    report={'status':'passed','numerical_errors':errors,'identity_checks':'all eight canonical feature packs sha256 + ids + finiteness; task supports/queries disjoint',
        'tested_real_tasks':2,'accuracy_not_used_for_selection':True,'command':'/opt/anaconda3/envs/torch/bin/python validate.py','torch':torch.__version__,'numpy':np.__version__,'timestamp':time.time()}
    ev.dump(HERE/'validation.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
