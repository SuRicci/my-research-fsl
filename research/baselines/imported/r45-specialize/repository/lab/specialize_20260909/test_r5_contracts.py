import torch
import torch.nn.functional as F
from campaign_common import HERE,read
from r5_candidates import run,support_cv,weighted_ridge,representation,ridge_scores

def test_weighted_ridge_against_primal_and_duplicate_mass():
    torch.manual_seed(197);s=torch.randn(2,11,7,dtype=torch.float64);x=torch.randn(2,9,7,dtype=torch.float64)
    y=torch.randn(2,11,3,dtype=torch.float64);w=torch.rand(2,11,dtype=torch.float64)+.1;lam=.3
    score=weighted_ridge(s,y,x,w,lam);sm=(s*w[:,:,None]).sum(1,keepdim=True)/w.sum(1)[:,None,None];ym=(y*w[:,:,None]).sum(1,keepdim=True)/w.sum(1)[:,None,None]
    c=s-sm;coef=torch.linalg.solve(c.transpose(1,2)@(c*w[:,:,None])+lam*torch.eye(7,dtype=s.dtype),c.transpose(1,2)@((y-ym)*w[:,:,None]))
    torch.testing.assert_close(score,(x-sm)@coef+ym,rtol=1e-9,atol=1e-9)
    duplicate=weighted_ridge(s.repeat_interleave(2,1),y.repeat_interleave(2,1),x,w.repeat_interleave(2,1)/2,lam)
    torch.testing.assert_close(score,duplicate,rtol=1e-9,atol=1e-9)

def test_candidate_episode_and_class_order_independence():
    torch.manual_seed(198);sc=F.normalize(torch.randn(2,5,1,6),dim=-1);sd=F.normalize(torch.randn(2,5,1,4),dim=-1)
    xc=F.normalize(torch.randn(2,9,6),dim=-1);xd=F.normalize(torch.randn(2,9,4),dim=-1);gc=F.normalize(torch.randn(92,6),dim=-1);gd=F.normalize(torch.randn(92,4),dim=-1)
    S=representation(sc,sd,.5);X=representation(xc,xd,.5);G=representation(gc,gd,.5);perm=torch.tensor([2,4,0,3,1])
    for name,cfg in read(HERE/'candidates.json').items():
        if not name.startswith('r5_') or name=='r5_support_cv':continue
        a,_=run(S,X,G,sc,sd,gc,gd,name,cfg)
        one,_=run(S[:1],X[:1],G,sc[:1],sd[:1],gc,gd,name,cfg);torch.testing.assert_close(a[:1],one,rtol=2e-5,atol=2e-5)
        b,_=run(S[:,perm],X,G,sc[:,perm],sd[:,perm],gc,gd,name,cfg);torch.testing.assert_close(a[:,:,perm],b,rtol=2e-5,atol=2e-5)

def test_support_loo_matches_explicit_refits():
    torch.manual_seed(199);cfg=read(HERE/'candidates.json')['r5_support_cv']
    sc=F.normalize(torch.randn(1,5,5,6),dim=-1);sd=F.normalize(torch.randn(1,5,5,4),dim=-1)
    supports=[representation(sc,sd,w) for w in cfg['weights']];queries=[s.flatten(1,2)[:,:3] for s in supports]
    score,diag=support_cv(supports,queries,cfg);Y=F.one_hot(torch.arange(5).repeat_interleave(5),5).float()[None]
    for j,S in enumerate(supports):
        a=S.flatten(1,2);values=[]
        for i in range(25):
            ids=[k for k in range(25) if k!=i];values.append(ridge_scores(a[:,ids],Y[:,ids],a[:,i:i+1],cfg['lambda']))
        expected=((torch.cat(values,1)-Y)**2).mean().item()
        assert abs(expected-float(diag['support_loo_mse'][0,j]))<2e-6
    assert torch.isfinite(score).all();assert abs(diag['fusion_weights'].sum()-1)<1e-6
