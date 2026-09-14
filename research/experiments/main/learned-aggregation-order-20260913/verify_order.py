from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
"""Independent NumPy forward/augmented-intercept solve and saved-result audit."""
import json,time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
import model

def transform(x,state,mode):
    if mode=='map_after_mean':x=x.mean(axis=-2)
    x=x+np.maximum(x@state['down.weight'].T,0)@state['up.weight'].T
    if mode=='map_before_mean':x=x.mean(axis=-2)
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def independent(s,q,state,mode,penalty=.1):
    s,q=[transform(x.astype(np.float64),state,mode) for x in [s,q]]
    result=[]
    for x,z in zip(s,q):
        k=x.shape[1];x=x.reshape(-1,x.shape[-1])
        y=np.eye(5)[np.repeat(np.arange(5),k)]
        n=len(x)
        system=np.zeros((n+1,n+1));system[:n,:n]=x@x.T+penalty*np.eye(n)
        system[:n,n]=1;system[n,:n]=1
        ab=np.linalg.solve(system,np.vstack([y,np.zeros((1,5))]))
        result.append(z@x.T@ab[:-1]+ab[-1])
    return np.stack(result)

def state_numpy(net):
    return {k:v.detach().numpy().astype(np.float64) for k,v in net.state_dict().items()}

def precheck(data):
    import train_eval as p
    torch.manual_seed(26091380)
    s=F.normalize(torch.randn(2,5,2,6,9,dtype=torch.double),dim=-1)
    q=F.normalize(torch.randn(2,10,6,9,dtype=torch.double),dim=-1)
    net=model.Map(9,3).double()
    with torch.no_grad():net.up.weight.normal_(0,.05)
    errors=[];graderrors=[]
    for mode in p.CFG['modes']:
        z=model.scores(net,s,q,mode).detach()
        other=independent(s.numpy(),q.numpy(),state_numpy(net),mode)
        errors.append(float(np.max(np.abs(z.numpy()-other))))
        assert errors[-1]<p.CFG['validation']['double_formula_atol']
        loss=F.cross_entropy(10*model.scores(net,s,q,mode).flatten(0,1),torch.arange(5).repeat_interleave(2).repeat(2))
        params=list(net.parameters());grads=torch.autograd.grad(loss,params)
        for param,grad in zip(params,grads):
            for idx in [(0,0),(1,2),(2,1)]:
                base=float(param[idx].detach());values=[]
                for sign in [1,-1]:
                    with torch.no_grad():param[idx]=base+sign*1e-6
                    values.append(float(F.cross_entropy(10*model.scores(net,s,q,mode).flatten(0,1),
                        torch.arange(5).repeat_interleave(2).repeat(2)).detach()))
                with torch.no_grad():param[idx]=base
                graderrors.append(abs((values[0]-values[1])/2e-6-float(grad[idx])))
        # Query independent and class equivariant; shared mean is view-order invariant.
        split=torch.cat([model.scores(net,s,part,mode) for part in q.split(3,1)],1)
        assert torch.allclose(z,split,atol=1e-10)
        perm=torch.tensor([3,1,4,0,2])
        assert torch.allclose(model.scores(net,s[:,perm],q,mode),z[...,perm],atol=1e-10)
        qp=torch.randperm(q.shape[1])
        assert torch.allclose(model.scores(net,s,q[:,qp],mode),z[:,qp],atol=1e-10)
        assert torch.allclose(model.scores(net,s.flip(-2),q.flip(-2),mode),z,atol=1e-10)
    assert max(graderrors)<p.CFG['validation']['gradient_atol']
    same=q.mean(-2,keepdim=True).expand_as(q)
    assert torch.allclose(net(same,p.CFG['modes'][0]),net(same,p.CFG['modes'][1]),atol=1e-10)
    linear=torch.randn(9,9,dtype=torch.double)
    assert torch.allclose((q@linear).mean(-2),q.mean(-2)@linear,atol=1e-10)
    fresh=model.Map(9,3).double()
    for mode in p.CFG['modes']:assert torch.allclose(fresh(q,mode),F.normalize(q.mean(-2),dim=-1),atol=1e-12)
    counts={};bench=[]
    for ds in p.CFG['source_domains']:
        for k in p.CFG['shots']:
            tt={role:p.tasks(data,ds,k,role) for role in ['train','selection','eval']}
            assert [len(tt[r]['seeds']) for r in tt]==[100,100,500]
            ids=[set(np.r_[t['support_indices'].ravel(),t['query_indices'].ravel()]) for t in [tt['train'],tt['selection']]]
            assert not ids[0]&ids[1]
            rgb=data[ds]['ident']['query_rgb']
            assert not {rgb[i] for i in ids[0]}&{rgb[i] for i in ids[1]}
            for role,t in tt.items():
                for si,qi in zip(t['support_indices'],t['query_indices']):assert not set(si.ravel())&set(qi.ravel())
            counts[f'{ds}_k{k}']={r:len(t['seeds']) for r,t in tt.items()}
    tr=p.tasks(data,'dtd',1,'train')
    rs,rq=p.batch(data['dtd']['query'],tr,np.arange(p.CFG['batch_size']))
    for mode in p.CFG['modes']:
        torch.manual_seed(26091381);net=model.Map()
        assert sum(x.numel() for x in net.parameters())==57344
        with torch.no_grad():
            direct=model.ridge(F.normalize(rs.mean(-2),dim=-1),F.normalize(rq.mean(-2),dim=-1),.1)
            assert torch.allclose(model.scores(net,rs,rq,mode),direct,atol=1e-6)
        start=time.time()
        for _ in range(10):
            loss=F.cross_entropy(10*model.scores(net,rs,rq,mode).flatten(0,1),p.label(tr).repeat(len(rs)))
            net.zero_grad();loss.backward()
        bench.append(dict(mode=mode,seconds_per_forward_backward=(time.time()-start)/10,batch_size=len(rs)))
    return dict(status='passed',max_numpy_score_error=max(errors),max_gradient_error=max(graderrors),
        identity_null=True,linear_commutation=True,repeated_view_limit=True,query_independence=True,
        class_view_permutation=True,disjoint_task_counts=counts,parameters=57344,source_timing=bench)

def bootstrap(d,seeds,cfg):
    rng=np.random.RandomState(cfg['gate']['bootstrap_seed'])
    bs=np.zeros(cfg['gate']['bootstrap_replicates'])
    for seed in sorted(set(seeds.tolist())):
        group=d[seeds==seed]
        sampled=rng.choice(group,size=(len(bs),len(group)),replace=True)
        bs+=sampled.mean(1)/len(set(seeds.tolist()))
    return dict(delta_pp=float(d.mean()*100),ci95_pp=(np.percentile(bs,[2.5,97.5])*100).tolist())

def postcheck():
    import train_eval as p
    torch.set_num_threads(p.CFG['resources']['threads'])
    choices=json.loads((p.OUT/'selection_lock.json').read_text())
    complete=json.loads((p.OUT/'complete.json').read_text())
    assert p.sha(p.OUT/'selection_lock.json')==complete['selection_sha256']
    assert p.hashes()==json.loads((p.HERE/'lock.json').read_text())
    data=p.prior.load();errors=[];rank_checks=0;stat_checks=0;max_stat_error=0.
    analysis=json.loads((p.OUT/'analysis.json').read_text())
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for k in p.CFG['shots']:
            z=np.load(p.OUT/'cells'/f'{source}_to_{target}_k{k}.npz')
            for key,val in p.tasks(data,target,k,'eval').items():assert np.array_equal(z[key],val)
            assert np.array_equal(z['predictions'],z['scores'].argmax(-1))
            assert np.array_equal(z['accuracy'],(z['predictions']==z['yq']).mean(-1))
            for mi,mode in enumerate(p.CFG['modes']):
                for ri,seed in enumerate(p.CFG['training_seeds']):
                    entry=choices[source][str(k)][mode+'_seed'+str(seed)]
                    net=model.Map();net.load_state_dict(torch.load(entry['model_path'],weights_only=True))
                    state=state_numpy(net)
                    ranks=[]
                    for row in entry['history']:
                        assert p.sha(row['model_path'])==row['sha256']
                        key=f'{source}_k{k}_{mode}_seed{seed}_step{row["step"]}'
                        with np.load(p.OUT/'selection'/f'{key}.npz') as sel:
                            sc=sel['scores'];correct=int((sc.argmax(-1)==sel['yq']).sum())
                            logits=sc.astype(np.float64)*p.CFG['logit_scale']
                            ce=float(np.mean(logsumexp(logits,axis=-1)-np.take_along_axis(logits,
                                np.broadcast_to(sel['yq'],logits.shape[:2])[...,None],axis=-1).squeeze(-1)))
                            assert correct==row['correct'] and abs(ce-row['ce'])<1e-5
                            ranks.append((correct,-row['ce'],-row['step']))
                            rank_checks+=1
                    assert max(ranks)==(entry['correct'],-entry['ce'],-entry['step'])
                    for ds,role,indices in [(target,'eval',p.CFG['validation']['audit_eval_indices']),
                        (source,'selection',p.CFG['validation']['audit_selection_indices'])]:
                        t=p.tasks(data,ds,k,role);s,q=p.batch(data[ds]['query'],t,np.array(indices))
                        independent_sc=independent(s.numpy(),q.numpy(),state,mode)
                        if role=='eval':saved=z['scores'][mi,ri,indices]
                        else:
                            key=f'{source}_k{k}_{mode}_seed{seed}_step{entry["step"]}'
                            with np.load(p.OUT/'selection'/f'{key}.npz') as sel:saved=sel['scores'][indices]
                        err=float(np.max(np.abs(independent_sc-saved)));errors.append(err)
                        assert err<=p.CFG['validation']['score_atol'],(source,target,k,mode,seed,role,err)
                        assert np.array_equal(independent_sc.argmax(-1),saved.argmax(-1))
            refs=json.loads((p.OUT/'references'/f'{source}_to_{target}_k{k}.json').read_text())
            banks=[]
            for gallery,ref in refs.items():
                assert p.sha(ref['path'])==ref['sha256']
                with np.load(ref['path']) as old:
                    names=old['names'].tolist();a=old['accuracy'].copy();banks.append(a)
                    ds={m:z['accuracy'][0].mean(0)-a[j] for j,m in enumerate(names)}
                    ds['map_after_mean']=(z['accuracy'][0]-z['accuracy'][1]).mean(0)
                    for name,d in ds.items():
                        v=bootstrap(d,z['seeds'],p.prior.CFG)
                        reported=analysis['cells'][f'{source}_to_{target}_k{k}_{gallery}']['comparisons'][name]
                        err=max(abs(v['delta_pp']-reported['delta_pp']),float(np.max(np.abs(np.array(v['ci95_pp'])-reported['ci95_pp']))))
                        max_stat_error=max(err,max_stat_error);stat_checks+=1;assert err<1e-10
            if k==1:
                ds={m:z['accuracy'][0].mean(0)-np.mean(banks,axis=0)[j] for j,m in enumerate(names)}
                ds['map_after_mean']=(z['accuracy'][0]-z['accuracy'][1]).mean(0)
                for name,d in ds.items():
                    v=bootstrap(d,z['seeds'],p.prior.CFG)
                    r=analysis['directions'][f'{source}_to_{target}']['comparisons'][name]
                    err=max(abs(v['delta_pp']-r['delta_pp']),float(np.max(np.abs(np.array(v['ci95_pp'])-r['ci95_pp']))))
                    max_stat_error=max(err,max_stat_error);stat_checks+=1;assert err<1e-10
    result=dict(status='passed',score_comparisons=len(errors),independent_episodes=24*8,
        max_score_error=max(errors),checkpoint_rank_checks=rank_checks,statistic_checks=stat_checks,max_statistic_error=max_stat_error,
        original_scatter_strict_audit='failed; retained separately',canonical_target_results=False)
    p.dump('independent_audit.json',result);print('AUDIT_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':postcheck()
