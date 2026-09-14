"""Independent NumPy distance kernels and intercept-constrained ridge."""
import json
import numpy as np
import torch
import view_kernel as vk

def normalized(x):
    x = np.asarray(x,dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def np_features(x,mode):
    x = normalized(x)
    if mode == 'original_rbf':
        return x[:,:1]
    if mode in ('mean_rbf','linear_mean'):
        return normalized(x.mean(1))[:,None]
    return x

def np_raw(a,b,beta,linear):
    out = np.empty((len(a),len(b)))
    for i,x in enumerate(a):
        for j,y in enumerate(b):
            out[i,j] = np.mean(x@y.T) if linear else np.exp(-.5*beta*((x[:,None]-y[None])**2).sum(-1)).mean()
    return out

def reference(s,q,mode,beta,penalty):
    labels = np.repeat(np.arange(5),s.shape[1])
    a,b = np_features(s.reshape(-1,*s.shape[-2:]),mode),np_features(q,mode)
    linear = mode=='linear_mean'
    k,t = np_raw(a,a,beta,linear),np_raw(b,a,beta,linear)
    sd = np.diag(k).copy()
    qd = np.array([np_raw(x[None],x[None],beta,linear)[0,0] for x in b])
    k /= np.sqrt(sd[:,None]*sd[None])
    t /= np.sqrt(qd[:,None]*sd[None])
    n = len(a)
    system = np.zeros((n+1,n+1))
    system[:n,:n] = k+penalty*np.eye(n)
    system[:n,n] = 1
    system[n,:n] = 1
    rhs = np.zeros((n+1,5))
    rhs[np.arange(n),labels]=1
    fitted = np.linalg.solve(system,rhs)
    return t@fitted[:n]+fitted[n]

def compare(s,q,mode,beta,penalty):
    actual = vk.scores(s,q,mode,beta,penalty).numpy()
    expected = reference(s.numpy(),q.numpy(),mode,beta,penalty)
    error = float(np.max(np.abs(actual-expected)))
    assert error < 1e-8, (mode,beta,penalty,error)
    assert np.array_equal(actual.argmax(-1),expected.argmax(-1))
    return error

def precheck(data):
    import run_kernel as run
    rng = np.random.RandomState(26091390)
    maxerr = 0
    checks = 0
    for shot in [1,5]:
        s = torch.tensor(rng.randn(5,shot,6,17))
        q = torch.tensor(rng.randn(7,6,17))
        for mode in run.MODES:
            for beta,penalty in run.params(mode):
                maxerr = max(maxerr,compare(s,q,mode,beta,penalty))
                checks += 1
        for beta in run.CFG['betas']:
            k,_ = vk.gram(s.flatten(0,1),q,'distribution_rbf',beta)
            assert torch.linalg.eigvalsh(k).min()>-1e-10
        a = vk.scores(s,q,'distribution_rbf',4,.1)
        b = torch.cat([vk.scores(s,q[i:i+1],'distribution_rbf',4,.1) for i in range(len(q))])
        assert torch.max(torch.abs(a-b))<1e-10
        reverse = vk.scores(s,q.flip(0),'distribution_rbf',4,.1).flip(0)
        assert torch.max(torch.abs(a-reverse))<1e-10
        repeated_s = s[:,:,:1].expand(-1,-1,6,-1)
        repeated_q = q[:,:1].expand(-1,6,-1)
        assert torch.max(torch.abs(vk.scores(repeated_s,repeated_q,'distribution_rbf',4,.1)-vk.scores(repeated_s,repeated_q,'original_rbf',4,.1)))<1e-10
        x,y = vk.unit(s.flatten(0,1)),vk.unit(q)
        assert torch.max(torch.abs(vk.raw_kernel(x,y,0,True)-x.mean(1)@y.mean(1).T))<1e-12
    for ds in run.CFG['source_domains']:
        for shot in run.CFG['shots']:
            t=run.task_data(data,ds,shot,'selection')
            s=data[ds]['query'][t['support_indices'][0]]
            q=data[ds]['query'][t['query_indices'][0].reshape(-1)][:3]
            for mode in run.MODES:
                for beta,penalty in run.params(mode):
                    maxerr=max(maxerr,compare(s,q,mode,beta,penalty))
                    checks+=1
    return dict(status='passed',independent_score_comparisons=checks,max_score_error=maxerr,
        tolerance=1e-8,query_partition_and_order_invariant=True,psd=True,
        repeated_view_limit=True,linear_kernel_equals_mean=True,classification_outcomes_used_for_selection=False)

def audit():
    import run_kernel as run
    assert run.sources()==json.loads((run.HERE/'lock.json').read_text())
    data=run.prior.load()
    choices=json.loads((run.OUT/'selection_lock.json').read_text())
    maxerr,checks=0,0
    for ds in run.CFG['source_domains']:
        for shot in run.CFG['shots']:
            for mode in run.MODES:
                z=np.load(run.OUT/f'{ds}_k{shot}_{mode}_selection.npz')
                counts=(z['scores'].argmax(-1)==z['yq']).sum((0,2))
                choice=choices[ds][str(shot)][mode]
                assert counts.tolist()==choice['integer_correct']
                assert int(counts.argmax())==choice['index']
                assert run.params(mode)[choice['index']]==(choice['beta'],choice['penalty'])
                for i in [0,49,99]:
                    s=data[ds]['query'][z['support_indices'][i]]
                    q=data[ds]['query'][z['query_indices'][i].reshape(-1)]
                    for j,(beta,penalty) in enumerate(run.params(mode)):
                        expected=reference(s.numpy(),q.numpy(),mode,beta,penalty)
                        err=float(np.max(np.abs(z['scores'][i,j]-expected)));assert err<1e-8
                        assert np.array_equal(z['scores'][i,j].argmax(-1),expected.argmax(-1))
                        maxerr=max(maxerr,err);checks+=1
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for shot in run.CFG['shots']:
            z=np.load(run.OUT/'cells'/f'{source}_to_{target}_{target}_k{shot}.npz')
            zz=np.load(run.OUT/'cells'/f'{source}_to_{target}_{source}_k{shot}.npz')
            names=z['names'].tolist()
            for cell in [z,zz]:
                assert np.array_equal(cell['predictions'],cell['scores'].argmax(-1))
                assert np.array_equal(cell['accuracy'],(cell['predictions']==cell['yq']).mean(-1))
                assert np.isfinite(cell['scores']).all()
            for mode in run.MODES:
                j=names.index(mode);choice=choices[source][str(shot)][mode]
                assert np.array_equal(z['scores'][j],zz['scores'][j])
                for i in [0,99,199,299,499]:
                    s=data[target]['query'][z['support_indices'][i]]
                    q=data[target]['query'][z['query_indices'][i].reshape(-1)]
                    expected=reference(s.numpy(),q.numpy(),mode,choice['beta'],choice['penalty'])
                    err=float(np.max(np.abs(z['scores'][j,i]-expected)));assert err<1e-8
                    assert np.array_equal(z['predictions'][j,i],expected.argmax(-1))
                    maxerr=max(maxerr,err);checks+=1
    result=dict(status='passed',independent_score_comparisons=checks,max_score_error=maxerr,
        selection_integer_choices_verified=True,all_output_consistency=True,
        historical_scatter_audit_unchanged='failed1/88',full_statistics_independent_audit_pending=True)
    run.dump('output_audit.json',result)
    print('OUTPUT_AUDIT',json.dumps(result),flush=True)

if __name__=='__main__':
    torch.set_num_threads(2)
    audit()
