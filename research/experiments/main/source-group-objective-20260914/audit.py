"""Independent source splits, objective gradients, ridge formula and paired statistics."""
from pathlib import Path
import importlib.util, json, sys
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('group_objective_study',HERE/'study.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
def ridge_numpy(state,support,query):
    down=state['down.weight'].numpy().astype(np.float64)
    up=state['up.weight'].numpy().astype(np.float64)
    def feature(x):
        x=x.astype(np.float64);x=x+np.maximum(x@down.T,0)@up.T
        return x/np.linalg.norm(x,axis=-1,keepdims=True)
    support=feature(support);query=feature(query)
    origin=support.mean(0);x=support-origin;y=np.eye(5)-.2
    return (query-origin)@x.T@np.linalg.solve(x@x.T+.1*np.eye(5),y)+.2
def check_math(data):
    for ds in s.CFG['source_domains']:s.get_source(data,ds)
    x,tr,va=s.get_source(data,'dtd')
    ix=np.array([0,50]);support,query=s.old.batch(x,tr,ix)
    support=support.double();query=query.double()
    torch.manual_seed(26091431);net=s.model.Map(896,32).double()
    initial=s.model.scores(net,support,query,'map_after_mean',.1)
    ref=s.ce_groups(initial).detach()
    assert float(s.objective(s.ce_groups(initial),ref,'reference_max'))==0.
    with torch.no_grad():net.up.weight.normal_(0,.01)
    errors={};gerrors={};sc=s.model.scores(net,support,query,'map_after_mean',.1)
    ref_manual=[]
    for row in sc.detach().numpy():
        logits=10*row;mx=logits.max(1,keepdims=True)
        logsum=np.log(np.exp(logits-mx).sum(1))+mx[:,0]
        ref_manual.append(float((logsum-logits[np.arange(75),s.Y]).mean()))
    assert np.allclose(s.ce_groups(sc).detach().numpy(),ref_manual,atol=1e-12)
    for i in range(2):
        manual=ridge_numpy(net.state_dict(),support[i].numpy().reshape(5,896),query[i].numpy().reshape(75,896))
        errors[str(i)]=float(abs(manual-sc[i].detach().numpy()).max());assert errors[str(i)]<1e-10
    for arm in s.OBJECTIVES:
        def f():return s.objective(s.ce_groups(s.model.scores(net,support,query,'map_after_mean',.1)),ref,arm)
        value=f();grad=torch.autograd.grad(value,net.up.weight)[0]
        flat=int(grad.abs().argmax());row,col=np.unravel_index(flat,grad.shape)
        eps=1e-5;v=float(net.up.weight[row,col])
        with torch.no_grad():net.up.weight[row,col]=v+eps
        hi=float(f())
        with torch.no_grad():net.up.weight[row,col]=v-eps
        lo=float(f())
        with torch.no_grad():net.up.weight[row,col]=v
        err=abs((hi-lo)/(2*eps)-float(grad[row,col]));gerrors[arm]=err;assert err<1e-6,(arm,err)
    # Subtracting a constant from the pooled objective has identical gradients.
    a=s.ce_groups(s.model.scores(net,support,query,'map_after_mean',.1))
    ga=torch.autograd.grad(a.mean(),net.up.weight,retain_graph=True)[0]
    gb=torch.autograd.grad((a-ref).mean(),net.up.weight)[0];assert torch.equal(ga,gb)
    return {'status':'passed','numpy_ridge_max_error':max(errors.values()),'gradient_errors':gerrors,
            'identity_reference_max_zero':True,'constant_mean_gradient_equal':True,
            'source_banks_exact_and_disjoint':True,'tasks_per_group':50,'device':'cpu',
            'torch':torch.__version__,'numpy':np.__version__}
def bootstrap(delta,seeds):
    rng=np.random.RandomState(26091430);values=np.zeros(5000)
    groups=[np.flatnonzero(seeds==v) for v in sorted(set(seeds.tolist()))]
    for ids in groups:
        values+=delta[ids[rng.randint(0,len(ids),size=(5000,len(ids)))]].sum(axis=1)/len(delta)
    return float(delta.mean()*100),(100*np.quantile(values,[.025,.975])).tolist()
def main():
    torch.set_num_threads(6);data,assets=s.old.load()
    lock=json.loads((HERE/'lock.json').read_text())
    assert lock=={'files':s.immutable(),'assets':assets}
    choices=json.loads((s.OUT/'selection_lock.json').read_text())
    done=json.loads((s.OUT/'complete.json').read_text())
    assert s.sha(s.OUT/'selection_lock.json')==done['selection_lock_sha256']
    result=json.loads((s.OUT/'analysis.json').read_text())
    maxerr=0.;ncases=0;nquery=0;ninterval=0
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        x,tr,va=s.get_source(data,source);ref=s.ce_groups(s.identity(x,va)).numpy()
        z=np.load(s.OUT/f'{source}_to_{target}.npz')
        t={k:z[k] for k in ['support_indices','query_indices','class_ids','seeds']}
        assert all(np.array_equal(t[k],v) for k,v in dict(np.load(s.old.BANK/f'eval_{target}_k1_tasks.npz')).items())
        xt=data[target]['x']
        for ia,arm in enumerate(s.OBJECTIVES):
            for isel,sel in enumerate(s.SELECTORS):
                for iz,seed in enumerate(s.CFG['seeds']):
                    record=choices[f'{source}_{seed}_{arm}'];e=record['selected'][sel]
                    assert s.sha(e['path'])==e['sha256']
                    def ranking(row):
                        loss=np.array(row['group_ce'])
                        return (float(loss.mean() if sel=='pooled_ce' else (loss-ref).max()),row['step'])
                    chosen=min(record['history'],key=ranking)
                    assert chosen['step']==e['step']
                    net=s.model.Map(896,32);state=torch.load(e['path'],weights_only=True);net.load_state_dict(state)
                    with torch.no_grad():vs=s.scores(net,x,va)
                    assert np.array_equal(vs.argmax(-1).numpy(),np.load(e['selection_path'])['predictions'])
                    assert np.allclose(s.ce_groups(vs).numpy(),chosen['group_ce'],atol=1e-7)
                    sc=s.old.predict(net,xt,t)
                    assert np.array_equal(sc.argmax(-1),z['predictions'][ia,isel,iz]);nquery+=sc.shape[0]*sc.shape[1]
                    for j,idx in enumerate(s.CFG['audit_indices']):
                        xx=xt.numpy()[:,0,:]
                        manual=ridge_numpy(state,xx[t['support_indices'][idx]].reshape(5,896),xx[t['query_indices'][idx]].reshape(75,896))
                        err=float(abs(manual-z['audit_scores'][ia,isel,iz,j]).max());maxerr=max(maxerr,err);assert err<3e-5
                        ncases+=1
        zero=s.old.predict(s.model.Map(896,32),xt,t).argmax(-1)
        assert np.array_equal(zero,z['identity_predictions'])
        accuracy=(z['predictions']==s.Y).mean(-1)
        avg=accuracy.mean(2);candidate=avg[2,1]
        controls={'identity':(zero==s.Y).mean(-1),'matched_mean':avg[0,1],'matched_raw_max':avg[1,1]}
        strong=[]
        for gallery in ['dtd','eurosat']:
            f=np.load(HERE.parent/'query-consistency-20260913/outputs'/f'{target}_{gallery}.npz')
            for k in t:assert np.array_equal(t[k],f[k])
            strong.append((f['predictions'][f['names'].tolist().index('consistency')]==s.Y).mean(-1))
        controls['strong_stack_macro']=np.mean(strong,axis=0)
        section=result['directions'][source+'_to_'+target]
        for ia,arm in enumerate(s.OBJECTIVES):
            for isel,sel in enumerate(s.SELECTORS):
                assert abs(section['arms'][arm+'/'+sel]['accuracy_pct']-avg[ia,isel].mean()*100)<1e-10
        primary=section['arms']['reference_max/reference_max_ce']
        passes=[]
        for name,values in controls.items():
            point,ci=bootstrap(candidate-values,t['seeds']);reported=primary['comparisons'][name]
            assert abs(point-reported['delta_pp'])<1e-10 and np.allclose(ci,reported['ci95_pp'],atol=1e-10)
            passes.append(point>=.5 and ci[0]>0);ninterval+=1
        for j,arm in enumerate(s.OBJECTIVES[:2]):
            effects=(accuracy[2,1]-accuracy[j,1]).mean(-1)*100
            assert np.allclose(effects,section['primary_seed_differences_pp'][arm],atol=1e-10)
            passes.append(int((effects>0).sum())>=2)
        assert bool(all(passes))==section['gate_passed']
    assert result['qualification_gate_passed']==all(v['gate_passed'] for v in result['directions'].values())
    assert lock=={'files':s.immutable(),'assets':s.old.load()[1]}
    out={'status':'passed','model_selector_checks':36,'prediction_checks':nquery,'independent_ridge_cases':ncases,
         'max_ridge_error':maxerr,'primary_intervals':ninterval,'input_locks_unchanged':True,
         'all_selected_checkpoint_ranks_verified':True,'gate_verified':True}
    s.dump(s.OUT/'independent_audit.json',out);print('AUDIT_PASSED',out,flush=True)
if __name__=='__main__':main()
