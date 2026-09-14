"""Independent NumPy score reconstruction and complete saved-outcome checks."""
from pathlib import Path
import json,sys
import numpy as np
import torch
import study as m

def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def score(state,s,q):
    down=state['down.weight'].numpy().astype(np.float64)
    up=state['up.weight'].numpy().astype(np.float64)
    def transform(x):
        x=x.astype(np.float64).mean(-2)
        return unit(x+np.maximum(x@down.T,0)@up.T)
    x=transform(s).reshape(-1,896);z=transform(q)
    y=np.repeat(np.eye(5),s.shape[1],axis=0)
    xc=x-x.mean(0);yc=y-y.mean(0)
    return (z-x.mean(0))@xc.T@np.linalg.solve(xc@xc.T+m.CFG['ridge_penalty']*np.eye(len(x)),yc)+y.mean(0)

def task_check(t,labels):
    for s,q,cs in zip(t['support_indices'],t['query_indices'],t['class_ids']):
        assert len(np.unique(np.r_[s.ravel(),q.ravel()]))==s.size+q.size
        assert np.array_equal(labels[s],np.repeat(cs[:,None],s.shape[1],axis=1))
        assert np.array_equal(labels[q],np.repeat(cs[:,None],q.shape[1],axis=1))

def check(data):
    errors=[];counts={};task_count=0
    ft=m.flower_tasks(data,5,'train');fv=m.flower_tasks(data,5,'selection')
    assert not set(ft['class_ids'].ravel())&set(fv['class_ids'].ravel())
    for ds in m.CFG['source_domains']:
        for k in m.CFG['shots']:
            x,labels,banks,va,offset=m.source_tasks(data,ds,k)
            task_check(va,labels);task_count+=len(va['seeds'])
            vi=set(np.r_[va['support_indices'].ravel(),va['query_indices'].ravel()])
            for arm,t in banks.items():
                task_check(t,labels);task_count+=len(t['seeds'])
                assert not vi&set(np.r_[t['support_indices'].ravel(),t['query_indices'].ravel()])
                assert len(t['seeds'])==100
                counts[f'{ds}_k{k}_{arm}']=m.exposure(t,labels,offset)
                torch.manual_seed(101);net=m.model.Map()
                with torch.no_grad():net.up.weight.normal_(0,.01)
                s,q=m.batch(x,t,np.array([0]))
                got=m.model.scores(net,s,q,m.MODE,.1).detach().numpy()[0]
                expected=score(net.state_dict(),s.numpy()[0],q.numpy()[0])
                e=float(np.max(np.abs(got-expected)));assert e<m.CFG['validation']['score_atol'],e;errors.append(e)
                # Each query is classified independently of the other queries.
                single=m.model.scores(net,s,q[:,:1],m.MODE,.1).detach().numpy()[0,0]
                assert np.max(np.abs(single-got[0]))<1e-6
            torch.manual_seed(101);net=m.model.Map()
            s,q=m.batch(x,banks['original'],np.array([0]))
            zero=m.model.scores(net,s,q,m.MODE,.1).detach().numpy()[0]
            expected=score(net.state_dict(),s.numpy()[0],q.numpy()[0])
            assert np.max(np.abs(zero-expected))<m.CFG['validation']['score_atol']
            sc=m.model.scores(net,s,q,m.MODE,.1);sc.square().mean().backward()
            assert all(torch.isfinite(p.grad).all() for p in net.parameters())
            assert float(net.up.weight.grad.abs().max())>0
    # Check canonical fused rows against independent block normalization.
    for ds in m.CFG['source_domains']:
        blocks=[]
        for b in ['clip_vitb16','dinov2_vits14']:
            z=torch.load(m.ASSET/(ds+'_'+b+'_query.pt'),weights_only=True)
            blocks.append(unit(z['features'][:7].numpy().astype(np.float64)))
        wanted=unit(np.concatenate(blocks,axis=-1))
        assert np.max(np.abs(wanted-data[ds]['x'][:7,0].numpy()))<1e-6
    return {'status':'passed','task_rows_checked':task_count,'numerical_cases':len(errors),
            'maximum_score_error':max(errors),'query_independence':'passed','finite_nonzero_gradient':'passed',
            'source_role_disjointness':'passed','exposure':counts}

def audit():
    torch.set_num_threads(m.CFG['resources']['threads'])
    data,assets=m.load();lock=json.loads((m.HERE/'lock.json').read_text())
    assert lock=={'code':m.inputs(),'assets':assets}
    complete=json.loads((m.OUT/'complete.json').read_text())
    assert m.sha(m.OUT/'selection_lock.json')==complete['selection_lock_sha256']
    choices=json.loads((m.OUT/'selection_lock.json').read_text());assert len(choices)==36
    initial={};errors=[];all_predictions=0
    for key,e in choices.items():
        token=(e['source'],e['shot'],e['seed'])
        initial.setdefault(token,set()).add(e['initial_state_sha256'])
        assert m.sha(e['model_path'])==e['sha256']
        best=max(e['history'],key=lambda v:(v['correct'],-v['ce'],-v['step']))
        assert e['step']==best['step']
        x,labels,banks,va,offset=m.source_tasks(data,e['source'],e['shot'])
        saved=np.load(m.OUT/'selection'/f'{key}.npz')
        for j,row in enumerate(e['history']):
            assert row['correct']==int((saved['predictions'][j]==saved['yq']).sum())
        net=m.model.Map();net.load_state_dict(torch.load(e['model_path'],weights_only=True))
        current=m.predict(net,x,va)
        step_index=m.CFG['checkpoints'].index(e['step'])
        assert np.array_equal(current.argmax(-1),saved['predictions'][step_index])
        all_predictions+=current.shape[0]*current.shape[1]
        for i in m.CFG['validation']['audit_selection_indices']:
            s,q=m.batch(x,va,np.array([i]));want=score(net.state_dict(),s.numpy()[0],q.numpy()[0])
            err=float(np.max(np.abs(want-current[i])));assert err<m.CFG['validation']['score_atol'],(key,i,err);errors.append(err)
        assert m.exposure(banks[e['arm']],labels,offset)==json.loads((m.OUT/'exposure.json').read_text())[f"{e['source']}_k{e['shot']}"][e['arm']]
    assert all(len(v)==1 for v in initial.values())
    task_scores=0
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for k in m.CFG['shots']:
            z=np.load(m.OUT/'cells'/f'{source}_to_{target}_k{k}.npz')
            t=dict(np.load(m.BANK/f'eval_{target}_k{k}_tasks.npz'))
            assert all(np.array_equal(z[a],v) for a,v in t.items())
            for j,arm in enumerate(m.ARMS):
                for h,seed in enumerate(m.CFG['training_seeds']):
                    e=choices[f'{source}_k{k}_{arm}_{seed}']
                    net=m.model.Map();net.load_state_dict(torch.load(e['model_path'],weights_only=True))
                    current=m.predict(net,data[target]['x'],t)
                    assert np.array_equal(current.argmax(-1),z['predictions'][j,h])
                    all_predictions+=current.shape[0]*current.shape[1]
                    for a,i in enumerate(z['audit_indices']):
                        s,q=m.batch(data[target]['x'],t,np.array([i]))
                        want=score(net.state_dict(),s.numpy()[0],q.numpy()[0])
                        err=float(np.max(np.abs(want-z['audit_scores'][j,h,a])))
                        assert err<m.CFG['validation']['score_atol'],(source,k,arm,seed,int(i),err)
                        errors.append(err);task_scores+=1
            net=m.model.Map();zero=m.predict(net,data[target]['x'],t)
            assert np.array_equal(zero.argmax(-1),z['zero_predictions'])
            for gallery in [source,target]:
                old=np.load(m.BANK/'cells'/f'eval_{target}_k{k}_{gallery}_v00.npz')
                assert old['predictions'].shape==(3,500,75)
            print('AUDIT_EVAL',source,target,k,flush=True)
    # Independent paired bootstrap without the vectorized weights implementation.
    report=json.loads((m.OUT/'analysis.json').read_text());stat_count=0
    def interval(delta,seeds):
        rng=np.random.RandomState(m.rv.CFG['bootstrap_seed']);v=np.zeros(m.rv.CFG['bootstrap_replicates'])
        groups=np.unique(seeds)
        for seed in groups:
            a=delta[seeds==seed];ix=rng.randint(len(a),size=(len(v),len(a)))
            v+=a[ix].mean(1)/len(groups)
        return {'delta_pp':float(delta.mean()*100),'ci95_pp':(np.quantile(v,[.025,.975])*100).tolist()}
    def compare(got,expected):
        assert abs(got['delta_pp']-expected['delta_pp'])<1e-10
        assert np.max(np.abs(np.array(got['ci95_pp'])-expected['ci95_pp']))<1e-10
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        passed_refs=[]
        for k in m.CFG['shots']:
            z=np.load(m.OUT/'cells'/f'{source}_to_{target}_k{k}.npz')
            a=(z['predictions']==z['yq']).mean(-1);mean=a.mean(1)
            local={'original':mean[0],'flowers':mean[1],'zero_ridge01':(z['zero_predictions']==z['yq']).mean(-1)}
            for gallery in [target,source]:
                old=np.load(m.BANK/'cells'/f'eval_{target}_k{k}_{gallery}_v00.npz')
                refs={**local,**dict(zip(m.rv.HEADS,(old['predictions']==z['yq']).mean(-1)))}
                row=report['cells'][f'{source}_to_{target}_{gallery}_k{k}']
                for name,val in refs.items():
                    compare(row['comparisons'][name],interval(mean[2]-val,z['seeds']));stat_count+=1
                if k==1:passed_refs.append(refs)
            if k==1:
                row=report['directions'][source+'_to_'+target]
                for name in local.keys()|set(m.rv.HEADS):
                    compare(row['comparisons'][name],interval(mean[2]-(passed_refs[0][name]+passed_refs[1][name])/2,z['seeds']));stat_count+=1
                assert np.max(np.abs(np.array(row['mixed_minus_original_by_training_seed_pp'])-(a[2]-a[0]).mean(-1)*100))<1e-10
    r={'status':'passed','fitted_models':len(choices),'recomputed_query_predictions':all_predictions,
       'independent_score_cases':len(errors),'max_score_error':max(errors),'independent_statistics':stat_count,
       'immutable_inputs':'passed','paired_initializations':'passed','selection_lock':'passed',
       'scope':'conditional development evidence; no new canonical target result'}
    m.dump(m.OUT/'independent_audit.json',r);print('AUDIT_COMPLETE',r,flush=True)
if __name__=='__main__':audit()
