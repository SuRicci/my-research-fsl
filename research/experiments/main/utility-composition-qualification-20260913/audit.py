"""Reconstruct saved controlled predictions and audit source/target data boundaries."""
from pathlib import Path
import json, sys, hashlib
import numpy as np
import torch
import run as exp
import utility
P=Path(__file__).resolve().parent;O=P/'outputs'
assert (O/'complete.json').exists(), 'Do not audit an incomplete run as complete.'
receipt=json.loads((P/'resume_receipt.json').read_text())
assert all(exp.old.sha(f)==h for f,h in receipt['retained_sha256'].items())
locks=json.loads((P/'locked_sources.json').read_text())
assert all(exp.old.sha(f)==h for f,h in locks.items())
data,manifest=exp.ref.assets()
rows=[]
for source in ['dtd','eurosat']:
    ident=data[source]['ident'];labels=np.array(ident['query_labels'])
    for shot in [1,5]:
        used={}
        for phase in ['train','selection']:
            with np.load(O/f'{source}_k{shot}_{phase}_tasks.npz') as z:
                assert len(z['seeds'])==100
                assert np.array_equal(labels[z['support_indices']],np.broadcast_to(z['class_ids'][...,None],z['support_indices'].shape))
                assert np.array_equal(labels[z['query_indices']],np.broadcast_to(z['class_ids'][...,None],z['query_indices'].shape))
                for s,q in zip(z['support_indices'],z['query_indices']):assert not set(s.flat)&set(q.flat)
                used[phase]=set(z['support_indices'].flat)|set(z['query_indices'].flat)
        assert not used['train']&used['selection']
        models=json.loads((O/f'models_{source}_k{shot}.json').read_text())
        for mode,m in models['models'].items():
            assert np.isfinite(m['coef']).all() and np.isfinite(m['alpha_logit'])
            h=models['history'][mode]
            best=max(h,key=lambda x:(x['selection_accuracy'],-x['selection_ce'],-x['epoch']))
            assert m['epoch']==best['epoch']
        assert models['grid_winner']==max(models['grid_means'],key=models['grid_means'].get)
with torch.no_grad():
    for p in sorted((O/'cells').glob('*.npz')):
        name=p.stem;source,other=name.split('_to_');target,gallery,k=other.split('_');shot=int(k[1:]);m=json.loads((O/f'models_{source}_k{shot}.json').read_text())
        with np.load(p) as z:
            assert len(z['seeds'])==500
            assert np.array_equal(np.unique(z['seeds'],return_counts=True)[1],np.repeat(100,5))
            assert len(z['names'])==len(set(z['names']))
            assert np.isfinite(z['scores']).all()
            assert np.array_equal(z['predictions'],z['scores'].argmax(-1))
            assert np.array_equal(z['accuracy'],(z['predictions']==z['yq']).mean(-1))
            labs=np.array(data[target]['ident']['query_labels'])
            assert np.array_equal(labs[z['query_indices']],np.broadcast_to(z['class_ids'][...,None],z['query_indices'].shape))
            assert np.array_equal(labs[z['support_indices']],np.broadcast_to(z['class_ids'][...,None],z['support_indices'].shape))
            for s,q in zip(z['support_indices'],z['query_indices']):assert not set(s.flat)&set(q.flat)
            # Re-evaluate three first/central/last tasks, trained nonzero weights, without target labels.
            ix=[0,249,499];t={key:z[key][ix] for key in ['support_indices','query_indices','class_ids','seeds']}
            S,Q,cache=exp.batch(data[target],t,data[gallery]['gallery']);names=z['names'].tolist();errors={}
            for mode,model in m['models'].items():
                c=torch.tensor(model['coef']);a=torch.tensor(model['alpha_logit'])
                sc=utility.scores(cache,Q,c,a,exp.ref.ridge_scores,mode=='scalar')
                error=float(np.abs(sc.numpy()-z['scores'][names.index(mode),ix]).max());assert error<2e-5
                split=torch.cat([utility.scores(cache,q,c,a,exp.ref.ridge_scores,mode=='scalar') for q in Q.split(13,1)],1)
                assert torch.allclose(sc,split,atol=2e-6)
                errors[mode]=error
            # Independently reconstruct trained utility neighbor means and dual solve in NumPy.
            model=m['models']['utility'];proto,neighbors,feat,_=cache
            logits=feat.numpy()@(5*np.tanh(model['coef']));w=np.exp(logits-logits.max(-1,keepdims=True));w/=w.sum(-1,keepdims=True)
            mu=(neighbors.numpy()*w[...,None]).sum(2);mu/=np.linalg.norm(mu,axis=-1,keepdims=True)
            alpha=1/(1+np.exp(-model['alpha_logit']));a=(1-alpha)*proto.numpy()+alpha*mu;a/=np.linalg.norm(a,axis=-1,keepdims=True)
            origin=a.mean(1,keepdims=True);xc=a-origin;targety=np.eye(5)-.2
            beta=np.linalg.solve(xc@xc.transpose(0,2,1)+.1*np.eye(5),np.broadcast_to(targety,(3,5,5)))
            sc=(Q.numpy()-origin)@xc.transpose(0,2,1)@beta+.2
            np_error=float(np.abs(sc-z['scores'][names.index('utility'),ix]).max());assert np_error<2e-5
            rows.append(dict(cell=name,task_count=500,methods=len(names),reconstruction_errors=errors,numpy_error=np_error,sha256=exp.old.sha(p)))
assert len(rows)==8
result=dict(status='passed',cell_count=8,task_condition_evaluations=4000,source_tasks_per_shot=100,source_selection_tasks_per_shot=100,
            source_train_selection_image_disjoint=True,source_only_checkpoint_selection=True,all_task_labels_checked=True,
            every_score_prediction_accuracy_checked=True,trained_query_partition_invariant=True,independent_numpy_reconstruction=True,rows=rows,
            limitations=['Only two exposed development domains','One deterministic training-task allocation; uncertainty conditional on it and fixed image pools','Fixed training horizon does not exhaust utility learning or BCE families'])
exp.old.dump(O/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
