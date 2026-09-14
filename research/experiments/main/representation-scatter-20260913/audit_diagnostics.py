"""Audit all aggregates and fixed sampled scores with existing independent NumPy algebra.

The verified cache loader is reused; candidate transforms use dense eigendecomposition,
not the measured low-rank Torch SVD. Run after complete.json. No fitting/tuning changes.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
import torch
import evaluate as e
from verify import dense_metric, numpy_r2

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
TOL=2e-5

_dense_metric = dense_metric
_context = None

def dense_metric(S,Q,G,gamma):
    global _context
    result = _dense_metric(S,Q,G,gamma)
    _context = (S,Q,G,gamma,result[0])
    return result

def compare(actual,expected,label,errors):
    error=float(np.max(np.abs(actual-expected)))
    row={'label':label,'max_score_error':error,'argmax_differences':int(np.sum(actual.argmax(-1)!=expected.argmax(-1))),'frozen_comparison_passed':bool(error<TOL and np.array_equal(actual.argmax(-1),expected.argmax(-1)))}
    if not row['frozen_comparison_passed']:
        S,Q,G,gamma,dense=_context
        factor=e.metric.scatter_factor(S,gamma)
        tr=[e.metric.transform(x.mean(-2),factor) for x in [S,Q,G]]
        row['transform_max_error']=max(float(abs(t.numpy()-d).max()) for t,d in zip(tr,dense))
        assert S.shape[1]==1,('unexpected five-shot discrepancy',label)
        ix=e.metric.ref.retrieve(e.metric.norm(tr[0].mean(1))[None],tr[2],64)[0].numpy()
        ds,dq,dg=dense
        proto=ds.mean(1);proto/=np.linalg.norm(proto,axis=-1,keepdims=True)
        sim=proto@dg.T;ind=np.argsort(-sim,axis=-1)[:,:64]
        changed=[]
        for cls,(x,y) in enumerate(zip(ix,ind)):
            diff=sorted(set(x)^set(y))
            if diff:changed.append({'class':cls,'symmetric_difference':[int(v) for v in diff],'dense_similarity_range':float(np.ptp(sim[cls,diff]))})
        row['neighbor_changes']=changed
        gal=dg[ix].mean(1);gal/=np.linalg.norm(gal,axis=-1,keepdims=True)
        X=.5*proto+.5*gal;X/=np.linalg.norm(X,axis=-1,keepdims=True)
        Y=np.eye(5);xc=X-X.mean(0);yc=Y-Y.mean(0)
        W=np.linalg.solve(xc.T@xc+.1*np.eye(X.shape[1]),xc.T@yc)
        conditional=(dq-X.mean(0))@W+Y.mean(0)
        row['fixed_neighbor_numpy_score_error']=float(abs(actual-conditional).max())
        row['fixed_neighbor_argmax_differences']=int(np.sum(actual.argmax(-1)!=conditional.argmax(-1)))
        assert row['fixed_neighbor_numpy_score_error']<TOL
        assert row['fixed_neighbor_argmax_differences']==0
    errors.append(row)

def main():
    started=time.time();torch.set_num_threads(1)
    assert json.loads((OUT/'complete.json').read_text())['status']=='success'
    locks=json.loads((HERE/'evaluation_lock.json').read_text())
    for path,digest in locks.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    manifest=json.loads((OUT/'evaluation_manifest.json').read_text());assert manifest['sources']==locks
    data=e.load();cfg=e.CFG;errors=[];selections={};paths={}
    for ds in cfg['source_domains']:
        for shot in cfg['shots']:
            f=OUT/f'{ds}_k{shot}_selection.npz';z=np.load(f);paths[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
            assert np.isfinite(z['scores']).all()
            t=e.sampler.tasks(data[ds]['ident'],shot,'selection')
            for key in t:assert np.array_equal(t[key],z[key]),(f,key)
            y=np.repeat(np.arange(5),z['query_indices'].shape[-1]);correct=z['scores'].argmax(-1)==y
            assert np.array_equal(z['accuracy'],correct.mean(-1))
            counts=correct.sum((0,1,3));best=min(g for g,n in zip(cfg['gamma_grid'],counts) if n==counts.max())
            saved=json.loads(f.with_suffix('.json').read_text());assert saved['gamma']==best
            means=counts/(correct.shape[0]*correct.shape[1]*correct.shape[3])
            assert np.allclose(saved['mean_accuracy'],means,atol=1e-14,rtol=0)
            old=np.load(OUT/'superseded_float_tie/outputs'/f.name)
            assert np.array_equal(z['scores'],old['scores']),('source predictions changed',f)
            labels=np.array(data[ds]['ident']['gallery_labels'])
            for i,classes in enumerate(z['class_ids']):
                for excluded in [0,1]:
                    pool=np.flatnonzero(~np.isin(labels,classes)) if excluded else np.arange(len(labels))
                    rng=np.random.RandomState(e.EXTRA['source_gallery_seed']+10000*excluded+i)
                    assert np.array_equal(z['gallery_indices'][i,excluded],rng.choice(pool,e.EXTRA['source_gallery_size'],replace=False))
            for i in [0,99]:
                s=data[ds]['query'][z['support_indices'][i]];q=data[ds]['query'][z['query_indices'][i].reshape(-1)]
                for condition in [0,1]:
                    g=data[ds]['gallery'][z['gallery_indices'][i,condition]]
                    for gi,gamma in enumerate(cfg['gamma_grid']):
                        dense,_=dense_metric(s,q,g,gamma)
                        compare(z['scores'][i,condition,gi],numpy_r2(*dense),f'{ds}/k{shot}/selection/{i}/{condition}/{gamma}',errors)
            selections[ds+f'_k{shot}']={'gamma':best,'correct_counts':counts.tolist()}
    summary=json.loads((OUT/'analysis.json').read_text());assert len(summary['cells'])==8
    task_conditions=0;historical_error=0.;candidate_samples=0
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for shot in cfg['shots']:
            t=e.sampler.tasks(data[target]['ident'],shot,'eval');saved_cells=[]
            for gallery in [target,source]:
                name=f'{source}_to_{target}_{gallery}_k{shot}';f=OUT/'cells'/(name+'.npz');z=np.load(f)
                paths[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest();names=z['names'].tolist();assert names==e.METHODS
                for key in t:assert np.array_equal(t[key],z[key]),(name,key)
                assert len(t['seeds'])==500 and set(t['seeds'])==set(cfg['eval_seeds'])
                assert not set(data[target]['ident']['query_rgb'])&set(data[gallery]['ident']['gallery_rgb'])
                assert np.isfinite(z['scores']).all() and np.array_equal(z['predictions'],z['scores'].argmax(-1))
                assert np.array_equal(z['yq'],np.repeat(np.arange(5),15))
                accuracy=(z['scores'].argmax(-1)==z['yq']).mean(-1);assert np.array_equal(accuracy,z['accuracy'])
                for j,method in enumerate(names):
                    assert abs(100*accuracy[j].mean()-summary['cells'][name]['accuracy_pct'][method])<1e-10
                gamma=selections[source+f'_k{shot}']['gamma'];assert z['chosen_gamma'].item()==gamma
                old=np.load(HERE.parent/'utility-composition-qualification-20260913/outputs/cells'/(name+'.npz'))
                for current,previous in [('original_r2','r2'),('original_CS_l2','cs_0.1')]:
                    error=float(np.max(abs(z['scores'][names.index(current)]-old['scores'][old['names'].tolist().index(previous)])))
                    assert error<2e-6,(name,current,error);historical_error=max(historical_error,error)
                for i in [0,100,200,300,400]:
                    s=data[target]['query'][t['support_indices'][i]];q=data[target]['query'][t['query_indices'][i].reshape(-1)];g=data[gallery]['gallery']
                    dense,_=dense_metric(s,q,g,gamma)
                    compare(z['scores'][names.index('scatter_r2'),i],numpy_r2(*dense),name+f'/scatter/{i}',errors)
                    candidate_samples+=1
                task_conditions+=len(t['seeds']);saved_cells.append((names,accuracy))
            if shot==1:
                names,first=saved_cells[0];macro=(first+saved_cells[1][1])/2;ci=names.index('scatter_r2')
                for j,method in enumerate(names):
                    if method!='scatter_r2':assert abs(100*(macro[ci]-macro[j]).mean()-summary['directions'][source+'_to_'+target][method]['delta_pp'])<1e-10
    assert task_conditions==4000
    gate=all(v['ci95_pp'][0]>=-.5 for c in summary['cells'].values() for v in c['comparisons'].values())
    gate=gate and all(v['delta_pp']>=.5 and v['ci95_pp'][0]>0 for d in summary['directions'].values() for v in d.values())
    assert (summary['status']=='passed')==gate
    result={'status':'diagnostic_only_original_frozen_audit_failed','task_conditions':task_conditions,'methods_per_cell':len(e.METHODS),'source_gamma_selections':selections,'source_and_eval_numpy_score_checks':len(errors),'frozen_score_failures':[r for r in errors if not r['frozen_comparison_passed']],'frozen_audit_passed':all(r['frozen_comparison_passed'] for r in errors),'candidate_sampled_eval_tasks':candidate_samples,'max_numpy_score_error':max(r['max_score_error'] for r in errors),'historical_original_score_max_error':historical_error,'all_source_predictions_equal_pre_repair':True,'score_tolerance':TOL,'frozen_gate_result':summary['status'],'elapsed_seconds':time.time()-started,'input_sha256':paths,'score_checks':errors,'scope':'Diagnostic derivative, NOT a passing frozen audit. No tolerance, method, selection, input or acceptance change. Fixed-neighbor algebra checks explicitly condition on float32 neighbor choices; end-to-end cross-precision failures remain failures. All aggregate values, source choices, tasks, original control scores verified; dense independent candidate scores on48source cases and40evaluation tasks. Reuses verified cache loader. Bootstrap interval algorithm not independently reimplemented. Fixed-pool development evidence only.'}
    (OUT/'output_audit_diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','score_checks']},indent=2),flush=True)

if __name__=='__main__':main()
