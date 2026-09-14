"""Post-failure diagnostic only; preserves the frozen audit and its failure."""
import json, time
import numpy as np
import torch
import torch.nn.functional as F
import caltech_audit as a
e=a.e
OUT=e.HERE/'diagnostics'
OUT.mkdir(exist_ok=True)

@torch.no_grad()
def tied_ensemble(sv,qv,gv,actual):
    s,q,g=[x.numpy().astype(np.float64) for x in [sv,qv,gv]]
    values=[]; boundaries=[]
    for v in range(6):
        p=a.norm(s[:,:,v].mean(1))
        ix64=np.argsort(-(p@g[:,v].T),axis=-1,kind='stable')[:,:64]
        pt=F.normalize(sv[:,:,v].mean(1),dim=-1)[None]
        ix32=e.metric.ref.retrieve(pt,gv[:,v],64)[0].numpy()
        for c in range(5):
            if set(ix32[c]) != set(ix64[c]):
                sims=p[c]@g[:,v].T; ordered=np.sort(sims)[::-1]
                diff=sorted(set(ix32[c]) ^ set(ix64[c]))
                boundaries.append({'view':v,'class':c,'symmetric_difference':diff,
                    'float64_boundary_gap':float(ordered[63]-ordered[64]),
                    'float64_scores_at_differing_ids':sims[diff].tolist()})
        x=a.norm(.5*p+.5*a.norm(g[:,v][ix32].mean(1)))
        values.append(a.ridge(x,q[:,v]))
    score=np.mean(values,axis=0)
    return {'neighbor_boundary_changes':boundaries,'fixed_float32_neighbors_dense_score_error':float(abs(actual-score).max()),
        'fixed_float32_neighbors_prediction_mismatches':int((actual.argmax(-1)!=score.argmax(-1)).sum()),
        'interpretation':'Diagnostic conditioning on original float32 selected identities; not a replacement for the failed fully independent audit.'}

def main():
    torch.set_num_threads(1);start=time.time();e.guard();e.check_lock(e.HERE/'evaluation_code_lock.json')
    failure_hash=e.sha(e.OUT/'audit_failure.json')
    target,galleries,tasks=e.load_data();summary=json.loads((e.OUT/'analysis.json').read_text())
    rows=[];interval_errors=[];bank_checks=[];controls={};primary=e.NAMES.index(e.CFG['primary_method'])
    for source,cfg in e.CFG['source_configurations'].items():
        accs=[]
        for gallery,gv in galleries.items():
            key=source+'_'+gallery;f=np.load(e.OUT/(key+'.npz'));score=f['scores'];pred=f['predictions']
            assert f['names'].tolist()==e.NAMES and score.shape==(10,500,75,5)
            assert np.isfinite(score).all() and np.array_equal(score.argmax(-1),pred)
            assert all(np.array_equal(f[k],tasks[k]) for k in e.KEYS)
            assert f['code_lock_sha256'].item()==e.sha(e.HERE/'evaluation_code_lock.json')
            acc=(pred==e.Y).mean(-1);assert np.array_equal(acc,f['accuracy']);accs.append(acc)
            assert np.allclose(acc.mean(-1)*100,list(summary['cells'][key]['accuracy_pct'].values()),rtol=0,atol=1e-10)
            if source=='dtd':controls[gallery]=score[:7].copy()
            else:assert np.array_equal(controls[gallery],score[:7])
            if cfg['eta']==0:assert np.array_equal(score[primary],score[e.NAMES.index('scatter_blend')])
            for k,name in enumerate(e.NAMES):
                if k==primary:continue
                got=a.independent_interval(acc[primary]-acc[k],f['seeds']);want=summary['cells'][key]['comparisons'][name]
                interval_errors.append(float(abs(got-np.r_[want['delta_pp'],want['ci95_pp']]).max()))
            for i in a.P['target_tasks']:
                e.guard();qi=a.P['target_query_indices_for_direct_primal'];sv=target[tasks['support_indices'][i]];qv=target[tasks['query_indices'][i].reshape(-1)[qi]]
                expected=a.independent(sv,qv,gv,**cfg)
                for name,want in expected.items():
                    got=score[e.NAMES.index(name),i,qi];err=float(abs(got-want).max())
                    tol=a.P['float64_stack_tolerance'] if name in ['scatter_r2','scatter_blend','query_consistency'] else a.P['float32_control_tolerance']
                    mismatch=int((got.argmax(-1)!=want.argmax(-1)).sum())
                    row={'cell':key,'task':i,'method':name,'max_error':err,'fixed_tolerance':tol,'prediction_mismatches':mismatch,'passed':err<tol and mismatch==0}
                    if name=='score_ensemble_r2' and not row['passed']:row['neighbor_diagnostic']=tied_ensemble(sv,qv,gv,got)
                    rows.append(row)
            bank_checks.append({'cell':key,'score_prediction_accuracy_identity':'passed','scores_sha256':e.sha(e.OUT/(key+'.npz'))})
            print('DIAG_CELL',key,'failures',sum(not x['passed'] for x in rows),flush=True)
        acc=np.mean(accs,0)
        for k,name in enumerate(e.NAMES):
            if k==primary:continue
            got=a.independent_interval(acc[primary]-acc[k],tasks['seeds']);want=summary['source_configurations'][source]['comparisons'][name]
            interval_errors.append(float(abs(got-np.r_[want['delta_pp'],want['ci95_pp']]).max()))
    assert max(interval_errors)<1e-10
    assert e.sha(e.OUT/'audit_failure.json')==failure_hash
    e.check_lock(e.HERE/'evaluation_code_lock.json')
    method_summary={n:{'checks':sum(x['method']==n for x in rows),'failures':sum(x['method']==n and not x['passed'] for x in rows),'max_error':max(x['max_error'] for x in rows if x['method']==n),'prediction_mismatches':sum(x['prediction_mismatches'] for x in rows if x['method']==n)} for n in sorted({x['method'] for x in rows})}
    result={'status':'diagnostic_complete_original_audit_still_failed','frozen_audit_failure_sha256':failure_hash,'fixed_sample_checks':len(rows),'failed_checks':sum(not x['passed'] for x in rows),'method_summary':method_summary,'bank_checks':bank_checks,'independent_intervals':len(interval_errors),'max_interval_error_pp':max(interval_errors),'checks':rows,'seconds':time.time()-start,'scope':'All original 30 cell-task audit samples and 72 paired intervals. Numerical diagnosis only: original failed audit unchanged; no tolerance relaxation, parameter selection, evaluator rerun or complete-validation claim.'}
    e.dump(OUT/'audit_diagnosis.json',result)
    print('DIAGNOSTIC_COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['checks','bank_checks']}),flush=True)

if __name__=='__main__':main()
