"""Audit frozen task outputs and compute all prespecified paired comparisons."""
from pathlib import Path
import json
import numpy as np
import torch
import run_eval as ev
import oslo
HERE, OUT, CFG, PARENT = ev.HERE, ev.OUT, ev.CFG, ev.PARENT
NAMES = CFG['methods']
N = len(CFG['seeds']) * CFG['episodes_per_seed']


def draws(values, indices):
    blocks = values.reshape(5, 100, -1)
    return blocks[np.arange(5)[None, :, None], indices].mean((1, 2)) * 100


def comparisons(values, boot):
    return {name: {'delta_pp': float((values[:, 0]-values[:, j]).mean()*100),
        'paired_ci95_pp': np.quantile(boot[:, 0]-boot[:, j], [.025,.975]).tolist(),
        'seed_deltas_pp': ((values[:,0]-values[:,j]).reshape(5,100).mean(1)*100).tolist(),
        'wins_ties_losses': [int((values[:,0]>values[:,j]).sum()),int((values[:,0]==values[:,j]).sum()),int((values[:,0]<values[:,j]).sum())]}
        for j,name in enumerate(NAMES) if j}


def main():
    torch.set_num_threads(CFG['cpu_threads'])
    signature = ev.lock_check()
    complete = json.loads((OUT/'complete.json').read_text())
    assert complete['status']=='success' and complete['cells']==8 and complete['signature']==signature
    for phase in ['encode','evaluate']:
        manifest=json.loads((OUT/('run_manifest_'+phase+'.json')).read_text())
        assert manifest['config']==CFG and manifest['signature']==signature
    assert json.loads((OUT/'source_validation.json').read_text())['status']=='passed'
    data,old_rows,_=ev.ref.load_data(); fresh,rows=ev.fresh_data()
    assert len(rows)==1804 and len(set(r['rgb'] for r in rows))==1804
    assert not set(r['rgb'] for r in rows)&set(r['rgb'] for side in ['query','gallery'] for r in old_rows[side])
    confirmation=json.loads((PARENT/'outputs/baseline/confirmed_comparator.json').read_text())
    assert ev.sha(confirmation['metric_contract_path'])==confirmation['metric_contract_sha256']
    assert ev.sha(PARENT/'outputs/baseline/validation.json')==confirmation['validation_sha256']
    validated=json.loads((PARENT/'outputs/baseline/validation.json').read_text());assert validated['status']=='passed'
    cells={};values_all={};boots={};hashes={};replays={};task_hashes={}
    rng=np.random.default_rng(CFG['bootstrap']['seed'])
    expected={s+'_'+g+'_k%d'%k for s in ['canonical','fresh'] for g in CFG['gallery_conditions'] for k in CFG['shots']}
    assert set(p.stem for p in (OUT/'cells').glob('*.npz'))==expected
    for scope,X,ident in [('canonical',data['query'],old_rows['query']),('fresh',fresh,rows)]:
        labels=np.array([r['label'] for r in ident])
        for shot in CFG['shots']:
            t=ev.tasks(scope,ident,shot);si,qi,cs,seeds=[t[k] for k in ['support_indices','query_indices','class_ids','seeds']]
            task_hashes[scope+'_k%d'%shot]=ev.sha(OUT/(scope+'_tasks_k%d.npz'%shot))
            expected_seeds=CFG['seeds' if scope=='canonical' else 'fresh_seeds']
            assert si.shape==(N,5,shot) and qi.shape==(N,5,15) and cs.shape==(N,5)
            assert np.array_equal(seeds,np.repeat(expected_seeds,100))
            assert np.all(labels[si]==cs[:,:,None]) and np.all(labels[qi]==cs[:,:,None])
            assert all(len(set(a.ravel())|set(b.ravel()))==5*(shot+15) for a,b in zip(si,qi))
            assert all(len(set(c))==5 for c in cs)
            indices=rng.integers(100,size=(CFG['bootstrap']['replicates'],5,100))
            constant=draws(np.tile([.25,.5],(N,1)),indices);assert np.allclose(constant[:,1]-constant[:,0],25)
            for gallery in CFG['gallery_conditions']:
                cell=scope+'_'+gallery+'_k%d'%shot;path=OUT/'cells'/(cell+'.npz');hashes[cell]=ev.sha(path)
                G=data['gallery' if gallery=='pets' else 'dtd'];assert len(G)==CFG['gallery_size']
                with np.load(path) as p:
                    assert p['names'].tolist()==NAMES and str(p['signature'])==signature
                    for k,v in t.items():assert np.array_equal(p[k],v)
                    scores=p['scores'];pred=p['predictions'];acc=p['accuracy'];y=np.repeat(np.arange(5),15)
                    assert scores.shape==(len(NAMES),N,75,5) and np.isfinite(scores).all()
                    assert np.array_equal(pred,scores.argmax(-1)) and np.array_equal(p['yq'],y)
                    correct=pred==y;assert np.array_equal(acc,correct.mean(-1))
                    for method in NAMES[:3]:
                        for stat in ['effective_mass','inlier_mean','prototype_movement']:assert np.isfinite(p[method+'_'+stat]).all()
                    diagnostics={m:{k:float(p[m+'_'+k].mean()) for k in ['effective_mass','inlier_mean','prototype_movement']} for m in NAMES[:3]}
                    if scope=='canonical':
                        parent_cell='pets_'+gallery+'_k%d'%shot;basepath=PARENT/'outputs/baseline'/(parent_cell+'.npz')
                        assert ev.sha(basepath)==validated['output_sha256'][parent_cell]
                        with np.load(basepath) as base:
                            for j,name in enumerate(base['names']):assert np.array_equal(scores[NAMES.index(str(name))],base['scores'][j])
                    replay={}
                    for i in [0,249,499]:
                        S=X[si[i:i+1]];Q=X[qi[i:i+1].reshape(1,75)]
                        for m,steps,enabled in [('OSLO_G',2,True),('closed_set',2,False),('zero_update',0,True)]:
                            state,_=oslo.fit(S,G,steps=steps,lambda_s=CFG['oslo']['lambda_s'],lambda_z=CFG['oslo']['lambda_z'],use_inlier=enabled)
                            actual=oslo.predict(state,Q);error=float(abs(actual.numpy()[0]-scores[NAMES.index(m),i]).max())
                            assert error<3e-6,(cell,i,m,error)
                            assert torch.allclose(oslo.predict(state,Q[:,:1]),actual[:,:1],atol=2e-6,rtol=0)
                            replay[str(i)+'_'+m]=error
                        if scope=='fresh':
                            checked={'r2':ev.ref.ref.r2_scores(S,Q,G).numpy()[0],
                                     'CS_l2':ev.ref.geo.head(ev.ref.geo.prepare(S,Q,G,'support'),.1).numpy()[0]}
                            for C in [1,10]:checked['support_logistic_C%d'%C]=ev.ref.logistic(S.flatten(0,2).numpy(),Q[0].numpy(),C)
                            for m,actual in checked.items():
                                error=float(abs(actual-scores[NAMES.index(m),i]).max());assert error<3e-6,(cell,i,m,error)
                                replay[str(i)+'_'+m]=error
                    values=acc.T;boot=draws(values,indices);values_all[cell]=values;boots[cell]=boot;replays[cell]=replay
                    breed_acc=correct.reshape(len(NAMES),N,5,15).mean(-1);per_breed={}
                    for breed in np.unique(cs):
                        mask=cs==breed;means=breed_acc[:,mask].mean(1)*100
                        per_breed[str(int(breed))]={'episode_occurrences':int(mask.sum()),'accuracy_pct':dict(zip(NAMES,means.tolist()))}
                    cells[cell]={'task_count':N,'accuracy_pct':dict(zip(NAMES,(values.mean(0)*100).tolist())),
                        'comparisons':comparisons(values,boot),'per_breed':per_breed,'diagnostics':diagnostics,
                        'per_seed_accuracy_pct':{str(s):dict(zip(NAMES,(values.reshape(5,100,-1)[j].mean(0)*100).tolist())) for j,s in enumerate(expected_seeds)}}
                print('AUDITED',cell,cells[cell]['accuracy_pct'],flush=True)
    metrics={};macro={}
    for scope in ['canonical','fresh']:
        prefix='pets_' if scope=='canonical' else 'pets_fresh_'
        for shot in CFG['shots']:
            for g in CFG['gallery_conditions']:
                key=prefix+('matched' if g=='pets' else 'mismatched')+'_%dshot_accuracy'%shot
                metrics[key]=cells[scope+'_'+g+'_k%d'%shot]['accuracy_pct']['OSLO_G']
        metrics[prefix+'macro_1shot_accuracy']=np.mean([metrics[prefix+condition+'_1shot_accuracy'] for condition in ['matched','mismatched']]).item()
        macro[scope]=comparisons(np.mean([values_all[scope+'_'+g+'_k1'] for g in CFG['gallery_conditions']],axis=0),np.mean([boots[scope+'_'+g+'_k1'] for g in CFG['gallery_conditions']],axis=0))
    gates={}
    for scope in ['canonical','fresh']:
        match=cells[scope+'_pets_k1']['comparisons']
        gain=all(match[n]['delta_pp']>=CFG['gates']['matched_gain_pp'] and match[n]['paired_ci95_pp'][0]>0 for n in ['r2','CS_l2'])
        heads=all(match[n]['paired_ci95_pp'][0]>0 for n in ['support_logistic_C1','support_logistic_C10'])
        violations=[{'cell':cell,'control':n,**cells[cell]['comparisons'][n]} for cell in cells if cell.startswith(scope) and cell!=scope+'_pets_k1' for n in NAMES[3:] if cells[cell]['comparisons'][n]['paired_ci95_pp'][0]<CFG['gates']['other_cells_ci_lower']]
        gates[scope]={'matched_strong_gate_passed':gain,'support_head_superiority_passed':heads,'other_cells_guard_passed':not violations,'other_cells_guard_violations':violations,
                      'inlier_mechanism_attribution_passed':match['closed_set']['paired_ci95_pp'][0]>0}
    passed=all(g[k] for g in gates.values() for k in ['matched_strong_gate_passed','support_head_superiority_passed','other_cells_guard_passed'])
    report={'metrics_summary':metrics,'cells':cells,'macro_1shot_comparisons':macro,'gates':gates,'primary_gate_passed':passed,'bootstrap':CFG['bootstrap'],
        'boundary':'Fixed source-default OSLO gallery transfer; no query adaptation. Canonical outcomes were exposed before direction selection; fresh images were prospectively encoded after freezing, but share domain/classes. Intervals condition on fixed pools and seeds. No independent-domain, universal-safety, algorithmic-novelty, or SOTA claim. Exact RGB overlap excluded; near-duplicate and pretraining overlap remain unmeasured.'}
    validation={'status':'passed','cells':8,'tasks':8*N,'signature':signature,'full_task_identity_prediction_accuracy_check':True,'canonical_comparators_bitwise_equal':True,
        'image_disjoint_confirmation_count':1804,'score_replays':replays,'output_sha256':hashes,'task_sha256':task_hashes,'audit_source_sha256':ev.sha(HERE/'analyze_results.py'),
        'confirmed_baseline_id':confirmation['baseline_id'],'confirmed_metric_contract_sha256':confirmation['metric_contract_sha256']}
    ev.dump(OUT/'analysis.json',report);ev.dump(OUT/'validation.json',validation);ev.dump(OUT/'metrics_summary.json',metrics)
    lines=['# Gallery-only OSLO source-default qualification','',report['boundary'],'','Primary qualification gate passed: '+str(passed),'',
        '| Pool / gallery / shot | OSLO_G | Closed set | Zero update | R2 | CS_l2 | Support C1 | Support C10 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for cell,row in cells.items():lines.append('| '+cell+' | '+' | '.join('%.4f'%row['accuracy_pct'][n] for n in NAMES)+' |')
    lines+=['','| Cell | Comparator | Difference (pp) | Paired 95% interval (pp) |','|---|---|---:|---|']
    for cell,row in cells.items():
        for name,v in row['comparisons'].items():lines.append('| %s | %s | %+.4f | [%+.4f, %+.4f] |'%(cell,name,v['delta_pp'],*v['paired_ci95_pp']))
    lines+=['','Full gate decisions, per-seed/per-breed results and fitting diagnostics: outputs/analysis.json.',
        'Complete eight-cell task, identity, prediction, frozen-baseline and score-replay audit: outputs/validation.json.',
        'Source parity and query permutation/split/insertion checks: outputs/source_validation.json.',
        'Protocol and inference source hashes were frozen before fresh feature extraction: protocol.json and locked_sources.json.',
        'Paired bootstrap: 5,000 resamples within five task-seed strata; tasks across gallery conditions use shared resamples. No independent-domain uncertainty estimate.',
        'Known OSLO source: Boudiaf et al., CVPR 2023, https://arxiv.org/abs/2301.08390; author code commit 9240a4e630874c50459e069db609db9163a81c03. Gallery-only fitting changes the task from source transductive query adaptation.',
        'All seven methods use identical frozen CLIP/DINO features. Existing encoders and images were reused; zero new downloads and no remote compute.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('FINAL',json.dumps({'metrics_summary':metrics,'gates':gates,'primary_gate_passed':passed}),flush=True)


if __name__=='__main__': main()
