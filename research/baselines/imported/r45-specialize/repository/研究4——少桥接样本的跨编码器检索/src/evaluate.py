from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from research_common.records import read_json,write_json,snapshot_run,sha256
from .data import PROJECT,prepare,load_features
from .metrics import ranks,per_query_ap


def evaluate(output,config_path=None):
    config=read_json(config_path or PROJECT/'configs/pilot_v1.json')
    manifest=prepare()
    output=snapshot_run(PROJECT,output,'R5G2 natural CUB compatibility development',config,
        {'manifest_sha256':sha256(PROJECT/'data/manifest_v1.json'),
         'feature_cache_meta':{n:sha256(PROJECT/'cache'/(n+'_all.json')) for n in config['encoders']}})
    features={n:load_features(n) for n in config['encoders']}
    labels={r['id']:r['label_evaluator_only'] for r in manifest['records']}
    gallery_ids=manifest['roles']['gallery'];query_ids=manifest['roles']['query']
    yq=np.array([labels[i] for i in query_ids]);yg=np.array([labels[i] for i in gallery_ids])
    def array(model,ids):return np.stack([features[model][i] for i in ids])
    oracle={};all_results=[]
    for old,new in config['pairs']:
        pair=old+'__'+new
        go,gn=array(old,gallery_ids),array(new,gallery_ids)
        qo,qn=array(old,query_ids),array(new,query_ids)
        old_ap,old_top=per_query_ap(ranks(qo@go.T),yq,yg)
        new_ap,new_top=per_query_ap(ranks(qn@gn.T),yq,yg)
        oracle[pair]={'old_map':float(old_ap.mean()),'new_map':float(new_ap.mean()),
                      'gap_pp':float(100*(new_ap.mean()-old_ap.mean())),
                      'old_top1':float(old_top.mean()),'new_top1':float(new_top.mean())}
        write_json(output/(pair+'_oracle.json'),oracle[pair])
        for condition in config['conditions']:
            pool=manifest['roles']['bridge' if condition=='same_domain' else 'stress_bridge']
            for seed in config['bridge_seeds']:
                permutation=np.random.default_rng(seed).permutation(pool)
                for budget in config['bridge_budgets']:
                    anchors=list(map(int,permutation[:budget]))
                    fit_count=budget-int(budget*config['calibration_fraction'])
                    job=f'{pair}__{condition}__m{budget}__s{seed}'
                    folder=output/'jobs'/job;folder.mkdir(parents=True)
                    input_file=folder/'method_input.npz';rank_file=folder/'rankings.npz'
                    np.savez_compressed(input_file,go=go,qo=qo,qn=qn,ao=array(old,anchors),an=array(new,anchors),fit_count=np.array(fit_count))
                    write_json(folder/'config.json',config)
                    write_json(folder/'input_receipt.json',{'anchors':anchors,'budget':budget,'fit_count':fit_count,
                        'query_ids':query_ids,'gallery_ids':gallery_ids,'input_sha256':sha256(input_file),
                        'labels_in_method_input':False,'gallery_new_in_method_input':False})
                    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONIOENCODING='utf-8')
                    completed=subprocess.run([sys.executable,str(PROJECT/'src/worker.py'),str(input_file),str(folder/'config.json'),
                        str(rank_file),str(PROJECT/'cache'/(new+'_all.npz'))],env=env,capture_output=True,text=True,encoding='utf-8')
                    (folder/'worker.log').write_text(completed.stdout+completed.stderr,encoding='utf-8')
                    if completed.returncode:raise RuntimeError(f'Worker failed: {job}; see worker.log')
                    metrics={}
                    arrays={}
                    with np.load(rank_file,allow_pickle=False) as z:
                        for method in z.files:
                            ap,top=per_query_ap(z[method],yq,yg)
                            metrics[method]={'map':float(ap.mean()),'top1':float(top.mean())}
                            arrays[method+'_ap']=ap;arrays[method+'_top1']=top
                    np.savez_compressed(folder/'evaluation_arrays.npz',**arrays)
                    info=read_json(rank_file.with_suffix('.json'))
                    record={'pair':pair,'condition':condition,'seed':seed,'bridge_budget':budget,'job':job,
                        'metrics':metrics,'diagnostics':info,'rankings_sha256':sha256(rank_file)}
                    write_json(folder/'result.json',record)
                    all_results.append(record)
                    print(json.dumps({'done':len(all_results),'pair':pair,'condition':condition,'m':budget,'seed':seed,
                        'old':metrics['old_old']['map'],'ridge':metrics['ridge']['map'],'candidate':metrics['candidate']['map']}),flush=True)
    rows=[]
    baseline_methods=['old_old','procrustes','ridge','relative','relative_row_centered','local_kernel','residual_all','residual_matched_fit']
    for pair in oracle:
        for condition in config['conditions']:
            for budget in config['bridge_budgets']:
                selected=[r for r in all_results if r['pair']==pair and r['condition']==condition and r['bridge_budget']==budget]
                means={method:float(np.mean([r['metrics'][method]['map'] for r in selected])) for method in selected[0]['metrics']}
                best=max(baseline_methods,key=lambda method:means[method])
                gain=means['candidate']-means[best]
                old=oracle[pair]['old_map'];new=oracle[pair]['new_map']
                row={'pair':pair,'condition':condition,'bridge_budget':budget,'map':means,'best_simple':best,
                    'candidate_minus_best_pp':100*gain,'candidate_minus_old_pp':100*(means['candidate']-old),
                    'oracle_gap_pp':100*(new-old),'oracle_gain_retention':(means['candidate']-old)/(new-old) if new-old>=.02 else None}
                # Paired query resampling with independent resampling of bridge seeds.
                differences=[]
                for r in selected:
                    with np.load(output/'jobs'/r['job']/'evaluation_arrays.npz',allow_pickle=False) as z:
                        differences.append(z['candidate_ap']-z[best+'_ap'])
                differences=np.stack(differences)
                rng=np.random.default_rng(55085)
                boot=[]
                for _ in range(config['bootstrap_replicates']):
                    si=rng.integers(len(selected),size=len(selected));qi=rng.integers(len(yq),size=len(yq))
                    boot.append(float(differences[np.ix_(si,qi)].mean())*100)
                row['candidate_minus_best_ci95_pp']=np.quantile(boot,[.025,.975]).tolist()
                rows.append(row)
    summary={'stage':'natural_CUB_development','jobs':len(all_results),'gallery_images':len(gallery_ids),'query_images':len(query_ids),
        'bridge_total_per_job_includes_calibration':True,'oracle_evaluation_separate':True,'confirmation_used':False,
        'oracle':oracle,'aggregate_rows':rows,'all_worker_read_probes_blocked':all(r['diagnostics']['access_audit']['blocked_probe'] for r in all_results),
        'any_same_domain_primary_gain_gate_passed':any(r['condition']=='same_domain' and r['bridge_budget']>=32 and r['oracle_gap_pp']>=2 and r['candidate_minus_best_pp']>=2
             and r['candidate_minus_best_ci95_pp'][0]>0 and r['oracle_gain_retention']>=.5 for r in rows),
        'paper_go':False,'gate_scope':'Development pilot; stress and external confirmation needed if positive. No unseen-dataset claim.'}
    write_json(output/'summary.json',summary)
    print(json.dumps({'completed':len(all_results),'primary_gain_gate':summary['any_same_domain_primary_gain_gate_passed'],'oracle':oracle}),flush=True)
