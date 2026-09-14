from pathlib import Path
import json,sys,subprocess,os
import numpy as np
from research_common.records import read_json,write_json,snapshot_run,sha256
from .data import PROJECT,prepare,load_features
from .metrics import per_query_ap,ranks

def run_grid(output,holdout=False,deduplicate_bridge=False):
    config=read_json(PROJECT/'configs/optimization_v1.json')
    config['worker_family']='optimization_menu'
    if holdout:
        manifest=read_json(PROJECT/'data/optimization_holdout_v1.json')
        selection=read_json(PROJECT/'runs/optimization_dev_v1/selection.json')
        if not selection['frozen']:raise RuntimeError('Selection not frozen')
        if selection['holdout_manifest_sha256']!=sha256(PROJECT/'data/optimization_holdout_v1.json'):raise ValueError('Holdout seal changed')
        config['selected_methods']=selection['holdout_methods']
        feature_dir=PROJECT/'cache/optimization_holdout'
    else:
        manifest=prepare();feature_dir=PROJECT/'cache'
    excluded=[]
    if deduplicate_bridge:
        assert holdout
        repair=read_json(PROJECT/'data/holdout_rgb_audit.json')
        assert repair['manifest_sha256']==sha256(PROJECT/'data/optimization_holdout_v1.json')
        assert repair['selection_sha256']==sha256(PROJECT/'runs/optimization_dev_v1/selection.json')
        excluded=repair['bridge_ids_to_exclude']
        config['data_repair']={'exact_rgb_duplicate_bridge_ids':excluded,'rule':'Keep gallery and queries fixed; skip duplicate bridge IDs in the original seeded permutation, then take the same budget','algorithm_retuned':False}
    output=snapshot_run(PROJECT,output,'R5 optimization holdout' if holdout else 'R5 optimization development menu',config,
        {'manifest_sha256':sha256(PROJECT/('data/optimization_holdout_v1.json' if holdout else 'data/manifest_v1.json'))})
    if holdout:write_json(output/'opening.json',{'selection_sha256':sha256(PROJECT/'runs/optimization_dev_v1/selection.json'),'no_post_open_tuning':True})
    features={}
    for model in ['clip_b32','clip_b16','dino_s14']:
        metadata=read_json(feature_dir/(model+'_all.json'))
        if metadata['npz_sha256']!=sha256(feature_dir/(model+'_all.npz')):raise ValueError('Feature cache corrupted')
        with np.load(feature_dir/(model+'_all.npz'),allow_pickle=False) as z:features[model]={int(i):v for i,v in zip(z['ids'],z['embeddings'])}
    labels={r['id']:r['label_evaluator_only'] for r in manifest['records']}
    gids=manifest['roles']['gallery'];qids=manifest['roles']['query'];qy=np.array([labels[i] for i in qids]);gy=np.array([labels[i] for i in gids])
    arr=lambda model,ids:np.stack([features[model][i] for i in ids])
    records=[];oracle={}
    for old,new in config['primary_pairs']:
        pair=old+'__'+new;go,qo,qn=arr(old,gids),arr(old,qids),arr(new,qids)
        oracle[pair]={'old_map':float(per_query_ap(ranks(qo@go.T),qy,gy)[0].mean()),'new_map':float(per_query_ap(ranks(qn@arr(new,gids).T),qy,gy)[0].mean())}
        for condition in config['conditions']:
            pool=manifest['roles']['bridge' if condition=='same_domain' else 'stress_bridge']
            for seed in config['bridge_seeds']:
                shuffled=np.random.default_rng(seed).permutation(pool)
                for budget in config['budgets']:
                    anchors=[int(i) for i in shuffled if int(i) not in excluded][:budget];job=f'{pair}__{condition}__m{budget}__s{seed}'
                    folder=output/'jobs'/job;folder.mkdir(parents=True)
                    input_file=folder/'input.npz';rank_file=folder/'rankings.npz'
                    np.savez_compressed(input_file,go=go,qo=qo,qn=qn,ao=arr(old,anchors),an=arr(new,anchors),fit_count=np.array(budget))
                    write_json(folder/'config.json',config)
                    write_json(folder/'input_receipt.json',{'anchors':anchors,'budget':budget,'input_sha256':sha256(input_file),'holdout':holdout,'labels_in_input':False,'new_gallery_in_input':False,'excluded_exact_rgb_bridge_ids':excluded})
                    result=subprocess.run([sys.executable,str(PROJECT/'src/worker.py'),str(input_file),str(folder/'config.json'),str(rank_file),str(feature_dir/(new+'_all.npz'))],
                        env=dict(os.environ,OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONIOENCODING='utf-8'),capture_output=True,text=True,encoding='utf-8')
                    (folder/'worker.log').write_text(result.stdout+result.stderr,encoding='utf-8')
                    if result.returncode:raise RuntimeError(job+' failed; see worker.log')
                    values={};arrays={}
                    with np.load(rank_file,allow_pickle=False) as z:
                        names=z.files if not holdout else selection['holdout_methods']
                        for method in names:
                            ap,top=per_query_ap(z[method],qy,gy);values[method]=float(ap.mean());arrays[method]=ap
                    np.savez_compressed(folder/'ap.npz',**arrays)
                    record={'pair':pair,'condition':condition,'budget':budget,'seed':seed,'job':job,'map':values}
                    write_json(folder/'result.json',record);records.append(record)
                    if len(records)%5==0:print(json.dumps({'done':len(records),'total':40,'last':job,'best':max(values,key=values.get),'map':max(values.values())}),flush=True)
    rows=[]
    for pair in oracle:
        for condition in config['conditions']:
            for budget in config['budgets']:
                rs=[r for r in records if r['pair']==pair and r['condition']==condition and r['budget']==budget]
                means={method:float(np.mean([r['map'][method] for r in rs])) for method in rs[0]['map']}
                rows.append({'pair':pair,'condition':condition,'budget':budget,'map':means})
    summary={'stage':'new_class_same_dataset_holdout' if holdout else 'seen_development_optimization','records':len(records),'rows':rows,'oracle':oracle,'all_data_budget_matched':True}
    write_json(output/'summary.json',summary)
    if not holdout:
        primary=[r for r in rows if r['condition']=='same_domain']
        macro={m:float(np.mean([r['map'][m] for r in primary])) for m in primary[0]['map']}
        candidate=max([m for m in macro if m.startswith('candidate_')],key=macro.get)
        control=max([m for m in macro if not m.startswith('candidate_')],key=macro.get)
        selection={'frozen':True,'candidate':candidate,'strongest_tuned_control':control,
            'holdout_methods':list(dict.fromkeys([candidate,control,'old_old','control_local_raw_k8_t0.05'])),
            'selection_rule':'Equal-weight same-domain mean over 2 encoder pairs and 2 bridge budgets; independent heldout untouched',
            'development_macro':macro,'holdout_manifest_sha256':sha256(PROJECT/'data/optimization_holdout_v1.json')}
        write_json(output/'selection.json',selection)
        print(json.dumps({'selected':candidate,'control':control,'dev_gain_pp':100*(macro[candidate]-macro[control])}),flush=True)
    else:print(json.dumps({'holdout_complete':True,'selected':selection['candidate']}),flush=True)

if __name__=='__main__':
    run_grid(PROJECT/'runs/optimization_dev_v1')
