"""Post-freeze evaluation audit and paired class/bridge-seed uncertainty.

This module never chooses a method or changes scoring parameters.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from PIL import Image
from research_common.records import read_json,write_json,sha256
from .data import PROJECT


def main():
    run=PROJECT/'runs/optimization_holdout_deduplicated_v1'
    manifest=read_json(PROJECT/'data/optimization_holdout_v1.json')
    development=read_json(PROJECT/'data/manifest_v1.json')
    selected=read_json(PROJECT/'runs/optimization_dev_v1/selection.json')
    assert sha256(PROJECT/'data/optimization_holdout_v1.json')==selected['holdout_manifest_sha256']
    assert read_json(run/'opening.json')['selection_sha256']==sha256(PROJECT/'runs/optimization_dev_v1/selection.json')
    labels={r['id']:r['label_evaluator_only'] for r in manifest['records']}
    qy=np.array([labels[i] for i in manifest['roles']['query']]);gy=np.array([labels[i] for i in manifest['roles']['gallery']])
    classes=np.unique(qy);cand=selected['candidate'];methods=selected['holdout_methods'];controls=[m for m in methods if m!=cand]
    audit={'model_rerun':False,'query_classes':len(classes),'bridge_seeds':5,'jobs_reconciled':0,'snapshot_files_checked':0,'all_checks_passed':False}
    assert not set(development['target_classes']+development['stress_classes']) & set(manifest['target_classes']+manifest['stress_classes'])
    assert not {r['id'] for r in development['records']} & {r['id'] for r in manifest['records']}
    rgb={};prior_rgb={r['rgb_sha256'] for r in development['records']}
    repaired=read_json(PROJECT/'data/holdout_rgb_audit.json')
    excluded=set(repaired['bridge_ids_to_exclude'])
    for r in manifest['records']:
        file=Path(manifest['source_directory'])/'images'/r['relative_path']
        assert sha256(file)==r['file_sha256']
        with Image.open(file) as im:
            arr=np.asarray(im.convert('RGB'))
            # Match the development identity protocol: shape followed by decoded RGB bytes.
            digest=hashlib.sha256(np.asarray(arr.shape,np.int64).tobytes()+arr.tobytes()).hexdigest()
        if r['id'] not in excluded:
            assert digest not in rgb and digest not in prior_rgb
            rgb[digest]=r['id']
    audit['holdout_images_hashed']=len(manifest['records']);audit['exact_rgb_duplicates_after_exclusion']=0
    audit['excluded_bridge_ids']=sorted(excluded);audit['data_repair']=repaired
    for name in ['optimization_dev_v1','optimization_holdout_v1','optimization_holdout_deduplicated_v1','optimization_legacy_controls_v1']:
        receipt=read_json(PROJECT/'runs'/name/'receipt.json')
        for p,digest in receipt['sources'].items():
            assert sha256(PROJECT/'runs'/name/'code_snapshot'/p)==digest
            audit['snapshot_files_checked']+=1
    jobs=[];arrays={};records=[]
    for folder in sorted((run/'jobs').iterdir()):
        r=read_json(folder/'result.json');jobs.append(r)
        info=read_json(folder/'rankings.json');inp=read_json(folder/'input_receipt.json')
        assert sha256(folder/'input.npz')==inp['input_sha256']
        assert len(inp['anchors'])==len(set(inp['anchors']))==r['budget']
        assert not set(inp['anchors']) & excluded
        assert set(inp['anchors']) <= set(manifest['roles']['bridge' if r['condition']=='same_domain' else 'stress_bridge'])
        assert info['access_audit']['blocked_probe'] and len(info['access_audit']['unexpected_data_reads'])==1
        assert info['gallery_image_reads']==info['gallery_new_embedding_reads']==info['gallery_new_calls']==0
        with np.load(folder/'input.npz',allow_pickle=False) as z:
            assert set(z.files)=={'go','qo','qn','ao','an','fit_count'}
            assert len(z['ao'])==len(z['an'])==int(z['fit_count'])==r['budget']
        with np.load(folder/'rankings.npz',allow_pickle=False) as z,np.load(folder/'ap.npz',allow_pickle=False) as ap:
            assert set(z.files)==set(methods)
            arrays[r['job']]={}
            for m in methods:
                assert np.all(np.sort(z[m],axis=1)==np.arange(len(gy)))
                relevant=gy[z[m]]==qy[:,None]
                actual=(np.cumsum(relevant,1)/np.arange(1,len(gy)+1)*relevant).sum(1)/relevant.sum(1)
                np.testing.assert_allclose(actual,ap[m],atol=1e-12)
                assert abs(actual.mean()-r['map'][m])<1e-12
                arrays[r['job']][m]=np.array([actual[qy==c].mean() for c in classes])
        audit['jobs_reconciled']+=1
        records.append({'job':r['job'],'input_sha256':sha256(folder/'input.npz'),'rankings_sha256':sha256(folder/'rankings.npz'),'ap_sha256':sha256(folder/'ap.npz')})
    legacy_names=['legacy_procrustes','legacy_ridge','legacy_relative','legacy_relative_row_centered','legacy_local_kernel']
    for r in jobs:
        f=PROJECT/'runs/optimization_legacy_controls_v1/jobs'/r['job'];legacy=read_json(f/'result.json')
        assert sha256(f/'input.npz')==legacy['input_sha256'] and sha256(f/'rankings.npz')==legacy['rankings_sha256']
        info=read_json(f/'rankings.json')
        assert info['access_audit']['blocked_probe'] and info['gallery_new_embedding_reads']==0
        with np.load(f/'input.npz',allow_pickle=False) as li,np.load(run/'jobs'/r['job']/'input.npz',allow_pickle=False) as original:
            for k in ['go','qo','qn','ao','an']:np.testing.assert_array_equal(li[k],original[k])
        with np.load(f/'ap.npz',allow_pickle=False) as ap,np.load(f/'rankings.npz',allow_pickle=False) as z:
            for m in legacy_names:
                relevant=gy[z[m.removeprefix('legacy_')]]==qy[:,None]
                actual=(np.cumsum(relevant,1)/np.arange(1,len(gy)+1)*relevant).sum(1)/relevant.sum(1)
                np.testing.assert_allclose(actual,ap[m],atol=1e-12)
                arrays[r['job']][m]=np.array([actual[qy==c].mean() for c in classes])
            np.testing.assert_allclose(ap['legacy_residual_all'].mean(),r['map']['control_local_raw_k8_t0.05'],atol=1e-12)
    methods=methods+legacy_names;controls=controls+legacy_names
    summary=read_json(run/'summary.json');rows=[];rng=np.random.default_rng(50870)
    # Resample shared bridge seeds and query classes, preserving all within-class queries.
    seed_draws=rng.integers(5,size=(10000,5));class_draws=rng.integers(len(classes),size=(10000,len(classes)))
    for row in summary['rows']:
        rs=sorted([r for r in jobs if all(r[k]==row[k] for k in ['pair','condition','budget'])],key=lambda r:r['seed'])
        assert len(rs)==5
        mats={m:np.stack([arrays[r['job']][m] for r in rs]) for m in methods}
        for m in legacy_names:row['map'][m]=float(mats[m].mean())
        for m in methods:assert abs(mats[m].mean()-row['map'][m])<1e-12
        boots={m:mats[m][seed_draws[:,:,None],class_draws[:,None,:]].mean((1,2)) for m in methods}
        comparisons={}
        for m in controls:
            gap=100*(mats[cand]-mats[m]).mean();b=100*(boots[cand]-boots[m])
            comparisons[m]={'gain_pp':float(gap),'ci95_pp':np.quantile(b,[.025,.975]).tolist(),'seed_gains_pp':(100*(mats[cand]-mats[m]).mean(1)).tolist()}
        strongest=max(controls,key=row['map'].get)
        conservative=100*(boots[cand]-np.max(np.stack([boots[m] for m in controls]),axis=0))
        oracle=summary['oracle'][row['pair']]
        retention=(row['map'][cand]-row['map']['old_old'])/(oracle['new_map']-row['map']['old_old'])
        rows.append(dict(row,comparisons=comparisons,strongest_frozen_control=strongest,
            gain_over_best_frozen_control_pp=100*(row['map'][cand]-row['map'][strongest]),
            best_control_resampled_ci95_pp=np.quantile(conservative,[.025,.975]).tolist(),
            oracle_gain_retention=float(retention)))
    frozen_source=read_json(PROJECT/'runs/optimization_dev_v1/source_freeze_verification.json')
    for name,digest in frozen_source['method_source_sha256'].items():assert sha256(PROJECT/'src'/name)==digest
    audit.update(all_checks_passed=True,legacy_comparator_jobs_reconciled=40,post_open_legacy_comparator_audit=True,source_freeze_parity=frozen_source,
        no_post_holdout_tuning=True,novel_class_holdout=True,external_dataset=False,
        isolation='Feature-only process and Python audit; not an adversarial OS sandbox')
    destination=PROJECT/'reports/evidence'
    write_json(destination/'optimization_integrity.json',audit,overwrite=True)
    write_json(destination/'optimization_artifact_hashes.json',records,overwrite=True)
    write_json(destination/'optimization_holdout_statistics.json',{'rows':rows,'bootstrap':'10000 paired bridge-seed and query-class cluster draws; 5 queries stay together per class; fixed 600-image gallery','candidate':cand,'no_multiplicity_adjustment':True},overwrite=True)
    for r in rows:print(json.dumps({k:r[k] for k in ['pair','condition','budget','gain_over_best_frozen_control_pp','best_control_resampled_ci95_pp','oracle_gain_retention']}),flush=True)

if __name__=='__main__':main()
