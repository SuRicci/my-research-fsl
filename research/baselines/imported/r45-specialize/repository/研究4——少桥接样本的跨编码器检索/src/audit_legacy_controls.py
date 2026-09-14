"""Replay the original registered baselines on the corrected frozen inputs.

Added comparator audit after opening; candidate/parameters remain frozen.
"""
import os,sys,subprocess,json
import numpy as np
from research_common.records import read_json,write_json,snapshot_run,sha256
from .data import PROJECT
from .metrics import per_query_ap

def main():
    source=PROJECT/'runs/optimization_holdout_deduplicated_v1'
    config=read_json(PROJECT/'configs/pilot_v2_centered.json')
    names=['old_old','procrustes','ridge','relative','relative_row_centered','local_kernel','residual_all']
    run=snapshot_run(PROJECT,PROJECT/'runs/optimization_legacy_controls_v1','Original registered comparator audit on repaired holdout',
        {'baseline_config':config,'methods':names,'candidate_not_retuned':True,'post_open_comparator_audit':True},
        {'repaired_run_receipt_sha256':sha256(source/'receipt.json')})
    m=read_json(PROJECT/'data/optimization_holdout_v1.json');labels={r['id']:r['label_evaluator_only'] for r in m['records']}
    qy=np.array([labels[i] for i in m['roles']['query']]);gy=np.array([labels[i] for i in m['roles']['gallery']])
    results=[]
    for index,folder in enumerate(sorted((source/'jobs').iterdir())):
        result=read_json(folder/'result.json');dest=run/'jobs'/folder.name;dest.mkdir(parents=True)
        with np.load(folder/'input.npz',allow_pickle=False) as z:inp={k:z[k] for k in z.files}
        inp['fit_count']=np.array(int(len(inp['ao'])*.75))
        np.savez_compressed(dest/'input.npz',**inp);write_json(dest/'config.json',config)
        p=subprocess.run([sys.executable,str(PROJECT/'src/worker.py'),str(dest/'input.npz'),str(dest/'config.json'),str(dest/'rankings.npz'),
            str(PROJECT/'cache/optimization_holdout/dino_s14_all.npz')],capture_output=True,text=True,encoding='utf-8',
            env=dict(os.environ,OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONIOENCODING='utf-8'))
        (dest/'worker.log').write_text(p.stdout+p.stderr,encoding='utf-8')
        if p.returncode:raise RuntimeError(folder.name)
        arrays={}
        with np.load(dest/'rankings.npz',allow_pickle=False) as z:
            for name in names:arrays['legacy_'+name]=per_query_ap(z[name],qy,gy)[0]
        np.savez_compressed(dest/'ap.npz',**arrays)
        result['map']={k:float(v.mean()) for k,v in arrays.items()}
        result.update(input_sha256=sha256(dest/'input.npz'),rankings_sha256=sha256(dest/'rankings.npz'))
        write_json(dest/'result.json',result);results.append(result)
        if (index+1)%10==0:print(json.dumps({'legacy_jobs':index+1,'total':40}),flush=True)
    rows=[]
    for r in read_json(source/'summary.json')['rows']:
        rs=[a for a in results if all(a[k]==r[k] for k in ['pair','condition','budget'])]
        values={name:float(np.mean([a['map'][name] for a in rs])) for name in rs[0]['map']}
        rows.append({k:r[k] for k in ['pair','condition','budget']} | {'map':values})
    write_json(run/'summary.json',{'rows':rows,'post_open_comparator_audit':True,'candidate_not_retuned':True})
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
