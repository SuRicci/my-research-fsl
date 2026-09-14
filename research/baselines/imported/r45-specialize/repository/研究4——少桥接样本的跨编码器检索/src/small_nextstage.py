"""Run a bounded development-only study, preserving old runs and held-out scores."""
from pathlib import Path
import json,os,sys,subprocess,shutil
import numpy as np
from research_common.records import read_json,write_json,snapshot_run,sha256
from .metrics import per_query_ap
from .nextstage import CANDIDATES

ROOT=Path(__file__).resolve().parents[1]
def main(output=None):
    source=ROOT/'runs/optimization_dev_v1';manifest=read_json(ROOT/'data/manifest_v1.json')
    config={'stage':'bounded_development_only','parent':'optimization_dev_v1','candidate_count':len(CANDIDATES),
        'candidate_menu':CANDIDATES,'regularization':[.001,.01,.1,1.,10.],'crossfit_folds':4,
        'rank_loss':'bridge-teacher pair inversions weighted by absolute teacher gap and top-quarter membership; MSE tie-break 1e-6',
        'primary':'equal-weight all 8 development cells','promotion_minimum_pp':.5,'maximum_cell_regression_pp':1.,
        'no_new_holdout_evaluation':True}
    out=snapshot_run(ROOT,output or ROOT/'runs/small_nextstage_v1','bounded R5 scale and ranking development study',config,
        {'parent_summary_sha256':sha256(source/'summary.json'),'manifest_sha256':sha256(ROOT/'data/manifest_v1.json')})
    labels={r['id']:r['label_evaluator_only'] for r in manifest['records']}
    qy=np.array([labels[i] for i in manifest['roles']['query']]);gy=np.array([labels[i] for i in manifest['roles']['gallery']])
    results=[];parity=0
    for i,folder in enumerate(sorted((source/'jobs').iterdir())):
        r=read_json(folder/'result.json');dest=out/'jobs'/folder.name;dest.mkdir(parents=True)
        shutil.copy2(folder/'input.npz',dest/'input.npz');write_json(dest/'config.json',config)
        p=subprocess.run([sys.executable,str(ROOT/'src/nextstage_worker.py'),str(dest/'input.npz'),str(dest/'config.json'),str(dest/'rankings.npz'),str(ROOT/'cache/dino_s14_all.npz')],
            capture_output=True,text=True,encoding='utf-8',env=dict(os.environ,OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONIOENCODING='utf-8'))
        (dest/'worker.log').write_text(p.stdout+p.stderr,encoding='utf-8')
        if p.returncode:raise RuntimeError(f'Worker failed: {folder.name}')
        arrays={}
        with np.load(dest/'rankings.npz',allow_pickle=False) as z,np.load(folder/'rankings.npz',allow_pickle=False) as previous:
            parity+=int(np.count_nonzero(z['v0_press']!=previous['candidate_rbf_0.5_std']))
            for method in z.files:arrays[method]=per_query_ap(z[method],qy,gy)[0]
        r['map']={m:float(v.mean()) for m,v in arrays.items()}
        r['input_sha256']=sha256(dest/'input.npz');r['rankings_sha256']=sha256(dest/'rankings.npz')
        np.savez_compressed(dest/'ap.npz',**arrays);write_json(dest/'result.json',r);results.append(r)
        if (i+1)%5==0:print(json.dumps({'jobs':i+1,'total':40,'v0_rank_position_changes':parity}),flush=True)
    assert parity==0,'Refactoring changed frozen V0 rankings'
    rows=[]
    for row in read_json(source/'summary.json')['rows']:
        rs=[r for r in results if all(r[k]==row[k] for k in ['pair','condition','budget'])]
        values={m:float(np.mean([r['map'][m] for r in rs])) for m in rs[0]['map']}
        rows.append({k:row[k] for k in ['pair','condition','budget']}|{'map':values})
    macro={m:float(np.mean([r['map'][m] for r in rows])) for m in rows[0]['map']}
    candidate=max(CANDIDATES,key=macro.get);gain=100*(macro[candidate]-macro['v0_press'])
    worst=min(100*(r['map'][candidate]-r['map']['v0_press']) for r in rows)
    promoted=gain>=.5 and worst>=-1.
    selection={'frozen_for_external_comparison':True,'candidate':candidate,'specification':list(CANDIDATES[candidate]),
        'primary_method':candidate if promoted else 'v0_press','promoted_on_development':promoted,'macro_gain_vs_v0_pp':gain,
        'worst_cell_gain_vs_v0_pp':worst,'not_independent_confirmation':True,
        'selection_rule':config,'scorer_sha256':sha256(ROOT/'src/nextstage.py')}
    write_json(out/'summary.json',{'rows':rows,'macro':macro,'selection':selection,'v0_rank_position_changes':parity,'jobs':40})
    write_json(out/'selection.json',selection)
    print(json.dumps(selection),flush=True)

if __name__=='__main__':main()
