"""Post-freeze analysis only. No tuning or model reruns."""
import json
from pathlib import Path
import numpy as np
import run_e12 as r
HERE=Path(__file__).resolve().parent

def main():
    rows=json.loads((HERE/'e12'/'summary.json').read_text(encoding='utf-8'))['rows']
    selections=json.loads((HERE/'e12'/'selection.json').read_text(encoding='utf-8'))
    report={'evidence_scope':'16 finite-pool class-disjoint within-round retest cells, 600 episodes each, 3 seeds; all benchmark datasets historically viewed','pooled':{}}
    for shot in [1,5]:
        deltas=[];weighted_delta=[];rawdelta=[];forgates=[]
        for path in sorted((HERE/'e12').glob(f'*_k{shot}_retest.npz')):
            d=np.load(path);names=d['config_names'].tolist();pred=d['predictions'];acc=(pred==d['yq']).mean(-1)
            b=next(b for b in r.BACKBONES if b in path.name);sel=selections[b+'_'+str(shot)]
            wg=acc[names.index(sel['weighted_gate']['name'])];um=acc[names.index(sel['uniform_mix']['name'])]
            wm=acc[names.index(sel['weighted_mix']['name'])];ug=acc[names.index(sel['uniform_gate']['name'])]
            deltas.append(wg-um);weighted_delta.append(wm-um);forgates.append(wg-ug);rawdelta.append(wg-acc[names.index('raw')])
        # Preserve shared episode identity for DTD across its two galleries: same bootstrap row across all cells.
        # This is a conditional, equal-cell macro estimate on these datasets, not independent-dataset inference.
        report['pooled'][str(shot)]={}
        for label,ds in [('weighted_gate_minus_uniform_mix',deltas),('weighted_mix_minus_uniform_mix',weighted_delta),('weighted_gate_minus_uniform_gate',forgates),('weighted_gate_minus_raw',rawdelta)]:
            d=np.stack(ds).mean(0)
            report['pooled'][str(shot)][label]={'mean':float(d.mean()),'paired_ci95':r.bootstrap(d)}
    shared=[d for d in rows if d['shot']==1 and (d['cell'].startswith('cifar_fs_') or d['cell'].startswith('dtd_dtd_'))]
    stress=[d for d in rows if not(d['cell'].startswith('cifar_fs_') or d['cell'].startswith('dtd_dtd_'))]
    p=report['pooled']['1']['weighted_gate_minus_uniform_mix']
    report['gates']={'shared_1shot_over_raw_at_least_3pp':all(d['weighted_gate_minus_raw']>=.03 for d in shared),
                     'beats_uniform_at_least_1pp_with_positive_ci':p['mean']>=.01 and p['paired_ci95'][0]>0,
                     'all_stress_losses_at_most_1pp':all(d['weighted_gate_minus_raw']>=-.01 for d in stress)}
    report['gate_passed']=all(report['gates'].values())
    report['decision']='retain_simple_retrieval_baseline_stop_current_weighting_and_gate_method'
    (HERE/'e12'/'decision.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
