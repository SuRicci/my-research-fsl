"""No-refit description of existing task outcomes; oracle numbers are not a method."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
ROOT=Path(__file__).resolve().parent
QUEST=Path('/Users/decoqwq/DeepScientist/quests/012')
SRC=QUEST/'.ds/worktrees/analysis-analysis-1985c6ec-ilpcz-gallery-control/experiments/analysis/analysis-1985c6ec/ilpcz-gallery-control'
OWN=QUEST/'.ds/worktrees/r2-ownership-canonical-20260912/experiments/main/r2-ownership-canonical-20260912'
IDENTITY=['support_indices','query_indices','class_ids','seeds','yq']
def load(p):
    with np.load(p) as a:return {k:a[k] for k in a.files}
result={'kind':'descriptive_saved_output_audit','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Already seen pools/tasks; no fitted gate, threshold or new prediction method. Oracle switching uses unavailable query labels and is a non-deployable upper bound.','development':{},'evaluation':{},'source_sha256':{}}
deltas={}
for phase in ['dev','eval']:
    cells=['dtd_dtd_k1','dtd_eurosat_k1','dtd_dtd_k5']
    if phase=='eval':cells+=['eurosat_dtd_k1','eurosat_eurosat_k1','eurosat_eurosat_k5']
    for cell in cells:
        p=SRC/'outputs'/phase/(cell+'_g2.npz'); a=load(p)
        q=OWN/'outputs/dev'/(cell+'.npz') if phase=='dev' else SRC/'outputs/controls'/(cell+'.npz')
        if phase=='dev' and cell.endswith('k5'):q=OWN.parent/'r2-geometry-canonical-20260912/outputs/dev'/(cell+'.npz')
        b=load(q)
        for key in IDENTITY:assert np.array_equal(a[key],b[key]),(phase,cell,key)
        acc=(a['predictions']==a['yq']).mean(-1)*100
        assert np.max(abs(acc-a['accuracy']*100))<1e-10
        i=list(b['names']).index('r2')
        base=b['accuracy'][i]*100 if phase=='dev' else b['accuracy_percent'][i]
        delta=acc-base;seeds=a['seeds'];deltas[(phase,cell)]=(delta,seeds)
        stats=json.loads((SRC/'outputs'/phase/(cell+'_g2_stats.json')).read_text())
        assert len(delta)==len(stats)==(200 if phase=='dev' else 1000)
        row={'tasks':len(delta),'ilpcz_accuracy_percent':float(acc.mean()),'r2_accuracy_percent':float(base.mean()),'delta_pp':float(delta.mean()),'win_tasks':int((delta>1e-9).sum()),'tie_tasks':int((abs(delta)<=1e-9).sum()),'loss_tasks':int((delta < -1e-9).sum()),'per_seed_delta_pp':{str(s):float(delta[seeds==s].mean()) for s in np.unique(seeds)},'median_selected_gallery_count':float(np.median([x['gallery_selected'] for x in stats]))}
        if phase=='dev':row['oracle_switch_upper_bound_pp']=float(np.maximum(delta,0).mean())
        result['development' if phase=='dev' else 'evaluation'][cell]=row
        for source in [p,q]:result['source_sha256'][str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
macro=[]
for domain in ['dtd','eurosat']:
    parts=[deltas[('eval',domain+'_'+g+'_k1')] for g in ['dtd','eurosat']]
    assert np.array_equal(parts[0][1],parts[1][1])
    avg=np.mean([x[0] for x in parts],axis=0);ss=parts[0][1]
    macro.append([avg[ss==s].mean() for s in np.unique(ss)])
result['macro_1shot_per_seed_delta_pp']=dict(zip(map(str,np.unique(ss)),map(float,np.mean(macro,axis=0))))
result['validated_counts']={'development_cells':len(result['development']),'evaluation_cells':len(result['evaluation']),'development_cell_tasks':sum(x['tasks'] for x in result['development'].values()),'evaluation_cell_tasks':sum(x['tasks'] for x in result['evaluation'].values())}
result['interpretation']='Seed consistency can retain a conditional effect, but task-level benefit frequencies and an oracle upper bound do not prove an observable, transferable selector. Original development is DTD-only; no EuroSAT selector may be tuned from these evaluation effects.'
(ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# iLPC-z enablement feasibility audit','',result['scope'],'','| Phase/cell | Tasks | Mean delta vs R2, pp | Wins/ties/losses | Seed delta range, pp |','|---|---:|---:|---|---|']
for phase in ['development','evaluation']:
    for cell,r in result[phase].items():
        v=list(r['per_seed_delta_pp'].values());lines.append(f"| {phase}/{cell} | {r['tasks']} | {r['delta_pp']:+.3f} | {r['win_tasks']}/{r['tie_tasks']}/{r['loss_tasks']} | {min(v):+.3f} to {max(v):+.3f} |")
lines+=['','Development oracle switching upper bounds (unavailable query-label knowledge): '+', '.join(f"{c}: {r['oracle_switch_upper_bound_pp']:.3f} pp" for c,r in result['development'].items()),'',result['interpretation']]
(ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:result[k] for k in ['validated_counts','development','evaluation','macro_1shot_per_seed_delta_pp']},indent=2))
