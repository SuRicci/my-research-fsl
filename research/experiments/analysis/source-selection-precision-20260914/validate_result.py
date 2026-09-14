"""Validate full saved source-selection banks and integer decisions, without model rerun."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
R=Path(__file__).resolve().parent;W=R.parents[2];O=R/'outputs';S=W/'experiments/main/representation-scatter-20260913/outputs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
z=json.loads((O/'RESULT.json').read_text());cfg=json.loads((R/'protocol.json').read_text())
assert z['status']=='completed' and z['comparison_count']==1200 and len(z['rows'])==1200
assert len({(x['domain'],x['episode'],x['condition'],x['gamma']) for x in z['rows']})==1200
for p,h in z['input_hashes'].items():assert sha(p)==h,p
for p,h in z['output_hashes'].items():assert sha(p)==h,p
checks=[];changed=0;fails=0;new_fails=0
for ds in cfg['domains']:
    old=np.load(S/f'{ds}_k1_selection.npz');new=np.load(O/f'{ds}.npz')
    for k in ['support_indices','query_indices','gallery_indices','class_ids','seeds']:assert np.array_equal(old[k],new[k]),(ds,k)
    assert new['scores'].shape==(100,2,3,75,5) and np.isfinite(new['scores']).all()
    assert np.array_equal(new['predictions'],np.argmax(new['scores'],axis=-1))
    tallies=[];old_tallies=[]
    for j,gamma in enumerate(cfg['gamma']):
        tally=0;old_tally=0
        for i in range(100):
            for c in range(2):
                labels=np.argmax(new['scores'][i,c,j],axis=-1);oldlabels=np.argmax(old['scores'][i,c,j],axis=-1)
                tally+=sum(int(int(p)==int(t//15)) for t,p in enumerate(labels))
                old_tally+=sum(int(int(p)==int(t//15)) for t,p in enumerate(oldlabels))
                er=float(np.max(np.abs(new['scores'][i,c,j]-old['scores'][i,c,j])))
                dif=int(np.count_nonzero(labels!=oldlabels));changed+=dif;fails+=int(er>=cfg['score_atol'] or dif>0)
                row=next(x for x in z['rows'] if (x['domain'],x['episode'],x['condition'],x['gamma'])==(ds,i,c,gamma))
                assert abs(er-row['historical_score_error'])<1e-12 and dif==row['historical_prediction_changes']
                new_fails+=int(row['float64_score_error']>=cfg['score_atol'] or row['float64_prediction_changes']>0)
        tallies.append(tally);old_tallies.append(old_tally)
    winner=sorted(zip(tallies,cfg['gamma']),key=lambda v:(-v[0],v[1]))[0][1]
    a=z['domains'][ds];assert tallies==a['float64_counts']==new['correct_counts'].tolist()
    assert old_tallies==a['old_counts'] and winner==a['float64_gamma']
    checks.append({'domain':ds,'counts':tallies,'old_counts':old_tallies,'gamma':winner,'old_gamma':a['old_gamma']})
assert changed==z['historical_prediction_changes'] and fails==z['historical_score_failures']
assert (new_fails==0)==z['float64_validation_passed']
prior=json.loads((S/'output_audit_diagnostics.json').read_text());assert prior['frozen_audit_passed'] is False
r={'status':'passed','comparisons':1200,'query_predictions_compared':90000,'source_episodes':200,'domain_selections':checks,'historical_prediction_changes':changed,'historical_full_score_failures':fails,'float64_full_score_failures':new_fails,'original_failure_preserved':True,'input_hashes_unchanged':True,'scope':'Complete output/selection arithmetic and identity audit; numeric independent model comparison was performed during full run.'}
(O/'validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
