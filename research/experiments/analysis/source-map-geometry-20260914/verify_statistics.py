"""Independent covariance-formula check of all descriptive correlations and saved counts."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent
r=json.loads((P/'RESULT.json').read_text())
assert r['model_count']==36 and len(r['rows'])==36
assert sum(x['selected_step']==0 for x in r['rows'])==6
pairs=np.load(P/'pairs.npz')
for ds,z in r['domains'].items():
    assert pairs[ds].shape==(4096,2)
    assert int(pairs[ds].min())>=0 and int(pairs[ds].max())<z['image_count']
    assert np.all(pairs[ds][:,0]!=pairs[ds][:,1])
    assert len(np.unique(np.sort(pairs[ds],axis=1),axis=0))==z['unique_sampled_pairs']
errors=[];checks=0
for key,v in r['correlations'].items():
    ds,shot=key.split('_k');rr=[x for x in r['rows'] if x['source']==ds and x['shot']==int(shot) and x['selected_step']>0]
    assert len(rr)==v['nonzero_model_count']
    y=[x['saved_accuracy']['transfer_delta_pp'] for x in rr]
    for pool,actual in v['geometry_pool'].items():
        a=np.array([[x['geometry'][pool]['mean_sq_displacement'],x['geometry'][pool]['pair_cosine_rms_change'],x['geometry'][pool]['class_gap_change'],y[i]] for i,x in enumerate(rr)])
        covariance=np.cov(a,rowvar=False);norm=np.sqrt(np.outer(np.diag(covariance),np.diag(covariance)));c=covariance/norm
        partial=(c[1,3]-c[0,1]*c[0,3])/np.sqrt((1-c[0,1]**2)*(1-c[0,3]**2))
        expected=dict(zip(['displacement_vs_transfer','distortion_vs_transfer','distortion_vs_transfer_adjusted_displacement','class_gap_vs_transfer'],[c[0,3],c[1,3],partial,c[2,3]]))
        for k,value in expected.items():errors.append(abs(float(value)-actual[k]));checks+=1
assert max(errors)<1e-10
inputs=json.loads((P/'inputs.json').read_text())
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in inputs.items())
summary={'status':'passed','correlation_checks':checks,'maximum_abs_error':max(errors),'models':36,'nonzero_models':30,'input_hashes_unchanged':True,'pair_bound_checks':True,'numeric_cases':r['validation']['independent_cases'],'formula':'sample covariance correlations and analytic partial correlation; independent of primary lstsq projection','boundary':'no inference p-values;6-9selectednonzero models per source/shot, repeated datasets, observational only'}
(P/'independent_statistics.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
