"""Post-hoc evidence decomposition and independent bootstrap arithmetic audit.
No new model, hyperparameter, task, target dataset or acceptance threshold.
"""
from pathlib import Path
import hashlib,json
import numpy as np
P=Path(__file__).resolve().parent; O=P/'outputs'
cfg=json.loads((P/'protocol.json').read_text());summary=json.loads((O/'analysis.json').read_text())
weights_cache={};checks=[];rows={};hashes={}
def weights(seeds):
    key=tuple(seeds.tolist())
    if key not in weights_cache:
        groups=np.unique(seeds); rng=np.random.RandomState(cfg['gate']['bootstrap_seed'])
        W=np.zeros((cfg['gate']['bootstrap_replicates'],len(seeds)))
        for seed in groups:
            ix=np.flatnonzero(seeds==seed);draw=rng.randint(len(ix),size=(len(W),len(ix)))
            W[:,ix]=np.stack([np.bincount(v,minlength=len(ix)) for v in draw])/(len(ix)*len(groups))
        assert np.allclose(W.sum(1),1)
        weights_cache[key]=W
    return weights_cache[key]
def ci(delta,seeds):
    return {'delta_pp':float(delta.mean()*100),'ci95_pp':(np.quantile(weights(seeds)@delta,[.025,.975])*100).tolist()}
def audit(a,names,seeds,saved,label):
    i=names.index('scatter_r2')
    for j,m in enumerate(names):
        if j==i:continue
        v=ci(a[i]-a[j],seeds);ref=saved[m]
        err=max(abs(v['delta_pp']-ref['delta_pp']),float(np.max(abs(np.array(v['ci95_pp'])-ref['ci95_pp']))))
        assert err<1e-10,(label,m,err)
        checks.append({'label':label+'/'+m,'max_stat_error':err})
for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
    pair=[]
    for shot in cfg['shots']:
        for gallery in [target,source]:
            name=f'{source}_to_{target}_{gallery}_k{shot}';f=O/'cells'/(name+'.npz');z=np.load(f)
            hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
            names=z['names'].tolist();a=(z['scores'].argmax(-1)==z['yq']).mean(-1);seeds=z['seeds']
            assert np.array_equal(a,z['accuracy'])
            audit(a,names,seeds,summary['cells'][name]['comparisons'],name)
            old,mean,scatter=[a[names.index(m)] for m in ['original_r2','mean_r2','scatter_r2']]
            rows[name]={'view_mean_minus_original':ci(mean-old,seeds),'scatter_minus_view_mean':ci(scatter-mean,seeds),'scatter_minus_mean_per_seed_pp':{str(seed):float((scatter-mean)[seeds==seed].mean()*100) for seed in np.unique(seeds)}}
            if shot==1:pair.append((a,names,seeds))
    a,names,seeds=pair[0];macro=(a+pair[1][0])/2
    audit(macro,names,seeds,summary['directions'][source+'_to_'+target],source+'_to_'+target)
assert len(checks)==90
result={'status':'passed_statistics_only','statistic_comparisons':len(checks),'max_stat_error':max(c['max_stat_error'] for c in checks),'cells':rows,'input_sha256':hashes,'scope':'Post-hoc fixed-pool decomposition. Bootstrap weights independently built with multinomial multiplicities; all90 predefined mean/interval comparisons recovered. Does not override failed dense end-to-end score audit, establish family impossibility, or add Pets/Caltech evidence.'}
(O/'statistical_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))
