from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
SRC=ROOT.parents[1]/'main/r4-transport-geometry-20260913/outputs'
groups={};hashes={}
for p in sorted((SRC/'dev').glob('*.npz')):
 m=json.loads(p.with_suffix('.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==m['npz_sha256']
 with np.load(p,allow_pickle=False) as z:
  names=z['methods'].tolist();labels=z['query_labels'];ap=z['ap']
  cluster=np.stack([ap[:,labels==c].mean(-1) for c in np.unique(labels)],-1)
  groups.setdefault((m['policy'],m['budget']),[]).append((m['seed'],cluster))
 hashes[str(p)]=m['npz_sha256']
assert len(groups)==4 and all(len(x)==5 for x in groups.values())
# Equal-cell averaging before class/seed resampling preserves paired development identities.
x=np.mean([np.stack([z[1] for z in sorted(g)]) for g in groups.values()],axis=0)
rng=np.random.default_rng(26091401);si=rng.integers(5,size=(5000,5));ci=rng.integers(23,size=(5000,23))
draw=x.transpose(0,2,1)[si[:,:,None],ci[:,None,:]].mean((1,2))*100
result={'scope':'No-refit descriptive audit of already seen development and evaluation outputs. Bootstrap argmax frequency is instability, not probability of generalization or a new selector.','source_jobs':20,'bootstrap_repeats':5000,'geometry':{},'source_sha256':hashes}
for g in ['raw','center']:
 ns=[g+'_a'+a for a in ['0','0.5','1','2']];ii=[names.index(n) for n in ns];b=draw[:,ii];gap=b[:,3]-b[:,2];counts=np.bincount(b.argmax(1),minlength=4)
 result['geometry'][g]={'development_alpha2_minus_alpha1_pp':float((x[:,ii[3]]-x[:,ii[2]]).mean()*100),'paired_ci95_pp':np.quantile(gap,[.025,.975]).tolist(),'bootstrap_argmax_counts':dict(zip(ns,counts.tolist()))}
a=json.loads((SRC/'analysis.json').read_text());result['primary_evaluation_macro_map']=a['metrics_summary']['macro_map'];result['primary_evaluation_delta_vs_old_centered_pp']=a['macro_comparisons']['old_centered']['delta_pp']
result['posthoc_alpha1_macro_map']={g:a['all_method_macro_map'][g+'_a1'] for g in ['raw','center']}
result['interpretation']='A small development preference plus overlapping uncertainty cannot establish reliable transfer selection. Posthoc evaluation alpha1 remains only a diagnostic; any new calibration rule needs prospective freezing and independent qualification.'
(ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
text=['# R4 development selection stability','',result['scope'],'','| Geometry | Development alpha2-alpha1, pp | Paired95% interval | Bootstrap best counts, alpha0/.5/1/2 |','|---|---:|---|---|']
for g,r in result['geometry'].items():text.append(f"| {g} | {r['development_alpha2_minus_alpha1_pp']:+.4f} | {r['paired_ci95_pp']} | {list(r['bootstrap_argmax_counts'].values())} |")
text+=['',result['interpretation'],'','Source: completed run-27bdcac3 on run/r4-transport-geometry-20260913; no new candidate or performance result.']
(ROOT/'REPORT.md').write_text('\n'.join(text)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)
