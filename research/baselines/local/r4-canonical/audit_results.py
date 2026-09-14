"""Independently reconcile full saved rankings, AP, identities and metric cells."""
from pathlib import Path
import json,time
import numpy as np
from evaluate import ROOT,P,OUT,sha,dump,load_data

data,identity_hashes=load_data()
completion=json.loads((OUT/'completion.json').read_text());assert completion['complete']
manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['validation']['identity_hashes']==identity_hashes
assert manifest['validation']['evaluator_sha256']==sha(ROOT/'evaluate.py')
assert manifest['validation']['protocol_sha256']==sha(ROOT/'protocol.json')
files=sorted(OUT.glob('*/*.npz'));assert len(files)==60
groups={};rows=0;max_error=0.;base_predictions={}
for p in files:
 meta=json.loads(p.with_suffix('.json').read_text());assert sha(p)==meta['npz_sha256']
 with np.load(p,allow_pickle=False) as archive:
  z={k:archive[k] for k in archive.files}
  ds,phase,policy=meta['dataset'],meta['phase'],meta['policy'];d=data[ds];m=meta['budget'];seed=meta['seed'];classes=d['splits'][phase]
  qi=d['query_indices'][np.isin(d['query_labels'][d['query_indices']],classes)]
  bd=ds if policy=='matched' else next(n for n in data if n!=ds);other=data[bd];pool=other['remaining']
  if policy=='matched':pool=pool[np.isin(other['query_labels'][pool],classes)]
  ai=np.random.default_rng(seed).permutation(pool)[:m]
  for key,expected in [('query_indices',qi),('query_ids',np.array(d['identity']['query_ids'])[qi]),('query_labels',d['query_labels'][qi]),('gallery_ids',d['identity']['gallery_ids']),('gallery_labels',d['gallery_labels']),('bridge_indices',ai),('bridge_ids',np.array(other['identity']['query_ids'])[ai])]:assert np.array_equal(z[key],expected),(p,key)
  assert str(z['bridge_domain'])==bd and int(z['seed'])==seed and int(z['budget'])==m
  assert z['methods'].tolist()==P['methods']
  assert z['ranks'].shape==(len(P['methods']),len(qi),len(d['gallery_labels']))
  assert np.array_equal(np.sort(z['ranks'],axis=-1),np.broadcast_to(np.arange(len(d['gallery_labels'])),z['ranks'].shape))
  independently=np.zeros_like(z['ap'])
  for k,name in enumerate(P['methods']):
   for i,label in enumerate(z['query_labels']):
    hit_ranks=np.flatnonzero(z['gallery_labels'][z['ranks'][k,i]]==label)
    npositive=np.count_nonzero(z['gallery_labels']==label);assert npositive>0
    independently[k,i]=np.sum(np.arange(1,len(hit_ranks)+1)/(hit_ranks+1))/npositive
   if name in ['old_old','old_centered','old_whiten','new_new_oracle']:
    key=(phase,ds,name)
    if key in base_predictions:assert np.array_equal(base_predictions[key],z['ranks'][k])
    else:base_predictions[key]=z['ranks'][k].copy()
  error=float(np.max(abs(independently-z['ap'])));assert error<1e-12
  max_error=max(max_error,error);rows+=len(qi)*len(P['methods'])
  key=(phase,ds,policy,m);groups.setdefault(key,[]).append((seed,independently,z['query_labels'].copy()))
assert len(groups)==12 and all(len(g)==5 for g in groups.values())
names=P['methods'];vi=names.index('v0_press');ci=[i for i,n in enumerate(names) if n not in ['v0_press','new_new_oracle']]
all_rows=[];metrics={};rng=np.random.default_rng(26091399)
for key,group in sorted(groups.items()):
 group.sort(key=lambda x:x[0]);assert [g[0] for g in group]==P['bridge_seeds']
 ap=np.stack([g[1] for g in group]);labels=group[0][2]
 mean=ap.mean((0,2))*100
 row={'phase':key[0],'dataset':key[1],'bridge_policy':key[2],'budget':key[3],'mean_map':dict(zip(names,mean.tolist())),'v0_seed_map':(ap[:,vi].mean(1)*100).tolist()}
 if key[0]=='eval':
  metric=f'{key[1]}_{key[2]}_bridge{key[3]}_map';metrics[metric]=float(mean[vi]);row['metric_id']=metric
  clustered=np.stack([ap[:,:,labels==c].mean(-1) for c in np.unique(labels)],axis=-1)
  si=rng.integers(5,size=(5000,5));qi=rng.integers(clustered.shape[-1],size=(5000,clustered.shape[-1]))
  draws=clustered.transpose(0,2,1)[si[:,:,None],qi[:,None,:]].mean((1,2))*100
  gap=draws[:,vi]-draws[:,ci].max(1)
  best=ci[int(np.argmax(mean[ci]))]
  row.update(best_control=names[best],v0_minus_best_control_pp=float(mean[vi]-mean[best]),v0_minus_resampled_best_ci95_pp=np.quantile(gap,[.025,.975]).tolist(),v0_minus_old_pp=float(mean[vi]-mean[names.index('old_old')]),oracle_minus_old_pp=float(mean[-1]-mean[0]))
 all_rows.append(row)
assert len(metrics)==8;metrics['macro_map']=float(np.mean(list(metrics.values())))
entries=[{'metric_id':k,'description':'Frozen V0 mean class-retrieval AP for '+k,'derivation':'100 times mean query AP then mean five bridge seeds; macro is equal mean eight evaluation cells. Full rankings retained.','direction':'maximize','unit':'%','required':True,'source_ref':str(OUT/'metrics_summary.json'),'origin_path':k} for k in metrics]
contract={'contract_id':'r4-canonical-local-semantic-mps32-v1','primary_metric_id':'macro_map','evaluation_protocol':{'scope_id':P['scope_id'],'protocol':P,'code_paths':[str(ROOT/'evaluate.py'),str(ROOT/'audit_results.py')],'code_hashes':{'evaluate.py':sha(ROOT/'evaluate.py'),'audit_results.py':sha(__file__)},'historical_parity':False},'metrics':entries}
dump(OUT/'metrics_summary.json',metrics);dump(OUT/'all_metrics.json',{'rows':all_rows})
dump(ROOT/'json/metric_contract_candidate.json',{'baseline_id':P['baseline_id'],'variant_id':P['variant_id'],'metric_contract':contract,'metrics_summary':metrics,'primary_metric':{'metric_id':'macro_map','value':metrics['macro_map'],'direction':'maximize'}})
validation={'passed':True,'completed_jobs':len(files),'development_cells':4,'evaluation_cells':8,'method_query_rankings_checked':rows,'max_saved_ap_error':max_error,'all_ranks_full_permutations':True,'all_task_and_bridge_ids_reconstructed':True,'fixed_zero_bridge_control_ranks_equal':True,'source_identity_and_rgb_checks':manifest['validation'],'no_new_encoding':True,'complete_metric_keys':list(metrics),'interval_scope':'class-cluster and bridge-seed paired bootstrap, same query identities across seeds, fixed gallery; resampled maximum over all controls; no external-domain or historical reproduction claim'}
dump(OUT/'validation_report.json',validation)
lines=['# Frozen V0 local comparison','', 'New DTD/EuroSAT semantic retrieval scope; canonical local feature variant, not historical CUB/Oxford reproduction. All methods frozen before this run. Values mAP%.','', '| Dataset | Bridge | Budget | Old | Center | White | V0 | Strongest control | Delta pp | Oracle |','|---|---|---:|---:|---:|---:|---:|---|---:|---:|']
for row in all_rows:
 if row['phase']!='eval':continue
 m=row['mean_map'];best=row['best_control'];lines.append(f"| {row['dataset']} | {row['bridge_policy']} | {row['budget']} | {m['old_old']:.3f} | {m['old_centered']:.3f} | {m['old_whiten']:.3f} | {m['v0_press']:.3f} | {best}: {m[best]:.3f} | {row['v0_minus_best_control_pp']:+.3f} | {m['new_new_oracle']:.3f} |")
lines+=['',f"Primary V0 equal-cell macro: {metrics['macro_map']:.6f}%.",'', 'Complete13method rows, development values, five seed values and paired intervals: outputs/all_metrics.json. Protocol and source hashes frozen; outputs/validation_report.json audits all60jobs. Queries: DTD240eval+230dev, EuroSAT100eval; five seeds reuse these same query images. AP compares semantic class matches, not instance labels. No gallery-new features or labels enter methods; evaluator-only oracle is an upper reference, not an allowed baseline. WIP is restricted m-bridge version, not full original training protocol.']
(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'validation':validation,'metrics':metrics}),flush=True)
