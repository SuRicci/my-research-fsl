"""Read-only numeric audit of seven tables against original recorded JSON."""
from pathlib import Path
import json,re,hashlib,datetime
W=Path(__file__).resolve().parents[2];P=W/'paper'
def read(p):return json.loads(p.read_text())
M=read(W/'experiments/main/oslo-gallery-pets-20260913/outputs/analysis.json')
I=read(W/'experiments/analysis/oslo_influence/attribution.json')
D=read(W/'experiments/analysis/oslo_final_step_intervention/analysis.json')
C=read(W/'experiments/main/prior-calibration-qualification-20260913/outputs/analysis.json')
expected={}
for stem in ['performance','interventions','influence','paired_primary','paired_interventions','calibration_accuracy','calibration_deltas']:expected[stem]=[]
for pool,plabel in [('canonical','Original'),('fresh','Confirm.')]:
 for shot in [1,5]:
  for gal,glabel in [('pets','Pets'),('dtd','DTD')]:
   k=f'{pool}_{gal}_k{shot}'; label=f'{plabel} / {glabel} / {shot}'
   expected['performance'].append((label,[M['cells'][k]['accuracy_pct'][n] for n in ['OSLO_G','closed_set','zero_update','r2','CS_l2','support_logistic_C1','support_logistic_C10']],2))
   expected['interventions'].append((label,[D['cells'][k]['accuracy_pct'][n] for n in ['original','mass_balanced','oracle_membership','oracle_balanced','zero_update','r2']],2))
   z=I['cells'][k];a=z['episode_mean_inlier_auc']
   expected['influence'].append((label,[100*z['gallery_averaging_coefficient']['mean'],100*z['retained_gallery_weight_partition']['out_of_episode_mass']]+([] if a is None else [a]),[3,3]+([] if a is None else [4])))
   z=M['cells'][k]['comparisons']['r2'];expected['paired_primary'].append((label,[z['delta_pp'],*z['paired_ci95_pp']],3))
 for n,name in [('mass_balanced','Mass'),('oracle_membership','Oracle'),('oracle_balanced','Oracle+mass')]:
  z=D['cells'][pool+'_pets_k1']['comparisons'][n]['r2']
  expected['paired_interventions'].append((plabel+' / Pets / 1 / '+name,[z['delta_pp'],*z['paired_ci95_pp']],3))
labels={'r2':'R2','logistic_1':'LR1','logistic_10':'LR10','calibrated':'Calibrated','same_mass':'Same mass','composition_only':'Composition only','fixed_prior':'Fixed prior'}
for n in next(iter(C['cells'].values()))['accuracy_pct']:
 if n.startswith('cs_'):label=r'CS$_{\ell_2}$, $\lambda='+n[3:]+'$'
 elif n.startswith('grid_'):
  _,mix,lam=n.split('_');label=r'Mix $'+mix+r'$, $\lambda='+lam+'$'
 else:label=labels[n]
 keys=['eurosat_to_dtd_dtd_k1','eurosat_to_dtd_eurosat_k1','eurosat_to_dtd_dtd_k5','eurosat_to_dtd_eurosat_k5','dtd_to_eurosat_eurosat_k1','dtd_to_eurosat_dtd_k1','dtd_to_eurosat_eurosat_k5','dtd_to_eurosat_dtd_k5']
 expected['calibration_accuracy'].append((label,[C['cells'][k]['accuracy_pct'][n] for k in keys],2))
checks=[];total=0
for stem,rows in expected.items():
 if stem=='calibration_deltas':continue
 text=(P/'latex/tables'/f'{stem}.tex').read_text()
 for label,values,precision in rows:
  lines=[line for line in text.splitlines() if line.split('&')[0].strip()==label];assert len(lines)==1,(stem,label)
  actual=[float(x) for x in re.findall(r'[-+]?\d*\.\d+|[-+]?\d+',lines[0].split('&',1)[1])]
  assert len(actual)==len(values),(stem,label,actual,values)
  ps=precision if isinstance(precision,list) else [precision]*len(values)
  for a,b,d in zip(actual,values,ps):assert abs(a-b)<=.5*10**(-d)+1e-9,(stem,label,a,b)
  total+=len(values)
 checks.append({'table':stem,'rows':len(rows),'passed':True})
text=(P/'latex/tables/calibration_deltas.tex').read_text()
for src,label in [('dtd','DTD'),('eurosat','EuroSAT')]:
 line=next(x for x in text.splitlines() if x.startswith(label+r' $\to$'))
 z=next(v for k,v in C['cells'].items() if k.startswith(src+'_to_'));d=C['directions'][src][z['source_winner']]
 actual=[float(x) for x in re.findall(r'[-+]?\d*\.\d+', '&'.join(line.split('&')[2:]))]
 assert len(actual)==4
 for a,b in zip(actual,[d['delta_pp'],*d['ci95_pp'],0.]):assert abs(a-b)<=.005+1e-9
 total+=4
checks.append({'table':'calibration_deltas','rows':2,'passed':True})
srcmap=read(P/'table_source_map.json')
for p,h in srcmap['sources'].items():assert hashlib.sha256((W/p).read_bytes()).hexdigest()==h
l=read(P/'evidence_ledger.json');inventory=[]
for r in l['items']:
 paths=r.get('source_paths',[]);resolved=[Path(x) if Path(x).is_absolute() else W/x for x in paths]
 if r['paper_role']!='reference_only':assert all(x.exists() for x in resolved),r['item_id']
 inventory.append({'item_id':r['item_id'],'status':r['status'],'paper_role':r['paper_role'],'claim_links':r.get('claim_links',[]),'classification':'appendix-only' if r['paper_role']=='appendix' else 'completed and written' if r['paper_role']=='main_text' else 'historical/reference-only; excluded from manuscript','source_paths':paths,'current_source_hashes':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in resolved if x.exists() and x.is_file() and x.stat().st_size<5000000}})
r={'status':'passed','updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'table_count':len(checks),'numeric_values_checked':total,'tables':checks,'paper_facing_groups':sum(x['paper_role']!='reference_only' for x in inventory),'ledger_items':len(inventory),'reference_only_items':sum(x['paper_role']=='reference_only' for x in inventory),'prior_validator_analysis_count':19,'count_boundary':'Table count is seven, not fourteen input/environment matches. Four measured evidence groups support five analysis questions; two diagnostics reuse the main task set. Historical inventory entries are not independent analyses.','inventory':inventory,'scope':'Table and prose/source mapping audit only; original raw-prediction audits remain authoritative. No new raw-prediction replay this review.'}
(P/'review/evidence_inventory.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({k:v for k,v in r.items() if k!='inventory'}))
