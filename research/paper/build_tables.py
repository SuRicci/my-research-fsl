"""Render recorded result JSON as tables; no new experiment."""
from pathlib import Path
import json,csv,hashlib
P=Path(__file__).resolve().parent; W=P.parent
sources=[W/'experiments/main/oslo-gallery-pets-20260913/outputs/analysis.json',W/'experiments/analysis/oslo_influence/attribution.json',W/'experiments/analysis/oslo_final_step_intervention/analysis.json']
M,I,D=[json.loads(p.read_text()) for p in sources]
def name(k):
 p,g,s=k.split('_')
 return f"{'Original' if p=='canonical' else 'Confirm.'} / {'Pets' if g=='pets' else 'DTD'} / {s[1:]}"
def table(stem,header,rows,caption,label):
 with (P/'tables'/f'{stem}.csv').open('w') as f:
  w=csv.writer(f);w.writerow(header);w.writerows(rows)
 body='\n'.join(' & '.join(str(v) for v in r)+r' \\' for r in rows)
 text=r'\begin{table}[t]'+'\n'+r'\centering\small\setlength{\tabcolsep}{3pt}'+'\n'+r'\begin{tabular}{l'+'r'*(len(header)-1)+'}\n'+r'\toprule'+'\n'+' & '.join(header)+r' \\ \midrule'+'\n'+body+'\n'+r'\bottomrule\end{tabular}'+'\n'+r'\caption{'+caption+r'}\label{'+label+'}\n'+r'\end{table}'+'\n'
 (P/'latex/tables'/f'{stem}.tex').write_text(text)
methods=['OSLO_G','closed_set','zero_update','r2','CS_l2','support_logistic_C1','support_logistic_C10']
table('performance',['Pool / gallery / shot',r'OSLO$_G$','Closed','Zero','R2',r'CS$_{\ell_2}$','LR1','LR10'],
 [[name(k)]+[f"{c['accuracy_pct'][x]:.2f}" for x in methods] for k,c in M['cells'].items()],
 'Accuracy (percent) for all conditions, with 500 paired tasks per cell. Confirmation is an image-disjoint pool from the same target dataset. Queries are always Pets images; DTD is the mismatched gallery. Closed removes inlier weighting; Zero retains initial centered support prototypes. LR1 and LR10 are support-only logistic controls with fixed regularization.','tab:performance')
rows=[]
for k,c in I['cells'].items():
 a=c['episode_mean_inlier_auc']
 rows.append([name(k),f"{100*c['gallery_averaging_coefficient']['mean']:.3f}",f"{100*c['retained_gallery_weight_partition']['out_of_episode_mass']:.3f}",'--' if a is None else f'{a:.4f}'])
table('influence',['Pool / gallery / shot',r'Gallery coeff. (\%)',r'External share (\%)','Mean AUROC'],rows,
 'Final-update weight accounting. Gallery coefficient is the mean class-specific gallery fraction of support-plus-gallery mass. External share is the fraction of gallery weight on classes outside the episode. AUROC is undefined if no in-episode class is present. These descriptive coefficients are not causal directional effects.','tab:influence')
table('interventions',['Pool / gallery / shot','Original','Mass','Oracle','Oracle+mass','Zero','R2'],
 [[name(k)]+[f"{c['accuracy_pct'][x]:.2f}" for x in ['original','mass_balanced','oracle_membership','oracle_balanced','zero_update','r2']] for k,c in D['cells'].items()],
 'Retrospective final-step interventions, accuracy in percent. Mass matches gallery weight to support mass per class. Oracle removes examples whose true class is outside the episode; Oracle+mass combines both. Oracle variants require privileged labels. Earlier fitted weights and centering stay fixed.','tab:interventions')
rows=[]
for k,c in M['cells'].items():
 v=c['comparisons']['r2'];a,b=v['paired_ci95_pp'];rows.append([name(k),f"{v['delta_pp']:+.3f}",f'[{a:+.3f}, {b:+.3f}]'])
table('paired_primary',['Pool / gallery / shot',r'$\Delta$ vs R2 (pp)',r'Paired 95\% interval'],rows,
 'Source-default transfer differences. Intervals use 5,000 paired bootstrap resamples stratified within task seeds and condition on fixed pools, not independent domains.','tab:paired')
rows=[]
for k in ['canonical_pets_k1','fresh_pets_k1']:
 for m,label in [('mass_balanced','Mass'),('oracle_membership','Oracle'),('oracle_balanced','Oracle+mass')]:
  v=D['cells'][k]['comparisons'][m]['r2'];a,b=v['paired_ci95_pp'];rows.append([name(k)+' / '+label,f"{v['delta_pp']:+.3f}",f'[{a:+.3f}, {b:+.3f}]'])
table('paired_interventions',['Condition / intervention',r'$\Delta$ vs R2 (pp)',r'Paired 95\% interval'],rows,
 'Matched one-shot diagnostic effects against R2. Intervals are conditional and retrospective; oracle gains do not qualify an admissible predictor.','tab:paired-interventions')
sub=[]
for k,c in M['cells'].items():
 vals=[x['accuracy_pct']['OSLO_G']-x['accuracy_pct']['r2'] for x in c['per_breed'].values()]
 sub.append({'cell':k,'breed_count':len(vals),'negative':sum(v<0 for v in vals),'equal':sum(v==0 for v in vals),'positive':sum(v>0 for v in vals),'seed_delta_pp':c['comparisons']['r2']['seed_deltas_pp']})
(P/'tables/subgroup_summary.json').write_text(json.dumps(sub,indent=2)+'\n')
(P/'table_source_map.json').write_text(json.dumps({'sources':{str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'tables':{'performance':str(sources[0]),'paired_primary':str(sources[0]),'influence':str(sources[1]),'interventions':str(sources[2]),'paired_interventions':str(sources[2])},'unique_evidence_groups':3,'condition_evaluations':4000,'sampled_episodes':2000},indent=2)+'\n')
print('Generated five tables from three audited sources.')
