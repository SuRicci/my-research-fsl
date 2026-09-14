"""Paired task-level effects; fixed pools, correlated gallery conditions."""
import numpy as np
import study as s

def interval(x,seeds):
    rng=np.random.RandomState(s.CFG['bootstrap_seed']);means=np.zeros(s.CFG['bootstrap_n'])
    for seed in np.unique(seeds):
        d=x[seeds==seed];means+=d[rng.randint(len(d),size=(len(means),len(d)))].mean(1)/len(np.unique(seeds))
    return {'delta_pp':float(x.mean()*100),'ci95_pp':(np.quantile(means,[.025,.975])*100).tolist(),'seed_deltas_pp':[float(x[seeds==seed].mean()*100) for seed in np.unique(seeds)]}

def run():
    cells={};macro={};gate=True
    for ds in s.CFG['domains']:
        source=next(d for d in s.CFG['domains'] if d!=ds)
        for k in s.CFG['shots']:
            all_acc=[];seeds=s.task(ds,k)['seeds']
            for gal in s.CFG['domains']:
                name=f'{ds}_{gal}_k{k}';z=np.load(s.OUT/f'{name}.npz')
                a={m:(z['predictions'][i]==z['yq']).mean(-1) for i,m in enumerate(z['names'])}
                old=np.load(s.MAIN/'representation-scatter-20260913/outputs/cells'/f'{source}_to_{ds}_{gal}_k{k}.npz')
                for i,m in enumerate(old['names']):
                    if m!='scatter_r2':a['prior_'+str(m)]=(old['predictions'][i]==old['yq']).mean(-1)
                refs=[m for m in a if m not in ['selected','view_selected']]
                comparisons={m:interval(a['selected']-a[m],seeds) for m in refs}
                if any(c['ci95_pp'][0]<-.5 for c in comparisons.values()):gate=False
                choice=z['selected'];cv=z['cv_accuracy'].mean(-1)
                # Descriptive same-data diagnostic, not a fitted predictor or new test.
                cv_chosen=cv[choice,np.arange(500)]
                choices={m:int((choice==i).sum()) for i,m in enumerate(['equal','dino','clip'])}
                pair=[]
                for j in [1,2]:
                    proxy=z['risk'][0].mean(-1)-z['risk'][j].mean(-1)
                    actual=a[['equal','dino','clip'][j]]-a['equal']
                    nonzero=(proxy!=0)&(actual!=0)
                    pair.append({'endpoint':['equal','dino','clip'][j],'nonzero_tasks':int(nonzero.sum()),'sign_agreement':float((np.sign(proxy[nonzero])==np.sign(actual[nonzero])).mean()) if nonzero.any() else None})
                cells[name]={'accuracy_pct':{m:float(v.mean()*100) for m,v in a.items()},'comparisons':comparisons,'choice_counts':choices,'chosen_cv_accuracy_pct':float(cv_chosen.mean()*100),'chosen_query_accuracy_pct':float(a['selected'].mean()*100),'validation_minus_query_pp':float((cv_chosen-a['selected']).mean()*100),'proxy_pairwise_agreement':pair,'view_vs_image_choice_disagreement':int((z['view_selected']!=choice).sum()),'view_minus_primary':interval(a['view_selected']-a['selected'],seeds)}
                all_acc.append(a)
            if k==1:
                ave={m:(all_acc[0][m]+all_acc[1][m])/2 for m in all_acc[0]}
                co={m:interval(ave['selected']-ave[m],seeds) for m in ave if m not in ['selected','view_selected']}
                if any(c['delta_pp']<.5 or c['ci95_pp'][0]<=0 for c in co.values()):gate=False
                macro[ds]={'accuracy_pct':{m:float(v.mean()*100) for m,v in ave.items()},'comparisons':co}
    result={'status':'passed' if gate else 'refuted_fixed_protocol','unique_tasks':2000,'cells':cells,'macro_1shot':macro,'interval_scope':'paired5000task bootstrap within5fixedseeds; galleryconditions averaged per task, fixed pools only','canonical_baseline':'unchanged; noPets/Caltech outcomes'}
    s.dump(s.OUT/'analysis.json',result)
    for ds,z in macro.items():print(ds,z['comparisons']['equal'],flush=True)
    print('VERDICT',result['status'],flush=True)
if __name__=='__main__':run()
