"""Independent full-ranking/identity audit and paired cluster/bridge-seed inference."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from run import ROOT,BASE,P,OUT,NAMES,ALPHAS,sha,dump

def audit():
    done=json.loads((OUT/'completion.json').read_text());assert done['complete'] and done['jobs']==60
    manifest=json.loads((OUT/'manifest.json').read_text());assert manifest['protocol']==P
    assert manifest['validation']['code_sha256']==sha(ROOT/'run.py')
    assert manifest['validation']['protocol_sha256']==sha(ROOT/'protocol.json')
    contract=json.loads(Path(P['canonical_contract_path']).read_text());assert sha(P['canonical_contract_path'])==P['canonical_contract_sha256']
    files=sorted(OUT.glob('*/*.npz'));assert len(files)==60
    groups={};dev={n:[] for n in NAMES};rows_checked=0;max_error=0.;ids_by_domain={}
    idkeys=['query_indices','query_ids','query_labels','gallery_ids','gallery_labels','bridge_indices','bridge_ids','bridge_domain','seed','budget']
    for p in files:
        meta=json.loads(p.with_suffix('.json').read_text());assert sha(p)==meta['npz_sha256']
        bp=Path(meta['baseline_npz']);assert sha(bp)==meta['baseline_npz_sha256']
        with np.load(p,allow_pickle=False) as z,np.load(bp,allow_pickle=False) as b:
            for k in idkeys:assert np.array_equal(z[k],b[k]),(p,k)
            assert z['methods'].tolist()==NAMES
            ranks=z['ranks'];ap=z['ap'];qy=z['query_labels'];gy=z['gallery_labels']
            assert ranks.shape==(len(NAMES),len(qy),len(gy))
            assert np.array_equal(np.sort(ranks,axis=-1),np.broadcast_to(np.arange(len(gy)),ranks.shape))
            independent=np.empty_like(ap)
            for mi,r in enumerate(ranks):
                for qi,label in enumerate(qy):
                    hit=np.flatnonzero(gy[r[qi]]==label)
                    independent[mi,qi]=np.sum(np.arange(1,len(hit)+1)/(hit+1))/np.count_nonzero(gy==label)
            err=float(np.max(abs(independent-ap)));assert err<1e-12;max_error=max(max_error,err);rows_checked+=len(NAMES)*len(qy)
            baseline_names=b['methods'].tolist();baseline_ap=b['ap']
            for n,bn in [('raw_a0','old_centered'),('center_a0','old_centered'),('raw_a2','centered_residual_2')]:
                assert np.max(abs(independent[NAMES.index(n)]-baseline_ap[baseline_names.index(bn)]))<1e-10
            if meta['phase']=='dev':
                for n,a in zip(NAMES,independent):dev[n].append(float(a.mean()*100))
            else:
                ds=meta['dataset'];key=(ds,meta['policy'],meta['budget'])
                # Canonical metadata names the policy as 'policy'.
                ids=z['query_ids'].copy()
                if ds in ids_by_domain:assert np.array_equal(ids_by_domain[ds],ids)
                else:ids_by_domain[ds]=ids
                groups.setdefault(key,[]).append((meta['seed'],np.concatenate([independent,baseline_ap]),qy.copy()))
        print('AUDITED',p.name,flush=True)
    selection=json.loads((OUT/'selection.json').read_text());assert selection['written_at']>=manifest['started_at']
    devmean={n:float(np.mean(v)) for n,v in dev.items()};expected={g:min(ALPHAS,key=lambda a:(-devmean[f'{g}_a{a:g}'],a)) for g in ['raw','center']}
    assert expected==selection['alphas'];assert all(abs(devmean[n]-selection['development_map'][n])<1e-12 for n in NAMES)
    names=NAMES+baseline_names;primary=f"center_a{expected['center']:g}";raw=f"raw_a{expected['raw']:g}";same=f"raw_a{expected['center']:g}"
    assert len(groups)==8 and all(len(x)==5 for x in groups.values())
    rng=np.random.default_rng(P['bootstrap_seed']);nboot=P['bootstrap_repeats'];si=rng.integers(5,size=(nboot,5));class_draws={};summaries=[];draws_by_cell={};values_by_method={n:{} for n in names}
    for key,group in sorted(groups.items()):
        group.sort(key=lambda x:x[0]);assert [x[0] for x in group]==P['baseline_protocol']['bridge_seeds']
        ap=np.stack([g[1] for g in group]);qy=group[0][2];classes=np.unique(qy)
        for g in group:assert np.array_equal(g[2],qy)
        clustered=np.stack([ap[:,:,qy==c].mean(-1) for c in classes],axis=-1)
        ds=key[0]
        if ds not in class_draws:class_draws[ds]=rng.integers(len(classes),size=(nboot,len(classes)))
        ci=class_draws[ds];draw=clustered.transpose(0,2,1)[si[:,:,None],ci[:,None,:]].mean((1,2))*100
        mean=ap.mean((0,2))*100;metric=f'{key[0]}_{key[1]}_bridge{key[2]}_map'
        for n,v in zip(names,mean):values_by_method[n][metric]=float(v)
        comparisons={n:{'delta_pp':float(mean[names.index(primary)]-mean[names.index(n)]),'ci95_pp':np.quantile(draw[:,names.index(primary)]-draw[:,names.index(n)],[.025,.975]).tolist()} for n in dict.fromkeys(['old_centered',raw,'centered_residual_2',same,'v0_press'])}
        summaries.append({'metric_id':metric,'dataset':ds,'policy':key[1],'budget':key[2],'mean_map':dict(zip(names,mean.tolist())),'comparisons':comparisons,'primary_seed_map':(ap[:,names.index(primary)].mean(1)*100).tolist()})
        draws_by_cell[key]=draw
    for n in names:values_by_method[n]['macro_map']=float(np.mean(list(values_by_method[n].values())))
    for n,variant in contract['metric_contract']['baseline_variants'].items():
        assert set(values_by_method[n])==set(variant['metrics_summary'])
        for metric,value in variant['metrics_summary'].items():assert abs(values_by_method[n][metric]-value)<1e-10,(n,metric)
    metric_values=values_by_method[primary];assert set(metric_values)==set(contract['metrics_summary']) and all(np.isfinite(list(metric_values.values())))
    macro_draw=np.mean(list(draws_by_cell.values()),axis=0)
    comparisons={n:{'delta_pp':metric_values['macro_map']-values_by_method[n]['macro_map'],'ci95_pp':np.quantile(macro_draw[:,names.index(primary)]-macro_draw[:,names.index(n)],[.025,.975]).tolist()} for n in dict.fromkeys(['old_centered',raw,'centered_residual_2',same,'v0_press'])}
    domains={ds:float(np.mean([r['comparisons']['old_centered']['delta_pp'] for r in summaries if r['dataset']==ds])) for ds in ['dtd','eurosat']}
    gate=expected['center']!=0 and comparisons['old_centered']['delta_pp']>=.5 and min(domains.values())>=-.5 and all(comparisons[n]['ci95_pp'][0]>0 for n in ['old_centered',raw,'centered_residual_2'])
    result={'primary_method':primary,'selected_raw_control':raw,'same_alpha_control':same,'selection':selection,'metrics_summary':metric_values,'all_method_macro_map':{n:v['macro_map'] for n,v in values_by_method.items()},'macro_comparisons':comparisons,'domain_delta_vs_old_centered_pp':domains,'strong_gate_passed':bool(gate),'interval_scope':'5000 paired bridge-seed and class-cluster resamples, shared query clusters across budget/policy, fixed galleries; exploratory seen pools','conclusion':'Strong local gate passed; independent confirmation and contribution audit still required.' if gate else 'Strong gate failed; retain attribution evidence and close this local kernel route without further exposed-grid tuning.'}
    validation={'passed':True,'jobs':60,'new_method_query_rankings_checked':rows_checked,'max_independent_ap_error':max_error,'full_rank_permutations':True,'all_saved_identities_match_canonical':True,'all13baseline_variants_all9metrics_reconciled':True,'selection_independently_reconstructed':True,'source_code_protocol_hashes_match':True,'metric_keys':list(metric_values),'audit_code_sha256':sha(__file__)}
    dump(OUT/'metrics_summary.json',metric_values);dump(OUT/'all_metrics.json',{'rows':summaries,'values_by_method':values_by_method});dump(OUT/'analysis.json',result);dump(OUT/'validation_report.json',validation)
    lines=['# R4 interpolation geometry paired results','',result['conclusion'],'','Exploratory DTD/EuroSAT semantic retrieval on previously seen pools; no new algorithm or historical CUB/Oxford parity claim.','',f"DTD development selected alpha: centered={expected['center']}, raw={expected['raw']}. Fixed minimum-lambda results are diagnostic only.",'','| Comparator | Primary minus comparator, pp | Paired95% interval, pp |','|---|---:|---|']
    for n,r in comparisons.items():lines.append(f"| {n} | {r['delta_pp']:+.4f} | [{r['ci95_pp'][0]:+.4f}, {r['ci95_pp'][1]:+.4f}] |")
    lines+=['','| Cell | Primary mAP | Old-centered | Equal-budget raw | Frozen centered residual2 |','|---|---:|---:|---:|---:|']
    for r in summaries:
        m=r['mean_map'];lines.append(f"| {r['metric_id']} | {m[primary]:.3f} | {m['old_centered']:.3f} | {m[raw]:.3f} | {m['centered_residual_2']:.3f} |")
    lines+=['',f"Primary macro mAP: {metric_values['macro_map']:.6f}%; strong gate: {bool(gate)}.",'',result['interval_scope'],f"Full independent audit: {rows_checked} new method-query ranking rows, maximum AP error {max_error:.3g}; all13variants and9metrics reconcile.",'','All raw-grid and fixed-regularizer values remain in outputs/all_metrics.json; none may be substituted as the primary after evaluation. Runtime and full inputs are in outputs/manifest.json and completion.json.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('RESULT',json.dumps(result),flush=True);print('VALIDATION',json.dumps(validation),flush=True)

if __name__=='__main__':audit()
