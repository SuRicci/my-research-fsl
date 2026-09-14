"""Independent output/contract audit and paired statistics; never changes predictions."""
from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def paired(arrs,seed=26091299,reps=5000):
    """One array per query domain; shared episode indices across its two galleries."""
    rng=np.random.RandomState(seed);samples=np.zeros(reps)
    for a in arrs:
        assert a.shape==(1000,)
        for s in range(5):
            block=a[s*200:(s+1)*200]
            samples+=block[rng.randint(0,200,size=(reps,200))].mean(1)/(5*len(arrs))
    return {'delta_pp':float(np.mean([a.mean() for a in arrs])),
        'ci95_pp':np.quantile(samples,[.025,.975]).tolist(),
        'seed_delta_pp':[float(np.mean([a[s*200:(s+1)*200].mean() for a in arrs])) for s in range(5)]}

def main():
    cfg=json.loads((HERE/'protocol.json').read_text());sel=json.loads((HERE/'selection.json').read_text())
    manifest=json.loads((OUT/'run_manifest.json').read_text());complete=json.loads((OUT/'completion.json').read_text())
    baseline=json.loads(Path(cfg['baseline_contract_path']).read_text())
    assert complete['status']=='completed' and complete['selection_sha256']==sha(HERE/'selection.json')
    assert manifest['config']==cfg and manifest['baseline_contract_sha256']==sha(Path(cfg['baseline_contract_path']))
    assert manifest['code_sha256']=={p.name:sha(p) for p in HERE.glob('*.py')}
    assert sel['code_sha256']==manifest['code_sha256'] and sel['protocol_sha256']==sha(HERE/'protocol.json')
    required=set(baseline['metrics_summary']);files=sorted((OUT/'eval').glob('*.npz'));assert len(files)==6
    data={};rows=[];metrics={};ids={};baseline_delta={};counts={}
    reference=HERE.parent/'r2-residual-canonical-20260912/outputs/baseline'
    for p in files:
        name,gallery,shotstr=p.stem.split('_');shot=int(shotstr[1:]);z=np.load(p);names=z['names'].tolist()
        assert z['accuracy'].shape==(len(names),1000) and z['scores'].shape==(len(names),1000,75,5)
        assert np.isfinite(z['scores']).all() and np.array_equal(z['predictions'],z['scores'].argmax(-1))
        assert np.allclose(z['accuracy'],(z['predictions']==z['yq']).mean(-1),rtol=0,atol=0)
        assert np.array_equal(z['seeds'],np.repeat(cfg['eval']['seeds'],200))
        ref=np.load(reference/p.name);ri=ref['names'].tolist().index('r2')
        for field in ['support_indices','query_indices','class_ids','seeds']:assert np.array_equal(z[field],ref[field]),(p,field)
        delta=float(np.max(np.abs(z['accuracy'][names.index('r2')]-ref['accuracy'][ri])))
        assert delta==0,(p,delta)
        key=name+('_5shot_accuracy' if shot==5 else '_'+('matched' if name==gallery else 'mismatched')+'_1shot_accuracy')
        baseline_delta[key]=float(100*z['accuracy'][names.index('r2')].mean()-baseline['metrics_summary'][key])
        assert abs(baseline_delta[key])<1e-10
        if shot==5:
            assert all(np.array_equal(x,z['scores'][names.index('r2')]) for x in z['scores'])
        else:
            assert z['shared_neighbor_indices'].shape==(1000,5,64)
            orig=HERE.parent/'r2-ownership-canonical-20260912/outputs/eval'/p.name
            own=np.load(orig);assert np.array_equal(z['shared_neighbor_indices'],own['neighbor_indices'])
            assert np.array_equal(z['gallery_pool_indices'],own['gallery_pool_indices'])
        data[p.stem]={m:100*z['accuracy'][i] for i,m in enumerate(names)}
        for i,m in enumerate(names):
            metrics.setdefault(m,{})[key]=float(100*z['accuracy'][i].mean())
            rows.append({'cell':p.stem,'metric_id':key,'method':m,'value':metrics[m][key],'inherited':shot==5})
        counts[p.stem]=len(z['seeds']);ids[p.stem]=sha(p)
    for m,mm in metrics.items():mm['macro_1shot_accuracy']=float(np.mean([v for k,v in mm.items() if '_1shot_' in k]));assert set(mm)==required
    baseline_delta['macro_1shot_accuracy']=metrics['r2']['macro_1shot_accuracy']-baseline['metrics_summary']['macro_1shot_accuracy']
    assert abs(baseline_delta['macro_1shot_accuracy'])<1e-10
    # Validate development summaries and exact ordered selector against complete saved arrays.
    weights=cfg['weights'];lambdas=cfg['lambda_grid'];settings=[(w,l) for w in weights for l in lambdas]
    devs=[np.load(p) for p in sorted((OUT/'dev').glob('*.npz'))];assert len(devs)==2
    for z in devs:
        assert z['accuracy'].shape[1]==200 and np.isfinite(z['scores']).all()
        assert np.array_equal(z['predictions'],z['scores'].argmax(-1))
        assert np.array_equal(z['accuracy'],(z['predictions']==z['yq']).mean(-1))
    dn=devs[0]['names'].tolist();assert dn==devs[1]['names'].tolist()
    dm={m:float(np.mean([z['accuracy'][i].mean() for z in devs])) for i,m in enumerate(dn)}
    for method,pair in sel['settings'].items():
        family='early_native' if method in ['clip_native','dino_native'] else method
        candidates=[(1. if method=='clip_native' else 0.,l) for l in lambdas] if method in ['clip_native','dino_native'] else settings
        winner=max(candidates,key=lambda c:dm[family+'_'+str(c[0])+'_'+str(c[1])])
        assert tuple(pair)==winner,(method,pair,winner)
    comparisons={}
    for other in metrics:
        if other=='late_shared':continue
        cells={};domains=[]
        for name in ['dtd','eurosat']:
            diffs=[]
            for gallery in ['dtd','eurosat']:
                cell=name+'_'+gallery+'_k1';a=data[cell]['late_shared']-data[cell][other]
                cells[cell]=paired([a]);diffs.append(a)
            domains.append(np.mean(diffs,axis=0))
        comparisons[other]={'macro':paired(domains),'cells':cells}
    r2=comparisons['r2'];best=sel['strongest_control'];w=sel['settings']['late_shared'][0]
    gate={'macro_at_least_half_pp':r2['macro']['delta_pp']>=.5,'macro_ci_positive':r2['macro']['ci95_pp'][0]>0,
       'no_cell_loss_over_half_pp':all(x['delta_pp']>=-.5 for x in r2['cells'].values()),
       'dev_selected_control':best,'control_ci_positive':comparisons[best]['macro']['ci95_pp'][0]>0,'interior_weight':0<w<1}
    gate['pass']=all(v for v in gate.values() if isinstance(v,bool))
    report={'status':'validated','task_count':sum(counts.values()),'tasks_by_cell':counts,'development_task_count':400,
       'feature_and_measurement_hashes_match':True,'all_task_ids_match_baseline':True,'all_shared_neighbors_match_original_r2':True,
       'baseline_reconciliation_pp':baseline_delta,'metric_count':len(required),'npz_sha256':ids,
       'five_shot':'2000 inherited R2 task rows per method; no fusion mechanism claim',
       'scope':'same historically seen image pools; paired task bootstrap stratified by seed, galleries paired within query domain; conditional intervals',
       'selection':sel['settings'],'strong_gate':gate,'comparisons':comparisons,'elapsed_seconds':complete['elapsed_seconds']}
    dump(OUT/'metrics_summary.json',metrics['late_shared']);dump(OUT/'all_metrics.json',metrics)
    dump(OUT/'metric_rows.json',[r for r in rows if r['method']=='late_shared']+[{'metric_id':'macro_1shot_accuracy','value':metrics['late_shared']['macro_1shot_accuracy'],'method':'late_shared','inherited':False}])
    dump(OUT/'validation_report.json',report)
    lines=['# Fusion-stage paired result','',f"Strong gate: {'PASS' if gate['pass'] else 'FAIL'}. Development-selected comparator: {best}.",'',
      '| Method | DTD matched | DTD mismatch | EuroSAT mismatch | EuroSAT matched | 1-shot macro | DTD 5-shot* | EuroSAT 5-shot* |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    order=['dtd_matched_1shot_accuracy','dtd_mismatched_1shot_accuracy','eurosat_mismatched_1shot_accuracy','eurosat_matched_1shot_accuracy','macro_1shot_accuracy','dtd_5shot_accuracy','eurosat_5shot_accuracy']
    for m,mm in metrics.items():lines.append('| '+m+' | '+' | '.join(f'{mm[k]:.4f}' for k in order)+' |')
    lines+=['','*Five-shot rows inherit R2 exactly and are not evidence for the fusion mechanism.','',
       '| Late_shared minus comparator | Macro delta, pp | 95% paired CI, pp |','|---|---:|---|']
    for m,c in comparisons.items():
        x=c['macro'];lo,hi=x['ci95_pp'];lines.append(f"| {m} | {x['delta_pp']:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    lines+=['','## Gate and limitations','',json.dumps(gate,indent=2),'',report['scope'],
       'The primary method was selected before evaluation. Reported comparator differences do not authorize post-hoc primary replacement. Fusion is an established technique; no new algorithm or independent-domain validation is implied.']
    (HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'validated':True,'task_count':sum(counts.values()),'gate':gate,'metrics':metrics['late_shared'],'macro_comparisons':{m:c['macro'] for m,c in comparisons.items()}}),flush=True)
if __name__=='__main__':main()
