"""Pre-registered auxiliary source qualification; target outcomes never tune."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
import metric
from extract_views import CFG,ASSET,OUT as ASSETS,HERE,sha,dump,guard
sys.path.insert(0,str(HERE.parent/'utility-composition-qualification-20260913'))
import run as sampler
sampler.CFG=CFG
OUT=HERE/'outputs';EXTRA=json.loads((HERE/'evaluation_contract.json').read_text());METHODS=EXTRA['methods']

def load():
    manifest=json.loads((ASSETS/'feature_manifest.json').read_text());assert manifest['status']=='completed'
    data={};original_manifest=json.loads((ASSET/'manifest.json').read_text())
    for ds in CFG['source_domains']:
        ident=json.loads((ASSET/(ds+'_identities.json')).read_text());data[ds]={'ident':ident}
        for side in ['query','gallery']:
            blocks=[]
            for b in ['clip_vitb16','dinov2_vits14']:
                original=ASSET/(ds+'_'+b+'_'+side+'.pt');entry=manifest['files'][ds+'_'+side+'_'+b]
                assert sha(original)==original_manifest['datasets'][ds]['backbones'][b+'_'+side]['sha256']
                assert sha(entry['path'])==entry['sha256']
                a=torch.load(original,weights_only=True);z=torch.load(entry['path'],weights_only=True)
                assert a['ids'].tolist()==z['ids'].tolist()==ident[side+'_ids']
                blocks.append(torch.cat([a['features'][:,None],z['features']],1).float())
            data[ds][side]=metric.ref.representation(*blocks,.5)
    return data

def select(ds,shot,data):
    d=data[ds];t=sampler.tasks(d['ident'],shot,'selection');gy=np.array(d['ident']['gallery_labels']);scores=[];galleries=[]
    for i,cs in enumerate(t['class_ids']):
        guard();s=d['query'][t['support_indices'][i]];q=d['query'][t['query_indices'][i].reshape(-1)];parts=[];gis=[]
        for excluded in [False,True]:
            rng=np.random.RandomState(EXTRA['source_gallery_seed']+10000*excluded+i)
            pool=np.flatnonzero(~np.isin(gy,cs)) if excluded else np.arange(len(gy))
            gi=rng.choice(pool,EXTRA['source_gallery_size'],replace=False);gis.append(gi)
            if excluded:assert not np.isin(gy[gi],cs).any()
            parts.append(torch.stack([metric.scatter(s,q,d['gallery'][gi],g) for g in CFG['gamma_grid']]).numpy())
        scores.append(np.stack(parts));galleries.append(np.stack(gis))
    sc=np.stack(scores);y=np.repeat(np.arange(5),t['query_indices'].shape[-1]);correct=sc.argmax(-1)==y;acc=correct.mean(-1)
    counts=correct.sum((0,1,3));means=counts/(correct.shape[0]*correct.shape[1]*correct.shape[3])
    chosen=min(g for g,n in zip(CFG['gamma_grid'],counts) if n==counts.max())
    np.savez_compressed(OUT/(ds+'_k%d_selection.npz'%shot),scores=sc,accuracy=acc,gallery_indices=np.stack(galleries),**t)
    result=dict(gamma=chosen,gamma_grid=CFG['gamma_grid'],mean_accuracy=means.tolist(),selection_domain=ds,target_outcomes_used=False)
    dump(OUT/(ds+'_k%d_selection.json'%shot),result);print('SELECTION_FROZEN',ds,shot,result,flush=True)
    return chosen

def interval(delta,seeds):
    rng=np.random.RandomState(CFG['gate']['bootstrap_seed']);samples=np.zeros(CFG['gate']['bootstrap_replicates'])
    for seed in np.unique(seeds):
        d=delta[seeds==seed];samples+=d[rng.randint(len(d),size=(len(samples),len(d)))].mean(1)/len(np.unique(seeds))
    return {'delta_pp':float(delta.mean()*100),'ci95_pp':(np.quantile(samples,[.025,.975])*100).tolist()}

def analyze():
    cells={};directions={};passed=True
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        pair=[]
        for shot in CFG['shots']:
            for gallery in [target,source]:
                name=f'{source}_to_{target}_{gallery}_k{shot}';z=np.load(OUT/'cells'/(name+'.npz'));names=z['names'].tolist()
                assert np.array_equal(z['predictions'],z['scores'].argmax(-1))
                assert np.array_equal(z['accuracy'],(z['predictions']==z['yq']).mean(-1))
                a=z['accuracy'];candidate=a[names.index('scatter_r2')]
                diff={m:interval(candidate-a[names.index(m)],z['seeds']) for m in names if m!='scatter_r2'}
                if any(v['ci95_pp'][0]<-.5 for v in diff.values()):passed=False
                cells[name]={'accuracy_pct':dict(zip(names,(a.mean(-1)*100).tolist())),'comparisons':diff}
                if shot==1:pair.append((names,a,z['seeds']))
        assert pair[0][0]==pair[1][0] and np.array_equal(pair[0][2],pair[1][2])
        names=pair[0][0];a=(pair[0][1]+pair[1][1])/2;c=a[names.index('scatter_r2')]
        co={m:interval(c-a[names.index(m)],pair[0][2]) for m in names if m!='scatter_r2'}
        if any(v['delta_pp']<.5 or v['ci95_pp'][0]<=0 for v in co.values()):passed=False
        directions[source+'_to_'+target]=co
    result={'status':'passed' if passed else 'refuted_fixed_source_gate','cells':cells,'directions':directions,'task_conditions':4000,'method_count':len(METHODS),'scope':'auxiliary/dev; fixed pools; source-selected strengths; no new Pets/Caltech result'}
    dump(OUT/'analysis.json',result);print('QUALIFICATION_VERDICT',result['status'],flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','run'],required=True);a=ap.parse_args();torch.set_num_threads(6)
    sources=[HERE/x for x in ['protocol.json','evaluation_contract.json','metric.py','evaluate.py','verify.py']]+[Path(sampler.__file__),Path(metric.ref.__file__),Path(metric.geo.__file__),Path(metric.ref.ridge_scores.__code__.co_filename)]
    hashes={str(p):sha(p) for p in sources}
    if a.phase=='check':
        from verify import verify
        result=verify(sampler);dump(OUT/'metric_validation.json',result);dump(HERE/'evaluation_lock.json',hashes);print(json.dumps(result),flush=True);return
    guard();assert hashes==json.loads((HERE/'evaluation_lock.json').read_text());assert json.loads((OUT/'metric_validation.json').read_text())['status']=='passed'
    dump(OUT/'evaluation_manifest.json',dict(command=[sys.executable,*sys.argv],sources=hashes,config=CFG,clarification=EXTRA,torch=torch.__version__,numpy=np.__version__))
    data=load();choices={}
    # All source choices frozen before any reported evaluation cell.
    for source in CFG['source_domains']:
        for shot in CFG['shots']:choices[(source,shot)]=select(source,shot,data)
    (OUT/'cells').mkdir(exist_ok=True)
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for shot in CFG['shots']:
            t=sampler.tasks(data[target]['ident'],shot,'eval');y=np.repeat(np.arange(5),15)
            for gallery in [target,source]:
                guard();assert not set(data[target]['ident']['query_rgb']) & set(data[gallery]['ident']['gallery_rgb'])
                name=f'{source}_to_{target}_{gallery}_k{shot}';values=[]
                for i in range(len(t['seeds'])):
                    guard();s=data[target]['query'][t['support_indices'][i]];q=data[target]['query'][t['query_indices'][i].reshape(-1)];g=data[gallery]['gallery']
                    with torch.no_grad():
                        out=metric.controls(s,q,g);out['scatter_r2']=metric.scatter(s,q,g,choices[(source,shot)])
                    values.append(np.stack([out[n].numpy() for n in METHODS]))
                    if i%100==0:print('EVALUATION',name,i,flush=True)
                sc=np.stack(values,1);assert np.isfinite(sc).all();pred=sc.argmax(-1);acc=(pred==y).mean(-1)
                np.savez_compressed(OUT/'cells'/(name+'.npz'),scores=sc,predictions=pred,accuracy=acc,names=METHODS,yq=y,chosen_gamma=choices[(source,shot)],**t)
                print('CELL_COMPLETE',name,dict(zip(METHODS,(acc.mean(-1)*100).tolist())),flush=True)
    result=analyze();dump(OUT/'complete.json',{'status':'success','verdict':result['status'],'source_hashes':hashes})

if __name__=='__main__':main()
