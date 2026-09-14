"""A single predeclared score average, with inherited endpoint verification."""
from pathlib import Path
import json,sys,time,hashlib,shutil,datetime,importlib.util
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)
CFG=json.loads((HERE/'protocol.json').read_text())
file=ROOT/'experiments/analysis/scatter-centering-attribution-20260913/analysis.py'
spec=importlib.util.spec_from_file_location('fixed_geometry_attribution',file);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
m=a.m;PARENT=a.PARENT;NAMES=['stack','raw','blend','mean_cs_reference']

def dump(path,x):path.write_text(json.dumps(x,indent=2)+'\n')
def analyze():
    cells={};domains={};vectors={};pooled={}
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        aa=[]
        for gallery in [target,source]:
            name=target+'_'+gallery;z=np.load(OUT/(name+'.npz'));acc=(z['predictions']==z['yq']).mean(-1);assert np.array_equal(acc,z['accuracy']);seeds=z['seeds'];aa.append(acc)
            cells[name]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'comparisons':{NAMES[j]:a.interval(acc[2]-acc[j],seeds) for j in [0,1,3]}}
        acc=np.mean(aa,0);domains[target]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'comparisons':{}}
        for j in [0,1,3]:
            delta=acc[2]-acc[j];domains[target]['comparisons'][NAMES[j]]=a.interval(delta,seeds);vectors[target+'/'+NAMES[j]]=(delta,seeds)
    for ref in ['stack','raw','mean_cs_reference']:
        delta=np.concatenate([vectors[d+'/'+ref][0] for d in ['dtd','eurosat']]);groups=np.concatenate([vectors[d+'/'+ref][1]+i*100000000 for i,d in enumerate(['dtd','eurosat'])]);pooled[ref]=a.interval(delta,groups)
    gates={}
    for ref,floor in [('stack',.5),('raw',0)]:
        tests={'pooled_gain':pooled[ref]['delta_pp']>=floor,'pooled_ci':pooled[ref]['ci95_pp'][0]>0,'domains':all(d['comparisons'][ref]['delta_pp']>=0 for d in domains.values()),'allcells':all(c['comparisons'][ref]['ci95_pp'][0]>=-.5 for c in cells.values())}
        gates[ref]={'passed':all(tests.values()),'tests':tests}
    return {'scope':CFG['scope'],'cells':cells,'domains':domains,'pooled':pooled,'gates':gates,'promotion':all(x['passed'] for x in gates.values())}

def main():
    start=time.time();torch.set_num_threads(6)
    for lockpath in [HERE/'code_lock.json',PARENT/'code_lock.json',a.HERE/'code_lock.json']:
        assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in json.loads(lockpath.read_text()).items())
    data=m.old.load();gm={d:data[d]['gallery'].double().mean(-2) for d in CFG['domains']};checks=[];count=0
    dump(OUT/'manifest.json',{'argv':[sys.executable,*sys.argv],'config':CFG,'torch':torch.__version__,'numpy':np.__version__,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        t=m.old.sampler.tasks(data[target]['ident'],1,'eval');gamma=a.PCFG['gamma_by_source'][source]['1'];galleries=[target,source]
        old={g:np.load(PARENT/'outputs'/f'{source}_to_{target}_{g}_k1.npz') for g in galleries};preds={g:[] for g in galleries};samples={g:[] for g in galleries}
        for g,z in old.items():
            for key in t:assert np.array_equal(t[key],z[key])
        for i in range(500):
            assert shutil.disk_usage(HERE).free/2**30>=10
            assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['resources']['deadline'])
            sv=data[target]['query'][t['support_indices'][i]];qv=data[target]['query'][t['query_indices'][i].reshape(-1)];_,(s,q),factor=m.prepare(sv,qv,gamma)
            for g in galleries:
                sc,_=a.heads(s,q,m.old.metric.transform(gm[g],factor));raw=sc[0];stack=sc[3];blend=.5*raw+.5*stack;v=torch.stack([stack,raw,blend]);assert torch.isfinite(v).all();assert torch.allclose(v.sum(-1),torch.ones_like(v.sum(-1)),atol=1e-10,rtol=0)
                pred=v.argmax(-1).numpy().astype(np.uint8)
                for j,ref in [(0,'scatter_cs'),(1,'scatter_r2')]:
                    assert np.array_equal(pred[j],old[g]['predictions'][m.NAMES.index(ref),i]);count+=1
                preds[g].append(np.concatenate([pred,old[g]['predictions'][m.NAMES.index('mean_cs'),i][None]],0))
                if i in CFG['audit_indices']:
                    dense,_=a.independent(sv,qv,gm[g],gamma);ind=.5*(dense[0]+dense[3]);err=float(abs(blend.numpy()-ind).max());assert err<2e-5 and np.array_equal(pred[2],ind.argmax(-1))
                    checks.append({'domain':target,'gallery':g,'task':i,'max_error':err});samples[g].append(v.numpy())
            if i%100==0:
                x={'domain':target,'tasks_done':i+1,'elapsed_seconds':time.time()-start};dump(OUT/'progress.json',x);print('PROGRESS',json.dumps(x),flush=True)
        y=np.repeat(np.arange(5),15)
        for g in galleries:
            pred=np.stack(preds[g],1);np.savez_compressed(OUT/(target+'_'+g+'.npz'),predictions=pred,accuracy=(pred==y).mean(-1),names=NAMES,yq=y,sample_scores=np.stack(samples[g],1),sample_indices=CFG['audit_indices'],**t);print('CELL_COMPLETE',target,g,flush=True)
    result=analyze();dump(OUT/'analysis.json',result);dump(OUT/'validation.json',{'status':'passed','parent_endpoint_task_checks':count,'independent_blend_score_checks':len(checks),'max_score_error':max(x['max_error'] for x in checks),'score_sum_one':True,'checks':checks,'statistics':'same already-independently-audited pairedbootstrap helper; all means recomputed from fullprediction arrays'})
    dump(OUT/'complete.json',{'status':'completed','elapsed_seconds':time.time()-start,'task_conditions':2000,'method_count':4,'promotion':result['promotion']});print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
