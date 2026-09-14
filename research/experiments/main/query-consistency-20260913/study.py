"""Finite source-only selection followed by complete paired development evaluation."""
from pathlib import Path
import datetime,hashlib,json,sys,time,shutil
import numpy as np
import torch
import consistency_model as z
HERE=z.HERE;OUT=HERE/'outputs';ROOT=z.ROOT
CFG=json.loads((HERE/'protocol.json').read_text());NAMES=['incumbent','consistency','isotropic','query_view_mean'];Y=np.repeat(np.arange(5),15)
KEYS=['support_indices','query_indices','class_ids','seeds']
OLD=ROOT/'experiments/main/query-competence-20260913/outputs'
def dump(name,x):(OUT/name).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
    assert shutil.disk_usage(HERE).free/2**30>=10
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline'])
def save(name,sc,names,t,gi=None):
    assert np.isfinite(sc).all();pred=sc.argmax(-1).astype(np.uint8)
    assert np.allclose(sc.sum(-1),1,atol=1e-10,rtol=0)
    np.savez_compressed(OUT/(name+'.npz'),scores=sc,predictions=pred,accuracy=(pred==Y).mean(-1),names=names,yq=Y,gallery_indices=np.array([]) if gi is None else gi,**t)
def summarize():
    cells={};domains={};vectors={}
    for d in CFG['domains']:
        aa=[]
        for g in CFG['domains']:
            f=np.load(OUT/(d+'_'+g+'.npz'));acc=(f['predictions']==Y).mean(-1);aa.append(acc)
            cells[d+'_'+g]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),
                'comparisons':{NAMES[j]:z.a.interval(acc[1]-acc[j],f['seeds']) for j in [0,2,3]},
                'repairs':int(((f['predictions'][0]!=Y)&(f['predictions'][1]==Y)).sum()),
                'spoiled':int(((f['predictions'][0]==Y)&(f['predictions'][1]!=Y)).sum())}
        acc=np.mean(aa,0);domains[d]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'comparisons':{}}
        for j in [0,2,3]:
            delta=acc[1]-acc[j];domains[d]['comparisons'][NAMES[j]]=z.a.interval(delta,f['seeds']);vectors[d+'/'+NAMES[j]]=(delta,f['seeds'])
    pooled={}
    for name in [NAMES[j] for j in [0,2,3]]:
        delta=np.concatenate([vectors[d+'/'+name][0] for d in CFG['domains']]);groups=np.concatenate([vectors[d+'/'+name][1]+i*100000000 for i,d in enumerate(CFG['domains'])]);pooled[name]=z.a.interval(delta,groups)
    tests={'increment':pooled['incumbent']['delta_pp']>=.5,'positive_ci':pooled['incumbent']['ci95_pp'][0]>0,
       'domains':all(d['comparisons']['incumbent']['delta_pp']>=0 for d in domains.values()),
       'cells':all(c['comparisons']['incumbent']['ci95_pp'][0]>=-.5 for c in cells.values()),
       'controls':all(pooled[n]['ci95_pp'][0]>0 for n in ['isotropic','query_view_mean'])}
    return {'cells':cells,'domains':domains,'pooled':pooled,'pooled_accuracy_pct':{n:float(np.mean([d['accuracy_pct'][n] for d in domains.values()])) for n in NAMES},'gate':{'passed':all(tests.values()),'tests':tests},'scope':CFG['canonical_metric_deviation']+' Repeatedly exposed development; paired intervals conditional on fixed pools.'}

def main():
    start=time.time();torch.set_num_threads(6);guard();assert json.loads((OUT/'numeric_validation.json').read_text())['status']=='passed'
    lock=json.loads((HERE/'code_lock.json').read_text());assert all(sha(k)==v for k,v in lock.items())
    dump('manifest.json',{'argv':[sys.executable,*sys.argv],'config':CFG,'code_hashes':lock,'torch':torch.__version__,'numpy':np.__version__,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    data=z.m.old.load();gm={d:data[d]['gallery'].double().mean(-2) for d in CFG['domains']};choices={};source_results={};parity=0;error=0
    grid=CFG['eta_grid'];source_names=['consistency_'+str(e) for e in grid]+['isotropic_'+str(e) for e in grid]+['incumbent','query_view_mean']
    bank_hashes={}
    for source in CFG['domains']:
        counts=np.zeros(len(source_names),dtype=np.int64)
        for cond in ['ordinary','excluded']:
            name=source+'_selection_'+cond;p=OLD/(name+'.npz');bank_hashes[name]=sha(p);old=np.load(p);t={k:old[k] for k in KEYS};bank=[]
            for i in range(len(t['seeds'])):
                sv=data[source]['query'][t['support_indices'][i]];qv=data[source]['query'][t['query_indices'][i].reshape(-1)]
                sc=z.evaluate(sv,qv,gm[source][old['gallery_indices'][i]],CFG['gamma_by_source'][source],grid,grid)
                err=float(abs(sc['incumbent'].numpy()-old['scores'][2,i]).max());error=max(error,err);assert err<1e-10
                assert np.array_equal(sc['incumbent'].argmax(-1).numpy(),old['predictions'][2,i]);parity+=1
                vals=torch.cat([sc['consistency'],sc['isotropic'],sc['incumbent'][None],sc['query_view_mean'][None]]).numpy();bank.append(vals);counts+=(vals.argmax(-1)==Y).sum(-1)
            save(name,np.stack(bank,1),source_names,t,old['gallery_indices']);guard();print('SOURCE_BANK',name,flush=True)
        choices[source]={'consistency_eta':grid[int(counts[:4].argmax())],'isotropic_eta':grid[int(counts[4:8].argmax())]};source_results[source]={'correct_counts':dict(zip(source_names,counts.tolist())),'query_count_per_method':30000//2,'choice':choices[source]}
    dump('source_selection.json',source_results);dump('selection_lock.json',{'choices':choices,'source_bank_hashes':bank_hashes,'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target_evaluation_started':False});choice_hash=sha(OUT/'selection_lock.json')
    print('ALL_SELECTIONS_FROZEN',json.dumps(choices),flush=True)
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        t=z.m.old.sampler.tasks(data[target]['ident'],1,'eval');banks={g:[] for g in CFG['domains']};olds={g:np.load(OLD/(target+'_'+g+'.npz')) for g in CFG['domains']}
        for g,old in olds.items():
            assert all(np.array_equal(t[k],old[k]) for k in KEYS)
            assert not set(data[target]['ident']['query_rgb'])&set(data[g]['ident']['gallery_rgb'])
        for i in range(len(t['seeds'])):
            sv=data[target]['query'][t['support_indices'][i]];qv=data[target]['query'][t['query_indices'][i].reshape(-1)]
            for g in CFG['domains']:
                c=choices[source];sc=z.evaluate(sv,qv,gm[g],CFG['gamma_by_source'][source],[c['consistency_eta']],[c['isotropic_eta']])
                err=float(abs(sc['incumbent'].numpy()-olds[g]['scores'][2,i]).max());error=max(error,err);assert err<1e-10
                assert np.array_equal(sc['incumbent'].argmax(-1).numpy(),olds[g]['predictions'][2,i]);parity+=1
                banks[g].append(torch.stack([sc['incumbent'],sc['consistency'][0],sc['isotropic'][0],sc['query_view_mean']]).numpy())
            if i%100==0:
                guard();p={'target':target,'tasks':i+1,'seconds':time.time()-start};dump('progress.json',p);print('PROGRESS',json.dumps(p),flush=True)
        for g in CFG['domains']:save(target+'_'+g,np.stack(banks[g],1),NAMES,t)
    assert sha(OUT/'selection_lock.json')==choice_hash
    result=summarize();dump('analysis.json',result);dump('validation.json',{'status':'passed','parent_full_task_checks':parity,'max_parent_score_error':error,'selection_hash':choice_hash,'source_choices_unchanged':True,'independent_audit':'pending'})
    result_end={'status':'completed','elapsed_seconds':time.time()-start,'source_task_conditions':400,'target_task_conditions':2000,'gate':result['gate']['passed']};dump('complete.json',result_end)
    print('COMPLETE',json.dumps({'accuracy':result['pooled_accuracy_pct'],'pooled':result['pooled'],'gate':result['gate'],'execution':result_end}),flush=True)
if __name__=='__main__':main()
