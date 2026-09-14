"""Exact empirical source-group objectives on immutable one-shot task banks."""
from pathlib import Path
import argparse, copy, hashlib, json, shutil, sys, time
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'source-composition-20260913'))
import study as old
model=old.model
CFG=json.loads((HERE/'protocol.json').read_text())
OUT=HERE/'outputs'
OBJECTIVES=CFG['objectives']; SELECTORS=CFG['selectors']
Y=np.repeat(np.arange(5),15)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,z):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(z,indent=2)+'\n')
def guard():
    assert shutil.disk_usage(HERE).free>=CFG['resources']['min_free_gib']*2**30,'disk floor'
    assert datetime.now(timezone.utc)<datetime.fromisoformat(CFG['resources']['deadline']),'deadline'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<CFG['resources']['max_output_mib']*2**20,'output budget'
def ce_groups(scores):
    losses=F.cross_entropy((CFG['logit_scale']*scores).flatten(0,1),
                          torch.as_tensor(np.tile(Y,len(scores))),reduction='none')
    return losses.reshape(2,-1).mean(1)
def objective(losses,reference,name):
    if name=='mean': return losses.mean()
    if name=='raw_max': return losses.max()
    if name=='reference_max': return (losses-reference).max()
    raise ValueError(name)
def get_source(data,ds):
    x,labels,banks,va,offset=old.source_tasks(data,ds,1)
    tr=banks['mixed']
    for name,t in [('mixed',tr),('selection',va)]:
        saved=dict(np.load(old.OUT/'tasks'/f'{ds}_k1_{name}.npz'))
        assert all(np.array_equal(t[k],saved[k]) for k in t)
        assert np.array_equal(labels[t['support_indices']],t['class_ids'][:,:,None])
        assert np.array_equal(labels[t['query_indices']],np.repeat(t['class_ids'][:,:,None],15,axis=2))
    assert not np.intersect1d(np.r_[tr['support_indices'].ravel(),tr['query_indices'].ravel()],
                              np.r_[va['support_indices'].ravel(),va['query_indices'].ravel()]).size
    return x,tr,va
def scores(net,x,t):
    return model.scores(net,*old.batch(x,t,np.arange(len(t['seeds']))),'map_after_mean',CFG['penalty'])
def identity(x,t):
    net=model.Map(CFG['dimension'],CFG['rank'])
    with torch.no_grad(): return scores(net,x,t)
def immutable():
    paths=[HERE/'study.py',HERE/'audit.py',HERE/'protocol.json',
           old.OUT/'tasks/dtd_k1_mixed.npz',old.OUT/'tasks/dtd_k1_selection.npz',
           old.OUT/'tasks/eurosat_k1_mixed.npz',old.OUT/'tasks/eurosat_k1_selection.npz']
    paths += list((HERE.parent/'query-consistency-20260913/outputs').glob('dtd_*.npz'))
    paths += list((HERE.parent/'query-consistency-20260913/outputs').glob('eurosat_*.npz'))
    return {**old.inputs(),**{str(p):sha(p) for p in paths}}
def interval(delta,seeds):
    rng=np.random.RandomState(CFG['bootstrap']['seed'])
    sampled=np.zeros(CFG['bootstrap']['replicates'])
    for seed in np.unique(seeds):
        ix=np.flatnonzero(seeds==seed)
        sampled+=delta[rng.choice(ix,(len(sampled),len(ix)))].sum(1)/len(delta)
    return {'delta_pp':float(delta.mean()*100),'ci95_pp':(np.percentile(sampled,[2.5,97.5])*100).tolist()}
def check(data):
    import importlib.util
    spec=importlib.util.spec_from_file_location("source_group_audit",HERE/"audit.py")
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    result=audit.check_math(data)
    dump(OUT/'precheck.json',result)
    dump(HERE/'lock.json',{'files':immutable(),'assets':old.load()[1]})
    print('PRECHECK',result,flush=True)
def train(data):
    choices={};started=time.time()
    for ds in CFG['source_domains']:
        x,tr,va=get_source(data,ds)
        with torch.no_grad():
            train_ref=ce_groups(identity(x,tr))
            val_ref=ce_groups(identity(x,va))
        for seed in CFG['seeds']:
            for arm in OBJECTIVES:
                guard();torch.manual_seed(seed);net=model.Map(CFG['dimension'],CFG['rank'])
                opt=torch.optim.Adam(net.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
                history=[];selected={};cache={};key=f'{ds}_{seed}_{arm}';begin=time.time()
                for step in range(CFG['updates']+1):
                    if step:
                        loss=objective(ce_groups(scores(net,x,tr)),train_ref,arm)
                        assert torch.isfinite(loss)
                        opt.zero_grad();loss.backward()
                        assert all(torch.isfinite(p.grad).all() for p in net.parameters())
                        opt.step()
                    if step in CFG['checkpoints']:
                        guard()
                        with torch.no_grad():
                            sc=scores(net,x,va);lg=ce_groups(sc);pred=sc.argmax(-1)
                            train_lg=ce_groups(scores(net,x,tr))
                        vals=[float(lg.mean()),float((lg-val_ref).max())]
                        row={'step':step,'group_ce':lg.tolist(),'group_correct':(pred==torch.as_tensor(Y)).reshape(2,-1).sum(1).tolist(),
                             'train_group_ce':train_lg.tolist(),'objective':float(objective(train_lg,train_ref,arm))}
                        history.append(row)
                        for sel,value in zip(SELECTORS,vals):
                            if sel not in selected or (value,step)<(selected[sel]['value'],selected[sel]['step']):
                                selected[sel]={'value':value,'step':step,'selection_predictions':pred.numpy().astype(np.uint8)}
                                cache[sel]=copy.deepcopy(net.state_dict())
                saved={}
                for sel in SELECTORS:
                    step=selected[sel]['step'];mp=OUT/'models'/f'{key}_step{step}.pt'
                    if not mp.exists():torch.save(cache[sel],mp)
                    sp=OUT/'models'/f'{key}_{sel}_selection.npz'
                    np.savez_compressed(sp,predictions=selected[sel].pop('selection_predictions'))
                    saved[sel]={**selected[sel],'path':str(mp),'sha256':sha(mp),'selection_path':str(sp)}
                choices[key]={'source':ds,'seed':seed,'objective':arm,'history':history,'selected':saved,
                              'train_identity_ce':train_ref.tolist(),'selection_identity_ce':val_ref.tolist(),'seconds':time.time()-begin}
                dump(OUT/'training_progress.json',choices)
                print('FIT_COMPLETE',key,{s:saved[s]['step'] for s in SELECTORS},round(time.time()-begin,2),flush=True)
    dump(OUT/'selection_lock.json',choices)
    print('ALL_CHOICES_LOCKED',len(choices),'seconds',round(time.time()-started,2),flush=True)
    return choices
def evaluate(data,choices):
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        guard();t=dict(np.load(old.BANK/f'eval_{target}_k1_tasks.npz'))
        pred=np.empty((3,2,3,500,75),dtype=np.uint8);audit=np.empty((3,2,3,5,75,5),dtype=np.float32)
        x=data[target]['x']
        for ia,arm in enumerate(OBJECTIVES):
            for isel,sel in enumerate(SELECTORS):
                for iz,seed in enumerate(CFG['seeds']):
                    e=choices[f'{source}_{seed}_{arm}']['selected'][sel]
                    assert sha(e['path'])==e['sha256']
                    net=model.Map(CFG['dimension'],CFG['rank']);net.load_state_dict(torch.load(e['path'],weights_only=True))
                    # Small chunks keep evaluation memory bounded; training risks remain full-bank.
                    sc=old.predict(net,x,t);pred[ia,isel,iz]=sc.argmax(-1)
                    audit[ia,isel,iz]=sc[CFG['audit_indices']]
        z=identity(x,t).detach().numpy()
        np.savez_compressed(OUT/f'{source}_to_{target}.npz',predictions=pred,audit_scores=audit,
            identity_predictions=z.argmax(-1).astype(np.uint8),yq=Y,**t)
        print('EVAL_COMPLETE',source,target,flush=True)
def analyze():
    results={};gate=True;summary=[]
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        z=np.load(OUT/f'{source}_to_{target}.npz')
        a=(z['predictions']==Y).mean(-1);mean=a.mean(2);zero=(z['identity_predictions']==Y).mean(-1)
        strong=[];scatter=[]
        for gallery in ['dtd','eurosat']:
            p=HERE.parent/'query-consistency-20260913/outputs'/f'{target}_{gallery}.npz'
            ref=np.load(p)
            for k in ['support_indices','query_indices','class_ids','seeds']:assert np.array_equal(ref[k],z[k]),k
            names=ref['names'].tolist()
            strong.append((ref['predictions'][names.index('consistency')]==Y).mean(-1))
            scatter.append((ref['predictions'][names.index('incumbent')]==Y).mean(-1))
        controls={'identity':zero,'strong_stack_macro':np.mean(strong,axis=0),'scatter_macro':np.mean(scatter,axis=0)}
        for gallery in [target,source]:
            ref=np.load(old.BANK/'cells'/f'eval_{target}_k1_{gallery}_v00.npz')
            taskref=np.load(old.BANK/f'eval_{target}_k1_tasks.npz')
            for k in ['support_indices','query_indices','class_ids','seeds']:assert np.array_equal(taskref[k],z[k]),k
            for name,prediction in zip(old.rv.HEADS,ref['predictions']):
                controls[f'{name}_{gallery}']=(prediction==Y).mean(-1)
        arms={}
        for ia,arm in enumerate(OBJECTIVES):
            for isel,sel in enumerate(SELECTORS):
                armcontrols={**controls,**{f'matched_{name}':mean[j,isel] for j,name in enumerate(OBJECTIVES) if j!=ia}}
                arms[arm+'/'+sel]={'accuracy_pct':float(mean[ia,isel].mean()*100),
                                  'selected_steps':[json.loads((OUT/'selection_lock.json').read_text())[f'{source}_{seed}_{arm}']['selected'][sel]['step'] for seed in CFG['seeds']],
                                  'comparisons':{n:interval(mean[ia,isel]-v,z['seeds']) for n,v in armcontrols.items()}}
        primary=arms['reference_max/reference_max_ce']
        required=['identity','strong_stack_macro','matched_mean','matched_raw_max']
        byseed={name:((a[2,1]-a[j,1]).mean(-1)*100).tolist() for j,name in enumerate(OBJECTIVES[:2])}
        passed=all(primary['comparisons'][n]['delta_pp']>=.5 and primary['comparisons'][n]['ci95_pp'][0]>0 for n in required)
        passed &= all(sum(v>0 for v in values)>=2 for values in byseed.values())
        gate &= passed
        results[source+'_to_'+target]={'arms':arms,'controls_pct':{n:float(v.mean()*100) for n,v in controls.items()},
                                     'primary_seed_differences_pp':byseed,'gate_passed':bool(passed)}
        summary.append((source,target,primary['accuracy_pct'],{n:primary['comparisons'][n] for n in required}))
    r={'status':'computed','qualification_gate_passed':bool(gate),'directions':results,'audit_pending':True,
       'scope':'exposed-development, one-shot only; no Pets/Caltech outcomes','fit_count':18,
       'unique_eval_tasks':1000,'selector_objective_arms':6}
    dump(OUT/'analysis.json',r);print('RESULT',summary,'GATE',bool(gate),flush=True)
    return r
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','run'],required=True);args=ap.parse_args()
    torch.set_num_threads(CFG['resources']['threads'])
    (OUT/'models').mkdir(parents=True,exist_ok=True);guard();data,assets=old.load()
    if args.phase=='check':check(data);return
    lock=json.loads((HERE/'lock.json').read_text());assert lock=={'files':immutable(),'assets':assets}
    assert json.loads((OUT/'precheck.json').read_text())['status']=='passed'
    assert not (OUT/'manifest.json').exists(),'Existing run: inspect instead of overwrite'
    dump(OUT/'manifest.json',{'command':[sys.executable,*sys.argv],'config':CFG,'inputs':lock,'started_at':datetime.now(timezone.utc).isoformat(),
                            'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'threads':torch.get_num_threads(),'device':'cpu'}})
    start=time.time();choices=train(data);h=sha(OUT/'selection_lock.json')
    evaluate(data,choices);result=analyze()
    assert h==sha(OUT/'selection_lock.json') and lock=={'files':immutable(),'assets':assets}
    dump(OUT/'complete.json',{'status':'computed','elapsed_seconds':time.time()-start,'selection_lock_sha256':h,
                             'qualification_gate_passed':result['qualification_gate_passed'],'audit_pending':True})
    print('COMPUTE_COMPLETE',round(time.time()-start,2),flush=True)
if __name__=='__main__':main()
