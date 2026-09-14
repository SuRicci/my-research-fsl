"""Matched source training with common validation; evaluation occurs after all locks."""
from pathlib import Path
import argparse, hashlib, json, sys, time, shutil
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'learned-aggregation-order-20260913'))
import model
sys.path.insert(0,str(HERE.parent/'reduced-view-efficiency-20260913'))
import rv_study as rv
CFG=json.loads((HERE/'protocol.json').read_text())
OUT=HERE/'outputs'
ROOT=Path('/Users/decoqwq/DeepScientist/quests/012')
ASSET=ROOT/'baselines/local/r2-canonical/assets'
FLOWER=ROOT/'baselines/local/r2-source-diversity/assets/flowers102'
BANK=HERE.parent/'reduced-view-efficiency-20260913/outputs'
TRAINCFG=HERE.parent/'utility-composition-qualification-20260913/protocol.json'
rv.prior.sampler.CFG={**json.loads(TRAINCFG.read_text()),**rv.prior.CFG}
ARMS=CFG['arms']
MODE='map_after_mean'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2)+'\n')
def guard():
    assert shutil.disk_usage(HERE).free>=10*2**30,'disk floor'
    assert datetime.now(timezone.utc)<datetime.fromisoformat(CFG['resources']['deadline_utc']),'deadline'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<CFG['resources']['added_output_gib_max']*2**30,'output limit'
def load():
    audit=json.loads((FLOWER/'independent_audit.json').read_text())
    assert audit['status']=='passed' and sha(FLOWER/'feature_manifest.json')==audit['feature_manifest_sha256']
    eligible=json.loads((FLOWER/'eligible_source_rows.json').read_text())
    assert sha(FLOWER/'eligible_source_rows.json')==audit['eligibility_sha256']
    rows=np.array(eligible['rows']);labels=np.array(eligible['labels'])
    data={};inputs={}
    manifest=json.loads((ASSET/'manifest.json').read_text())
    for ds in ['dtd','eurosat','flowers']:
        blocks=[]
        ident=json.loads((ASSET/(ds+'_identities.json')).read_text()) if ds!='flowers' else None
        for b in ['clip_vitb16','dinov2_vits14']:
            p=FLOWER/(b+'.pt') if ds=='flowers' else ASSET/(ds+'_'+b+'_query.pt')
            h=sha(p);inputs[str(p)]=h
            expected=audit['feature_checks'][b]['sha256'] if ds=='flowers' else manifest['datasets'][ds]['backbones'][b+'_query']['sha256']
            assert h==expected,p
            z=torch.load(p,weights_only=True)
            if ds=='flowers':
                assert z['labels'][rows].tolist()==labels.tolist()
                x=z['features'][rows].float()
            else:
                assert z['ids'].tolist()==ident['query_ids']
                x=z['features'].float()
            assert torch.isfinite(x).all()
            blocks.append(x)
        x=rv.prior.metric.ref.representation(*blocks,.5)[:,None,:]
        data[ds]={'x':x,'labels':labels if ds=='flowers' else np.array(ident['query_labels']),'ident':ident}
    return data,inputs

def flower_tasks(data,k,phase):
    lab=data['flowers']['labels']
    cs=np.random.RandomState(CFG['flower_split_seed']).permutation(np.unique(lab))
    cs=cs[:CFG['flower_train_classes']] if phase=='train' else cs[CFG['flower_train_classes']:]
    c=json.loads(TRAINCFG.read_text());seed=c['train_seed'] if phase=='train' else c['selection_seed']
    rng=np.random.RandomState(seed+k);nq=15 if k==1 else 13
    s,q,y=[],[],[]
    for _ in range(100):
        chosen=rng.choice(sorted(cs),5,replace=False)
        ix=np.stack([rng.choice(np.flatnonzero(lab==v),k+nq,replace=False) for v in chosen])
        s.append(ix[:,:k]);q.append(ix[:,k:]);y.append(chosen)
    return dict(support_indices=np.array(s),query_indices=np.array(q),class_ids=np.array(y),seeds=np.full(100,seed))
def shift(t,offset,class_offset):
    return {k:v+(offset if k in ['support_indices','query_indices'] else class_offset if k=='class_ids' else 0) for k,v in t.items()}
def join(a,b): return {k:np.concatenate([a[k][:50],b[k][:50]]) for k in a}
def source_tasks(data,ds,k):
    offset=len(data[ds]['x']);co=int(data[ds]['labels'].max())+1
    a={p:rv.prior.sampler.tasks(data[ds]['ident'],k,p) for p in ['train','selection']}
    f={p:shift(flower_tasks(data,k,p),offset,co) for p in ['train','selection']}
    banks={'original':a['train'],'flowers':f['train'],'mixed':join(a['train'],f['train'])}
    va=join(a['selection'],f['selection'])
    x=torch.cat([data[ds]['x'],data['flowers']['x']])
    labels=np.r_[data[ds]['labels'],data['flowers']['labels']+co]
    return x,labels,banks,va,offset

def batch(x,t,ix):
    return x[t['support_indices'][ix]],x[t['query_indices'][ix].reshape(len(ix),-1)]
def yq(t): return np.repeat(np.arange(5),t['query_indices'].shape[-1])
def predict(net,x,t):
    values=[]
    with torch.no_grad():
        for lo in range(0,len(t['seeds']),CFG['batch_size']):
            s,q=batch(x,t,np.arange(lo,min(lo+CFG['batch_size'],len(t['seeds']))))
            values.append(model.scores(net,s,q,MODE,CFG['ridge_penalty']).numpy())
    return np.concatenate(values)
def inputs():
    paths=[HERE/'study.py',HERE/'verify_composition.py',HERE/'protocol.json',Path(model.__file__),Path(rv.__file__),Path(rv.prior.sampler.__file__),TRAINCFG,
           ROOT/'baselines/local/r2-pets-transfer/json/metric_contract.json',FLOWER/'independent_audit.json',FLOWER/'eligible_source_rows.json',FLOWER/'identities.json']
    paths += [Path(rv.prior.__file__),Path(rv.prior.metric.__file__),Path(rv.prior.metric.ref.__file__),Path(rv.prior.metric.ref.ridge_scores.__code__.co_filename),rv.PARENT/'protocol.json',rv.HERE/'protocol.json',rv.LINEAR,ASSET/'manifest.json',ASSET/'dtd_identities.json',ASSET/'eurosat_identities.json']
    paths+=list(BANK.glob('eval*task*.npz'))+list((BANK/'cells').glob('eval*_v00.npz'))
    return {str(p):sha(p) for p in paths}
def exposure(t,labels,offset):
    s=t['support_indices'];q=t['query_indices'];ids=np.unique(np.r_[s.ravel(),q.ravel()])
    result={}
    for name,ix in [('original',ids[ids<offset]),('flowers',ids[ids>=offset])]:
        result[name]={'unique_images':len(ix),'unique_classes':len(np.unique(labels[ix])),
            'support_presentations':int(((s<offset) if name=='original' else (s>=offset)).sum()*10),
            'query_presentations':int(((q<offset) if name=='original' else (q>=offset)).sum()*10)}
    return result

def train(data):
    choices={};exposures={}
    for ds in CFG['source_domains']:
        for k in CFG['shots']:
            x,labels,banks,va,offset=source_tasks(data,ds,k)
            prefix=f'{ds}_k{k}'
            np.savez_compressed(OUT/'tasks'/f'{prefix}_selection.npz',**va)
            exposures[prefix]={}
            for arm,t in banks.items():
                np.savez_compressed(OUT/'tasks'/f'{prefix}_{arm}.npz',**t)
                exposures[prefix][arm]=exposure(t,labels,offset)
            for seed in CFG['training_seeds']:
                rng=np.random.RandomState(seed)
                order=np.concatenate([rng.permutation(100) for _ in range(10)]).reshape(200,5)
                np.save(OUT/'tasks'/f'{prefix}_{seed}_order.npy',order)
                for arm,t in banks.items():
                    guard();torch.manual_seed(seed);net=model.Map(CFG['dimension'],CFG['rank'])
                    initial_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
                    opt=torch.optim.Adam(net.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
                    key=f'{prefix}_{arm}_{seed}';best=None;history=[];preds=[];audits=[];started=time.time()
                    model_path=OUT/'models'/f'{key}.pt'
                    for step in range(CFG['updates']+1):
                        if step:
                            s,q=batch(x,t,order[step-1])
                            sc=model.scores(net,s,q,MODE,CFG['ridge_penalty'])
                            loss=F.cross_entropy((CFG['logit_scale']*sc).flatten(0,1),
                                torch.from_numpy(np.tile(yq(t),len(s))))
                            assert torch.isfinite(loss)
                            opt.zero_grad();loss.backward()
                            assert all(torch.isfinite(p.grad).all() for p in net.parameters())
                            opt.step()
                        if step in CFG['checkpoints']:
                            guard();sc=predict(net,x,va);pred=sc.argmax(-1)
                            count=int((pred==yq(va)).sum())
                            ce=float(F.cross_entropy(torch.from_numpy(CFG['logit_scale']*sc).flatten(0,1),
                                torch.from_numpy(np.tile(yq(va),len(sc)))))
                            rank=(count,-ce,-step)
                            row=dict(step=step,correct=count,ce=ce)
                            history.append(row);preds.append(pred.astype(np.uint8));audits.append(sc[CFG['validation']['audit_selection_indices']])
                            if best is None or rank>best[0]:
                                torch.save(net.state_dict(),model_path);best=(rank,row)
                    np.savez_compressed(OUT/'selection'/f'{key}.npz',predictions=preds,audit_scores=audits,
                        audit_indices=CFG['validation']['audit_selection_indices'],yq=yq(va))
                    choices[key]=dict(**best[1],model_path=str(model_path),sha256=sha(model_path),history=history,
                        initial_state_sha256=initial_hash,source=ds,shot=k,arm=arm,seed=seed,seconds=time.time()-started)
                    dump(OUT/'training_progress.json',choices);dump(OUT/'exposure.json',exposures)
                    print('FIT_COMPLETE',key,'selected_step',best[1]['step'],'seconds',round(time.time()-started,2),flush=True)
    dump(OUT/'selection_lock.json',choices)
    return choices

def evaluate(data,choices):
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for k in CFG['shots']:
            t=dict(np.load(BANK/f'eval_{target}_k{k}_tasks.npz'))
            check=rv.prior.sampler.tasks(data[target]['ident'],k,'eval')
            assert all(np.array_equal(t[a],v) for a,v in check.items())
            pred=[];audit=[]
            for arm in ARMS:
                pp=[];aa=[]
                for seed in CFG['training_seeds']:
                    e=choices[f'{source}_k{k}_{arm}_{seed}'];assert sha(e['model_path'])==e['sha256']
                    net=model.Map(CFG['dimension'],CFG['rank']);net.load_state_dict(torch.load(e['model_path'],weights_only=True))
                    sc=predict(net,data[target]['x'],t);assert np.isfinite(sc).all()
                    pp.append(sc.argmax(-1).astype(np.uint8));aa.append(sc[CFG['validation']['audit_eval_indices']])
                pred.append(pp);audit.append(aa)
            net=model.Map(CFG['dimension'],CFG['rank']);sc0=predict(net,data[target]['x'],t)
            name=f'{source}_to_{target}_k{k}'
            np.savez_compressed(OUT/'cells'/f'{name}.npz',predictions=pred,audit_scores=audit,
                audit_indices=CFG['validation']['audit_eval_indices'],zero_predictions=sc0.argmax(-1).astype(np.uint8),
                zero_audit_scores=sc0[CFG['validation']['audit_eval_indices']],yq=yq(t),**t)
            print('EVAL_COMPLETE',name,flush=True)

def analyze():
    cells={};directions={};qualified=True;complement=True
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        refs1=[]
        for k in CFG['shots']:
            z=np.load(OUT/'cells'/f'{source}_to_{target}_k{k}.npz')
            a=(z['predictions']==z['yq']).mean(-1);mean=a.mean(1);candidate=mean[2]
            local={'original':mean[0],'flowers':mean[1],'zero_ridge01':(z['zero_predictions']==z['yq']).mean(-1)}
            per={}
            for gallery in [target,source]:
                p=BANK/'cells'/f'eval_{target}_k{k}_{gallery}_v00.npz'
                old=np.load(p);ref=(old['predictions']==z['yq']).mean(-1)
                controls={**local,**dict(zip(rv.HEADS,ref))}
                co={m:rv.interval(candidate-v,z['seeds']) for m,v in controls.items()}
                qualified &= all(v['ci95_pp'][0]>=-.5 for m,v in co.items() if m!='flowers')
                complement &= co['flowers']['ci95_pp'][0]>=-.5
                cells[f'{source}_to_{target}_{gallery}_k{k}']={'comparisons':co,'accuracy_pct':{**{arm:float(mean[j].mean()*100) for j,arm in enumerate(ARMS)},**{m:float(v.mean()*100) for m,v in controls.items()}}}
                per[gallery]=controls
                if k==1:refs1.append(controls)
            if k==1:
                controls={m:(refs1[0][m]+refs1[1][m])/2 for m in refs1[0]}
                co={m:rv.interval(candidate-v,z['seeds']) for m,v in controls.items()}
                seed_effect=(a[2]-a[0]).mean(-1)*100
                qualified &= all(v['delta_pp']>=.5 and v['ci95_pp'][0]>0 for m,v in co.items() if m!='flowers')
                qualified &= int((seed_effect>0).sum())>=2
                complement &= co['flowers']['delta_pp']>=.5 and co['flowers']['ci95_pp'][0]>0
                directions[source+'_to_'+target]={'comparisons':co,'mixed_minus_original_by_training_seed_pp':seed_effect.tolist()}
    r={'qualification_gate_passed':bool(qualified),'mixture_complementarity_gate_passed':bool(qualified and complement),
       'cells':cells,'directions':directions,'scope':CFG['scope'],'audit_pending':True}
    dump(OUT/'analysis.json',r)
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','run'],required=True);args=ap.parse_args()
    torch.set_num_threads(CFG['resources']['threads'])
    for d in ['tasks','models','selection','cells']:(OUT/d).mkdir(parents=True,exist_ok=True)
    guard();data,assets=load()
    if args.phase=='check':
        import verify_composition
        v=verify_composition.check(data);dump(OUT/'precheck.json',v)
        dump(HERE/'lock.json',{'code':inputs(),'assets':assets})
        dump(OUT/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':torch.get_num_threads()})
        print('PRECHECK',v,flush=True);return
    lock=json.loads((HERE/'lock.json').read_text());assert lock=={'code':inputs(),'assets':assets}
    assert json.loads((OUT/'precheck.json').read_text())['status']=='passed'
    assert not (OUT/'manifest.json').exists(),'real run exists; inspect rather than overwrite'
    dump(OUT/'manifest.json',{'config':CFG,'inputs':lock,'command':[sys.executable,*sys.argv]})
    started=time.time();choices=train(data);locked=sha(OUT/'selection_lock.json')
    evaluate(data,choices);result=analyze()
    assert locked==sha(OUT/'selection_lock.json') and lock=={'code':inputs(),'assets':assets}
    dump(OUT/'complete.json',{'status':'computed','elapsed_seconds':time.time()-started,
        'qualification_gate_passed':result['qualification_gate_passed'],'selection_lock_sha256':locked,'audit_pending':True})
    print('COMPUTE_COMPLETE',time.time()-started,flush=True)
if __name__=='__main__':main()
