"""Frozen auxiliary study: source selection is complete before target evaluation."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import model
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent/'representation-scatter-20260913'
BANK = HERE.parent/'view-distribution-kernel-20260913'/'outputs/cells'
sys.path.insert(0,str(PARENT))
import evaluate as prior
TRAIN_SOURCE = HERE.parent/'utility-composition-qualification-20260913'/'protocol.json'
prior.sampler.CFG = {**json.loads(TRAIN_SOURCE.read_text()), **prior.CFG}
sys.path.insert(0,str(HERE))
CFG = json.loads((HERE/'protocol.json').read_text())
OUT = HERE/'outputs'

def dump(name,value):
    p=OUT/name; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2)+'\n')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def guard():
    prior.guard()
    size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert size<CFG['resources']['added_output_gib_max']*2**30, size

def tasks(data,ds,k,role): return prior.sampler.tasks(data[ds]['ident'],k,role)

def hashes():
    paths=list(HERE.glob('*.py'))+[TRAIN_SOURCE,HERE/'protocol.json',PARENT/'evaluate.py',PARENT/'protocol.json',
        PARENT/'metric.py',PARENT/'extract_views.py',PARENT/'assets/feature_manifest.json',
        Path(prior.sampler.__file__),Path(prior.metric.ref.__file__),
        Path(prior.metric.ref.ridge_scores.__code__.co_filename),
        HERE.parents[2]/'baselines/local/r2-pets-transfer/json/metric_contract.json']
    paths+=sorted(BANK.glob('*.npz'))
    return {str(p):sha(p) for p in paths}

def batch(x,t,ids):
    s=x[t['support_indices'][ids]]
    ix=t['query_indices'][ids]
    q=x[ix.reshape(len(ids),-1)]
    return s,q

def label(t): return torch.arange(5).repeat_interleave(t['query_indices'].shape[-1])

def predict(net,x,t,mode):
    parts=[]
    with torch.no_grad():
        for lo in range(0,len(t['seeds']),CFG['batch_size']):
            s,q=batch(x,t,np.arange(lo,min(lo+CFG['batch_size'],len(t['seeds']))))
            parts.append(model.scores(net,s,q,mode,CFG['ridge_penalty']).numpy())
    return np.concatenate(parts)

def train(data):
    choices={}; t0=time.time()
    for ds in CFG['source_domains']:
        choices[ds]={}
        for k in CFG['shots']:
            tr,va=[tasks(data,ds,k,role) for role in ['train','selection']]
            for role,t in [('train',tr),('selection',va)]:
                np.savez_compressed(OUT/f'{ds}_k{k}_{role}_tasks.npz',**t)
            choices[ds][str(k)]={}
            x=data[ds]['query']
            for seed in CFG['training_seeds']:
                rng=np.random.RandomState(seed)
                order=np.concatenate([rng.permutation(len(tr['seeds'])) for _ in range(
                    CFG['updates']*CFG['batch_size']//len(tr['seeds']))]).reshape(CFG['updates'],-1)
                np.save(OUT/f'{ds}_k{k}_seed{seed}_order.npy',order)
                for mode in CFG['modes']:
                    guard(); torch.manual_seed(seed); net=model.Map(CFG['dimension'],CFG['rank'])
                    opt=torch.optim.Adam(net.parameters(),lr=CFG['learning_rate'],weight_decay=CFG['weight_decay'])
                    key=f'{ds}_k{k}_{mode}_seed{seed}'; best=None; history=[]; started=time.time()
                    for step in range(CFG['updates']+1):
                        if step:
                            s,q=batch(x,tr,order[step-1])
                            sc=model.scores(net,s,q,mode,CFG['ridge_penalty'])
                            loss=F.cross_entropy((CFG['logit_scale']*sc).flatten(0,1),label(tr).repeat(len(s)))
                            assert torch.isfinite(loss)
                            opt.zero_grad();loss.backward()
                            assert all(torch.isfinite(p.grad).all() for p in net.parameters())
                            opt.step()
                        if step in CFG['checkpoints']:
                            guard();sc=predict(net,x,va,mode);y=label(va).numpy()
                            count=int((sc.argmax(-1)==y).sum())
                            ce=float(F.cross_entropy(torch.from_numpy(CFG['logit_scale']*sc).flatten(0,1),
                                                     torch.from_numpy(np.tile(y,len(sc)))))
                            path=OUT/'models'/f'{key}_step{step}.pt'
                            torch.save(net.state_dict(),path)
                            np.savez_compressed(OUT/'selection'/f'{key}_step{step}.npz',scores=sc,yq=y)
                            rank=(count,-ce,-step)
                            row=dict(step=step,correct=count,ce=ce,model_path=str(path),sha256=sha(path))
                            history.append(row)
                            if best is None or rank>best[0]:best=(rank,row)
                            print('CHECKPOINT',key,step,count,ce,round(time.time()-started,2),flush=True)
                    choices[ds][str(k)][mode+'_seed'+str(seed)]=dict(
                        **best[1],mode=mode,seed=seed,history=history,training_seconds=time.time()-started,
                        source_order_sha256=sha(OUT/f'{ds}_k{k}_seed{seed}_order.npy'))
                    dump('training_progress.json',choices)
            print('SOURCE_SHOT_LOCKED',ds,k,round(time.time()-t0,2),flush=True)
    dump('selection_lock.json',choices)
    return choices

def evaluate(data,choices):
    timing=[]
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for k in CFG['shots']:
            guard();t=tasks(data,target,k,'eval');values=[]
            for mode in CFG['modes']:
                members=[]
                for seed in CFG['training_seeds']:
                    entry=choices[source][str(k)][mode+'_seed'+str(seed)]
                    assert sha(entry['model_path'])==entry['sha256']
                    net=model.Map(CFG['dimension'],CFG['rank'])
                    net.load_state_dict(torch.load(entry['model_path'],weights_only=True))
                    start=time.time();sc=predict(net,data[target]['query'],t,mode)
                    assert np.isfinite(sc).all();members.append(sc)
                    timing.append(dict(source=source,target=target,shot=k,mode=mode,seed=seed,
                                       seconds=time.time()-start,episodes=len(t['seeds'])))
                values.append(members)
            sc=np.asarray(values);y=label(t).numpy();pred=sc.argmax(-1)
            references={}
            for gallery in [target,source]:
                p=BANK/f'{source}_to_{target}_{gallery}_k{k}.npz'
                with np.load(p) as old:
                    for key,value in t.items():assert np.array_equal(value,old[key]),(p,key)
                    assert np.array_equal(old['yq'],y)
                    assert np.array_equal(old['predictions'],old['scores'].argmax(-1))
                    assert np.array_equal(old['accuracy'],(old['predictions']==y).mean(-1))
                    references[gallery]=dict(path=str(p),sha256=sha(p),names=old['names'].tolist())
            name=f'{source}_to_{target}_k{k}'
            np.savez_compressed(OUT/'cells'/f'{name}.npz',scores=sc,predictions=pred,
                accuracy=(pred==y).mean(-1),yq=y,modes=CFG['modes'],training_seeds=CFG['training_seeds'],**t)
            dump('references/'+name+'.json',references)
            print('EVAL_COMPLETE',name,((pred==y).mean((2,3))*100).tolist(),flush=True)
    dump('inference_timing.json',timing)

def analyze():
    cells={};directions={};passed=True
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        pairs=[]
        for k in CFG['shots']:
            name=f'{source}_to_{target}_k{k}'
            with np.load(OUT/'cells'/f'{name}.npz') as z:
                a=z['accuracy'];seeds=z['seeds'];candidate=a[0].mean(0)
                assert np.array_equal(z['predictions'],z['scores'].argmax(-1))
                assert np.array_equal(a,(z['predictions']==z['yq']).mean(-1))
                for gallery,entry in json.loads((OUT/'references'/f'{name}.json').read_text()).items():
                    assert sha(entry['path'])==entry['sha256']
                    with np.load(entry['path']) as old:
                        co={m:prior.interval(candidate-old['accuracy'][j],seeds) for j,m in enumerate(old['names'].tolist())}
                        co['map_after_mean']=prior.interval(candidate-a[1].mean(0),seeds)
                        passed &= all(v['ci95_pp'][0]>=-.5 for v in co.values())
                        cells[name+'_'+gallery]=dict(accuracy_pct=(a.mean(-1)*100).tolist(),comparisons=co)
                        if k==1:pairs.append((old['accuracy'].copy(),old['names'].tolist()))
                if k==1:
                    seed_effects=(a[0]-a[1]).mean(-1)*100
                    co={'map_after_mean':prior.interval(candidate-a[1].mean(0),seeds)}
                    for j,m in enumerate(pairs[0][1]):
                        co[m]=prior.interval(candidate-np.mean([p[0][j] for p in pairs],axis=0),seeds)
                    passed &= all(v['delta_pp']>=.5 and v['ci95_pp'][0]>0 for v in co.values())
                    passed &= int((seed_effects>0).sum())>=2
                    directions[source+'_to_'+target]=dict(comparisons=co,paired_training_seed_effect_pp=seed_effects.tolist())
    result=dict(metric_gate_passed=bool(passed),cells=cells,directions=directions,
        unique_eval_episodes=2000,training_seeds=CFG['training_seeds'],scope=CFG['scope'],
        uncertainty=CFG['bootstrap'],promotion_ready=False,independent_audit_pending=True)
    dump('analysis.json',result)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','run'],required=True);args=ap.parse_args()
    torch.set_num_threads(CFG['resources']['threads'])
    for d in ['models','selection','cells','references']:(OUT/d).mkdir(parents=True,exist_ok=True)
    guard();data=prior.load()
    if args.phase=='check':
        import verify_order
        result=verify_order.precheck(data)
        dump('validation.json',result)
        dump('environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
            threads=torch.get_num_threads(),device='cpu'))
        (HERE/'lock.json').write_text(json.dumps(hashes(),indent=2)+'\n')
        print('VALIDATION',json.dumps(result),flush=True);return
    assert hashes()==json.loads((HERE/'lock.json').read_text())
    assert json.loads((OUT/'validation.json').read_text())['status']=='passed'
    assert not (OUT/'manifest.json').exists(),'Do not overwrite a real run'
    start=time.time();dump('manifest.json',dict(config=CFG,source_hashes=hashes(),command=[sys.executable,*sys.argv]))
    choices=train(data);selection_sha=sha(OUT/'selection_lock.json')
    evaluate(data,choices);result=analyze()
    assert sha(OUT/'selection_lock.json')==selection_sha
    assert hashes()==json.loads((HERE/'lock.json').read_text())
    guard();dump('complete.json',dict(status='computed',elapsed_seconds=time.time()-start,
        metric_gate_passed=result['metric_gate_passed'],selection_sha256=selection_sha,independent_audit_pending=True))
    print('COMPUTE_COMPLETE',result['metric_gate_passed'],time.time()-start,flush=True)

if __name__=='__main__':main()
