"""Frozen canonical and prospective Pets evaluation, using existing assets."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse, hashlib, json, os, platform, shutil, sys, time
import numpy as np
import torch
import oslo
HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs'
CFG = json.loads((HERE/'protocol.json').read_text())
PARENT = Path(CFG['parent_root'])
sys.path.insert(0, str(PARENT))
import evaluate as ref


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 2**20), b''): h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); os.replace(tmp, path)


def guard():
    assert shutil.disk_usage(HERE).free >= 10 * 2**30
    assert datetime.now(timezone.utc) < datetime.fromisoformat(CFG['resources']['deadline_utc'])


def lock_check():
    lock = json.loads((HERE/'locked_sources.json').read_text())
    for name, expected in lock['files'].items(): assert sha(name) == expected, name
    return sha(HERE/'locked_sources.json')


def encode():
    sys.path.insert(0, str(Path('/Users/decoqwq/DeepScientist/quests/012/baselines/local/r2-replay')))
    import recover_gallery as encoder
    rows = json.loads(Path(CFG['fresh_identity_manifest']).read_text())['identities']
    weights = json.loads((PARENT/'outputs/asset_validation.json').read_text())['model_weights_sha256']
    for path, expected in weights.items(): assert sha(path) == expected
    previous = json.loads((PARENT/'assets/identities.json').read_text())
    assert len(rows) == 1804 and len({r['rgb'] for r in rows}) == len(rows)
    assert not {r['rgb'] for r in rows} & {r['rgb'] for side in ['query', 'gallery'] for r in previous[side]}
    assert torch.backends.mps.is_available()
    manifest = {'identity_sha256': sha(CFG['fresh_identity_manifest']), 'files': {}, 'weights_sha256': weights,
                'precision': 'MPS float32; normalized float16 storage', 'scope': CFG['scope_boundary']}
    for index, backbone in enumerate(encoder.BACKBONES):
        guard(); model = encoder.model(backbone)
        tf = encoder.T.Compose([encoder.T.Resize(224, interpolation=encoder.T.InterpolationMode.BICUBIC), encoder.T.CenterCrop(224), encoder.T.ToTensor(), encoder.T.Normalize(*encoder.NORM[index])])
        chunks = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            for start in range(0, len(rows), 32):
                guard(); path = HERE/'assets/chunks'/backbone/('%06d.pt' % start); path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists(): x = torch.load(path, weights_only=True)
                else:
                    x = encoder.encode(model, backbone, [r['path'] for r in rows[start:start+32]], tf, pool).half()
                    tmp=path.with_suffix('.tmp');torch.save(x,tmp);os.replace(tmp,path)
                assert len(x) == min(32, len(rows)-start) and torch.isfinite(x).all()
                chunks.append(x)
                if start % 320 == 0: print('ENCODE', backbone, start, len(rows), flush=True)
        full = torch.cat(chunks)
        assert float((full.float().norm(dim=-1)-1).abs().max()) < .002
        path = HERE/'assets'/(backbone+'_fresh.pt');torch.save({'features':full,'ids':torch.arange(len(rows))},path)
        manifest['files'][backbone]={'path':str(path),'sha256':sha(path),'shape':list(full.shape)}
        del model;torch.mps.empty_cache()
    dump(HERE/'assets/feature_manifest.json',manifest)
    print('FRESH_FEATURES_COMPLETE',len(rows),flush=True)


def fresh_data():
    rows=json.loads(Path(CFG['fresh_identity_manifest']).read_text())['identities']
    manifest=json.loads((HERE/'assets/feature_manifest.json').read_text());blocks=[]
    assert manifest['identity_sha256']==sha(CFG['fresh_identity_manifest'])
    for b in ['clip_vitb16','dinov2_vits14']:
        r=manifest['files'][b];assert sha(r['path'])==r['sha256'];p=torch.load(r['path'],weights_only=True)
        assert p['ids'].tolist()==list(range(len(rows)));blocks.append(p['features'].float())
    return ref.representation(*blocks,.5),rows


def tasks(scope, rows, shot):
    path=OUT/(scope+'_tasks_k%d.npz'%shot)
    if scope=='canonical':
        with np.load(PARENT/'outputs'/('tasks_k%d.npz'%shot)) as p: result={k:p[k] for k in p.files}
    else:
        ref.ref.CFG={'eval':{'seeds':CFG['fresh_seeds'],'episodes_per_seed':CFG['episodes_per_seed']}}
        si,qi,cs,seeds=ref.ref.tasks({'query_labels':[r['label'] for r in rows]},'pets',shot,'eval')
        result=dict(support_indices=si,query_indices=qi,class_ids=cs,seeds=seeds)
    if path.exists():
        with np.load(path) as p:
            for k,v in result.items():assert np.array_equal(p[k],v)
    else: np.savez_compressed(path,**result)
    return result


def evaluate():
    data, old_rows, _=ref.load_data();fresh,rows=fresh_data();results={}
    for scope,X,ident in [('canonical',data['query'],old_rows['query']),('fresh',fresh,rows)]:
        for shot in CFG['shots']:
            t=tasks(scope,ident,shot);si,qi=t['support_indices'],t['query_indices']
            S=X[si];Q=X[qi.reshape(len(qi),75)];pure={}
            if scope=='fresh':
                for C in [1,10]:
                    pure['support_logistic_C%d'%C]=np.stack([ref.logistic(s.flatten(0,1).numpy(),q.numpy(),C) for s,q in zip(S,Q)])
                print('FRESH_SUPPORT_HEADS',shot,flush=True)
            for gallery in CFG['gallery_conditions']:
                guard();cell=scope+'_'+gallery+'_k%d'%shot;target=OUT/'cells'/(cell+'.npz')
                if target.exists():
                    with np.load(target) as p:assert str(p['signature'])==lock_check()
                    continue
                started=time.time();G=data['gallery' if gallery=='pets' else 'dtd'];names=CFG['methods'];scores={m:[] for m in names[:3]};stats={m:{k:[] for k in ['effective_mass','inlier_mean','prototype_movement']} for m in names[:3]}
                for start in range(0,len(S),CFG['batch_size']):
                    guard();s=S[start:start+CFG['batch_size']];q=Q[start:start+CFG['batch_size']]
                    for name,steps,enabled in [('OSLO_G',2,True),('closed_set',2,False),('zero_update',0,True)]:
                        state,st=oslo.fit(s,G,steps=steps,lambda_s=CFG['oslo']['lambda_s'],lambda_z=CFG['oslo']['lambda_z'],use_inlier=enabled)
                        scores[name].append(oslo.predict(state,q).numpy())
                        for key,value in st.items():stats[name][key].append(value.numpy())
                    if (start+len(s))%128==0: print('EVAL_PROGRESS',cell,start+len(s),len(S),flush=True)
                scores={m:np.concatenate(v) for m,v in scores.items()}
                if scope=='canonical':
                    with np.load(PARENT/'outputs/baseline'/('pets_'+gallery+'_k%d.npz'%shot)) as p:
                        assert np.array_equal(p['support_indices'],si) and np.array_equal(p['query_indices'],qi)
                        scores.update({str(name):p['scores'][i] for i,name in enumerate(p['names'])})
                else:
                    scores.update(pure)
                    scores['r2']=np.concatenate([ref.ref.r2_scores(S[i:i+16],Q[i:i+16],G).numpy() for i in range(0,len(S),16)])
                    scores['CS_l2']=np.concatenate([ref.geo.head(ref.geo.prepare(S[i:i+16],Q[i:i+16],G,'support'),.1).numpy() for i in range(0,len(S),16)])
                values=np.stack([scores[m] for m in names]);assert np.isfinite(values).all();pred=values.argmax(-1);yq=np.repeat(np.arange(5),15);acc=(pred==yq).mean(-1)
                target.parent.mkdir(exist_ok=True);temp=target.with_suffix('.tmp')
                diag={m+'_'+k:np.concatenate(v) for m,row in stats.items() for k,v in row.items()}
                with temp.open('wb') as f:np.savez_compressed(f,names=names,scores=values,predictions=pred,accuracy=acc,yq=yq,signature=lock_check(),**t,**diag)
                os.replace(temp,target)
                results[cell]={'seconds':time.time()-started,'tasks':len(S),'sha256':sha(target)}
                dump(OUT/'progress.json',results)
                print('CELL_COMPLETE',cell,'tasks',len(S),'seconds',round(time.time()-started,2),flush=True)
    files=list((OUT/'cells').glob('*.npz'));assert len(files)==8
    used=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
    assert used < CFG['resources']['added_gib_max']*2**30
    dump(OUT/'complete.json',{'status':'success','cells':8,'signature':lock_check(),'bytes':used})
    print('EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['encode','evaluate'],required=True);args=ap.parse_args()
    guard();signature=lock_check();OUT.mkdir(exist_ok=True);torch.set_num_threads(CFG['cpu_threads']);start=time.time()
    assert json.loads((OUT/'source_validation.json').read_text())['status']=='passed'
    dump(OUT/('run_manifest_'+args.phase+'.json'),{'command':[sys.executable,*sys.argv],'config':CFG,'signature':signature,'started_at':datetime.now(timezone.utc).isoformat(),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()})
    if args.phase=='encode':encode()
    else:evaluate()
    print('PHASE_COMPLETE',args.phase,round(time.time()-start,2),flush=True)
