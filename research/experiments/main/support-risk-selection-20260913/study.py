"""Frozen finite representation selection from held-out support information."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, shutil, sys, time
import numpy as np
import torch
import torch.nn.functional as F
HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
sys.path.insert(0, str(MAIN/'r2-fusion-stage-canonical-20260912'))
import evaluate as asset_module
import geometry_reference as geo
CFG = json.loads((HERE/'protocol.json').read_text())
OUT = HERE/'outputs'
BANK = MAIN/'reduced-view-efficiency-20260913/outputs'
CACHE = MAIN/'representation-scatter-20260913/assets'
ASSET = Path(asset_module.CFG['asset_root'])
NAMES = CFG['names']
torch.set_num_threads(CFG['threads'])

def norm(x): return F.normalize(x, dim=-1)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, indent=2)+'\n')
def guard():
    assert shutil.disk_usage(HERE).free >= CFG['resources']['min_free_gib']*2**30, 'disk floor'
    assert datetime.now(timezone.utc) < datetime.fromisoformat(CFG['resources']['deadline'])
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < CFG['resources']['output_max_mib']*2**20

def load():
    mf = json.loads((CACHE/'feature_manifest.json').read_text())
    orig = json.loads((ASSET/'manifest.json').read_text())
    assert mf['status'] == 'completed'
    paths = [HERE/'study.py', HERE/'protocol.json', CACHE/'feature_manifest.json', ASSET/'manifest.json', Path(asset_module.__file__), Path(geo.__file__), Path(geo.ref.__file__), Path(geo.ref.ridge_scores.__code__.co_filename), Path('/Users/decoqwq/DeepScientist/quests/012/baselines/local/r2-pets-transfer/json/metric_contract.json')]
    data = {}
    for ds in CFG['domains']:
        ip = ASSET/(ds+'_identities.json'); paths.append(ip)
        ident = json.loads(ip.read_text()); data[ds] = {'ident':ident}
        for side in ['query','gallery']:
            blocks = []
            for backbone in ['clip_vitb16','dinov2_vits14']:
                op = ASSET/f'{ds}_{backbone}_{side}.pt'
                entry = mf['files'][f'{ds}_{side}_{backbone}']; ep = Path(entry['path'])
                assert sha(op) == orig['datasets'][ds]['backbones'][backbone+'_'+side]['sha256']
                assert sha(ep) == entry['sha256']
                a = torch.load(op, weights_only=True); e = torch.load(ep, weights_only=True)
                assert a['ids'].tolist() == e['ids'].tolist() == ident[side+'_ids']
                z = norm(torch.cat([a['features'][:,None],e['features']],1).float())
                assert z.shape[1] == 6 and torch.isfinite(z).all()
                blocks.append(z); paths += [op,ep]
            data[ds][side] = blocks
    paths += list(BANK.glob('eval*task*.npz'))
    return data, {str(p):sha(p) for p in paths}

def task(ds,k): return dict(np.load(BANK/f'eval_{ds}_k{k}_tasks.npz'))
def representation(a,b,w):
    return norm(torch.cat([a*(w**.5),b*((1-w)**.5)],-1))

def head(s,q,g):
    return geo.head(geo.prepare(s,q,g,'support'),CFG['ridge'])

@torch.no_grad()
def risks(sc,sd,gc,gd,w,image_folds=False):
    # No real query tensors, labels or gallery labels are accepted here.
    k = sc.shape[2]
    gallery = representation(gc.mean(-2),gd.mean(-2),w)
    losses=[]; correct=[]; scores=[]
    for f in range(k if image_folds else 6):
        if image_folds:
            assert k > 1
            keep=[j for j in range(k) if j != f]
            sa,sb=sc[:,:,keep].mean(-2),sd[:,:,keep].mean(-2)
            qa,qb=sc[:,:,f].mean(-2),sd[:,:,f].mean(-2)
            y=torch.arange(5)
        else:
            keep=[j for j in range(6) if j != f]
            sa,sb=sc[:,:,:,keep].mean(-2),sd[:,:,:,keep].mean(-2)
            qa,qb=sc[:,:,:,f].flatten(1,2),sd[:,:,:,f].flatten(1,2)
            y=torch.arange(5).repeat_interleave(k)
        z=head(representation(sa,sb,w),representation(qa,qb,w),gallery)
        target=F.one_hot(y,5).to(z.dtype)
        losses.append((z-target).square().mean((-2,-1)))
        correct.append((z.argmax(-1)==y).float().mean(-1))
        scores.append(z)
    return torch.stack(losses,-1),torch.stack(correct,-1),torch.stack(scores,1)

@torch.no_grad()
def batch(sc,sd,qc,qd,gc,gd):
    k=sc.shape[2]; predictions=[]; query_scores=[]; vr=[]; ir=[]; va=[]; ia=[]; audit=[]
    for w in CFG['weights']:
        v,a,vscore=risks(sc,sd,gc,gd,w);vr.append(v);va.append(a)
        if k==5:
            i,b,iscore=risks(sc,sd,gc,gd,w,True)
        else: i,b,iscore=v,a,vscore
        ir.append(i);ia.append(b);audit.append(iscore)
        z=head(representation(sc.mean(-2),sd.mean(-2),w),representation(qc.mean(-2),qd.mean(-2),w),representation(gc.mean(-2),gd.mean(-2),w))
        query_scores.append(z);predictions.append(z.argmax(-1))
    vr,ir=torch.stack(vr),torch.stack(ir)
    vsel=vr.mean(-1).argmin(0);sel=ir.mean(-1).argmin(0)
    qs=torch.stack(query_scores);ix=torch.arange(len(sc))
    all_scores=torch.cat([qs,qs[sel,ix][None],qs[vsel,ix][None]])
    return {'predictions':all_scores.argmax(-1),'scores':all_scores,'view_risk':vr,'risk':ir,'view_accuracy':torch.stack(va),'cv_accuracy':torch.stack(ia),'selected':sel,'view_selected':vsel,'fold_scores':torch.stack(audit)}

def gallery(data,ds,g):
    ids=data[g]['ident']['gallery_rgb']; forbidden=set(data[ds]['ident']['query_rgb'])
    keep=np.array([i for i,h in enumerate(ids) if h not in forbidden])
    assert len(keep)>=64
    return [z[keep] for z in data[g]['gallery']],keep

def args(data,ds,t,ix):
    s=t['support_indices'][ix];q=t['query_indices'][ix].reshape(len(ix),-1)
    return [z[s] for z in data[ds]['query']]+[z[q] for z in data[ds]['query']]

def precheck(data,lock):
    checks=[];count=0
    for ds in CFG['domains']:
        labels=np.array(data[ds]['ident']['query_labels'])
        for k in CFG['shots']:
            t=task(ds,k);s=t['support_indices'];q=t['query_indices']
            assert len(s)==500 and len(np.unique(t['seeds']))==5
            assert np.array_equal(labels[s],np.repeat(t['class_ids'][:,:,None],k,2))
            assert np.array_equal(labels[q],np.repeat(t['class_ids'][:,:,None],15,2))
            for a,b in zip(s,q): assert len(np.unique(np.r_[a.ravel(),b.ravel()]))==5*(k+15)
            count+=len(s)
            g,_=gallery(data,ds,ds);aa=args(data,ds,t,np.array([0,100,499]))
            out=batch(*aa,*g)
            assert all(torch.isfinite(z).all() for z in out.values())
            individual=torch.cat([batch(*[v[j:j+1] for v in aa],*g)['risk'] for j in range(3)],dim=1)
            assert torch.allclose(out['risk'],individual,atol=3e-6)
            # Exact previous equal-view aggregation; independent implementation of concatenation order.
            fused=[geo.ref.representation(a,b,.5) for a,b in [aa[:2],aa[2:],g]]
            S,Q,G=[norm(z.mean(-2)) for z in fused]
            z=geo.head(geo.prepare(S,Q,G,'support'),.1)
            assert torch.allclose(out['scores'][0],z,atol=3e-5)
            checks.append({'domain':ds,'shot':k,'equal_reference_score_error':float((out['scores'][0]-z).abs().max()),'batch_error':float((individual-out['risk']).abs().max())})
    dump(OUT/'precheck.json',{'status':'passed','task_count':count,'checks':checks,'numpy':np.__version__,'torch':torch.__version__,'fold_contract':'one heldout identity per class at5shot; no heldoutview in training mean at1shot; sameimage dependence explicit'})
    dump(HERE/'lock.json',lock);print('PRECHECK_PASS',count,flush=True)

def run(data,lock):
    assert lock==json.loads((HERE/'lock.json').read_text())
    assert json.loads((OUT/'precheck.json').read_text())['status']=='passed'
    started=time.time();dump(OUT/'manifest.json',{'command':[sys.executable,*sys.argv],'cwd':str(Path.cwd()),'numpy':np.__version__,'torch':torch.__version__,'protocol':CFG,'lock_sha256':sha(HERE/'lock.json'),'start':datetime.now(timezone.utc).isoformat()})
    for ds in CFG['domains']:
        for k in CFG['shots']:
            t=task(ds,k);np.savez_compressed(OUT/f'tasks_{ds}_k{k}.npz',**t)
            for gal in CFG['domains']:
                guard();g,keep=gallery(data,ds,gal);parts={};audit=[]
                for lo in range(0,len(t['seeds']),CFG['batch_size']):
                    ix=np.arange(lo,min(lo+CFG['batch_size'],len(t['seeds'])))
                    z=batch(*args(data,ds,t,ix),*g)
                    for key,value in z.items():
                        if key not in ['scores','fold_scores']:parts.setdefault(key,[]).append(value.numpy())
                    for i in [0,100,499]:
                        if i in ix:audit.append((i,z['scores'][:,i-lo].numpy(),z['fold_scores'][:,i-lo].numpy()))
                audit.sort(key=lambda z:z[0]);cell=f'{ds}_{gal}_k{k}'
                arrays={key:np.concatenate(values,axis=0 if key in ['selected','view_selected'] else 1) for key,values in parts.items()}
                arrays['predictions']=arrays['predictions'].astype('int8')
                np.savez_compressed(OUT/f'{cell}.npz',**arrays,names=NAMES,yq=np.repeat(np.arange(5),15),gallery_pool_indices=keep,audit_indices=[v[0] for v in audit],audit_scores=np.stack([v[1] for v in audit],1),audit_fold_scores=np.stack([v[2] for v in audit],1))
                dump(OUT/'progress.json',{'cell':cell,'seconds':time.time()-started});print('CELL_COMPLETE',cell,round(time.time()-started,2),flush=True)
    dump(OUT/'completion.json',{'status':'completed','seconds':time.time()-started,'cells':8,'unique_tasks':2000});print('RUN_COMPLETE',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['precheck','run'],required=True);a=ap.parse_args()
    OUT.mkdir(exist_ok=True);guard();data,lock=load()
    (precheck if a.phase=='precheck' else run)(data,lock)
if __name__=='__main__':main()
