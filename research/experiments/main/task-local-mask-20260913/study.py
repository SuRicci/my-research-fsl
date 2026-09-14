"""Fixed development qualification; no source/target-outcome tuning."""
from pathlib import Path
import argparse,hashlib,json,sys,time,shutil
from datetime import datetime,timezone
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'r2-fusion-stage-canonical-20260912'))
import evaluate as assets_module
import geometry_reference as geo
import taskmask as tm
CFG=json.loads((HERE/'protocol.json').read_text());torch.set_num_threads(CFG['threads'])
OUT=HERE/'outputs';BANK=HERE.parent/'reduced-view-efficiency-20260913/outputs'
NAMES=['mask_cs','equal_cs','clip_cs','dino_cs','r2','mask_r2','equal_ncc','mask_ncc']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def guard():
    assert shutil.disk_usage(HERE).free>=CFG['resources']['free_gib_min']*2**30,'diskfloor'
    assert datetime.now(timezone.utc)<datetime.fromisoformat(CFG['resources']['deadline']),'deadline'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<CFG['resources']['output_max_mib']*2**20

def inputs(manifest):
    paths=[*HERE.glob('*.py'),HERE/'protocol.json',Path(assets_module.__file__),Path(geo.__file__),Path(geo.ref.__file__),Path(geo.ref.ridge_scores.__code__.co_filename),Path(assets_module.CFG['asset_root'])/'manifest.json',Path('/Users/decoqwq/DeepScientist/quests/012/baselines/local/r2-pets-transfer/json/metric_contract.json')]
    paths+=list(BANK.glob('eval*task*.npz'))
    return {'files':{str(p):sha(p) for p in paths},'features':manifest}

def task(ds,k):return dict(np.load(BANK/f'eval_{ds}_k{k}_tasks.npz'))
def take(data,ds,t,ix):
    si=t['support_indices'][ix];qi=t['query_indices'][ix].reshape(len(si),-1)
    sc,sd=[x[si] for x in data[ds]['query']];qc,qd=[x[qi] for x in data[ds]['query']]
    return sc,sd,qc,qd

def gal(data,ds,gallery):
    forbidden=set(data[ds]['ident']['query_rgb']);keep=np.array([i for i,h in enumerate(data[gallery]['ident']['gallery_rgb']) if h not in forbidden])
    return [x[keep] for x in data[gallery]['gallery']],keep

def all_scores(args,g,w):
    z={};half=torch.full_like(w,.5)
    for name,ww,origin in [('mask_cs',w,'support'),('equal_cs',half,'support'),('clip_cs',torch.ones_like(w),'support'),('dino_cs',torch.zeros_like(w),'support'),('r2',half,'none'),('mask_r2',w,'none'),('equal_ncc',half,'ncc'),('mask_ncc',w,'ncc')]:
        z[name]=tm.scores(*args,*g,ww,origin,geo.ref.ridge_scores)
    assert all(torch.isfinite(v).all() for v in z.values())
    return torch.stack([z[m] for m in NAMES])

def precheck(data,manifest):
    import torch.nn.functional as F
    checks=[];tasks_checked=0
    for ds in CFG['domains']:
        for k in CFG['shots']:
            t=task(ds,k);s=t['support_indices'];q=t['query_indices'];lab=np.array(data[ds]['ident']['query_labels'])
            assert len(s)==500 and len(np.unique(t['seeds']))==5
            assert np.array_equal(lab[s],np.repeat(t['class_ids'][:,:,None],k,2))
            assert np.array_equal(lab[q],np.repeat(t['class_ids'][:,:,None],15,2))
            for a,b in zip(s,q):assert len(np.unique(np.r_[a.ravel(),b.ravel()]))==5*(k+15)
            tasks_checked+=len(s);args=take(data,ds,t,np.arange(3));sc,sd=args[:2]
            w,bef,aft=tm.masks(sc,sd,CFG)
            one=torch.cat([tm.masks(sc[i:i+1],sd[i:i+1],CFG)[0] for i in range(3)])
            assert torch.allclose(w,one,atol=2e-5)
            perm=torch.tensor([3,0,4,1,2]);pw=tm.masks(sc[:,perm],sd[:,perm],CFG)[0]
            assert torch.allclose(w,pw,atol=2e-5)
            alpha=torch.tensor([[.2,-.1]]*3,requires_grad=True);ww=tm.weight(alpha)
            grams=[tm.n(v.flatten(1,2))@tm.n(v.flatten(1,2)).transpose(1,2) for v in [sc,sd]]
            loss=tm.support_loss(grams,ww,k);gg=torch.autograd.grad(loss.sum(),alpha)[0]
            aa=alpha.detach().clone().requires_grad_();ss=tm.combine(sc,sd,tm.weight(aa));pp=tm.n(ss.mean(2));sims=tm.n(ss.flatten(1,2))@pp.transpose(1,2)
            y=torch.arange(5).repeat_interleave(k);direct=-sims.log_softmax(-1)[:,torch.arange(5*k),y].mean(-1)
            gd=torch.autograd.grad(direct.sum(),aa)[0]
            assert torch.allclose(loss,direct,atol=2e-6) and torch.allclose(gg,gd,atol=2e-6)
            for gallery in CFG['domains']:
                g,_=gal(data,ds,gallery);scor=all_scores(args,g,w)
                ss=geo.ref.representation(sc,sd,.5);qq=geo.ref.representation(*args[2:],.5);ggg=geo.ref.representation(*g,.5)
                ref=geo.head(geo.prepare(ss,qq,ggg,'support'),.1)
                raw=geo.ref.r2_scores(ss,qq,ggg)
                err=float((ref-scor[1]).abs().max());eraw=float((raw-scor[4]).abs().max())
                assert err<3e-5 and eraw<3e-5
                checks.append({'ds':ds,'shot':k,'gallery':gallery,'equal_cs_error':err,'r2_error':eraw,'mask_batch_error':float((w-one).abs().max())})
    dump(OUT/'precheck.json',{'status':'passed','tasks_checked':tasks_checked,'checks':checks,'query_independence':'mask function accepts supports only; class permutation and per-task batch equality verified','torch':torch.__version__,'numpy':np.__version__})
    dump(HERE/'lock.json',inputs(manifest));print('PRECHECK_PASS',tasks_checked,flush=True)

def run(data,manifest):
    assert json.loads((OUT/'precheck.json').read_text())['status']=='passed'
    lock=json.loads((HERE/'lock.json').read_text());assert lock==inputs(manifest)
    start=time.time();dump(OUT/'run_manifest.json',{'argv':[sys.executable,*sys.argv],'cwd':str(Path.cwd()),'torch':torch.__version__,'numpy':np.__version__,'config':CFG,'lock_sha256':sha(HERE/'lock.json'),'started_utc':datetime.now(timezone.utc).isoformat()})
    for ds in CFG['domains']:
        for k in CFG['shots']:
            t=task(ds,k);ws=[];before=[];after=[]
            for lo in range(0,500,CFG['batch_size']):
                args=take(data,ds,t,np.arange(lo,min(lo+CFG['batch_size'],500)))
                w,b,a=tm.masks(*args[:2],CFG);ws.append(w);before.append(b);after.append(a)
            w=torch.cat(ws);np.savez_compressed(OUT/f'masks_{ds}_k{k}.npz',weights=w.numpy(),before=torch.cat(before).numpy(),after=torch.cat(after).numpy(),**t)
            for gallery in CFG['domains']:
                guard();g,keep=gal(data,ds,gallery);pred=[];audit=[]
                for lo in range(0,500,CFG['batch_size']):
                    ix=np.arange(lo,min(lo+CFG['batch_size'],500));args=take(data,ds,t,ix)
                    z=all_scores(args,g,w[ix]);pred.append(z.argmax(-1).numpy().astype('int8'))
                    for index in [0,100,499]:
                        if index in ix:audit.append((index,z[:,index-lo].numpy()))
                audit.sort(key=lambda a:a[0]);name=f'{ds}_{gallery}_k{k}'
                np.savez_compressed(OUT/f'{name}.npz',predictions=np.concatenate(pred,axis=1),names=NAMES,yq=np.repeat(np.arange(5),15),audit_indices=[a[0] for a in audit],audit_scores=np.stack([a[1] for a in audit],axis=1),gallery_pool_indices=keep)
                dump(OUT/'progress.json',{'cell':name,'elapsed_seconds':time.time()-start});print('CELL_COMPLETE',name,round(time.time()-start,2),flush=True)
    assert lock==inputs(manifest)
    dump(OUT/'completion.json',{'status':'completed','elapsed_seconds':time.time()-start,'cells':8,'unique_tasks':2000});print('RUN_COMPLETE',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['precheck','run'],required=True);a=ap.parse_args()
    OUT.mkdir(exist_ok=True);guard();data,manifest=assets_module.assets()
    (precheck if a.phase=='precheck' else run)(data,manifest)
if __name__=='__main__':main()
