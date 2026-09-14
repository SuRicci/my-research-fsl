"""Stream three intermediate DINOv2 CLS blocks with full identity/final parity."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, datetime, hashlib, importlib.util, json, os, shutil, sys, time
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];ASSET=HERE/'assets';OUT=HERE/'outputs'
CFG=json.loads((HERE/'protocol.json').read_text())
spec=importlib.util.spec_from_file_location('depth_existing_views',HERE.parent/'representation-scatter-20260913/extract_views.py')
ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');os.replace(t,p)
def guard(reserve=0):
    assert shutil.disk_usage(HERE).free>=CFG['min_free_gib']*2**30+reserve,'disk floor'
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline']),'deadline'
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<CFG['max_new_mib']*2**20,'output cap'
def run(resume):
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1'
    assert json.loads((OUT/'precheck.json').read_text())['status']=='passed';torch.set_num_threads(CFG['threads'])
    rows=ex.source_rows();assert sum(len(r['ids']) for r in rows.values())==CFG['expected_images']
    inherited=json.loads((ex.HERE/'encoding_lock.json').read_text())['files']
    for p,h in inherited.items():assert sha(p)==h,p
    m=ex.enc.model('dinov2_vits14')
    import inspect
    lock={**inherited,**{str(p):sha(p) for p in [HERE/'protocol.json',Path(__file__),Path(inspect.getfile(type(m)))]}}
    validation=json.loads((ex.OUT/'feature_manifest.json').read_text());old_manifest=json.loads((ex.ASSET/'manifest.json').read_text())
    recordpath=ASSET/'manifest.json'
    if resume:
        record=json.loads(recordpath.read_text());assert record['status']=='running' and record['lock']==lock
    else:
        assert not recordpath.exists(),'Adopt current session or explicit --resume; never overwrite'
        guard(CFG['expected_images']*6*3*384*2+40*2**20)
        record={'status':'running','lock':lock,'config':CFG,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pools':{},'active':None,'images':0,'command':[sys.executable,*sys.argv],'elapsed_seconds':0,'software':{'torch':torch.__version__,'numpy':np.__version__,'mps':torch.backends.mps.is_available(),'mps_fallback':'1'}}
        dump(recordpath,record)
    started=time.time();previous_elapsed=record['elapsed_seconds']
    tf=ex.enc.T.Compose([ex.enc.T.ToTensor(),ex.enc.T.Normalize(*ex.enc.NORM[1])])
    with ThreadPoolExecutor(max_workers=CFG['encoding_workers']) as pool:
        for key,row in rows.items():
            target=ASSET/(key+'.npy');partial=target.with_suffix('.partial.npy');n=len(row['ids']);shape=(n,6,3,384)
            if key in record['pools']:
                assert target.exists() and sha(target)==record['pools'][key]['sha256'];continue
            ds,side=key.split('_');oldpath=ex.ASSET/f'{ds}_dinov2_vits14_{side}.pt';other=validation['files'][key+'_dinov2_vits14']
            assert sha(oldpath)==old_manifest['datasets'][ds]['backbones']['dinov2_vits14_'+side]['sha256']
            assert sha(other['path'])==other['sha256']
            base=torch.load(oldpath,weights_only=True);extra=torch.load(other['path'],weights_only=True)
            assert base['ids'].tolist()==extra['ids'].tolist()==row['ids']
            old=F.normalize(torch.cat([base['features'][:,None],extra['features']],1).float(),dim=-1)
            active=record.get('active')
            if resume and active and active['pool']==key:
                offset=active['done'];f=np.load(partial,mmap_mode='r+');assert f.shape==shape and f.dtype==np.float16
                assert hashlib.sha256(f[:offset].tobytes()).hexdigest()==active['prefix_sha256'];maxerr=active['max_abs'];mincos=active['min_cosine'];nativeerr=active['native_error']
            else:
                assert not target.exists() and not partial.exists(),'Unexpected existing feature file'
                offset=0;maxerr=0.;mincos=1.;nativeerr=0.;f=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float16,shape=shape)
            for start in range(offset,n,CFG['batch']):
                guard();end=min(start+CFG['batch'],n);paths=row['paths'][start:end]
                assert list(pool.map(ex.enc.rgb,paths))==row['rgb'][start:end]
                images=list(pool.map(ex.views,paths));parts=[];lasts=[]
                with torch.inference_mode():
                    for view in range(CFG['views']):
                        x=torch.stack([tf(ims[view]) for ims in images]).to('mps')
                        blocks=m.get_intermediate_layers(x,n=4,return_class_token=True,norm=True)
                        cls=torch.stack([F.normalize(v.float().cpu(),dim=-1) for _,v in blocks],1)
                        assert cls.shape==(end-start,4,384) and torch.isfinite(cls).all()
                        if start==0 and view==0:
                            direct=F.normalize(m(x).float().cpu(),dim=-1);nativeerr=float((cls[:,-1]-direct).abs().max());assert nativeerr<CFG['encoding_parity']['native_intermediate_last_abs']
                        parts.append(cls[:,:3]);lasts.append(cls[:,-1])
                output=torch.stack(parts,1).half().numpy();last=torch.stack(lasts,1)
                error=float((last-old[start:end]).abs().max());cos=float((last*old[start:end]).sum(-1).min())
                assert error<CFG['encoding_parity']['max_abs'] and cos>CFG['encoding_parity']['min_cosine'],(key,start,error,cos)
                assert abs(np.linalg.norm(output.astype(np.float32),axis=-1)-1).max()<.002
                f[start:end]=output;f.flush();assert np.array_equal(f[start:end],output)
                maxerr=max(maxerr,error);mincos=min(mincos,cos)
                record['elapsed_seconds']=previous_elapsed+time.time()-started
                record['active']={'pool':key,'done':end,'total':n,'max_abs':maxerr,'min_cosine':mincos,'native_error':nativeerr,'prefix_sha256':hashlib.sha256(f[:end].tobytes()).hexdigest()}
                dump(recordpath,record)
                progress={'pool':key,'pool_done':end,'pool_total':n,'images_done':record['images']+end,'images_total':CFG['expected_images'],'elapsed_seconds':record['elapsed_seconds'],'free_gib':shutil.disk_usage(HERE).free/2**30}
                dump(OUT/'encoding_progress.json',progress)
                if start%320==0:print('ENCODING',json.dumps(progress),flush=True)
            del f;os.replace(partial,target)
            identity=ASSET/(key+'_identities.json');dump(identity,{k:row[k] for k in ['ids','rgb']})
            record['pools'][key]={'shape':list(shape),'sha256':sha(target),'identities_sha256':sha(identity),'all_rgb_verified':True,'all_final_layer_parity_checked':True,'max_abs':maxerr,'min_cosine':mincos,'native_error':nativeerr}
            record['images']+=n;record['active']=None;dump(recordpath,record);print('POOL_COMPLETE',key,n,flush=True)
    assert record['images']==CFG['expected_images']
    for p,h in lock.items():assert sha(p)==h
    record.update(status='completed',elapsed_seconds=previous_elapsed+time.time()-started,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dump(recordpath,record);dump(OUT/'encoding_complete.json',record);print('ENCODING_COMPLETE',record['images'],record['elapsed_seconds'],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args();run(args.resume)
