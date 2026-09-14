"""Actual source cache extraction with inline identity, CLS and pooling checks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,json,os,shutil,sys,time
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CFG=json.loads((HERE/'protocol.json').read_text())
sys.path.insert(0,str(ROOT/'experiments/main/representation-scatter-20260913'))
import extract_views as ex
ASSET=HERE/'assets';OUT=HERE/'outputs'
def dump(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
    assert shutil.disk_usage(HERE).free>=CFG['min_free_gib']*2**30
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline'])
def main():
    guard();assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1';torch.set_num_threads(CFG['threads'])
    rows={k:v for k,v in ex.source_rows().items() if k.endswith('_query')}
    assert sum(len(r['ids']) for r in rows.values())==CFG['expected_images']
    lock=json.loads((ROOT/'experiments/main/representation-scatter-20260913/encoding_lock.json').read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    started=time.time();m=ex.enc.model(CFG['backbone']);torch.set_num_threads(CFG['threads'])
    tf=ex.enc.T.Compose([ex.enc.T.Resize(224,interpolation=ex.enc.T.InterpolationMode.BICUBIC),ex.enc.T.CenterCrop(224),ex.enc.T.ToTensor(),ex.enc.T.Normalize(*ex.enc.NORM[1])])
    record={'status':'running','config':CFG,'inherited_encoding_lock':lock,'source_code':{str(HERE/'extract.py'):sha(HERE/'extract.py'),str(HERE/'protocol.json'):sha(HERE/'protocol.json')},'software':{'torch':torch.__version__,'numpy':np.__version__,'mps_fallback':os.environ['PYTORCH_ENABLE_MPS_FALLBACK']},'pools':{},'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    total=0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for key,r in rows.items():
            target=ASSET/(key+'.npy');partial=target.with_suffix('.partial.npy');assert not target.exists() and not partial.exists(), 'Adopt existing run; do not overwrite partial cache'
            n=len(r['ids']);f=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float16,shape=(n,16,384));guard()
            old=torch.load(ex.ASSET/(key.split('_')[0]+'_'+CFG['backbone']+'_query.pt'),weights_only=True);assert old['ids'].tolist()==r['ids']
            errors=[];cosines=[];poolerr=0.;nativeerr=0.
            for start in range(0,n,CFG['encoding_batch']):
                guard();end=min(start+CFG['encoding_batch'],n);paths=r['paths'][start:end]
                assert list(pool.map(ex.enc.rgb,paths))==r['rgb'][start:end]
                x=torch.stack(list(pool.map(lambda p:ex.enc.image(p,tf),paths))).to('mps')
                with torch.inference_mode():
                    z=m.forward_features(x);cls=F.normalize(z['x_norm_clstoken'].float().cpu(),dim=-1);p=z['x_norm_patchtokens'].float().cpu();assert p.shape==(end-start,256,384)
                    if start==0:
                        native=F.normalize(m(x).float().cpu(),dim=-1);nativeerr=float((native-cls).abs().max());assert nativeerr<1e-5
                    grid=p.reshape(len(x),16,16,384);a=grid.reshape(len(x),4,4,4,4,384).mean((2,4)).reshape(len(x),16,384)
                    b=F.avg_pool2d(grid.permute(0,3,1,2),4,4).flatten(2).transpose(1,2)
                    e=float((a-b).abs().max());assert e<2e-5;poolerr=max(poolerr,e)
                    a=F.normalize(a,dim=-1);assert torch.isfinite(a).all();stored=a.half().numpy();f[start:end]=stored
                    assert np.array_equal(f[start:end],stored);assert abs(np.linalg.norm(stored.astype(np.float32),axis=-1)-1).max()<.002
                cached=F.normalize(old['features'][start:end].float(),dim=-1);error=float((cls-cached).abs().max());cos=float((cls*cached).sum(-1).min());assert error<.001 and cos>.99999,(key,start,error,cos)
                errors.append(error);cosines.append(cos);total+=end-start;f.flush()
                progress={'pool':key,'pool_done':end,'images_done':total,'images_total':CFG['expected_images'],'elapsed_seconds':time.time()-started,'free_gib':shutil.disk_usage(HERE).free/2**30}
                dump(OUT/'encoding_progress.json',progress)
                if start%320==0:print('ENCODING',json.dumps(progress),flush=True)
            del f;os.replace(partial,target)
            identities={k:r[k] for k in ['ids','rgb']};dump(ASSET/(key+'_identities.json'),identities)
            record['pools'][key]={'count':n,'shape':[n,16,384],'sha256':sha(target),'identity_sha256':sha(ASSET/(key+'_identities.json')),'cached_cls_max_abs':max(errors),'cached_cls_min_cosine':min(cosines),'native_cls_max_abs':nativeerr,'pooling_max_abs':poolerr,'all_image_hashes_verified':True}
            dump(ASSET/'manifest.json',record);print('POOL_COMPLETE',key,json.dumps(record['pools'][key]),flush=True)
    assert total==CFG['expected_images'];record.update(status='completed',elapsed_seconds=time.time()-started,images=total,free_gib=shutil.disk_usage(HERE).free/2**30)
    dump(ASSET/'manifest.json',record);dump(OUT/'encoding_complete.json',record);print('ENCODING_COMPLETE',record['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
