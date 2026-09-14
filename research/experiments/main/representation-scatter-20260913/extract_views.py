"""Deterministic additional views; same image identities and frozen encoders."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse, hashlib, json, os, sys, time, shutil
import numpy as np
import torch
from PIL import Image
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'protocol.json').read_text())
sys.path.insert(0,CFG['encoder_root'])
import recover_gallery as enc
ASSET=Path(CFG['asset_root']); OUT=HERE/'assets'; OUT.mkdir(exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()

def dump(p,a):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(a,indent=2));os.replace(t,p)

def guard():
    assert shutil.disk_usage(HERE).free>=10*2**30, 'disk floor'
    assert datetime.now(timezone.utc)<datetime.fromisoformat(CFG['resources']['deadline_utc']), 'deadline'

def source_rows():
    result={}
    for ds in CFG['source_domains']:
        ident=json.loads((ASSET/(ds+'_identities.json')).read_text())
        for side,split in [('query','test'),('gallery','train')]:
            paths,labels=enc.dataset(ds,split);ids=ident[side+'_ids']
            assert labels[ids].tolist()==ident[side+'_labels']
            result[ds+'_'+side]=dict(paths=[str(paths[i]) for i in ids],ids=ids,labels=ident[side+'_labels'],rgb=ident[side+'_rgb'])
    assert sum(len(r['ids']) for r in result.values())==14153
    return result

def views(path):
    with Image.open(path) as im:
        im=im.convert('RGB')
        large=enc.T.Resize(256,interpolation=enc.T.InterpolationMode.BICUBIC)(im)
        width,height=large.size
        positions={(0,0),(width-224,0),(0,height-224),(width-224,height-224),(round((width-224)/2),round((height-224)/2))}
        assert len(positions)==5 and min(width,height)>=256
        crops=enc.T.FiveCrop(224)(large)
        original=enc.T.CenterCrop(224)(enc.T.Resize(224,interpolation=enc.T.InterpolationMode.BICUBIC)(im))
        return (original,*crops)

def batch(model,backbone,bi,images,indices):
    tf=enc.T.Compose([enc.T.ToTensor(),enc.T.Normalize(*enc.NORM[bi])]);fs=[]
    with torch.inference_mode():
        for v in indices:
            x=torch.stack([tf(im[v]) for im in images]).to('mps')
            y=model.encode_image(x) if backbone=='clip_vitb16' else model(x)
            fs.append(torch.nn.functional.normalize(y.float().cpu(),dim=-1))
    return torch.stack(fs,1)

def check():
    guard();rows=source_rows(); result={'status':'passed','checks':{},'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'mps':torch.backends.mps.is_available()}
    assert result['mps']
    for bi,b in enumerate(enc.BACKBONES):
        model=enc.model(b)
        for key,r in rows.items():
            with ThreadPoolExecutor(max_workers=6) as pool: ims=list(pool.map(views,r['paths'][:4]))
            distinct=[len({np.asarray(v).tobytes() for v in vs}) for vs in ims]
            assert min(distinct)>=2, (key,distinct)  # Intrinsic texture symmetries can repeat pixels at distinct positions.
            fresh=batch(model,b,bi,ims,range(6));ds,side=key.split('_');old=torch.load(ASSET/(ds+'_'+b+'_'+side+'.pt'),weights_only=True)
            assert old['ids'].tolist()==r['ids']
            old=torch.nn.functional.normalize(old['features'][:4].float(),dim=-1)
            error=float((fresh[:,0]-old).abs().max());cos=float((fresh[:,0]*old).sum(-1).min())
            assert error<.001 and cos>.99999,(key,b,error,cos)
            result['checks'][key+'_'+b]=dict(distinct_views=distinct,original_max_abs=error,original_min_cosine=cos,feature_shape=list(fresh.shape))
        del model;torch.mps.empty_cache()
    inputs=[HERE/'protocol.json',Path(__file__),Path(enc.__file__)]+[ASSET/(ds+'_identities.json') for ds in CFG['source_domains']]
    weights=[Path('/Users/decoqwq/.cache/clip/ViT-B-16.pt'),Path('/Users/decoqwq/.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m/snapshots/4610ca143709d58a633b6397a74412c2c3842454/model.safetensors')]
    dump(HERE/'encoding_lock.json',{'files':{str(p):sha(p) for p in inputs+weights}})
    dump(HERE/'outputs/encoding_validation.json',result);print(json.dumps(result),flush=True)

def run():
    guard();lock=json.loads((HERE/'encoding_lock.json').read_text())
    for p,h in lock['files'].items():assert sha(p)==h,p
    assert json.loads((HERE/'outputs/encoding_validation.json').read_text())['status']=='passed'
    rows=source_rows();dump(OUT/'identities.json',rows);started=time.time()
    manifest={'status':'running','config_sha256':sha(HERE/'protocol.json'),'lock':lock,'files':{},'total_source_images':14153,'extra_views':5,'command':[sys.executable,*sys.argv]}
    for bi,b in enumerate(enc.BACKBONES):
        model=enc.model(b)
        with ThreadPoolExecutor(max_workers=6) as pool:
            for key,r in rows.items():
                parts=[];hist={};chunkdir=OUT/'chunks'/(key+'_'+b);chunkdir.mkdir(parents=True,exist_ok=True)
                for start in range(0,len(r['ids']),32):
                    guard();path=chunkdir/('%06d.pt'%start);n=min(32,len(r['ids'])-start)
                    ims=list(pool.map(views,r['paths'][start:start+32]))
                    for vs in ims:
                        count=len({hashlib.sha256(np.asarray(v).tobytes()).digest() for v in vs});hist[str(count)]=hist.get(str(count),0)+1
                    if path.exists():f=torch.load(path,weights_only=True)
                    else:
                        f=batch(model,b,bi,ims,range(1,6)).half()
                        tmp=path.with_suffix('.tmp');torch.save(f,tmp);os.replace(tmp,path)
                    assert f.shape==(n,5,512 if bi==0 else 384) and torch.isfinite(f).all()
                    assert float((f.float().norm(dim=-1)-1).abs().max())<.002
                    parts.append(f)
                    progress=dict(status='running',backbone=b,pool=key,images_done=start+n,pool_size=len(r['ids']),elapsed_seconds=time.time()-started,free_gib=shutil.disk_usage(HERE).free/2**30)
                    dump(HERE/'outputs/encoding_progress.json',progress)
                    if start%320==0:print('ENCODING',json.dumps(progress),flush=True)
                f=torch.cat(parts);target=OUT/(key+'_'+b+'.pt');tmp=target.with_suffix('.tmp')
                torch.save({'features':f,'ids':torch.tensor(r['ids'])},tmp);os.replace(tmp,target)
                manifest['files'][key+'_'+b]={'path':str(target),'sha256':sha(target),'shape':list(f.shape),'unique_pixel_view_histogram':hist}
                dump(OUT/'feature_manifest.json',manifest)
                # Only this run's verified intermediate chunks, identical to final, are redundant.
                saved=torch.load(target,weights_only=True)['features'];offset=0
                for part in sorted(chunkdir.glob('*.pt')):
                    z=torch.load(part,weights_only=True);assert torch.equal(z,saved[offset:offset+len(z)]);offset+=len(z)
                assert offset==len(saved);shutil.rmtree(chunkdir)
                print('POOL_COMPLETE',key,b,len(f),flush=True)
        del model;torch.mps.empty_cache()
    manifest.update(status='completed',elapsed_seconds=time.time()-started)
    dump(OUT/'feature_manifest.json',manifest);dump(HERE/'outputs/encoding_complete.json',manifest)
    print('ENCODING_COMPLETE',manifest['elapsed_seconds'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','run'],required=True);args=ap.parse_args()
    check() if args.phase=='check' else run()
