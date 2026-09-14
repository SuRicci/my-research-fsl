from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,importlib.util,io,json,os,shutil,sys,tarfile,time,zipfile
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];CFG=json.loads((HERE/'protocol.json').read_text());OUT=HERE/'assets';OUT.mkdir(exist_ok=True)
REP=ROOT/'experiments/main/representation-scatter-20260913';spec=importlib.util.spec_from_file_location('caltech_frozen_views',REP/'extract_views.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def dump(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,p)
def guard():
    assert shutil.disk_usage(HERE).free/2**30>=10
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['resources']['deadline'])
def batches(rows):
    indices={r['member']:i for i,r in enumerate(rows)};parts=[]
    with zipfile.ZipFile(CFG['archive_path']) as z,z.open('caltech-101/101_ObjectCategories.tar.gz') as inner,tarfile.open(fileobj=inner,mode='r|gz') as t:
        for member in t:
            if member.name not in indices:continue
            raw=t.extractfile(member).read();i=indices[member.name]
            assert hashlib.sha256(raw).hexdigest()==rows[i]['jpeg_sha256']
            parts.append((i,raw))
            if len(parts)==32:yield parts;parts=[]
        if parts:yield parts

def main():
    start=time.time();guard();torch.set_num_threads(6);assert torch.backends.mps.is_available()
    design=json.loads((HERE/'design_lock.json').read_text())
    for p,h in design['sha256'].items():assert sha(p)==h,p
    for p,h in json.loads((REP/'encoding_lock.json').read_text())['files'].items():assert sha(p)==h,p
    assert sha(CFG['archive_path'])==CFG['archive_sha256']
    rows=json.loads((HERE/'image_order.json').read_text())['rows'];n=len(rows)
    lock=json.loads((HERE/'encoding_code_lock.json').read_text())
    for p,h in lock.items():assert sha(p)==h,p
    mf=OUT/'manifest.json';manifest=json.loads(mf.read_text()) if mf.exists() else {'status':'running','design_sha256':sha(HERE/'design_lock.json'),'encoding_code_lock':lock,'image_count':n,'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'files':{},'precision':'same frozen source MPSfloat32 normalized/storedfloat16','first_started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    assert manifest['design_sha256']==sha(HERE/'design_lock.json') and manifest['encoding_code_lock']==lock
    dump(mf,manifest)
    for bi,b in enumerate(old.enc.BACKBONES):
        target=OUT/(b+'.pt')
        if b in manifest['files']:
            assert sha(target)==manifest['files'][b]['sha256'];print('REUSE_COMPLETED',b,flush=True);continue
        model=old.enc.model(b);chunkdir=OUT/(b+'_chunks');chunkdir.mkdir(exist_ok=True);pieces=[];ids=[];hist={};done=0
        with ThreadPoolExecutor(max_workers=6) as pool:
            for number,items in enumerate(batches(rows)):
                guard();ix=[i for i,raw in items];path=chunkdir/f'{number:05d}.pt'
                if path.exists():
                    saved=torch.load(path,weights_only=True);assert saved['indices']==ix;f=saved['features'];counts=saved['unique_view_counts']
                else:
                    images=list(pool.map(lambda pair:old.views(io.BytesIO(pair[1])),items))
                    counts=[len({hashlib.sha256(np.asarray(v).tobytes()).digest() for v in vv}) for vv in images]
                    f=old.batch(model,b,bi,images,range(6)).half();saved={'indices':ix,'features':f,'unique_view_counts':counts};tmp=path.with_suffix('.tmp');torch.save(saved,tmp);os.replace(tmp,path)
                assert f.shape==(len(ix),6,512 if bi==0 else 384) and torch.isfinite(f).all()
                assert float((f.float().norm(dim=-1)-1).abs().max())<.002
                pieces.append(f);ids.extend(ix);done+=len(ix)
                for v in counts:hist[str(v)]=hist.get(str(v),0)+1
                progress={'status':'running','backbone':b,'images_done':done,'image_count':n,'elapsed_seconds':time.time()-start,'free_gib':shutil.disk_usage(HERE).free/2**30};dump(OUT/'progress.json',progress)
                if number%10==0:print('ENCODING',json.dumps(progress),flush=True)
        assert sorted(ids)==list(range(n));joined=torch.cat(pieces);order=np.argsort(ids);features=joined[order];tmp=target.with_suffix('.tmp');torch.save({'features':features,'ids':torch.arange(n)},tmp);os.replace(tmp,target)
        verified=torch.load(target,weights_only=True);assert torch.equal(verified['features'],features) and verified['ids'].tolist()==list(range(n))
        offset=0
        for path in sorted(chunkdir.glob('*.pt')):
            saved=torch.load(path,weights_only=True);assert torch.equal(saved['features'],verified['features'][saved['indices']]);offset+=len(saved['indices'])
        assert offset==n
        manifest['files'][b]={'path':str(target),'sha256':sha(target),'shape':list(features.shape),'dtype':str(features.dtype),'unique_view_histogram':hist,'all_chunk_rows_verified':True};dump(mf,manifest)
        shutil.rmtree(chunkdir);print('BACKBONE_COMPLETE',b,n,flush=True);del model,joined,features,verified,pieces;torch.mps.empty_cache()
    manifest.update(status='completed',last_elapsed_seconds=time.time()-start,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),free_gib=shutil.disk_usage(HERE).free/2**30);dump(mf,manifest);dump(OUT/'complete.json',manifest);print('ENCODING_COMPLETE',json.dumps({'seconds':time.time()-start,'images':n,'files':manifest['files']}),flush=True)
if __name__=='__main__':main()
