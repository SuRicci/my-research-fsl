"""One public frozen base checkpoint, exactly the retained RGB identities/views."""
from concurrent.futures import ThreadPoolExecutor
import argparse, inspect, platform, time
from common import *
def images(model,paths,device):
    tf=ex.enc.T.Compose([ex.enc.T.ToTensor(),ex.enc.T.Normalize(*ex.enc.NORM[1])])
    with ThreadPoolExecutor(max_workers=CFG['encoding_workers']) as pool:ims=list(pool.map(ex.views,paths))
    outputs=[]
    with torch.inference_mode():
        for v in range(6):
            x=torch.stack([tf(a[v]) for a in ims]).to(device)
            outputs.append(F.normalize(model(x).float().cpu(),dim=-1))
    return torch.stack(outputs,1)
def lock(model):
    repo=ex.enc.Q/'baselines/imported/r45-specialize/repository/research_common/extern/dinov2'
    files=list(repo.rglob('*.py'))+[HERE/'protocol.json',HERE/'common.py',Path(__file__),Path(inspect.getfile(type(model))),ASSET/'weight_manifest.json']
    old=json.loads((ex.HERE/'encoding_lock.json').read_text())['files']
    for p,h in old.items():assert sha(p)==h,p
    return {**old,**{str(p):sha(p) for p in files}}
def check():
    guard();torch.set_num_threads(CFG['threads']);assert torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1'
    rows=ex.source_rows();paths=[r['paths'][0] for r in rows.values()];m=backbone('cpu');params=sum(p.numel() for p in m.parameters());cpu=images(m,paths,'cpu');m.to('mps');gpu=images(m,paths,'mps')
    err=float(abs(cpu-gpu).max());cos=float((cpu*gpu).sum(-1).min());assert err<CFG['encoding_checks']['cpu_mps_abs'] and cos>CFG['encoding_checks']['cpu_mps_cosine']
    sample=[p for r in rows.values() for p in r['paths'][:8]];assert len(sample)==32
    torch.mps.synchronize();start=time.perf_counter();timed=images(m,sample,'mps');torch.mps.synchronize();sec=time.perf_counter()-start;hours=sec/len(sample)*CFG['expected_images']/3600
    assert timed.shape==(32,6,768) and torch.isfinite(timed).all() and abs(timed.norm(dim=-1)-1).max()<1e-5
    result={'status':'passed','cpu_mps_max_abs':err,'cpu_mps_min_cosine':cos,'checked_images':4,'checked_views':6,'timed_images':32,'seconds':sec,'projected_encoding_hours':hours,'projection_scope':'one32image warm sample; fullrun throughput may differ','parameters':params,'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'platform':platform.platform(),'mps':True,'threads':CFG['threads'],'weight_strict_load':True,'source_rgb_count':sum(len(r['ids']) for r in rows.values()),'lock':lock(m),'free_gib':shutil.disk_usage(HERE).free/2**30}
    if hours>CFG['max_encoding_hours']:result['status']='failed_cost_gate'
    dump(OUT/'encoding_precheck.json',result);print(json.dumps(result),flush=True);assert result['status']=='passed'
def run(resume):
    assert json.loads((OUT/'precheck.json').read_text())['status']=='passed'
    pre=json.loads((OUT/'encoding_precheck.json').read_text());assert pre['status']=='passed';guard(CFG['expected_images']*6*768*2+30*2**20);torch.set_num_threads(CFG['threads']);m=backbone('mps');locked=lock(m);assert locked==pre['lock'];rows=ex.source_rows();mp=ASSET/'manifest.json'
    if resume:
        rec=json.loads(mp.read_text());assert rec['status']=='running' and rec['lock']==locked
    else:
        assert not mp.exists(),'adopt existing session or explicit resume'
        rec={'status':'running','lock':locked,'config':CFG,'pools':{},'images':0,'active':None,'seconds':0,'command':[sys.executable,*sys.argv]};dump(mp,rec)
    started=time.time();prior=rec['seconds']
    for key,row in rows.items():
        p=ASSET/(key+'.npy');part=p.with_suffix('.partial.npy');shape=(len(row['ids']),6,768)
        if key in rec['pools']:assert sha(p)==rec['pools'][key]['sha256'];continue
        if resume and rec['active'] and rec['active']['pool']==key:
            offset=rec['active']['done'];f=np.load(part,mmap_mode='r+');assert f.shape==shape and sha(part)==rec['active']['partial_sha256']
        else:
            assert not part.exists() and not p.exists();offset=0;f=np.lib.format.open_memmap(part,mode='w+',dtype=np.float16,shape=shape)
        for i in range(offset,len(row['ids']),CFG['batch']):
            guard();end=min(i+CFG['batch'],len(row['ids']));paths=row['paths'][i:end]
            with ThreadPoolExecutor(max_workers=CFG['encoding_workers']) as pool:assert list(pool.map(ex.enc.rgb,paths))==row['rgb'][i:end]
            v=images(m,paths,'mps');assert torch.isfinite(v).all();data=v.half().numpy();assert abs(np.linalg.norm(data.astype(np.float32),axis=-1)-1).max()<CFG['encoding_checks']['unit_norm_error'];f[i:end]=data;f.flush();assert np.array_equal(f[i:end],data)
            rec['seconds']=prior+time.time()-started;assert rec['seconds']<CFG['max_encoding_hours']*3600,'encoding time gate';rec['active']={'pool':key,'done':end,'partial_sha256':sha(part)};dump(mp,rec)
            progress={'images_done':rec['images']+end,'images_total':CFG['expected_images'],'pool':key,'pool_done':end,'seconds':rec['seconds'],'free_gib':shutil.disk_usage(HERE).free/2**30};dump(OUT/'encoding_progress.json',progress)
            if i%320==0:print('ENCODING',json.dumps(progress),flush=True)
        del f;os.replace(part,p);ids=ASSET/(key+'_identities.json');dump(ids,{'ids':row['ids'],'rgb':row['rgb']});rec['pools'][key]={'shape':list(shape),'sha256':sha(p),'identity_sha256':sha(ids),'all_rgb_verified':True};rec['images']+=len(row['ids']);rec['active']=None;dump(mp,rec);print('POOL_COMPLETE',key,flush=True)
    assert rec['images']==CFG['expected_images'] and lock(m)==locked;rec['status']='completed';rec['seconds']=prior+time.time()-started;dump(mp,rec);dump(OUT/'encoding_complete.json',rec);print('ENCODING_COMPLETE',rec['images'],rec['seconds'],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','run'],required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();check() if a.phase=='check' else run(a.resume)
