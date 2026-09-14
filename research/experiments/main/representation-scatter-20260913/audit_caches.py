"""Audit the completed cache snapshot without modifying live extraction assets."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil
import torch

HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*2**20),b''): h.update(chunk)
    return h.hexdigest()

def main():
    torch.set_num_threads(2)
    cfg=json.loads((HERE/'protocol.json').read_text())
    raw=(HERE/'assets/feature_manifest.json').read_bytes()
    manifest=json.loads(raw)
    rows=json.loads((HERE/'assets/identities.json').read_text())
    locks=json.loads((HERE/'encoding_lock.json').read_text())['files']
    locks.update(json.loads((HERE/'evaluation_lock.json').read_text()))
    for path,digest in locks.items(): assert sha(path)==digest, path
    assert manifest['config_sha256']==sha(HERE/'protocol.json')
    # Derive actual DINO spelling from the manifest-independent canonical asset names.
    source=Path(cfg['asset_root'])
    backbones=sorted({x.name[len('dtd_'):-len('_query.pt')] for x in source.glob('dtd_*_query.pt')})
    assert len(backbones)==2, backbones
    expected={key+'_'+b for key in rows for b in backbones}
    assert set(manifest['files'])<=expected
    checks={}
    for key,entry in manifest['files'].items():
        pool=next(k for k in rows if key.startswith(k+'_'))
        backbone=key[len(pool)+1:]
        ds,side=pool.split('_')
        canonical=json.loads((source/(ds+'_identities.json')).read_text())
        assert rows[pool]['ids']==canonical[side+'_ids']
        assert rows[pool]['labels']==canonical[side+'_labels']
        path=Path(entry['path']);digest=sha(path);assert digest==entry['sha256']
        cache=torch.load(path,map_location='cpu',weights_only=True)
        f=cache['features'];n=len(rows[pool]['ids'])
        old=torch.load(source/(ds+'_'+backbone+'_'+side+'.pt'),map_location='cpu',weights_only=True)
        assert torch.equal(cache['ids'],old['ids']) and cache['ids'].tolist()==rows[pool]['ids']
        assert len(set(rows[pool]['ids']))==n
        assert list(f.shape)==entry['shape']==[n,5,old['features'].shape[-1]]
        assert f.dtype==torch.float16 and torch.isfinite(f).all()
        norm_error=float((f.float().norm(dim=-1)-1).abs().max());assert norm_error<.002
        assert sum(entry['unique_pixel_view_histogram'].values())==n
        checks[key]={'sha256':digest,'images':n,'shape':list(f.shape),'max_norm_error':norm_error,'ids_exact':True,'bytes':path.stat().st_size}
    complete=manifest['status']=='completed' and set(manifest['files'])==expected
    result={'checked_at':datetime.now(timezone.utc).isoformat(),'status':'passed_for_completed_files','encoding_complete':complete,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'manifest_snapshot':manifest,'complete_cache_count':len(checks),'expected_cache_count':len(expected),'frozen_file_count':len(locks),'validated_image_encoder_pairs':sum(v['images'] for v in checks.values()),'checks':checks,'free_gib':shutil.disk_usage(HERE).free/2**30,'torch_version':torch.__version__,'scope':'All completed files in the saved manifest snapshot; no classification accuracy claim.'}
    if complete:
        assert (HERE/'outputs/encoding_complete.json').exists()
        assert json.loads((HERE/'outputs/encoding_complete.json').read_text())==manifest
    out=HERE/'outputs/cache_integrity.json';out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='manifest_snapshot'},indent=2))

if __name__=='__main__': main()
