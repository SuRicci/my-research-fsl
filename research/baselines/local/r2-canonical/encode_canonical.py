"""Canonical fresh features: no mixing with historical query embeddings."""
from pathlib import Path
import sys,json,hashlib,time,shutil
from concurrent.futures import ThreadPoolExecutor
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"r2-replay"))
import recover_gallery as r
B=Path(__file__).resolve().parent
def main():
    t=time.perf_counter()
    manifest={"variant":"canonical-openai-quickgelu-native-dinov2","historical_parity":False,"precision":"float32 MPS with bicubic CPU fallback; normalized float16 storage","source_code_sha256":hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest(),"datasets":{}}
    (B/"assets").mkdir(exist_ok=True)
    for name in ["eurosat","dtd"]:
        qp,ql=r.dataset(name,"test");gp,gl=r.dataset(name,"train")
        gis=r.np.arange(len(gp))
        if name=="eurosat":
            rng=r.np.random.RandomState(129209)
            gis=r.np.array(sorted(int(i) for c in r.np.unique(gl) for i in rng.choice(r.np.flatnonzero(gl==c),500,replace=False)))
        with ThreadPoolExecutor(max_workers=8) as pool:
            qh=list(pool.map(r.rgb,qp));gh=list(pool.map(r.rgb,[gp[i] for i in gis]))
        seen=set();qids=[]
        for i,h in enumerate(qh):
            if h not in seen:seen.add(h);qids.append(i)
        banned=set(qh);seen=set();gids=[];hashes=[]
        for i,h in zip(gis,gh):
            if h not in banned and h not in seen:seen.add(h);gids.append(int(i));hashes.append(h)
        if name=="dtd":assert sorted(set(range(len(qp)))-set(qids))==[427,436,439]
        ident={"query_ids":qids,"query_labels":ql[qids].tolist(),"query_rgb":[qh[i] for i in qids],"gallery_ids":gids,"gallery_labels":gl[gids].tolist(),"gallery_rgb":hashes}
        r.dump(B/"assets"/(name+"_identities.json"),ident)
        row={"query_unique":len(qids),"gallery_unique":len(gids),"backbones":{}};manifest["datasets"][name]=row
        for bi,b in enumerate(r.BACKBONES):
            m=r.model(b);tf=r.T.Compose([r.T.Resize(224,interpolation=r.T.InterpolationMode.BICUBIC),r.T.CenterCrop(224),r.T.ToTensor(),r.T.Normalize(*r.NORM[bi])])
            with ThreadPoolExecutor(max_workers=6) as pool:
                for side,paths,ids in [("query",qp,qids),("gallery",gp,gids)]:
                    target=B/"assets"/(name+"_"+b+"_"+side+".pt");chunkdir=B/"assets"/(name+"_"+b+"_"+side+"_chunks");chunkdir.mkdir(exist_ok=True)
                    if not target.exists():
                        chunks=[]
                        for st in range(0,len(ids),32):
                            assert shutil.disk_usage(B).free>10*1024**3
                            p=chunkdir/("%06d.pt"%st)
                            if p.exists():f=torch.load(p,weights_only=True)
                            else:
                                f=r.encode(m,b,[paths[i] for i in ids[st:st+32]],tf,pool).half()
                                assert torch.isfinite(f).all();torch.save(f,p)
                            chunks.append(f)
                            if st%320==0:print("ENCODE",name,b,side,st,len(ids),"elapsed",round(time.perf_counter()-t,1),flush=True)
                        torch.save({"ids":torch.tensor(ids),"features":torch.cat(chunks)},target)
                    row["backbones"][b+"_"+side]={"path":str(target),"sha256":hashlib.sha256(target.read_bytes()).hexdigest(),"n":len(ids)}
                    r.dump(B/"assets/manifest.json",manifest)
            del m;torch.mps.empty_cache()
    manifest["elapsed_seconds"]=time.perf_counter()-t;r.dump(B/"assets/manifest.json",manifest);print("CANONICAL_FEATURES_COMPLETE",flush=True)
if __name__=="__main__":main()
