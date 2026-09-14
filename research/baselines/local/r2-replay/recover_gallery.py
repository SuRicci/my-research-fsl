"""Restore R2 gallery features and validate public images against archived query features."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import clip
from safetensors.torch import load_file

B=Path(__file__).resolve().parent
Q=B.parents[2]
torch.set_num_threads(6)
BACKBONES=["clip_vitb16","dinov2_vits14"]
NORM=[((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711)),
      ((.485,.456,.406),(.229,.224,.225))]
def dump(p,x):
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def dataset(name,split):
    root=B/"data"/("EuroSAT_RGB" if name=="eurosat" else "dtd")
    cats=sorted(x.name for x in (root if name=="eurosat" else root/"images").iterdir() if x.is_dir())
    paths=[];labels=[]
    if name=="eurosat":
        for c,cat in enumerate(cats):
            files=sorted((root/cat).glob("*.jpg"));order=np.random.RandomState(42).permutation(len(files));n=int(.8*len(files))
            take=order[:n] if split=="train" else order[n:]
            paths.extend(files[i] for i in take);labels.extend([c]*len(take))
    else:
        for rel in (root/"labels"/(split+"1.txt")).read_text().splitlines():
            paths.append(root/"images"/rel);labels.append(cats.index(rel.split("/")[0]))
    return paths,np.array(labels)
def rgb(path):
    with Image.open(path) as im:
        a=np.asarray(im.convert("RGB"))
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()
def image(path,tf):
    with Image.open(path) as im:return tf(im.convert("RGB"))
def model(backbone):
    if backbone=="clip_vitb16":
        m,_=clip.load("/Users/decoqwq/.cache/clip/ViT-B-16.pt",device="cpu",jit=False)
    else:
        repo=Q/"baselines/imported/r45-specialize/repository/research_common/extern/dinov2"
        m=torch.hub.load(str(repo),"dinov2_vits14",source="local",pretrained=False)
        p=Path("/Users/decoqwq/.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m/snapshots/4610ca143709d58a633b6397a74412c2c3842454/model.safetensors")
        state=load_file(str(p));state["mask_token"]=m.mask_token.detach();m.load_state_dict(state,strict=True)
    return m.float().eval().to("mps")
@torch.inference_mode()
def encode(m,b,paths,tf,pool):
    x=torch.stack(list(pool.map(lambda p:image(p,tf),paths))).to("mps")
    y=m.encode_image(x) if b=="clip_vitb16" else m(x)
    return F.normalize(y.float().cpu(),dim=-1)
def main():
    manifest={"precision":"MPS float32; unsupported bicubic CPU fallback; stored normalized float16","datasets":{}}
    for name,scope in [("eurosat","eurosat"),("dtd","dtd_test")]:
        qp,ql=dataset(name,"test");gp,gl=dataset(name,"train")
        gis=np.arange(len(gp))
        if name=="eurosat":
            rng=np.random.RandomState(129209)
            gis=np.array(sorted(int(i) for c in np.unique(gl) for i in rng.choice(np.flatnonzero(gl==c),500,replace=False)))
        with ThreadPoolExecutor(max_workers=8) as pool:
            qh=list(pool.map(rgb,qp));gh=list(pool.map(rgb,[gp[i] for i in gis]))
        seen=set();qkeep=[]
        for i,h in enumerate(qh):
            if h not in seen:seen.add(h);qkeep.append(i)
        banned=set(qh);seen=set();gkeep=[]
        for j,h in enumerate(gh):
            if h not in banned and h not in seen:seen.add(h);gkeep.append(j)
        if name=="dtd":assert sorted(set(range(len(qp)))-set(qkeep))==[427,436,439]
        ident={"query_ids":qkeep,"query_labels":ql[qkeep].tolist(),"query_rgb":[qh[i] for i in qkeep],
               "gallery_ids":gis[gkeep].tolist(),"gallery_labels":gl[gis[gkeep]].tolist(),"gallery_rgb":[gh[j] for j in gkeep],
               "gallery_removed":len(gis)-len(gkeep)}
        dump(B/"assets"/(name+"_local_identities.json"),ident)
        sample=np.array([np.flatnonzero(ql==c)[0] for c in np.unique(ql)])
        row={"query_total":len(qp),"query_unique":len(qkeep),"gallery_unique":len(gkeep),"parity":{}}
        manifest["datasets"][name]=row
        for bi,b in enumerate(BACKBONES):
            t=time.perf_counter();m=model(b);tf=T.Compose([T.Resize(224,interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(224),T.ToTensor(),T.Normalize(*NORM[bi])])
            with ThreadPoolExecutor(max_workers=6) as pool:
                new=torch.cat([encode(m,b,[qp[i] for i in sample[s:s+32]],tf,pool) for s in range(0,len(sample),32)])
                old=F.normalize(torch.load(B/"assets"/(scope+"_"+b+"_query.pt"),weights_only=True)["features"][sample].float(),dim=-1)
                cos=(old*new).sum(1);stats={"n":len(sample),"min_cosine":cos.min().item(),"mean_cosine":cos.mean().item(),"max_abs_difference":(old-new).abs().max().item(),"threshold_min_cosine":.999}
                row["parity"][b]=stats;dump(B/"assets/gallery_recovery_manifest.json",manifest)
                print("QUERY_PARITY",name,b,stats,flush=True)
                assert stats["min_cosine"]>=.999,"Image/cache/encoder parity insufficient"
                target=B/"assets"/(name+"_"+b+"_gallery_local.pt")
                chunks=[];n=len(gkeep)
                if target.exists():print("REUSE",target.name,flush=True)
                else:
                    chunkdir=B/"assets"/(name+"_"+b+"_gallery_chunks");chunkdir.mkdir(exist_ok=True)
                    for st in range(0,n,32):
                        part=chunkdir/("%06d.pt"%st)
                        if part.exists():f=torch.load(part,weights_only=True)
                        else:
                            f=encode(m,b,[gp[int(gis[gkeep[j]])] for j in range(st,min(st+32,n))],tf,pool).half()
                            assert torch.isfinite(f).all();torch.save(f,part)
                        chunks.append(f)
                        if st%320==0:print("GALLERY",name,b,st,n,"seconds",round(time.perf_counter()-t,1),flush=True)
                    torch.save({"ids":torch.tensor(ident["gallery_ids"]),"features":torch.cat(chunks)},target)
                row["parity"][b]["seconds"]=time.perf_counter()-t
            del m;torch.mps.empty_cache()
        dump(B/"assets/gallery_recovery_manifest.json",manifest)
    print("ASSET_RECOVERY_COMPLETE",flush=True)
if __name__=="__main__":main()
