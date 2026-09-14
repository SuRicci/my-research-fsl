"""Spot-check reused features against real frozen encoders, retain provenance."""
import sys,json,hashlib,platform,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_e12 as e12
from src.config import CLIP_CKPT_DIR,DINOV2_VITS14_PATH
torch.set_num_threads(4)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    rows=[]
    for b in e12.BACKBONES:
        tf=e12.bridge.center_transform(b);xs=[];expect=[];items=[]
        for q,g in [('cifar_fs','cifar100'),('dtd','dtd'),('miniimagenet','miniimagenet')]:
            qs=e12.load_query(q,b);gs,_,_=e12.load_gallery(g,b)
            qds=e12.bridge.load_cls_dataset(q,'test');gds=e12.bridge.load_cls_dataset(g,'train')
            for ds,side,ids,features in [(qds,'query',list(qs[4]),qs[0]),(gds,'gallery',gs['ids'].tolist(),gs['features'])]:
                for j in [0,len(ids)//2]:
                    ix=ids[j];xs.append(tf(ds.get_image(ix)));expect.append(features[j]);items.append((q,g,side,ix))
        batch=torch.stack(xs)
        fp32=e12.bridge.encode(b,batch,'cuda')
        # R3 scripts/extract_features_cls.py explicitly uses CUDA fp16 autocast for query caches;
        # E12 run_pilot.py encodes the gallery in fp32. Reproduce each saved source's precision.
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            amp=e12.bridge.encode(b,batch,'cuda')
        fresh=torch.stack([amp[j] if item[2]=='query' else fp32[j] for j,item in enumerate(items)])
        old=torch.stack(expect)
        cos=F.cosine_similarity(fresh,old).numpy();err=(F.normalize(fresh,dim=-1)-F.normalize(old,dim=-1)).abs().amax(-1).numpy()
        fp32cos=F.cosine_similarity(fp32,old).numpy()
        for item,c,v,pc in zip(items,cos,err,fp32cos):rows.append({'backbone':b,'item':item,'cosine':float(c),'max_absolute_error':float(v),'fp32_comparison_cosine':float(pc),'reproduced_precision':'fp16_autocast' if item[2]=='query' else 'fp32'})
        assert np.min(cos)>.99999 and np.max(err)<.001,(b,cos,err)
    weights=[CLIP_CKPT_DIR/'ViT-B-16.pt',Path(DINOV2_VITS14_PATH)]
    r={'status':'passed','n_images_compared':len(rows),'rows':rows,'torch':torch.__version__,'python':sys.version,'gpu':torch.cuda.get_device_name(0),'weights':{str(p):digest(p) for p in weights},'created_at':time.time()}
    (HERE/'feature_parity.json').write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf-8')
    print('feature parity',len(rows),'passed',flush=True)

if __name__=='__main__':main()
