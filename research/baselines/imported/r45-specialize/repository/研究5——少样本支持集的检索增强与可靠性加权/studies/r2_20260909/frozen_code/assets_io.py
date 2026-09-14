# -*- coding: utf-8 -*-
"""Read-only R1 assets and own R2 external-dataset feature preparation."""
import sys,json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
R1=ROOT/'续验11与12_20260909'
E12=ROOT/'实验12——少样本支持集的检索增强与可靠性加权'
sys.path.insert(0,str(E12))
from e12 import r3bridge as bridge
torch.set_num_threads(4)
BACKBONES=['clip_vitb16','dinov2_vits14']
ASSET=HERE/'assets';ASSET.mkdir(exist_ok=True)

def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def rgb(image):
    a=np.asarray(image.convert('RGB'))
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()

class Assets:
    def __init__(self):self.q={};self.g={}
    def query(self,name):
        if name in self.q:return self.q[name]
        ds=bridge.load_cls_dataset(name,'test')
        if name in ['cub200','eurosat']:
            pack=torch.load(ASSET/(name+'_query.pt'),weights_only=True)
            ids=pack['ids'].tolist();feats=pack['features'];hashes=pack['rgb']
        else:
            packs=[torch.load(R1/'assets'/'e12'/(name+'_test_'+b+'.pt'),weights_only=True) for b in BACKBONES]
            assert torch.equal(packs[0]['ids'],packs[1]['ids'])
            ids=packs[0]['ids'].tolist();feats=[F.normalize(p['features'],dim=-1) for p in packs]
            oldident=np.load(R1/f'identities_{name}_{"cifar100" if name=="cifar_fs" else name}.npz')['query']
            hashes=[str(oldident[i]) for i in ids]
        seen=set();keep=[]
        for j,h in enumerate(hashes):
            if h not in seen:seen.add(h);keep.append(j)
        ids=np.asarray(ids)[keep];labels=np.asarray(ds.labels)[ids];hs=[hashes[j] for j in keep]
        value={'features':[f[keep] for f in feats],'ids':ids,'labels':labels,'names':list(ds.classnames),'rgb':hs,
               'pools':{int(c):np.where(labels==c)[0] for c in np.unique(labels)}}
        self.q[name]=value;return value
    def gallery(self,name):
        if name in self.g:return self.g[name]
        ds=bridge.load_cls_dataset(name,'train')
        if name in ['cub200','eurosat']:
            pack=torch.load(ASSET/(name+'_gallery.pt'),weights_only=True);ids=pack['ids'].tolist();features=pack['features'];hashes=pack['rgb']
        else:
            packs=[torch.load(R1/'assets'/'e12'/(name+'_train_'+b+'_clean.pt'),weights_only=True) for b in BACKBONES]
            assert torch.equal(packs[0]['ids'],packs[1]['ids']);ids=packs[0]['ids'].tolist();features=[F.normalize(p['features'],dim=-1) for p in packs]
            ident=np.load(R1/f'identities_{"cifar_fs" if name=="cifar100" else name}_{name}.npz')['gallery'];hashes=[str(ident[i]) for i in ids]
        self.g[name]={'features':features,'ids':np.array(ids),'labels':np.asarray(ds.labels)[ids],'names':list(ds.classnames),'rgb':hashes}
        return self.g[name]
    def pair(self,qname,gname):
        q=self.query(qname);g=self.gallery(gname)
        banned=set(q['rgb']);seen=set();keep=[]
        for i,h in enumerate(g['rgb']):
            if h not in banned and h not in seen:keep.append(i);seen.add(h)
        return q,{**g,'features':[f[keep] for f in g['features']],'ids':g['ids'][keep],'labels':g['labels'][keep],'rgb':[g['rgb'][i] for i in keep]}

def prepare_external():
    t=time.perf_counter();manifest={}
    for name in ['cub200','eurosat']:
        queries=bridge.load_cls_dataset(name,'test');gallery=bridge.load_cls_dataset(name,'train')
        qis=list(range(len(queries)))
        if name=='eurosat':
            rng=np.random.RandomState(129209)
            gis=sorted(int(i) for c in np.unique(gallery.labels) for i in rng.choice(np.where(gallery.labels==c)[0],500,replace=False))
        else:gis=list(range(len(gallery)))
        def hash_ids(ds,ids):
            def one(i):return rgb(ds.get_image(i))
            with ThreadPoolExecutor(max_workers=8) as pool:return list(pool.map(one,ids))
        hpath=ASSET/(name+'_identities.json')
        if hpath.exists():ident=json.loads(hpath.read_text(encoding='utf-8'))
        else:
            ident={'query_ids':qis,'gallery_ids':gis,'query_rgb':hash_ids(queries,qis),'gallery_rgb':hash_ids(gallery,gis)};dump(hpath,ident)
        assert qis==ident['query_ids'] and gis==ident['gallery_ids']
        for side,ds,ids in [('query',queries,qis),('gallery',gallery,gis)]:
            target=ASSET/(name+'_'+side+'.pt')
            if target.exists():continue
            fs=[];timing={}
            for b in BACKBONES:
                tp=time.perf_counter()
                partial=ASSET/(name+'_'+side+'_'+b+'.pt')
                if partial.exists():f=torch.load(partial,weights_only=True)
                elif side=='query':
                    scope='cub200_test' if name=='cub200' else 'eurosat'
                    fc=bridge.FeatureCache('cls/'+scope,b,root=bridge.R3_CACHE_ROOT,read_only=True)
                    keys=[f'{b}|{scope}#{i}|v0|r224' for i in ids]
                    f=F.normalize(fc.get_many(keys),dim=-1);torch.save(f,partial)
                else:
                    tf=bridge.center_transform(b);chunks=[]
                    for st in range(0,len(ids),64):
                        xs=torch.stack([tf(ds.get_image(i)) for i in ids[st:st+64]])
                        # Match R3 cached queries' original fp16 autocast extraction precision.
                        with torch.autocast(device_type='cuda',dtype=torch.float16):f=bridge.encode(b,xs,'cuda')
                        chunks.append(f.half())
                        if st%640==0:print('external encode',name,b,st,'/',len(ids),flush=True)
                    f=F.normalize(torch.cat(chunks).float(),dim=-1);torch.save(f,partial)
                fs.append(f);timing[b]=time.perf_counter()-tp
            torch.save({'ids':torch.tensor(ids),'features':fs,'rgb':ident[side+'_rgb'],'timing':timing},target)
        manifest[name]={'query_pool':len(qis),'gallery_pool':len(gis),'query_unique_rgb':len(set(ident['query_rgb'])),'gallery_unique_rgb':len(set(ident['gallery_rgb'])),'rgb_overlap':len(set(ident['query_rgb'])&set(ident['gallery_rgb']))}
        print('external prepared',name,manifest[name],flush=True)
    dump(HERE/'external_preparation.json',{'datasets':manifest,'elapsed_seconds':time.perf_counter()-t,'gradient_updates':0})

if __name__=='__main__':prepare_external()
