# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from assets_io import Assets,bridge,HERE,BACKBONES,dump

def main():
    store=Assets();rows=[]
    for bi,b in enumerate(BACKBONES):
        xs=[];expected=[];items=[];tf=bridge.center_transform(b)
        for name in ['cub200','eurosat']:
            for side,split in [('query','test'),('gallery','train')]:
                state=getattr(store,side)(name);ds=bridge.load_cls_dataset(name,split)
                for j in [0,len(state['ids'])//2]:
                    i=int(state['ids'][j]);xs.append(tf(ds.get_image(i)));expected.append(state['features'][bi][j]);items.append([name,side,i])
        with torch.autocast(device_type='cuda',dtype=torch.float16):fresh=bridge.encode(b,torch.stack(xs),'cuda')
        old=torch.stack(expected);cos=F.cosine_similarity(old,fresh);err=(F.normalize(old,dim=-1)-F.normalize(fresh,dim=-1)).abs().amax(-1)
        for item,c,e in zip(items,cos,err):rows.append({'backbone':b,'item':item,'cosine':float(c),'max_abs_error':float(e)})
        assert float(cos.min())>.99999 and float(err.max())<.001,(b,cos,err)
    dump(HERE/'external_feature_parity.json',{'status':'passed','n_samples':len(rows),'rows':rows})
    print('external feature parity passed',len(rows),flush=True)

if __name__=='__main__':main()
