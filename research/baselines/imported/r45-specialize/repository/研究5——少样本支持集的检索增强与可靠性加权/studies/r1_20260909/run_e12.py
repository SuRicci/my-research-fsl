"""Bounded, label-blind retrieval continuation. Predictions and manifests retained."""
import os, sys, json, time, hashlib, argparse
from pathlib import Path
from itertools import product
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'research_directions.json').is_file())
sys.path.insert(0,str(ROOT/'研究5——少样本支持集的检索增强与可靠性加权'))
from e12 import r3bridge as bridge
from e12 import augment

BACKBONES=['clip_vitb16','dinov2_vits14']
CELLS=[('cifar_fs','cifar100'),('dtd','dtd'),('miniimagenet','miniimagenet'),('dtd','miniimagenet')]
OUT=HERE/'e12'; OUT.mkdir(exist_ok=True)
CACHE=HERE/'assets'/'e12';CACHE.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)

def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def bootstrap(d,seed=909):
    d=np.asarray(d,dtype=float);rng=np.random.default_rng(seed)
    vals=np.concatenate([d[rng.integers(len(d),size=(100,len(d)))].mean(1) for _ in range(30)])
    return [float(x) for x in np.quantile(vals,[.025,.975])]

def load_query(name,b):
    scope=name+'_test';labels,names=bridge.load_classnames(scope)
    pools=bridge.query_cached_pools(labels,scope,b)
    ids=sorted(set(int(i) for p in pools.values() for i in p))
    fp=CACHE/(scope+'_'+b+'.pt')
    if fp.exists():data=torch.load(fp,weights_only=True)
    else:
        data={'ids':torch.tensor(ids),'features':bridge.load_query_features(scope,b,ids)}
        torch.save(data,fp)
    assert data['ids'].tolist()==ids
    lut={v:i for i,v in enumerate(ids)}
    ident=np.load(HERE/f'identities_{name}_{"cifar100" if name=="cifar_fs" else name}.npz')['query']
    seen=set();valid=set()
    for i in ids:
        if str(ident[i]) not in seen:
            valid.add(i);seen.add(str(ident[i]))
    pools={c:np.array([i for i in v if int(i) in valid],dtype=np.int64) for c,v in pools.items()}
    dump(OUT/(scope+'_query_dedup.json'),{'excluded_indices':sorted(set(ids)-valid),'remaining':len(valid)})
    return F.normalize(data['features'],dim=-1),pools,labels,names,lut

def load_gallery(name,b):
    ds=bridge.load_cls_dataset(name,'train'); scope=name+'_train'
    ident=np.load(HERE/f'identities_{"cifar_fs" if name=="cifar100" else name}_{name}.npz')
    hg=ident['gallery'].tolist()
    forbidden=set()
    for path in HERE.glob('identities_*.npz'):
        forbidden.update(np.load(path)['query'].tolist())
    seen=set();keep=[]
    for i,h in enumerate(hg):
        if h in forbidden or h in seen:continue
        seen.add(h);keep.append(i)
    fp=CACHE/(scope+'_'+b+'_clean.pt')
    if fp.exists():data=torch.load(fp,weights_only=True)
    else:
        fc=bridge.gallery_cache(scope,b)
        G=fc.get_many([bridge.gallery_key(b,scope,i) for i in keep])
        G=F.normalize(G,dim=-1)
        t=time.perf_counter();print('density start',name,b,len(keep),flush=True)
        hub=augment.precompute_hub(G,k=16,chunk=512)
        data={'features':G,'hub':hub,'ids':torch.tensor(keep),'density_seconds':time.perf_counter()-t}
        torch.save(data,fp)
    assert data['ids'].tolist()==keep
    return data, np.asarray(ds.labels)[keep], list(ds.classnames)

def episodes(pools,phase,shot):
    classes=np.array(sorted(c for c,v in pools.items() if len(v)>=shot+16))
    order=np.random.RandomState(120909).permutation(classes)
    allowed=order[:len(order)//2] if phase=='dev' else order[len(order)//2:]
    perseed=200;seeds=[129090] if phase=='dev' else [129091,129092,129093]
    es=[]
    for seed in seeds:
        rng=np.random.RandomState(seed+shot)
        for _ in range(perseed):
            cs=rng.choice(allowed,5,replace=False)
            sidx=[];qidx=[];tidx=[]
            for c in cs:
                nk=min(25,len(pools[int(c)])-15)
                ix=rng.choice(pools[int(c)],nk+15,replace=False)
                sidx.append(ix[:shot]);qidx.append(ix[nk:]);tidx.append(np.pad(ix[:nk],(0,25-nk),constant_values=-1))
            # Nested original support, at most 25 real labeled examples, never pad with real images.
            es.append((seed,cs,np.stack(sidx),np.stack(qidx),np.stack(tidx)))
    return es,allowed.tolist()

def config_grid():
    cfg=[{'name':'raw','family':'raw','r':0,'mix':0.,'tau':-100.}]
    for family in ['uniform_mix','weighted_mix','uniform_gate','weighted_gate']:
        taus=[-100.] if family.endswith('mix') else [-100.,-.3,-.2,-.1,0.,.1,100.]
        for r,mix,tau in product([16,64],[0.,.25,.5,.75,1.],taus):
            cfg.append({'name':f'{family}_r{r}_m{mix}_t{tau}','family':family,'r':r,'mix':mix,'tau':tau})
    return cfg

CONFIGS=config_grid()

@torch.no_grad()
def evaluate_state(qstate,gstate,es,glabels,gnames,qnames,phase,cell,b,shot):
    feats,pools,labels,names,lut=qstate
    G=gstate['features'].cuda();hub=gstate['hub'].cuda()
    ss=torch.stack([feats[[lut[int(i)] for i in e[2].ravel()]].reshape(5,shot,-1) for e in es]).cuda()
    xx=torch.stack([feats[[lut[int(i)] for i in e[3].ravel()]] for e in es]).cuda()
    tt=torch.stack([torch.stack([feats[lut[int(i)]] if i>=0 else torch.zeros(feats.shape[-1]) for i in e[4].ravel()]).reshape(5,25,-1) for e in es]).cuda()
    p=F.normalize(ss.mean(2),dim=-1)
    val=[];ind=[]; t0=time.perf_counter()
    for st in range(0,len(es),32):
        sim=(p[st:st+32].reshape(-1,p.shape[-1])@G.T).reshape(-1,5,len(G))
        v,i=sim.topk(64,dim=-1);val.append(v);ind.append(i)
    coh=torch.cat(val);idx=torch.cat(ind);cand=G[idx]
    torch.cuda.synchronize(); retrieval_s=time.perf_counter()-t0
    yq=np.repeat(np.arange(5),15)[None,:]
    rawpred=(xx@p.transpose(1,2)).argmax(-1).cpu().numpy().astype('int8')
    pred={'raw':rawpred,'true25_diag':(xx@F.normalize(tt.mean(2),dim=-1).transpose(1,2)).argmax(-1).cpu().numpy().astype('int8')}
    diag={}; rprotos={}
    for r in [16,64]:
        cc=cand[:,:,:r];cv=coh[:,:,:r];hh=hub[idx[:,:,:r]]
        sim=torch.einsum('ecrd,ekd->ecrk',cc,p)
        mask=torch.eye(5,device='cuda',dtype=torch.bool)[None,:,None,:]
        margin=cv-sim.masked_fill(mask,-torch.inf).amax(-1)
        w=torch.sigmoid(8*margin+2*(cv-cv.mean(-1,keepdim=True)))/(1+hh)
        un=F.normalize(cc.mean(2),dim=-1)
        we=F.normalize((w[...,None]*cc).sum(2)/w.sum(2)[...,None],dim=-1)
        coverage=(cv-hh).mean(-1)
        rprotos[r]=(un,we,coverage)
        gid=idx[:,:,:r].cpu().numpy(); purity=[]
        namemap={n:i for i,n in enumerate(gnames)}
        for j,e in enumerate(es):
            match=np.array([namemap.get(qnames[int(c)],-1) for c in e[1]])
            purity.append((glabels[gid[j]]==match[:,None]).mean(-1))
        diag[str(r)]={'coverage':coverage.cpu().numpy(),'purity':np.asarray(purity),'weights_mean':w.mean(-1).cpu().numpy()}
    if phase=='dev':selected=CONFIGS
    else:
        sel=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))[b+'_'+str(shot)]
        selected=[CONFIGS[0]]+[sel[f] for f in ['uniform_mix','weighted_mix','uniform_gate','weighted_gate']]
    for cfg in selected:
        if cfg['family']=='raw':continue
        un,we,cov=rprotos[cfg['r']]
        proto=we if cfg['family'].startswith('weighted') else un
        mix=cfg['mix']*(cov>=cfg['tau']).float()
        adjusted=F.normalize((1-mix[...,None])*p+mix[...,None]*proto,dim=-1)
        pred[cfg['name']]=(xx@adjusted.transpose(1,2)).argmax(-1).cpu().numpy().astype('int8')
    # Original formula at previously selected pilot configuration: historical reference only.
    old=json.loads(((ROOT/'研究5——少样本支持集的检索增强与可靠性加权')/'results'/'pilot_main.json').read_text(encoding='utf-8'))
    meta=next(m for m in old['metas'] if m['query']==cell[0]+'_test' and m['gallery']==cell[1]+'_train' and m['backbone']==b and m['shot']==shot)
    cfg=meta['dev_best_cfg'];r=cfg['r'];cc=cand[:,:,:r];cv=coh[:,:,:r];hh=hub[idx[:,:,:r]]
    sim=torch.einsum('ecrd,ekd->ecrk',cc,p)
    mask=torch.eye(5,device='cuda',dtype=torch.bool)[None,:,None,:]
    margin=cv-sim.masked_fill(mask,-torch.inf).amax(-1)
    w=torch.sigmoid(cfg['alpha']*margin+cfg['beta']*(cv-cv.mean(-1,keepdim=True)))/(1+cfg['gamma']*hh)
    for key,add in [('legacy_uniform',cc.sum(2)),('legacy_weighted',(w[...,None]*cc).sum(2))]:
        pr=F.normalize(ss.sum(2)+add,dim=-1)
        pred[key]=(xx@pr.transpose(1,2)).argmax(-1).cpu().numpy().astype('int8')
    acc={key:(v==yq).mean(1) for key,v in pred.items()}
    stem=f'{cell[0]}_{cell[1]}_{b}_k{shot}_{phase}'
    payload={'support_indices':np.stack([e[2] for e in es]),'query_indices':np.stack([e[3] for e in es]),
             'true25_indices':np.stack([e[4] for e in es]),'class_ids':np.stack([e[1] for e in es]),
             'seed':np.array([e[0] for e in es]),'yq':yq,'config_names':np.array(list(pred)),
             'predictions':np.stack(list(pred.values()))}
    for r,d in diag.items():
        for key,value in d.items():payload[f'r{r}_{key}']=value
    np.savez_compressed(OUT/(stem+'.npz'),**payload)
    dump(OUT/(stem+'.json'),{'means':{k:float(v.mean()) for k,v in acc.items()},'retrieval_seconds':retrieval_s,
                             'gallery_n':len(G),'density_seconds':gstate['density_seconds'],'episodes':len(es)})
    print(stem,'raw',round(float(acc['raw'].mean()),4),'done',flush=True)
    return acc

def summarize():
    selections=json.loads((OUT/'selection.json').read_text(encoding='utf-8'));rows=[]
    for path in sorted(OUT.glob('*_retest.npz')):
        d=np.load(path);names=d['config_names'].tolist();pred=d['predictions'];corr=pred==d['yq'];acc=corr.mean(-1)
        stem=path.stem
        for b in BACKBONES:
            if b in stem:break
        shot=int(stem.split('_k')[1].split('_')[0]);sel=selections[b+'_'+str(shot)]
        mapping={'raw':'raw','true25_diag':'true25_diag','legacy_uniform':'legacy_uniform','legacy_weighted':'legacy_weighted'}
        mapping.update({f:c['name'] for f,c in sel.items()})
        row={'cell':stem,'shot':shot,'backbone':b,'n_episodes':len(d['seed'])}
        for family,key in mapping.items():
            j=names.index(key);v=acc[j];row[family]=float(v.mean())
            row[family+'_seed_means']={str(s):float(v[d['seed']==s].mean()) for s in np.unique(d['seed'])}
        for a,ref in [('uniform_mix','raw'),('weighted_mix','uniform_mix'),('uniform_gate','uniform_mix'),('weighted_gate','uniform_gate'),('weighted_gate','uniform_mix'),('weighted_gate','raw')]:
            j,k=names.index(mapping[a]),names.index(mapping[ref]);diff=acc[j]-acc[k]
            key=a+'_minus_'+ref;row[key]=float(diff.mean());row[key+'_ci95']=bootstrap(diff)
            classdiff=[]
            for c in np.unique(d['class_ids']):
                z=[]
                for e,cs in enumerate(d['class_ids']):
                    at=np.where(cs==c)[0]
                    if len(at):
                        sl=slice(int(at[0])*15,(int(at[0])+1)*15)
                        z.append(float(corr[j,e,sl].mean()-corr[k,e,sl].mean()))
                classdiff.append(np.mean(z))
            row[key+'_class_macro_ci95']=bootstrap(classdiff)
        for family in ['uniform_gate','weighted_gate']:
            cfg=sel[family];cv=d[f'r{cfg["r"]}_coverage'];row[family+'_fallback_rate']=float((cv<cfg['tau']).mean())
        rows.append(row)
    dump(OUT/'summary.json',{'rows':rows,'ci_scope':'Conditional finite-pool episode bootstrap plus class-macro sensitivity; no new external dataset confirmation.'})
    print('summary saved',len(rows),'cells',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','retest','summarize'],required=True);args=ap.parse_args()
    if args.phase=='summarize':summarize();return
    scores={};manifest={}
    for b in BACKBONES:
        qcache={};gcache={}
        for q,g in CELLS:
            if q not in qcache:qcache[q]=load_query(q,b)
            if g not in gcache:gcache[g]=load_gallery(g,b)
            qs=qcache[q];gs,gl,gn=gcache[g]
            for shot in [1,5]:
                es,allowed=episodes(qs[1],args.phase,shot)
                manifest[f'{q}_{b}_{shot}']={'classes':allowed,'episodes':len(es)}
                acc=evaluate_state(qs,gs,es,gl,gn,qs[3],args.phase,(q,g),b,shot)
                scores.setdefault(b+'_'+str(shot),[]).append({k:float(v.mean()) for k,v in acc.items()})
    dump(OUT/('manifest_'+args.phase+'.json'),manifest)
    if args.phase=='dev':
        selections={}
        for key,cells in scores.items():
            selections[key]={}
            for family in ['uniform_mix','weighted_mix','uniform_gate','weighted_gate']:
                cc=[c for c in CONFIGS if c['family']==family]
                # Stable tie break: grid order prioritizes no augmentation, shallow retrieval.
                best=max(cc,key=lambda c:np.mean([d[c['name']] for d in cells]))
                selections[key][family]={**best,'dev_macro':float(np.mean([d[best['name']] for d in cells]))}
        dump(OUT/'selection.json',selections)
        dump(OUT/'selection_receipt.json',{'protocol_sha256':hashlib.sha256((HERE/'protocol_v1.json').read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'created_at':time.time()})
        print('configuration frozen',selections,flush=True)
    else:summarize()

if __name__=='__main__':main()
