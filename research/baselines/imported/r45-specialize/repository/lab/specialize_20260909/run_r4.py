import argparse,sys,time
from pathlib import Path
import numpy as np
from campaign_common import ROOT,HERE,OUT,read,dump,sha,validate_sources
from r4_candidates import run,prepare
from r4_refine import read_dataset,METHODS
from src.lab_metrics import evaluate_rank

FIRST=Path('/data/liuhaoyu/r45-refine-20260909')
HOL=Path('/data/liuhaoyu/r45-followup-20260909/r4_holidays')

def dataset(name):
    if name=='holidays':
        m=read(HOL/'manifest.json');a={}
        for n in ['clip_b32','clip_b16','dino_s14']:
            p=HOL/'cache'/n/'embeddings.npy';meta=read(p.with_suffix('.json'))
            assert sha(p)==meta['array_sha256'] and sha(HOL/'manifest.json')==meta['manifest_sha256'];a[n]=np.load(p)
        qi=[m['gallery_names'].index(q) for q in m['query_names']]
        truth=[[m['truth_evaluator_only'][q]] for q in m['query_names']]
        return m,a,np.arange(len(m['records'])),qi,m['query_names'],truth,['map']
    m,a=read_dataset(name);ix={r['id']:i for i,r in enumerate(m['records'])}
    qids=m['roles']['query'];truth=[[m['truth_evaluator_only'][q][c] for c in ['medium','hard']] for q in qids]
    return m,a,[ix[i] for i in m['roles']['gallery']],[ix[i] for i in qids],qids,truth,['medium','hard']

def bridge_lookup(name,m,a):
    if name!='holidays':
        other='rparis' if name=='roxford' else 'roxford';em,ea=read_dataset(other)
        lookup={r['id']:(a,i) for i,r in enumerate(m['records'])}
        lookup.update({other+':'+r['id']:(ea,i) for i,r in enumerate(em['records'])});return lookup
    lookup={}
    for ds in ['roxford','rparis']:
        mm,aa=read_dataset(ds);permitted=set(mm['roles']['gallery'])
        for i,r in enumerate(mm['records']):
            if r['id'] in permitted:lookup[ds+':'+r['id']]=(aa,i)
            elif ds=='roxford' and r['id'] in mm['roles']['bridge']:lookup[r['id']]=(aa,i)
    return lookup

def measure(rank,truth):
    return np.array([[evaluate_rank(r,t,'revisited')['ap'] for t in ts] for r,ts in zip(rank,truth)])

def parent(name,path):
    meta=read(path);f=path.parent
    if name=='holidays':
        p=f/'predictions.npz';assert sha(p)==meta['sha256'];z=np.load(p,allow_pickle=False)
        names=list(z['names']);arrays={n:z['ap'][i,:,None] for i,n in enumerate(names)};inputs=[dict(path=str(p),sha256=sha(p))]
    else:
        p=f/'aps.npz';assert sha(p)==meta['aps_sha256'];z=np.load(p,allow_pickle=False)
        arrays={n:z['ap'][i] for i,n in enumerate(z['names'])};gp=FIRST/'r4_geometry'/name/f.name/'aps.npz';gz=np.load(gp,allow_pickle=False)
        gm=read(gp.parent/'complete.json');assert sha(gp)==gm['aps_sha256']
        arrays.update({n:gz['ap'][i] for i,n in enumerate(gz['names'])});inputs=[dict(path=str(p),sha256=sha(p)),dict(path=str(gp),sha256=sha(gp))]
    return meta,arrays,inputs

def execute(method):
    validate_sources();cfg=read(HERE/'candidates.json')[method];dest=OUT/method;dest.mkdir(parents=True,exist_ok=True)
    for name in ['roxford','rparis','holidays']:
        m,a,gi,qi,qids,truth,conditions=dataset(name);lookup=bridge_lookup(name,m,a)
        source=HOL/'evaluation' if name=='holidays' else FIRST/'r4'/name
        paths=sorted(source.glob('*/complete.json'));assert len(paths)==60
        prepared={}
        for p in paths:
            folder=dest/name/p.parent.name
            if (folder/'complete.json').exists():
                assert sha(folder/'predictions.npz')==read(folder/'complete.json')['sha256'];continue
            meta,bases,inputs=parent(name,p);old,new=meta['pair'];go,qo,qn=a[old][gi],a[old][qi],a[new][qi]
            chosen=[lookup[i] for i in meta['bridge_ids']];ao=np.stack([aa[old][i] for aa,i in chosen]);an=np.stack([aa[new][i] for aa,i in chosen])
            if old not in prepared:
                t=time.time();prepared[old]=prepare(go,qo,method,cfg);print('PREPARED',method,name,old,round(time.time()-t,2),flush=True)
            t=time.time();scores,rank,base,info=run(go,qo,qn,ao,an,method,cfg,prepared[old])
            if rank is None:
                assert np.isfinite(scores).all();rank=np.argsort(-scores.astype('float32'),axis=1,kind='stable')
            ap=measure(rank,truth);parity={}
            for n in ['old_centered','v0_press','centered_residual_2.0']:
                rr=np.argsort(-base[n].astype('float32'),axis=1,kind='stable');b=measure(rr,truth)
                error=float(np.nanmax(np.abs(np.nanmean(b,0)-np.nanmean(bases[n],0))));assert error<3e-5,(name,method,n,error)
                parity[n]=error
            bn=list(METHODS)+['old_centered','centered_residual_2.0','oracle_new_new']
            folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'predictions.npz',rankings=rank.astype('uint16'),ap=ap,
                baseline_names=bn,baseline_ap=np.stack([bases[n] for n in bn]),query_ids=qids,conditions=conditions)
            dump(folder/'complete.json',dict(method=method,dataset=name,pair=meta['pair'],budget=meta['budget'],seed=meta['seed'],policy=meta['policy'],
                bridge_ids=meta['bridge_ids'],parent_complete_sha256=sha(p),inputs=inputs,sha256=sha(folder/'predictions.npz'),
                means=np.nanmean(ap,0).tolist(),conditions=conditions,baseline_replay_max_map_error=parity,diagnostics=info,seconds=time.time()-t))
            print('COMPLETE',method,name,p.parent.name,np.round(np.nanmean(ap,0)*100,3).tolist(),flush=True)
    summarize(method)

def summarize(method):
    groups={};paths=sorted((OUT/method).glob('*/*/complete.json'));assert len(paths)==180
    for p in paths:
        m=read(p);assert sha(p.parent/'predictions.npz')==m['sha256'];z=np.load(p.parent/'predictions.npz',allow_pickle=False)
        groups.setdefault((m['dataset'],*m['pair'],m['budget'],m['policy']),[]).append((m,z))
    rows=[]
    for key,packs in sorted(groups.items()):
        packs.sort(key=lambda x:x[0]['seed']);assert len(packs)==5
        ap=np.stack([z['ap'] for m,z in packs]);bas=np.stack([z['baseline_ap'] for m,z in packs]);bn=list(packs[0][1]['baseline_names'])
        for ci,c in enumerate(packs[0][1]['conditions']):
            means={n:float(np.nanmean(bas[:,i,:,ci])) for i,n in enumerate(bn)}
            strongest=max(list(METHODS)+['old_centered','centered_residual_2.0'],key=means.get);comparisons={}
            for base in dict.fromkeys(['old_old','old_centered','v0_press','centered_residual_2.0',strongest]):
                d=ap[:,:,ci]-bas[:,bn.index(base),:,ci];rng=np.random.default_rng(902191)
                boot=[np.nanmean(d[rng.integers(5,size=5)][:,rng.integers(d.shape[1],size=d.shape[1])]) for _ in range(1000)]
                gain=float(np.nanmean(d)*100);ci95=(100*np.nanquantile(boot,[.025,.975])).tolist();seedg=np.nanmean(d,1)*100
                comparisons[base]=dict(gain_pp=gain,ci95_pp=ci95,seed_gain_pp=seedg.tolist(),positive_seeds=int((seedg>0).sum()),
                    clear_local_gain=bool(gain>=1 and ci95[0]>0 and (seedg>0).sum()>=4))
            rows.append(dict(dataset=key[0],pair=list(key[1:3]),budget=key[3],policy=key[4],condition=str(c),map=float(np.nanmean(ap[:,:,ci])),
                baseline_means=means,strongest_baseline=strongest,comparisons=comparisons))
    dump(OUT/method/'summary.json',dict(complete=True,method=method,cells=180,rows=rows,protocol_sha256=sha(OUT/'protocol.json'),
        scope='Historically viewed images; full condition reporting; unadjusted paired query/scene and bridge-seed bootstrap; Oxford/Paris same-landmark dependence remains'))
    print('METHOD_ALL_COMPLETE',method,'rows',len(rows),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('method');args=parser.parse_args();execute(args.method)
