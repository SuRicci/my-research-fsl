# -*- coding: utf-8 -*-
"""Develop once, freeze, retest and validate E12 on new dataset conditions."""
import argparse,json,time,hashlib
from pathlib import Path
from layout_compat import verify_artifact
import numpy as np
import torch
from assets_io import Assets,HERE,R1,dump
from methods import grid,representation,evaluate_configs,config_name

OUT=HERE/'outputs';OUT.mkdir(exist_ok=True)
DEV_CELLS=[('cifar_fs','cifar100'),('dtd','dtd'),('miniimagenet','miniimagenet'),('dtd','miniimagenet')]
EXT_CELLS=[('cub200','cub200'),('eurosat','eurosat'),('cub200','miniimagenet'),('eurosat','miniimagenet')]
CONTROLS={'raw','raw_ridge','uniform'}

def episodes(q,phase,shot):
    classes=np.array(sorted(c for c,v in q['pools'].items() if len(v)>=shot+15))
    if phase!='external':
        order=np.random.RandomState(120909).permutation(classes)
        classes=order[:len(order)//2] if phase=='dev' else order[len(order)//2:]
    seeds={'dev':[129190],'retest':[129191,129192,129193],'external':[129291,129292,129293]}[phase]
    records=[]
    for seed in seeds:
        rng=np.random.RandomState(seed+shot)
        for _ in range(200):
            cs=rng.choice(classes,5,replace=False)
            ix=np.stack([rng.choice(q['pools'][int(c)],shot+15,replace=False) for c in cs])
            records.append({'seed':seed,'classes':cs,'support':ix[:,:shot],'query':ix[:,shot:]})
    return records,classes

def r1_configs(shot):
    sel=json.loads((R1/'e12'/'selection.json').read_text(encoding='utf-8'));out=[]
    for b,w in [('dinov2_vits14',0.),('clip_vitb16',1.)]:
        c=sel[b+'_'+str(shot)]['uniform_mix'];cfg={'family':'uniform','w':w,'r':c['r'],'mix':c['mix'],'name':'r1_'+b}
        out.append(cfg)
    return out

def evaluate_cell(store,cell,shot,phase,configs):
    q,g=store.pair(*cell);es,allowed=episodes(q,phase,shot)
    sidx=np.stack([e['support'] for e in es]);qidx=np.stack([e['query'] for e in es]);classids=np.stack([e['classes'] for e in es]);seed=np.array([e['seed'] for e in es])
    yq=np.repeat(np.arange(5),15)
    preds={};scores={};times={};total=time.perf_counter()
    for w in sorted({c['w'] for c in configs}):
        qf=representation(*q['features'],w);G=representation(*g['features'],w).cuda()
        S=qf[sidx].cuda();X=qf[qidx.reshape(len(es),-1)].cuda()
        selected=[c for c in configs if c['w']==w]
        pp,ss,tt=evaluate_configs(S,X,G,selected,return_scores=(phase!='dev'))
        preds.update({n:v.numpy().astype('int8') for n,v in pp.items()})
        scores.update({n:v.numpy().astype('float32') for n,v in ss.items()});times[str(w)]=tt
        del S,X,G
    stem=f'{cell[0]}_{cell[1]}_k{shot}_{phase}'
    payload={'names':np.array(list(preds)),'predictions':np.stack(list(preds.values())),'yq':yq,'seed':seed,'class_ids':classids,
             'support_indices':q['ids'][sidx],'query_indices':q['ids'][qidx],
             'support_rgb':np.array(q['rgb'])[sidx],'query_rgb':np.array(q['rgb'])[qidx],
             'gallery_ids':g['ids'],'gallery_rgb':np.array(g['rgb'])}
    if scores:payload['scores']=np.stack([scores[n] for n in preds])
    np.savez_compressed(OUT/(stem+'.npz'),**payload)
    means={n:float((p==yq).mean()) for n,p in preds.items()}
    dump(OUT/(stem+'.json'),{'means':means,'elapsed_seconds':time.perf_counter()-total,'timings':times,'gallery_n':len(g['ids']),'query_pool_n':len(q['ids']),'query_classes':allowed.tolist(),'episodes':len(es),'shot':shot,'cell':list(cell),'phase':phase})
    print(stem,'best',max(means,key=means.get),round(max(means.values()),4),'R1_DINO',round(means['r1_dinov2_vits14'],4),flush=True)
    return means

def freeze(allmeans):
    configs=grid();result={}
    for shot,rows in allmeans.items():
        macro={c['name']:float(np.mean([r[c['name']] for r in rows])) for c in configs}
        control=max([c for c in configs if c['family'] in CONTROLS],key=lambda c:macro[c['name']])
        candidate=max([c for c in configs if c['family'] not in CONTROLS],key=lambda c:macro[c['name']])
        overall=max(configs,key=lambda c:macro[c['name']])
        # Include best single-DINO policies to attribute any upgrade to representation vs mechanism.
        single=max([c for c in configs if c['w']==0],key=lambda c:macro[c['name']])
        result[str(shot)]={'control':control,'candidate':candidate,'overall':overall,'single_dino':single,
                          'dev_control':macro[control['name']],'dev_candidate':macro[candidate['name']],
                          'dev_overall':macro[overall['name']],'dev_single_dino':macro[single['name']]}
    dump(HERE/'selection.json',result)
    frozen=HERE/'frozen_code';frozen.mkdir(exist_ok=True);hashes={}
    for name in ['methods.py','run.py','assets_io.py','protocol.json']:
        src=(HERE/name).read_bytes();(frozen/name).write_bytes(src);hashes[name]=hashlib.sha256(src).hexdigest()
    dump(HERE/'selection_receipt.json',{'time':time.time(),'hashes':hashes,'configs':len(configs),'no_external_score_viewed':True})
    print('FROZEN',json.dumps(result),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','retest','external'],required=True);args=ap.parse_args()
    store=Assets();allmeans={1:[],5:[]}
    for cell in EXT_CELLS if args.phase=='external' else DEV_CELLS:
        for shot in [1,5]:
            if args.phase=='dev':configs=grid()+r1_configs(shot)
            else:
                sel=json.loads((HERE/'selection.json').read_text())[str(shot)]
                selected=[sel[k] for k in ['control','candidate','overall','single_dino']]
                # Ablation: raw prototype with the exact selected representation, no gallery.
                raw={'family':'raw','w':sel['overall']['w']};raw['name']=config_name(raw)
                configs=list({c['name']:c for c in selected+r1_configs(shot)+[raw]}.values())
                rec=json.loads((HERE/'selection_receipt.json').read_text())
                for name,h in rec['hashes'].items():verify_artifact(HERE/name,h)
            allmeans[shot].append(evaluate_cell(store,cell,shot,args.phase,configs))
    if args.phase=='dev':freeze(allmeans)
    dump(HERE/(args.phase+'_completion.json'),{'complete':True,'time':time.time(),'cells':8})

if __name__=='__main__':main()
