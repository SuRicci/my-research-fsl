# -*- coding: utf-8 -*-
"""Reaggregate paired predictions; never choose configurations here."""
import json
import numpy as np
from assets_io import HERE,dump

def bootstrap(d,seed=129):
    d=np.asarray(d,float);rng=np.random.default_rng(seed)
    out=np.concatenate([d[rng.integers(len(d),size=(100,len(d)))].mean(1) for _ in range(50)])
    return np.quantile(out,[.025,.975]).tolist()

def main():
    sel=json.loads((HERE/'selection.json').read_text());rows=[];paired={}
    for path in sorted((HERE/'outputs').glob('*.npz')):
        if '_dev.' in path.name:continue
        d=np.load(path);meta=json.loads(path.with_suffix('.json').read_text());shot=str(meta['shot']);names=d['names'].tolist()
        corr=d['predictions']==d['yq'];acc=corr.mean(-1);s=sel[shot]
        labels={'r1_dino':'r1_dinov2_vits14','r1_clip':'r1_clip_vitb16',**{k:s[k]['name'] for k in ['control','candidate','overall','single_dino']}}
        row={'cell':meta['cell'],'shot':int(shot),'phase':meta['phase'],'n_episodes':len(d['seed']),'n_distinct_query_images':len(np.unique(d['query_rgb'])),'gallery_n':meta['gallery_n']}
        for label,name in labels.items():
            a=acc[names.index(name)];row[label]=float(a.mean());row[label+'_seed_means']={str(k):float(a[d['seed']==k].mean()) for k in np.unique(d['seed'])}
        for a,b in [('overall','r1_dino'),('candidate','control'),('overall','control'),('single_dino','r1_dino')]:
            ia=names.index(labels[a]);ib=names.index(labels[b]);diff=acc[ia]-acc[ib];key=a+'_minus_'+b
            row[key]={'mean':float(diff.mean()),'ci95':bootstrap(diff)}
            classdiff=[]
            for c in np.unique(d['class_ids']):
                sub=[]
                for j,cs in enumerate(d['class_ids']):
                    loc=np.where(cs==c)[0]
                    if len(loc):
                        q=slice(int(loc[0])*15,(int(loc[0])+1)*15)
                        sub.append(float(corr[ia,j,q].mean()-corr[ib,j,q].mean()))
                classdiff.append(float(np.mean(sub)))
            row[key]['class_macro_mean']=float(np.mean(classdiff));row[key]['class_macro_ci95']=bootstrap(classdiff)
            paired.setdefault((meta['phase'],shot,key),[]).append(diff)
        rows.append(row)
    pooled={}
    for (phase,shot,key),values in paired.items():
        # One shared episode-resampling index preserves the paired DTD/CUB/EuroSAT cross-gallery draws.
        d=np.stack(values).mean(0)
        pooled.setdefault(phase,{}).setdefault(shot,{})[key]={'mean':float(d.mean()),'ci95':bootstrap(d)}
    gates={}
    for phase in ['retest','external']:
        gates[phase]={}
        for shot in [1,5]:
            rr=[r for r in rows if r['phase']==phase and r['shot']==shot]
            v=pooled[phase][str(shot)]['overall_minus_r1_dino']
            shared=[r for r in rr if r['cell'] in [['cifar_fs','cifar100'],['dtd','dtd'],['cub200','cub200'],['eurosat','eurosat']]]
            stress=[r for r in rr if r not in shared]
            g={'macro_gain_ge_0_5pp_and_ci_positive':v['mean']>=.005 and v['ci95'][0]>0,
               'shared_loss_within_0_5pp':all(r['overall_minus_r1_dino']['mean']>=-.005 for r in shared),
               'stress_loss_within_1pp':all(r['overall_minus_r1_dino']['mean']>=-.01 for r in stress)}
            g['practical_upgrade_passed']=all(g.values());gates[phase][str(shot)]=g
    practical=all(x['practical_upgrade_passed'] for p in gates.values() for x in p.values())
    same_new=[r for r in rows if r['phase']=='external' and r['cell'][0]==r['cell'][1]]
    scientific={str(k):all(pooled[p][str(k)]['candidate_minus_control']['mean']>=.01 and pooled[p][str(k)]['candidate_minus_control']['ci95'][0]>0 for p in ['retest','external']) and all(r['candidate_minus_control']['mean']>0 for r in same_new if r['shot']==k) for k in [1,5]}
    result={'rows':rows,'pooled':pooled,'gates':gates,'practical_upgrade_passed':practical,'new_method_increment_gate':scientific,
            'scope':'Conditional paired finite-pool estimates. CUB/EuroSAT new to E12 evaluation but previously used by other projects. No selection on retest/external.'}
    dump(HERE/'summary.json',result)
    print(json.dumps({'pooled':pooled,'gates':gates,'practical_upgrade_passed':practical,'method_increment':scientific},indent=2),flush=True)

if __name__=='__main__':main()
