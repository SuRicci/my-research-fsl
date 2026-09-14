"""Retrospective final-step mass/composition intervention; no algorithm promotion."""
from pathlib import Path
import sys,json
import torch
import numpy as np
HERE=Path(__file__).resolve().parent;MAIN=HERE.parents[1]/'main/oslo-gallery-pets-20260913'
sys.path.insert(0,str(MAIN))
import run_eval as ev
import oslo
sys.path.insert(0,str(MAIN))
from analyze_results import draws
source=Path(oslo.__file__).read_text();needle='return (origin, proto), stats';assert source.count(needle)==1
ns={};exec(compile(source.replace(needle,needle+', weight, initial, g'),str(oslo.__file__)+'[diagnostic_return_only]','exec'),ns)
NAMES=['original','mass_balanced','oracle_membership','oracle_balanced','zero_update','r2','CS_l2']


def main():
    torch.set_num_threads(4);ev.guard();signature=ev.lock_check();data,old,_=ev.ref.load_data();fresh,rows=ev.fresh_data();result={};hashes={};replay={}
    rng=np.random.default_rng(ev.CFG['bootstrap']['seed'])
    for scope,X,ident in [('canonical',data['query'],old['query']),('fresh',fresh,rows)]:
        for shot in [1,5]:
            t=ev.tasks(scope,ident,shot);si,qi,cs=[t[k] for k in ['support_indices','query_indices','class_ids']]
            indices=rng.integers(100,size=(5000,5,100))
            for gallery in ['pets','dtd']:
                ev.guard();cell=scope+'_'+gallery+'_k%d'%shot;G=data['gallery' if gallery=='pets' else 'dtd']
                with np.load(MAIN/'outputs/cells'/(cell+'.npz')) as p:parent={name:p['scores'][j] for j,name in enumerate(p['names'])}
                gy=torch.tensor([r['label'] for r in old['gallery']]) if gallery=='pets' else torch.full((len(G),),-1)
                values={m:[] for m in NAMES[:4]};errors=[]
                for start in range(0,500,16):
                    ix=slice(start,start+16);S=X[si[ix]];Q=X[qi[ix].reshape(len(S),75)]
                    (origin,proto),stats,weight,initial,g=ns['fit'](S,G)
                    original=oslo.predict((origin,proto),Q);err=float(abs(original.numpy()-parent['OSLO_G'][ix]).max());assert err<3e-6;errors.append(err)
                    inside=(gy[None,:,None]==torch.tensor(cs[ix])[:,None,:]).any(-1);clean=weight*inside[:,:,None]
                    outputs={'original':original}
                    for name,w,balance in [('mass_balanced',weight,True),('oracle_membership',clean,False),('oracle_balanced',clean,True)]:
                        mass=w.sum(1)[...,None];numerator=w.transpose(1,2)@g
                        if balance:
                            mean=numerator/mass.clamp_min(1e-12);p2=torch.where(mass>0,(initial+mean)/2,initial)
                        else:p2=(initial*shot+numerator)/(shot+mass)
                        outputs[name]=oslo.predict((origin,p2),Q)
                        if gallery=='dtd' and name.startswith('oracle'):
                            assert float(abs(outputs[name].numpy()-parent['zero_update'][ix]).max())<3e-6
                    for name,x in outputs.items():values[name].append(x.numpy())
                scores={m:np.concatenate(v) for m,v in values.items()};scores.update({m:parent[m] for m in NAMES[4:]})
                stacked=np.stack([scores[m] for m in NAMES]);assert stacked.shape==(7,500,75,5) and np.isfinite(stacked).all()
                pred=stacked.argmax(-1);acc=(pred==np.repeat(np.arange(5),15)).mean(-1).T
                boot=draws(acc,indices);comparison={}
                for j,m in enumerate(NAMES[:4]):
                    comparison[m]={}
                    for control in ['original','zero_update','r2','CS_l2']:
                        k=NAMES.index(control);comparison[m][control]={'delta_pp':float((acc[:,j]-acc[:,k]).mean()*100),'paired_ci95_pp':np.quantile(boot[:,j]-boot[:,k],[.025,.975]).tolist()}
                path=HERE/(cell+'.npz');np.savez_compressed(path,names=NAMES,accuracy=acc,predictions=pred,**t)
                hashes[cell]=ev.sha(path);replay[cell]=max(errors)
                result[cell]={'tasks':500,'accuracy_pct':dict(zip(NAMES,(acc.mean(0)*100).tolist())),'comparisons':comparison}
                print('COUNTERFACTUAL_COMPLETE',cell,result[cell]['accuracy_pct'],flush=True)
    final={'status':'passed','parent_run_id':'oslo-gallery-pets-20260913','cells':result,'task_count':4000,'maximum_original_replay_error':max(replay.values()),'oracle_mismatch_equals_zero_update':True,'output_sha256':hashes,'source_sha256':ev.sha(HERE/'evaluate.py'),'frozen_parent_signature':signature,'bootstrap':ev.CFG['bootstrap'],'boundary':'Retrospective final-step intervention; same fitted weights and centering. Oracle composition uses labels unavailable to inference. Balanced variants are diagnostic interventions, not selected improvements. No changes to earlier iterations or prospective confirmation claim.'}
    ev.dump(HERE/'analysis.json',final)
    print('COUNTERFACTUAL_AUDIT_PASSED',4000,flush=True)


if __name__=='__main__':main()
