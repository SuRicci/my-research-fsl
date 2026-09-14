"""Read-only instrumentation of the frozen fit, for class-membership attribution."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[1]/'main/oslo-gallery-pets-20260913'
sys.path.insert(0,str(MAIN))
import run_eval as ev
import oslo
source=Path(oslo.__file__).read_text();needle='return (origin, proto), stats'
assert source.count(needle)==1
namespace={};exec(compile(source.replace(needle,needle+', weight, xi, z'),str(oslo.__file__)+'[diagnostic_return_only]','exec'),namespace)
fit=namespace['fit'];torch.set_num_threads(4)


def summary(x):
    x=np.asarray(x);return {'mean':float(x.mean()),'p10_median_p90':np.quantile(x,[.1,.5,.9]).tolist(),'min':float(x.min()),'max':float(x.max())}


def main():
    ev.guard();signature=ev.lock_check();data,old,_=ev.ref.load_data();fresh,rows=ev.fresh_data();result={};saved={}
    for scope,X,ident in [('canonical',data['query'],old['query']),('fresh',fresh,rows)]:
        for shot in [1,5]:
            t=ev.tasks(scope,ident,shot);si,qi,cs=[t[k] for k in ['support_indices','query_indices','class_ids']]
            for gallery in ['pets','dtd']:
                ev.guard();cell=scope+'_'+gallery+'_k%d'%shot
                G=data['gallery' if gallery=='pets' else 'dtd'];labels=torch.tensor([r['label'] for r in old['gallery']]) if gallery=='pets' else torch.full((len(G),),-1)
                with np.load(MAIN/'outputs/cells'/(cell+'.npz')) as p:original=p['scores'][0];original_mass=p['OSLO_G_effective_mass']
                values={k:[] for k in ['mass','correct_class_mass','other_in_episode_mass','out_of_episode_mass','gallery_share','out_of_episode_coefficient','xi_in','xi_out','inlier_auc']};errors=[]
                for start in range(0,500,16):
                    idx=slice(start,start+16);S=X[si[idx]];Q=X[qi[idx].reshape(len(S),75)]
                    state,stats,weight,xi,z=fit(S,G,**{'steps':2,'lambda_s':.05,'lambda_z':.1,'use_inlier':True})
                    score=oslo.predict(state,Q).numpy();error=float(abs(score-original[idx]).max());assert error<3e-6
                    mass=weight.sum(1);assert np.allclose(mass.numpy(),original_mass[idx],rtol=1e-6,atol=1e-4)
                    correct=labels[None,:,None]==torch.tensor(cs[idx])[:,None,:];inside=correct.any(-1)
                    cm=(weight*correct).sum(1);wm=(weight*(inside[:,:,None]&~correct)).sum(1);om=(weight*(~inside[:,:,None])).sum(1)
                    assert torch.allclose(cm+wm+om,mass,rtol=1e-6,atol=1e-4)
                    for k,x in [('mass',mass),('correct_class_mass',cm),('other_in_episode_mass',wm),('out_of_episode_mass',om),('gallery_share',mass/(shot+mass)),('out_of_episode_coefficient',om/(shot+mass))]:values[k].append(x.numpy())
                    for j in range(len(S)):
                        indicator=inside[j].numpy();inlier=xi[j,:,0].numpy()
                        if indicator.any():values['xi_in'].append(inlier[indicator].mean());values['inlier_auc'].append(roc_auc_score(indicator,inlier))
                        values['xi_out'].append(inlier[~indicator].mean())
                    errors.append(error)
                arrays={k:np.concatenate(v) if k not in ['xi_in','xi_out','inlier_auc'] else np.asarray(v) for k,v in values.items()}
                for k,a in arrays.items():saved[cell+'_'+k]=a
                mass_total=arrays['mass'].sum();shares={k:float(arrays[k].sum()/mass_total) for k in ['correct_class_mass','other_in_episode_mass','out_of_episode_mass']};assert abs(sum(shares.values())-1)<2e-6
                result[cell]={'tasks':500,'final_step_mass':summary(arrays['mass']),'gallery_averaging_coefficient':summary(arrays['gallery_share']),'out_of_episode_averaging_coefficient':summary(arrays['out_of_episode_coefficient']),
                    'retained_gallery_weight_partition':shares,'xi_in_mean':float(arrays['xi_in'].mean()) if len(arrays['xi_in']) else None,'xi_out_mean':float(arrays['xi_out'].mean()),
                    'episode_mean_inlier_auc':float(arrays['inlier_auc'].mean()) if len(arrays['inlier_auc']) else None,'maximum_score_replay_error':max(errors)}
                print('TRACE_COMPLETE',cell,json.dumps(result[cell]),flush=True)
    np.savez_compressed(HERE/'task_attribution.npz',**saved)
    result={'status':'passed','parent_run_id':'oslo-gallery-pets-20260913','task_count':4000,'cells':result,'frozen_signature':signature,'source_sha256':ev.sha(HERE/'trace.py'),
        'array_sha256':ev.sha(HERE/'task_attribution.npz'),'boundary':'Final-step averaging coefficients are algebraic contributions, not counterfactual performance. Latent xi is a source score, not a calibrated membership probability. Labels are used only for post-fit diagnostics. All predictions agree with frozen outputs.'}
    ev.dump(HERE/'attribution.json',result)
    print('TRACE_AUDIT_PASSED',4000,flush=True)


if __name__=='__main__':main()
