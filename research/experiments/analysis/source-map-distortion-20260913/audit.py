"""Read-only geometry audit of the previously selected source maps."""
from pathlib import Path
import importlib.util, json, hashlib, datetime, shutil, time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
PARENT=ROOT/'experiments/main/source-composition-20260913'
spec=importlib.util.spec_from_file_location('source_composition_audit_parent',PARENT/'study.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
CFG=json.loads((HERE/'protocol.json').read_text())

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalize(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
def describe(x,z):
    base=x@x.T; changed=z@z.T;mask=~np.eye(len(x),dtype=bool)
    a,b=base[mask],changed[mask]
    return {'mean_one_minus_cosine':float(np.mean(1-np.sum(x*z,axis=-1))),
            'gram_offdiagonal_rmse':float(np.sqrt(np.mean((a-b)**2))),
            'gram_offdiagonal_correlation':float(np.corrcoef(a,b)[0,1])}

def main():
    start=time.time();torch.set_num_threads(CFG['resources']['threads'])
    assert shutil.disk_usage(HERE).free/2**30>=10
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['resources']['deadline'])
    choices=json.loads((PARENT/'outputs/selection_lock.json').read_text())
    assert len(choices)==36
    data,inputs=old.load()
    sample={}
    for source in ['dtd','eurosat']:
        target='eurosat' if source=='dtd' else 'dtd'
        for shot in [1,5]:
            va=np.load(PARENT/'outputs/tasks'/f'{source}_k{shot}_selection.npz')
            offset=len(data[source]['x'])
            ids=np.unique(np.r_[va['support_indices'].ravel(),va['query_indices'].ravel()])
            task=np.load(old.BANK/f'eval_{target}_k{shot}_tasks.npz')
            targetids=np.unique(np.r_[task['support_indices'].ravel(),task['query_indices'].ravel()])
            pools={'source_validation':ids[ids<offset],'flowers_validation':ids[ids>=offset]-offset,'target_evaluation':targetids}
            sample[source+f'_k{shot}']={}
            for name,pool in pools.items():
                ix=np.random.RandomState(26091339).choice(pool,min(256,len(pool)),replace=False)
                ds=source if name=='source_validation' else 'flowers' if name=='flowers_validation' else target
                sample[source+f'_k{shot}'][name]={'dataset':ds,'pool_count':len(pool),'indices':ix.tolist(),'pool_sha256':hashlib.sha256(pool.tobytes()).hexdigest()}
    rng=np.random.RandomState(26091339);sx=normalize(rng.randn(30,8));orth=np.linalg.qr(rng.randn(8,8))[0]
    null=describe(sx,sx@orth)
    assert null['gram_offdiagonal_rmse']<1e-12 and null['mean_one_minus_cosine']>.1
    identity=describe(sx,sx)
    assert identity['gram_offdiagonal_rmse']==0
    rows=[];max_error=0.
    with torch.no_grad():
        for key,e in choices.items():
            path=Path(e['model_path']);assert sha(path)==e['sha256']
            state=torch.load(path,weights_only=True)
            net=old.model.Map(old.CFG['dimension'],old.CFG['rank']);net.load_state_dict(state)
            source=e['source'];target='eurosat' if source=='dtd' else 'dtd';shot=e['shot']
            descriptors={}
            down,up=[state[k].numpy().astype(float) for k in ['down.weight','up.weight']]
            for name,s in sample[source+f'_k{shot}'].items():
                tensor=data[s['dataset']]['x'][s['indices']]
                x=tensor[:,0].numpy().astype(float)
                z=normalize(x+np.maximum(x@down.T,0)@up.T)
                actual=net(tensor,old.MODE).numpy()
                error=float(abs(z-actual).max());max_error=max(max_error,error);assert error<3e-5
                descriptors[name]=describe(normalize(x),z)
            cell=np.load(PARENT/'outputs/cells'/f'{source}_to_{target}_k{shot}.npz')
            ai=old.ARMS.index(e['arm']);si=old.CFG['training_seeds'].index(e['seed'])
            pred=cell['predictions'][ai,si];zero=cell['zero_predictions'];truth=cell['yq']
            delta=float(((pred==truth).mean()-(zero==truth).mean())*100)
            sel=np.load(PARENT/'outputs/selection'/(key+'.npz'))
            hidx=old.CFG['checkpoints'].index(e['step'])
            assert int((sel['predictions'][hidx]==sel['yq']).sum())==e['correct']
            selection_delta=(e['correct']-e['history'][0]['correct'])/sel['predictions'][0].size*100
            rows.append({'id':key,'source':source,'target':target,'shot':shot,'arm':e['arm'],'seed':e['seed'],
                         'selected_step':e['step'],'weight_sha256':e['sha256'],'selection_delta_pp':selection_delta,
                         'target_delta_vs_zero_pp':delta,'geometry':descriptors})
    summaries={}
    for source in ['dtd','eurosat']:
        for shot in [1,5]:
            for arm in old.ARMS:
                rr=[r for r in rows if (r['source'],r['shot'],r['arm'])==(source,shot,arm)]
                summaries[f'{source}_k{shot}_{arm}']={'models':len(rr),'selected_steps':[r['selected_step'] for r in rr],
                    'mean_selection_delta_pp':float(np.mean([r['selection_delta_pp'] for r in rr])),
                    'mean_target_delta_pp':float(np.mean([r['target_delta_vs_zero_pp'] for r in rr])),
                    'mean_target_gram_rmse':float(np.mean([r['geometry']['target_evaluation']['gram_offdiagonal_rmse'] for r in rr])),
                    'mean_source_gram_rmse':float(np.mean([r['geometry']['source_validation']['gram_offdiagonal_rmse'] for r in rr]))}
    report={'status':'completed','scope':CFG['limits'],'sample':sample,'model_count':len(rows),'rows':rows,'groups':summaries,
            'validation':{'status':'passed','independent_numpy_forward_max_error':max_error,'orthogonal_null':null,'identity_null':identity},
            'input_hashes':inputs,'selection_lock_sha256':sha(PARENT/'outputs/selection_lock.json'),
            'elapsed_seconds':time.time()-start}
    (HERE/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SOURCE_GEOMETRY_AUDIT_COMPLETE',json.dumps({'model_count':len(rows),'groups':summaries,'validation':report['validation'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
if __name__=='__main__':main()
