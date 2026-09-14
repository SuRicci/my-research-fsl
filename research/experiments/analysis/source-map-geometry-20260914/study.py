"""Observational geometry of immutable selected maps; no fit or new predictions."""
from pathlib import Path
import json,sys,time,hashlib,shutil
import numpy as np
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
MAIN=P.parents[1]/'main/source-composition-20260913'
sys.path.insert(0,str(MAIN))
import study as parent
CFG=json.loads((P/'protocol.json').read_text())

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def corr(a,b,control=None):
    a=np.array(a,dtype=float);b=np.array(b,dtype=float)
    if len(a)<4:return None
    if control is not None:
        design=np.c_[np.ones(len(a)),control]
        a=a-design@np.linalg.lstsq(design,a,rcond=None)[0]
        b=b-design@np.linalg.lstsq(design,b,rcond=None)[0]
    a=a-a.mean();b=b-b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den>1e-14 else None

def pairs(labels,seed):
    rng=np.random.RandomState(seed);classes=np.unique(labels)
    groups={c:np.flatnonzero(labels==c) for c in classes}
    ans=[]
    for same in [True,False]:
        for _ in range(CFG['pair_count_per_domain']//2):
            c=rng.choice(classes)
            if same:i,j=rng.choice(groups[c],2,replace=False)
            else:
                d=rng.choice(classes[classes!=c]);i=rng.choice(groups[c]);j=rng.choice(groups[d])
            ans.append((i,j))
    return np.array(ans,dtype=np.int64)

def main():
    started=time.time();torch.set_num_threads(CFG['constraints']['threads'])
    assert not (P/'RESULT.json').exists(),'Completed result exists; do not rerun'
    assert shutil.disk_usage(P).free>=10*2**30
    data,assets=parent.load()
    selection_path=MAIN/'outputs/selection_lock.json'
    choices=json.loads(selection_path.read_text());assert len(choices)==36
    hashes={**assets,str(selection_path):sha(selection_path),str(P/'protocol.json'):sha(P/'protocol.json'),str(P/'study.py'):sha(P/'study.py')}
    norms={};pairbank={};nulls={};rows=[];numeric=[];accuracy={}
    for di,ds in enumerate(CFG['domains']):
        raw=data[ds]['x'][:,0,:].double()
        norms[ds]=F.normalize(raw,dim=-1)
        pairbank[ds]=pairs(data[ds]['labels'],CFG['pair_seed']+di)
        x=norms[ds];ij=pairbank[ds];base=(x[ij[:,0]]*x[ij[:,1]]).sum(-1)
        # Fixed signed coordinate permutation is orthogonal but generally moves vectors.
        signs=torch.where(torch.arange(x.shape[1])%2==0,1.,-1.).double()
        rotated=x.flip(-1)*signs
        dot=(rotated[ij[:,0]]*rotated[ij[:,1]]).sum(-1)
        err=float((dot-base).abs().max());assert err<1e-12
        nulls[ds]={'image_count':len(x),'sampled_pairs':len(ij),'unique_sampled_pairs':len(np.unique(np.sort(ij,axis=1),axis=0)),
                   'orthogonal_cosine_max_error':err,'orthogonal_mean_sq_displacement':float(((rotated-x)**2).sum(-1).mean())}
    np.savez_compressed(P/'pairs.npz',**pairbank)
    hashes[str(P/'pairs.npz')]=sha(P/'pairs.npz')
    for key,c in choices.items():
        mp=Path(c['model_path']);assert sha(mp)==c['sha256'];hashes[str(mp)]=c['sha256']
        state=torch.load(mp,weights_only=True);down=state['down.weight'].double();up=state['up.weight'].double()
        geom={}
        for ds,x in norms.items():
            residual=F.relu(x@down.T)@up.T
            z=F.normalize(x+residual,dim=-1)
            ij=pairbank[ds];base=(x[ij[:,0]]*x[ij[:,1]]).sum(-1)
            delta=(z[ij[:,0]]*z[ij[:,1]]).sum(-1)-base
            half=len(ij)//2
            g={'mean_sq_displacement':float(((z-x)**2).sum(-1).mean()),'raw_residual_energy':float((residual**2).sum(-1).mean()),
               'pair_cosine_rms_change':float(torch.sqrt((delta**2).mean())),
               'within_cosine_change':float(delta[:half].mean()),'between_cosine_change':float(delta[half:].mean()),
               'class_gap_change':float(delta[:half].mean()-delta[half:].mean())}
            assert all(np.isfinite(v) for v in g.values())
            if c['step']==0:
                assert max(abs(v) for v in g.values())<1e-12
            # Independent NumPy evaluation at fixed scattered rows and pairs.
            ids=np.unique(np.r_[np.linspace(0,len(x)-1,16,dtype=int),ij[:64].ravel()])
            xn=x[ids].numpy();rn=np.maximum(xn@down.numpy().T,0)@up.numpy().T
            zn=xn+rn;zn/=np.maximum(np.linalg.norm(zn,axis=-1,keepdims=True),1e-12)
            e=float(np.max(np.abs(zn-z[ids].numpy())))
            lookup={int(v):i for i,v in enumerate(ids)}
            a=np.array([lookup[int(v)] for v in ij[:64,0]]);b=np.array([lookup[int(v)] for v in ij[:64,1]])
            nd=np.einsum('ij,ij->i',zn[a],zn[b])-np.einsum('ij,ij->i',xn[a],xn[b])
            pe=float(np.max(np.abs(nd-delta[:64].numpy())))
            assert e<1e-12 and pe<1e-12
            numeric.append({'key':key,'domain':ds,'transform_max_abs':e,'pair_delta_max_abs':pe,'rows':len(ids),'pairs':64})
            geom[ds]=g
        ds=c['source'];k=c['shot'];target='eurosat' if ds=='dtd' else 'dtd'
        sp=MAIN/'outputs/selection'/f'{key}.npz';tp=MAIN/'outputs/cells'/f'{ds}_to_{target}_k{k}.npz'
        for f in [sp,tp]:hashes[str(f)]=sha(f)
        sel=np.load(sp);test=np.load(tp)
        index=[h['step'] for h in c['history']].index(c['step'])
        ss=(sel['predictions'][index]==sel['yq']).mean(-1)
        s0=(sel['predictions'][0]==sel['yq']).mean(-1)
        assert int((sel['predictions'][index]==sel['yq']).sum())==c['correct']
        arm=parent.ARMS.index(c['arm']);seed=parent.CFG['training_seeds'].index(c['seed'])
        tt=(test['predictions'][arm,seed]==test['yq']).mean(-1);t0=(test['zero_predictions']==test['yq']).mean(-1)
        saved={'source_macro_delta_pp':float((ss-s0).mean()*100),'source_original_delta_pp':float((ss[:50]-s0[:50]).mean()*100),
               'source_flowers_delta_pp':float((ss[50:]-s0[50:]).mean()*100),'transfer_delta_pp':float((tt-t0).mean()*100),
               'transfer_accuracy_pct':float(tt.mean()*100),'transfer_zero_accuracy_pct':float(t0.mean()*100)}
        rows.append({'key':key,'source':ds,'target':target,'shot':k,'arm':c['arm'],'seed':c['seed'],'selected_step':c['step'],'geometry':geom,'saved_accuracy':saved})
        print('MODEL_COMPLETE',key,'transfer_delta_pp',round(saved['transfer_delta_pp'],4),flush=True)
    correlations={}
    for ds in ['dtd','eurosat']:
        for k in [1,5]:
            rr=[r for r in rows if r['source']==ds and r['shot']==k and r['selected_step']>0]
            target='eurosat' if ds=='dtd' else 'dtd'
            y=[r['saved_accuracy']['transfer_delta_pp'] for r in rr]
            blocks={}
            for pool in [ds,'flowers',target]:
                disp=[r['geometry'][pool]['mean_sq_displacement'] for r in rr]
                dist=[r['geometry'][pool]['pair_cosine_rms_change'] for r in rr]
                gap=[r['geometry'][pool]['class_gap_change'] for r in rr]
                blocks[pool]={'displacement_vs_transfer':corr(disp,y),'distortion_vs_transfer':corr(dist,y),
                              'distortion_vs_transfer_adjusted_displacement':corr(dist,y,disp),
                              'class_gap_vs_transfer':corr(gap,y)}
            correlations[f'{ds}_k{k}']={'nonzero_model_count':len(rr),'geometry_pool':blocks}
    assert all(sha(p)==h for p,h in hashes.items()),'Input mutation'
    out={'status':'computed','scope':CFG['scope'],'model_count':len(rows),'zero_step_models':sum(r['selected_step']==0 for r in rows),'domains':nulls,'rows':rows,
         'correlations':correlations,'validation':{'status':'passed','independent_cases':len(numeric),'maximum_transform_error':max(r['transform_max_abs'] for r in numeric),
         'maximum_pair_delta_error':max(r['pair_delta_max_abs'] for r in numeric),'all_input_hashes_match':True,'identity_models_zero':True,'orthogonal_null_passed':True},
         'elapsed_seconds':time.time()-started,'free_gib':shutil.disk_usage(P).free/2**30}
    dump(P/'inputs.json',hashes);dump(P/'validation_cases.json',numeric);dump(P/'RESULT.json',out)
    print('DIAGNOSTIC_COMPLETE',out['model_count'],'seconds',round(out['elapsed_seconds'],2),flush=True)
    print('CORRELATIONS',json.dumps(correlations),flush=True)
if __name__=='__main__':main()
