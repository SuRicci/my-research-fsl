"""Frozen development then paired evaluation, retaining per-view canonical assets."""
from pathlib import Path
import hashlib,json,sys,time,shutil
import numpy as np
import torch
import reference_eval as ref
import fusion
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/'protocol.json').read_text());ref.CFG=CFG
torch.set_num_threads(CFG['threads'])
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def assets():
    root=Path(CFG['asset_root']);manifest=json.loads((root/'manifest.json').read_text());data={}
    assert 'elapsed_seconds' in manifest
    for name in ['dtd','eurosat']:
        ident=json.loads((root/(name+'_identities.json')).read_text());data[name]={'ident':ident}
        for side in ['query','gallery']:
            views=[]
            for backbone in ['clip_vitb16','dinov2_vits14']:
                p=root/(name+'_'+backbone+'_'+side+'.pt');record=manifest['datasets'][name]['backbones'][backbone+'_'+side]
                assert sha(p)==record['sha256'],str(p)
                x=torch.load(p,weights_only=True)
                assert x['ids'].tolist()==ident[side+'_ids'] and torch.isfinite(x['features']).all()
                views.append(x['features'].float())
            data[name][side]=views
    return data,manifest

def run():
    started=time.time();data,manifest=assets();out=HERE/'outputs';out.mkdir(exist_ok=True)
    sources={p.name:sha(p) for p in HERE.glob('*.py')};contract=Path(CFG['baseline_contract_path'])
    dump(out/'run_manifest.json',{'config':CFG,'code_sha256':sources,'feature_manifest':manifest,
       'baseline_contract_sha256':sha(contract),'command_argv':[sys.executable,*sys.argv],
       'cwd':str(Path.cwd()),'torch':torch.__version__,'numpy':np.__version__,
       'started_at':time.time(),'free_disk_gib':shutil.disk_usage(HERE).free/2**30})
    selection=None
    for phase in ['dev','eval']:
        folder=out/phase;folder.mkdir(exist_ok=True);summary={};dev={};spec=CFG[phase]
        for name in spec['datasets']:
            shots=[1] if phase=='dev' else [1,5]
            for shot in shots:
                si,qi,cs,seeds=ref.tasks(data[name]['ident'],name,shot,phase);y=np.repeat(np.arange(5),15)
                for gallery in (spec['gallery_datasets'] if shot==1 else [name]):
                    hashes=set(data[name]['ident']['query_rgb'])
                    keep=np.array([i for i,h in enumerate(data[gallery]['ident']['gallery_rgb']) if h not in hashes])
                    Gc,Gd=[x[keep] for x in data[gallery]['gallery']];cell=name+'_'+gallery+'_k'+str(shot)
                    records={};neighbors=[]
                    for st in range(0,len(si),CFG['batch_size']):
                        assert shutil.disk_usage(HERE).free>=10*2**30,'disk floor'
                        ids=si[st:st+CFG['batch_size']];qids=qi[st:st+CFG['batch_size']].reshape(-1,75)
                        Sc,Sd=[x[ids] for x in data[name]['query']];Qc,Qd=[x[qids] for x in data[name]['query']]
                        if shot==1:
                            choices={f:fusion.SETTINGS for f in fusion.FAMILIES} if phase=='dev' else {f:[tuple(selection['settings'][f])] for f in fusion.FAMILIES}
                            if phase=='eval':
                                choices['early_native']=list(dict.fromkeys(choices['early_native']+[tuple(selection['settings'][f]) for f in ['clip_native','dino_native']]))
                            values,idx=fusion.batch_scores(Sc,Sd,Qc,Qd,Gc,Gd,choices)
                            neighbors.append(idx.numpy())
                            if phase=='eval':
                                values={**{f:values[fusion.key('early_native' if f in ['clip_native','dino_native'] else f,*c)] for f,c in selection['settings'].items()},
                                        'r2':values['r2'],'cs_l2_fixed':values['cs_l2_fixed']}
                        else:
                            S=ref.representation(Sc,Sd,.5);Q=ref.representation(Qc,Qd,.5);G=ref.representation(Gc,Gd,.5)
                            score=ref.r2_scores(S,Q,G)
                            values={f:score for f in list(selection['settings'])+['r2','cs_l2_fixed']}
                        for method,score in values.items():records.setdefault(method,[]).append(score.numpy())
                        dump(out/'progress.json',{'phase':phase,'cell':cell,'completed_tasks_in_cell':min(st+CFG['batch_size'],len(si)),'cell_total':len(si),'elapsed_seconds':time.time()-started})
                    names=list(records);scores=np.stack([np.concatenate(records[m]) for m in names]);pred=scores.argmax(-1);acc=(pred==y).mean(-1)
                    extra={'shared_neighbor_indices':np.concatenate(neighbors)} if shot==1 else {}
                    np.savez_compressed(folder/(cell+'.npz'),scores=scores,predictions=pred,accuracy=acc,names=names,yq=y,
                         support_indices=si,query_indices=qi,class_ids=cs,seeds=seeds,gallery_pool_indices=keep,**extra)
                    summary[ref.metric_key(name,gallery,shot)]={m:100*float(acc[i].mean()) for i,m in enumerate(names)}
                    for i,m in enumerate(names):dev.setdefault(m,[]).append(float(acc[i].mean()))
                    dump(folder/'summary.json',summary)
                    print('CELL_COMPLETE',phase,cell,{k:round(v,4) for k,v in summary[ref.metric_key(name,gallery,shot)].items()} if phase=='eval' else len(names),flush=True)
        if phase=='dev':
            means={m:float(np.mean(x)) for m,x in dev.items()}
            settings={f:max(fusion.SETTINGS,key=lambda c:means[fusion.key(f,*c)]) for f in fusion.FAMILIES}
            for method,w in [('clip_native',1.),('dino_native',0.)]:settings[method]=(w,max(fusion.LAMBDAS,key=lambda l:means[fusion.key('early_native',w,l)]))
            simple=['early_shared','early_native','centered_native','clip_native','dino_native']
            selected_scores={f:means[fusion.key('early_native' if f in ['clip_native','dino_native'] else f,*c)] for f,c in settings.items()}
            selection={'settings':settings,'development_accuracy':means,'selected_development_accuracy':selected_scores,
               'strongest_control':max(simple,key=lambda f:selected_scores[f]),'locked_before_eval':True,
               'created_at':time.time(),'code_sha256':sources,'protocol_sha256':sha(HERE/'protocol.json'),
               'rule':CFG['parameter_selection'],'grid_count_per_family':20,'endpoint_lambda_count':4}
            dump(HERE/'selection.json',selection)
            print('SELECTION_FROZEN',settings,'control',selection['strongest_control'],flush=True)
        dump(folder/'completion.json',{'status':'completed','cells':len(summary),'elapsed_seconds':time.time()-started})
    assert sources=={p.name:sha(p) for p in HERE.glob('*.py')},'measurement source changed during run'
    dump(out/'completion.json',{'status':'completed','elapsed_seconds':time.time()-started,'selection_sha256':sha(HERE/'selection.json')})
    print('RUN_COMPLETE',time.time()-started,flush=True)
if __name__=='__main__':run()
