"""Chunked, resumable evaluation. Truth remains outside pure feature-map adapters."""
from pathlib import Path
import json,time,os,uuid
import numpy as np
from research_common.records import read_json,write_json,sha256,object_hash
from .lab_methods import FeatureMap,METHODS
from .lab_metrics import evaluate_rank

def cache_array(cache,encoder,manifest_hash):
    path=Path(cache)/encoder/'embeddings.npy';meta=read_json(path.with_suffix('.json'))
    if meta['manifest_sha256']!=manifest_hash or sha256(path)!=meta['array_sha256']:raise ValueError('Feature cache identity mismatch: '+encoder)
    return np.load(path,mmap_mode='r'),meta

def evaluate(manifest_path,cache,output,config_path,device='cpu',shard=0,shards=1,query_batch=16,dtype='float32',max_queries=None):
    manifest=read_json(manifest_path);config=read_json(config_path);root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if not config.get('frozen'):raise ValueError('Freeze method configuration before external evaluation')
    manifest_hash=sha256(manifest_path);config_hash=sha256(config_path)
    code={n:sha256(Path(__file__).parent/n) for n in ['lab_methods.py','lab_metrics.py','nextstage.py','improved.py','methods.py']}
    signature=object_hash({'manifest':manifest_hash,'config':config_hash,'dtype':dtype,'max_queries':max_queries,'code':code})
    opening=root/'opening.json'
    if opening.exists():
        if read_json(opening)['signature']!=signature:raise ValueError('Output already binds a different protocol')
    else:
        value={'signature':signature,'manifest_sha256':manifest_hash,'config_sha256':config_hash,
        'methods':config['methods'],'code_sha256':code,'expected_method_jobs':len(config['pairs'])*len(config['budgets'])*len(config['seeds'])*(len(config['methods'])+int(config.get('oracle',True))),
        'stage':'external_frozen_evaluation' if not max_queries else 'limited_pipeline_diagnostic',
        'isolation':'Pure numerical feature-map adapter receives only old gallery chunks, paired bridges and current query batch; evaluator-only labels and new gallery never passed. Not an OS sandbox.',
        'queries_mutually_source_disjoint_from_gallery':True}
        temporary=root/('.opening_'+uuid.uuid4().hex+'.json');write_json(temporary,value)
        try:os.link(temporary,opening)
        except FileExistsError:
            if read_json(opening)['signature']!=signature:raise ValueError('Concurrent output protocol differs')
        finally:temporary.unlink()
    rows=manifest['records'];lookup={r['id']:i for i,r in enumerate(rows)};roles=manifest['roles'];gids=roles['gallery'];qids=roles['query'][:max_queries]
    gi=np.array([lookup[i] for i in gids]);qi=np.array([lookup[i] for i in qids])
    labels=manifest.get('labels_evaluator_only',{});gl=np.array([labels[i] for i in gids]) if labels else None
    def truth(q):
        if manifest['protocol']=='revisited':return manifest['truth_evaluator_only'][q]
        return {'class':{'positive':np.flatnonzero(gl==labels[q]).tolist(),'ignore':[]}}
    arrays={};models=sorted(set(x for pair in config['pairs'] for x in pair))
    for name in models:arrays[name]=cache_array(cache,name,manifest_hash)[0]
    jobs=[(a,b,m,s) for a,b in config['pairs'] for m in config['budgets'] for s in config['seeds']]
    for job_number,(old,new,budget,seed) in enumerate(jobs):
        if job_number%shards!=shard:continue
        job=root/f'{old}__{new}__m{budget}__s{seed}';job.mkdir(parents=True,exist_ok=True)
        bridge_role=config.get('bridge_role','bridge');pool=roles[bridge_role]
        if len(pool)<budget:raise ValueError('Insufficient independent bridges')
        anchors=list(map(str,np.random.default_rng(seed).permutation(pool)[:budget]));ai=[lookup[i] for i in anchors]
        ao=arrays[old][ai];an=arrays[new][ai]
        receipt=job/'input_identity.json'
        if not receipt.exists():write_json(receipt,{'bridge_ids':anchors,'budget':budget,'signature':signature,'gallery_count':len(gids),'query_count':len(qids),
            'gallery_new_passed_to_method':False,'class_labels_passed_to_method':False})
        elif read_json(receipt)['bridge_ids']!=anchors:raise ValueError('Resume anchors changed')
        for method in config['methods']+(['oracle_new_new'] if config.get('oracle',True) else []):
            summary_path=job/(method+'.summary.json')
            if summary_path.exists():continue
            start=time.perf_counter();fm=None if method=='oracle_new_new' else FeatureMap(ao,an,method)
            feature_file=job/('_gallery_'+method+'.npy');gallery=None
            for begin in range(0,len(gi),4096):
                values=arrays[new][gi[begin:begin+4096]] if fm is None else fm.gallery(arrays[old][gi[begin:begin+4096]])
                if gallery is None:gallery=np.lib.format.open_memmap(feature_file,mode='w+',dtype=dtype,shape=(len(gi),values.shape[1]))
                gallery[begin:begin+len(values)]=values
            gallery.flush();build_seconds=time.perf_counter()-start
            gpu=None
            if device.startswith('cuda'):
                import torch
                gpu=torch.from_numpy(np.array(gallery)).to(device)
            predictions=[];times=[]
            for begin in range(0,len(qi),query_batch):
                q=qi[begin:begin+query_batch];start=time.perf_counter()
                fquery=arrays[new][q] if fm is None else fm.query(arrays[old][q],arrays[new][q])
                fquery=np.asarray(fquery,dtype=dtype)
                if gpu is None:scores=fquery@gallery.T
                else:
                    import torch
                    with torch.inference_mode():scores=(torch.from_numpy(fquery).to(device)@gpu.T).cpu().numpy()
                ranked=np.argsort(-scores,axis=1,kind='stable');times.append(time.perf_counter()-start)
                for k,rank in enumerate(ranked):
                    qid=qids[begin+k];metrics={name:evaluate_rank(rank,t,'revisited' if manifest['protocol']=='revisited' else 'class') for name,t in truth(qid).items()}
                    for values in metrics.values():
                        for name,value in list(values.items()):
                            if isinstance(value,float) and not np.isfinite(value):values[name]=None
                    predictions.append({'query_id':qid,'metrics':metrics,'top100':rank[:100].tolist()})
                if begin%(query_batch*50)==0:print(json.dumps({'job':job.name,'method':method,'queries':begin+len(q),'total':len(qi)}),flush=True)
            prediction_path=job/(method+'.predictions.json');write_json(prediction_path,predictions,overwrite=True)
            summary={}
            for condition in predictions[0]['metrics']:
                ms=[r['metrics'][condition] for r in predictions];aps=np.array([r['ap'] for r in ms],dtype=float);valid=np.isfinite(aps)
                summary[condition]={'map':float(np.mean(aps[valid])) if valid.any() else None,'valid_queries':int(valid.sum())}
                for k in ['recall_at_1','recall_at_10','recall_at_100','precision_at_1','precision_at_5','precision_at_10']:
                    vals=np.array([r[k] for r in ms],dtype=float);summary[condition][k]=float(np.nanmean(vals[valid])) if valid.any() else None
            write_json(summary_path,{'metrics':summary,'method':method,'bridge_budget':budget,'seed':seed,'old':old,'new':new,
                'predictions_sha256':sha256(prediction_path),'gallery_feature_build_seconds':build_seconds,'batch_seconds':times,
                'query_batch':query_batch,'score_dtype':dtype,'scoring_device':device,'timing_excludes_encoders':True,'gallery_feature_bytes':int(gallery.nbytes)})
            del gallery,gpu
            # Only this invocation's named temporary array is removed; evidence stays immutable.
            feature_file.unlink()
    print(json.dumps({'shard_complete':shard,'shards':shards}),flush=True)

def summarize(output):
    root=Path(output);items=[]
    for p in root.glob('*/*.summary.json'):items.append(read_json(p)|{'job':p.parent.name})
    if not items:raise ValueError('No completed method results')
    rows=[]
    keys=sorted({(r['old'],r['new'],r['bridge_budget'],r['method']) for r in items})
    for old,new,m,method in keys:
        part=[r for r in items if (r['old'],r['new'],r['bridge_budget'],r['method'])==(old,new,m,method)]
        for condition in part[0]['metrics']:
            maps=[r['metrics'][condition]['map'] for r in part if r['metrics'][condition]['map'] is not None]
            rows.append({'old':old,'new':new,'budget':m,'method':method,'condition':condition,'completed_seeds':len(part),'map':float(np.mean(maps)) if maps else None})
    expected=read_json(root/'opening.json')['expected_method_jobs']
    write_json(root/'summary.json',{'rows':rows,'completed_method_jobs':len(items),'expected_method_jobs':expected,'complete':len(items)==expected},overwrite=True)
    print(json.dumps({'completed_method_jobs':len(items),'summary_rows':len(rows)}),flush=True)
