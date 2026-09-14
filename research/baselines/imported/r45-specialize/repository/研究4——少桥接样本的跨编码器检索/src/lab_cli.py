"""Portable laboratory entrypoints. No remote jobs are launched automatically."""
from pathlib import Path
import argparse,json,os,sys,shutil
import numpy as np
from research_common.records import read_json,write_json,sha256,download_public

ROOT=Path(__file__).resolve().parents[1]

def import_pilot(manifest_path,cache,output):
    source=read_json(manifest_path);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[dict(r,id=str(r['id'])) for r in source['records']]
    m={'schema':'r5_lab_manifest_v1','dataset':'cub_observed_development','protocol':'class','records':rows,
       'roles':{k:list(map(str,v)) for k,v in source['roles'].items()},'labels_evaluator_only':{str(r['id']):r['label_evaluator_only'] for r in rows},
       'source_manifest_sha256':sha256(manifest_path),'performance_opened':True,'not_external_confirmation':True}
    path=out/'manifest.json'
    if path.exists():
        if read_json(path)!=m:raise ValueError('Imported manifest changed')
    else:write_json(path,m)
    ids=[int(r['id']) for r in rows]
    for name in ['clip_b32','clip_b16','dino_s14']:
        folder=out/'cache'/name;folder.mkdir(parents=True,exist_ok=True);dst=folder/'embeddings.npy'
        with np.load(Path(cache)/(name+'_all.npz'),allow_pickle=False) as z:
            lookup={int(v):i for i,v in enumerate(z['ids'])};np.save(dst,z['embeddings'][[lookup[i] for i in ids]])
        write_json(dst.with_suffix('.json'),{'manifest_sha256':sha256(path),'array_sha256':sha256(dst),'rows':len(ids),'model_signature':read_json(Path(cache)/(name+'_all.json'))['model']},overwrite=True)
    print(json.dumps({'manifest':str(path),'cache':str(out/'cache'),'development_only':True}),flush=True)

def preflight(require_gpus=1):
    import importlib,torch
    checks={};errors=[]
    for name in ['numpy','scipy','PIL','torch','torchvision','clip','requests']:
        try:m=importlib.import_module(name);checks[name]=getattr(m,'__version__','import-ok')
        except Exception as e:errors.append(f'{name}: {e}')
    gpu=torch.cuda.device_count()
    if gpu<require_gpus:errors.append(f'Found {gpu} GPU(s), required {require_gpus}')
    weights=Path(os.environ.get('R5_CLIP_WEIGHTS_DIR',str(Path.home()/'.cache/torch/hub/checkpoints')))
    paths=[weights/'ViT-B-32.pt',weights/'ViT-B-16.pt',Path(os.environ.get('R5_DINO_WEIGHTS',str(ROOT.parent/'研究1——CLIP结合DinoV2/pretrained/dinov2_vits14_pretrain.pth')))]
    for p in paths:
        if not p.is_file():errors.append('Missing checkpoint: '+str(p))
    result={'versions':checks,'gpus':[torch.cuda.get_device_name(i) for i in range(gpu)],'disk_free_gib':shutil.disk_usage(ROOT).free/2**30,
        'checkpoint_paths':[str(p) for p in paths],'failed':len(errors),'errors':errors}
    print(json.dumps(result,ensure_ascii=False,indent=2));return int(bool(errors))

def assets():
    import clip
    from research_common.encoders import prepare_dino_source
    weights=Path(os.environ.get('R5_CLIP_WEIGHTS_DIR',str(Path.home()/'.cache/torch/hub/checkpoints')))
    for name in ['ViT-B/32','ViT-B/16']:
        url=clip._MODELS[name];dest=weights/(name.replace('/','-')+'.pt')
        download_public(url,dest,expected_sha256=url.split('/')[-2])
    path=Path(os.environ.get('R5_DINO_WEIGHTS',str(ROOT.parent/'研究1——CLIP结合DinoV2/pretrained/dinov2_vits14_pretrain.pth')))
    download_public('https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',path,'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9')
    prepare_dino_source();print('Encoder assets verified',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('preflight');a.add_argument('--require-gpus',type=int,default=1)
    sub.add_parser('assets',help='Explicitly download/verify official checkpoints and pinned source')
    a=sub.add_parser('import-pilot');a.add_argument('--manifest',required=True);a.add_argument('--cache',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('prepare-revisited');a.add_argument('--dataset',choices=['roxford','rparis'],required=True);a.add_argument('--images',required=True);a.add_argument('--annotations',required=True);a.add_argument('--bridge-root',required=True);a.add_argument('--output',required=True);a.add_argument('--bridge-count',type=int,default=256);a.add_argument('--seed',type=int,default=51850)
    a=sub.add_parser('prepare-sop');a.add_argument('--root',required=True);a.add_argument('--bridge-root');a.add_argument('--output',required=True);a.add_argument('--bridge-count',type=int,default=256);a.add_argument('--seed',type=int,default=51850);a.add_argument('--split',choices=['train','test'],default='test')
    a=sub.add_parser('add-distractors');a.add_argument('--manifest',required=True);a.add_argument('--images',required=True);a.add_argument('--count',type=int,required=True);a.add_argument('--output',required=True);a.add_argument('--seed',type=int,default=51890)
    a=sub.add_parser('encode');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);a.add_argument('--encoder',choices=['clip_b32','clip_b16','dino_s14'],required=True);a.add_argument('--device',default='cuda');a.add_argument('--batch-size',type=int,default=32);a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=1)
    a=sub.add_parser('join');a.add_argument('--manifest',required=True);a.add_argument('--output',required=True);a.add_argument('--encoder',required=True)
    a=sub.add_parser('evaluate');a.add_argument('--manifest',required=True);a.add_argument('--cache',required=True);a.add_argument('--output',required=True);a.add_argument('--config',default=str(ROOT/'configs/lab_frozen_v1.json'));a.add_argument('--device',default='cpu');a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=1);a.add_argument('--query-batch',type=int,default=16);a.add_argument('--dtype',choices=['float32','float64'],default='float32');a.add_argument('--max-queries',type=int)
    a=sub.add_parser('summarize');a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='preflight':return preflight(args.require_gpus)
    if args.command=='assets':assets()
    elif args.command=='import-pilot':import_pilot(args.manifest,args.cache,args.output)
    elif args.command=='prepare-revisited':
        from .lab_data import prepare_revisited
        prepare_revisited(args.dataset,args.images,args.annotations,args.bridge_root,args.output,args.bridge_count,args.seed)
    elif args.command=='prepare-sop':
        from .lab_data import prepare_sop
        prepare_sop(args.root,args.bridge_root,args.output,args.bridge_count,args.seed,args.split)
    elif args.command=='add-distractors':
        from .lab_data import add_distractors
        add_distractors(args.manifest,args.images,args.count,args.output,args.seed)
    elif args.command=='encode':
        from .lab_data import encode_manifest
        encode_manifest(args.manifest,args.output,args.encoder,args.device,args.batch_size,args.shard,args.shards)
    elif args.command=='join':
        from .lab_data import join_features
        join_features(args.manifest,args.output,args.encoder)
    elif args.command=='evaluate':
        from .lab_evaluate import evaluate
        evaluate(args.manifest,args.cache,args.output,args.config,args.device,args.shard,args.shards,args.query_batch,args.dtype,args.max_queries)
    elif args.command=='summarize':
        from .lab_evaluate import summarize
        summarize(args.output)
    return 0

if __name__=='__main__':raise SystemExit(main())
