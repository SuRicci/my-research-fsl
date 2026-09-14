"""Portable benchmark manifests and resumable sharded frozen feature extraction."""
from pathlib import Path
import hashlib,json,pickle,time
import numpy as np
from PIL import Image
from research_common.records import read_json,write_json,sha256,object_hash

class RestrictedAnnotations(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy.core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','scalar')}
        if (module,name) not in allowed:raise pickle.UnpicklingError(f'Forbidden annotation object: {module}.{name}')
        return super().find_class(module,name)

def identity(path):
    with Image.open(path) as im:
        a=np.asarray(im.convert('RGB'));small=np.asarray(im.convert('RGB').resize((9,8)).convert('L'))
    bits=(small[:,1:]>small[:,:-1]).ravel()
    dh=sum(int(v)<<i for i,v in enumerate(bits))
    return {'file_sha256':sha256(path),'rgb_sha256':hashlib.sha256(np.asarray(a.shape,np.int64).tobytes()+a.tobytes()).hexdigest(),'dhash64':f'{dh:016x}'}

def ordered(rows,seed):return sorted(rows,key=lambda r:hashlib.sha256(f"{seed}:{r['id']}".encode()).digest())

def bridge_records(root,count,seed):
    root=Path(root).resolve()
    paths=[p for p in root.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp']]
    rows=[{'id':'bridge:'+p.relative_to(root).as_posix(),'path':p.relative_to(root).as_posix(),'root':'bridge'} for p in paths]
    return ordered(rows,seed),root

def seal_manifest(manifest,destination,bridge_candidates,bridge_count):
    roots={k:Path(v) for k,v in manifest['roots'].items()};seen={};buckets={};excluded=[]
    role_for={i:role for role,ids in manifest['roles'].items() for i in ids};same_role_duplicates=[]
    for i,row in enumerate(manifest['records']):
        path=roots[row['root']]/row['path']
        if 'rgb_sha256' not in row:row.update(identity(path))
        elif sha256(path)!=row['file_sha256']:raise ValueError('Source changed during identity audit')
        if row['rgb_sha256'] in seen:
            previous=seen[row['rgb_sha256']]
            if role_for[previous]!=role_for[row['id']]:raise ValueError('Exact source identity collision across benchmark roles: '+row['id'])
            same_role_duplicates.append([previous,row['id']])
        seen[row['rgb_sha256']]=row['id']
        val=int(row['dhash64'],16)
        for j in range(4):buckets.setdefault((j,(val>>(16*j))&65535),[]).append(val)
        if (i+1)%1000==0:print(json.dumps({'identity_images':i+1}),flush=True)
    bridge=[]
    for row in bridge_candidates:
        if len(bridge)>=bridge_count:break
        row=dict(row);row.update(identity(roots['bridge']/row['path']));val=int(row['dhash64'],16)
        possible=set(v for j in range(4) for v in buckets.get((j,(val>>(16*j))&65535),[]))
        near=any(bin(val^v).count('1')<=2 for v in possible)
        if row['rgb_sha256'] in seen or near:
            excluded.append({'id':row['id'],'reason':'exact_rgb_or_conservative_dhash_distance_le_2'});continue
        seen[row['rgb_sha256']]=row['id'];bridge.append(row)
    if len(bridge)<bridge_count:raise ValueError(f'Only {len(bridge)} independent bridge images; need {bridge_count}')
    manifest['roles']['bridge']=[r['id'] for r in bridge];manifest['records'].extend(bridge)
    manifest.update(schema='r5_lab_manifest_v1',identity_audit={'exact_rgb_roles_disjoint':True,'bridge_near_duplicate_exclusions':excluded,
        'bridge_dhash_threshold':2,'benchmark_same_role_duplicate_rgb_groups':same_role_duplicates,'query_gallery_near_views':'Preserved according to the official retrieval protocol; no claim of manual near-duplicate review'},
        performance_opened=False)
    manifest['content_sha256']=object_hash(manifest);write_json(destination,manifest)
    print(json.dumps({'manifest':str(destination),'roles':{k:len(v) for k,v in manifest['roles'].items()}}),flush=True)

def prepare_revisited(name,image_root,annotations,bridge_root,destination,bridge_count=256,seed=51850):
    with open(annotations,'rb') as f:gnd=RestrictedAnnotations(f).load()
    images=Path(image_root).resolve();lookup={p.stem:p for p in images.rglob('*.jpg')}
    records=[];roles={'gallery':[],'query':[]}
    for role,key in [('gallery','imlist'),('query','qimlist')]:
        for i,n in enumerate(gnd[key]):
            if n not in lookup:raise FileNotFoundError(n)
            row={'id':role+':'+n,'path':lookup[n].relative_to(images).as_posix(),'root':'dataset'}
            if role=='query':row['bbox_xyxy']=list(map(float,gnd['gnd'][i]['bbx']))
            records.append(row);roles[role].append(row['id'])
    expected=(4993,70) if name=='roxford' else (6322,70)
    if (len(roles['gallery']),len(roles['query']))!=expected:raise ValueError('Unexpected official annotation version')
    truth={}
    for i,q in enumerate(roles['query']):
        entry=gnd['gnd'][i];easy=list(map(int,entry['easy']));hard=list(map(int,entry['hard']));junk=list(map(int,entry['junk']))
        truth[q]={'medium':{'positive':easy+hard,'ignore':junk},'hard':{'positive':hard,'ignore':easy+junk}}
    candidates,br=bridge_records(bridge_root,bridge_count,seed)
    m={'dataset':name,'protocol':'revisited','roots':{'dataset':str(images),'bridge':str(br)},'roles':roles,'records':records,
       'truth_evaluator_only':truth,'annotation_sha256':sha256(annotations),'seed':seed,'query_bbox_required':True,
       'bridge_semantic_domain':'User-specified external pool; semantic/landmark exclusion needs a separately supplied audit before class-disjoint claims'}
    seal_manifest(m,destination,candidates,bridge_count)

def add_distractors(manifest_path,image_root,count,destination,seed=51890):
    m=read_json(manifest_path)
    if m['protocol']!='revisited':raise ValueError('R1M appending is only defined for revisited protocols')
    root=Path(image_root).resolve();paths=sorted(root.rglob('*.jpg'),key=lambda p:hashlib.sha256(f'{seed}:{p.relative_to(root).as_posix()}'.encode()).digest())
    if len(paths)<count:raise ValueError(f'Only {len(paths)} distractors, need {count}')
    protected=set(m['roles']['query']+m['roles']['bridge'])
    protected_rgb={r['rgb_sha256'] for r in m['records'] if r['id'] in protected}
    if 'distractors' in m['roots']:raise ValueError('Append from the original base manifest, not an already expanded one')
    m['roots']['distractors']=str(root);new=[]
    for i,path in enumerate(paths[:count]):
        row={'id':'r1m:'+path.relative_to(root).as_posix(),'root':'distractors','path':path.relative_to(root).as_posix()};row.update(identity(path))
        if row['rgb_sha256'] in protected_rgb:raise ValueError('Distractor overlaps a query or bridge source: '+row['id'])
        new.append(row)
        if (i+1)%1000==0:print(json.dumps({'distractors_audited':i+1,'total':count}),flush=True)
    m['records'].extend(new);m['roles']['gallery'].extend(r['id'] for r in new)
    m['distractor_expansion']={'parent_manifest_sha256':sha256(manifest_path),'count':count,'seed':seed,'official_full_R1M':count==1001001,
        'ground_truth_unchanged':True,'bridge_distractor_near_duplicate_manual_audit':'not_completed; exact RGB protection enforced'}
    m.pop('content_sha256',None);m['content_sha256']=object_hash(m);write_json(destination,m)

def prepare_sop(root,bridge_root,destination,bridge_count=256,seed=51850,split='test'):
    root=Path(root).resolve();meta=root/('Ebay_'+split+'.txt');groups={}
    for line in meta.read_text(encoding='utf-8').splitlines()[1:]:
        parts=line.split()
        if len(parts)<4:continue
        row={'id':'sop:'+parts[0],'label':int(parts[1]),'path':parts[3],'root':'dataset'}
        groups.setdefault(row['label'],[]).append(row)
    roles={'gallery':[],'query':[]};records=[];labels={};skipped=[];rgb_labels={}
    for rows in groups.values():
        for r in rows:
            r.update(identity(root/r['path']));rgb_labels.setdefault(r['rgb_sha256'],set()).add(r['label'])
    ambiguous={h for h,l in rgb_labels.items() if len(l)>1}
    # Duplicates within a product are collapsed before deterministic role assignment.
    for label,rows in sorted(groups.items()):
        unique={}
        for r in ordered(rows,seed):
            if r['rgb_sha256'] not in ambiguous:unique.setdefault(r['rgb_sha256'],r)
        rows=list(unique.values())
        if len(rows)<2:skipped.append(label);continue
        for i,r in enumerate(rows):roles['query' if i==0 else 'gallery'].append(r['id']);records.append(r);labels[r['id']]=label
    if bridge_root is None:
        if split!='test':raise ValueError('Training-development manifests require a separately disjoint bridge root')
        br=root;candidates=[]
        for line in (root/'Ebay_train.txt').read_text(encoding='utf-8').splitlines()[1:]:
            p=line.split()
            if len(p)>=4:candidates.append({'id':'bridge:'+p[0],'path':p[3],'root':'bridge'})
        candidates=ordered(candidates,seed)
    else:candidates,br=bridge_records(bridge_root,bridge_count,seed)
    m={'dataset':'sop_disjoint_'+split,'protocol':'class','roots':{'dataset':str(root),'bridge':str(br)},'roles':roles,'records':records,
       'labels_evaluator_only':labels,'annotation_sha256':sha256(meta),'official_all_to_all':False,'seed':seed,'classes_skipped_for_identity_count':skipped,
       'ambiguous_cross_product_rgb_groups_removed':len(ambiguous)}
    seal_manifest(m,destination,candidates,bridge_count)

def encode_manifest(manifest_path,output,encoder,device='cuda',batch_size=32,shard=0,shards=1):
    import torch
    from research_common.encoders import FrozenEncoder
    manifest=read_json(manifest_path);records=manifest['records'];indices=list(range(shard,len(records),shards))
    if not 0<=shard<shards or batch_size<1 or not indices:raise ValueError('Invalid or empty encoder shard')
    folder=Path(output)/encoder;folder.mkdir(parents=True,exist_ok=True);path=folder/f'part_{shard:02d}_of_{shards:02d}.npy';meta_path=path.with_suffix('.json')
    seal=sha256(manifest_path)
    if meta_path.exists():
        meta=read_json(meta_path)
        if meta.get('complete'):
            if meta['manifest_sha256']!=seal or sha256(path)!=meta['array_sha256']:raise ValueError('Feature cache mismatch')
            print('Verified complete feature shard',flush=True);return
    model=FrozenEncoder(encoder,device);model_sig=object_hash({k:v for k,v in model.identity.items() if k!='device'})
    progress=read_json(meta_path) if meta_path.exists() else {'completed_rows':0,'manifest_sha256':seal,'model_signature':model_sig}
    if progress['manifest_sha256']!=seal or progress['model_signature']!=model_sig:raise ValueError('Resume identity changed')
    completed=progress['completed_rows'];array=np.load(path,mmap_mode='r+') if path.exists() else None;start_time=time.perf_counter()
    for start in range(completed,len(indices),batch_size):
        images=[]
        for i in indices[start:start+batch_size]:
            r=records[i];source=Path(manifest['roots'][r['root']])/r['path']
            if sha256(source)!=r['file_sha256']:raise ValueError('Image changed: '+r['id'])
            with Image.open(source) as im:
                im=im.convert('RGB')
                if 'bbox_xyxy' in r:im=im.crop(tuple(r['bbox_xyxy']))
                images.append(im.copy())
        value=model.encode_pil(images)
        if array is None:array=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(indices),value.shape[1]))
        array[start:start+len(value)]=value;array.flush()
        progress.update(completed_rows=start+len(value),shard=shard,shards=shards,total_records=len(records),model=model.identity,complete=False)
        write_json(meta_path,progress,overwrite=True)
        if start%(batch_size*16)==0:print(json.dumps({'encoder':encoder,'shard':shard,'rows':start+len(value),'total':len(indices)}),flush=True)
    progress.update(complete=True,array_sha256=sha256(path),seconds_this_invocation=time.perf_counter()-start_time,
       role_count={k:len(v) for k,v in manifest['roles'].items()},new_gallery_encoding='Evaluator-only oracle preparation; never given to method adapters')
    write_json(meta_path,progress,overwrite=True);model.close()

def join_features(manifest_path,output,encoder):
    folder=Path(output)/encoder;meta_files=sorted(folder.glob('part_*.json'));manifest=read_json(manifest_path)
    target=folder/'embeddings.npy';meta_target=folder/'embeddings.json'
    if meta_target.exists():
        meta=read_json(meta_target)
        if meta['manifest_sha256']==sha256(manifest_path) and sha256(target)==meta['array_sha256']:return
        raise ValueError('Merged cache identity mismatch')
    seen=set();array=None;signatures=set()
    for f in meta_files:
        m=read_json(f)
        if not m.get('complete') or m['manifest_sha256']!=sha256(manifest_path):raise ValueError('Incomplete/mismatched shard')
        data=np.load(f.with_suffix('.npy'),mmap_mode='r');assert sha256(f.with_suffix('.npy'))==m['array_sha256']
        indices=m['indices'] if 'indices' in m else list(range(m['shard'],m['total_records'],m['shards']))
        if seen&set(indices):raise ValueError('Duplicate shard indices')
        seen.update(indices);signatures.add(m['model_signature'])
        if array is None:array=np.lib.format.open_memmap(target,mode='w+',dtype='float32',shape=(len(manifest['records']),data.shape[1]))
        array[indices]=data
    if seen!=set(range(len(manifest['records']))) or len(signatures)!=1:raise ValueError('Missing rows or encoder mismatch')
    array.flush();write_json(meta_target,{'manifest_sha256':sha256(manifest_path),'array_sha256':sha256(target),'model_signature':next(iter(signatures)),'rows':len(seen)})
