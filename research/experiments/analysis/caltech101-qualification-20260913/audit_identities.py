from pathlib import Path
from collections import Counter,defaultdict
import datetime,hashlib,io,json,shutil,tarfile,time,zipfile
import numpy as np
from PIL import Image
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];Q=Path('/Users/decoqwq/DeepScientist/quests/012')
C=json.loads((HERE/'CONTRACT.json').read_text());A=json.loads((HERE/'acquisition.json').read_text());assert A['status']=='passed'
assert shutil.disk_usage(Q).free/2**30>=10.05
assert hashlib.sha256(Path(A['archive_path']).read_bytes()).hexdigest()==A['sha256']
def rgb_values(x):
    values=[]
    if isinstance(x,dict):
        for k,v in x.items():
            if 'rgb' in k.lower():
                if isinstance(v,str) and len(v)==64:values.append(v)
                elif isinstance(v,list):values.extend(a for a in v if isinstance(a,str) and len(a)==64)
            if isinstance(v,(dict,list)):values.extend(rgb_values(v))
    elif isinstance(x,list):
        for v in x:
            if isinstance(v,(dict,list)):values.extend(rgb_values(v))
    return values
paths=sorted((Q/'baselines/local').rglob('*identities*.json'))+[ROOT/'experiments/main/fsl-pets-transfer-20260913/assets/identities.json']
reference={};known=set()
for p in paths:
    v=set(rgb_values(json.loads(p.read_text())));assert v,p
    reference[str(p)]={'count':len(v),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};known.update(v)
rows=[];errors=[];non_images=[];image_bytes=0;start=time.time()
with zipfile.ZipFile(A['archive_path']) as z,z.open('caltech-101/101_ObjectCategories.tar.gz') as inner,tarfile.open(fileobj=inner,mode='r|gz') as t:
    for member in t:
        name=Path(member.name);assert not name.is_absolute() and '..' not in name.parts
        assert not member.issym() and not member.islnk()
        if not member.isfile():continue
        if name.suffix.lower() not in ['.jpg','.jpeg','.png']:non_images.append(member.name);continue
        assert name.parts[0]=='101_ObjectCategories' and len(name.parts)==3
        raw=t.extractfile(member).read();image_bytes+=len(raw)
        try:
            with Image.open(io.BytesIO(raw)) as im:arr=np.asarray(im.convert('RGB'));shape=arr.shape
            rgb=hashlib.sha256(str(shape).encode()+arr.tobytes()).hexdigest()
            rows.append({'member':member.name,'class_name':name.parts[1],'rgb':rgb,'shape':list(shape),'jpeg_sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
        except Exception as e:errors.append({'member':member.name,'error':repr(e)})
        if len(rows)%1000==0:print('DECODED',len(rows),flush=True)
assert not errors,errors
obj=[r for r in rows if r['class_name']!='BACKGROUND_Google'];groups=defaultdict(list)
for r in obj:groups[r['rgb']].append(r)
kept=[];excluded=[];duplicate_groups=0;conflicts=0
for rgb,group in sorted(groups.items()):
    group.sort(key=lambda r:r['member']);classes={r['class_name'] for r in group}
    duplicate_groups+=len(group)>1
    if rgb in known:excluded.extend(dict(r,reason='prior exact RGB overlap') for r in group)
    elif len(classes)>1:conflicts+=1;excluded.extend(dict(r,reason='cross-class exact RGB conflict') for r in group)
    else:kept.append(group[0]);excluded.extend(dict(r,reason='within-class exact RGB duplicate') for r in group[1:])
classes=defaultdict(list)
for r in kept:classes[r['class_name']].append(r)
query=[];gallery=[];counts={};ineligible=[]
for c,rr in sorted(classes.items()):
    rr.sort(key=lambda r:hashlib.sha256(('26091393:'+r['rgb']).encode()).hexdigest());n=len(rr)//3
    counts[c]={'unique':len(rr),'gallery':n,'query_pool':len(rr)-n}
    if len(rr)-n<16:ineligible.append(c);continue
    gallery.extend(rr[:n]);query.extend(rr[n:])
assert not ({r['rgb'] for r in query}&{r['rgb'] for r in gallery})
assert not ({r['rgb'] for r in query+gallery}&known)
identities={'query':query,'gallery':gallery,'excluded':excluded,'classes':counts,'ineligible_classes':ineligible,'split_seed':26091393,'source_archive_sha256':A['sha256'],'prior_manifests':reference}
(HERE/'identities.json').write_text(json.dumps(identities,indent=2)+'\n')
result={'status':'passed' if not ineligible and len(classes)>=5 and len(gallery)>=1024 else 'partial','decoded_images':len(rows),'object_images':len(obj),'background_images':len(rows)-len(obj),'object_classes':len(classes),'query_pool_images':len(query),'gallery_pool_images':len(gallery),'class_count_min':min(x['unique'] for x in counts.values()),'query_class_min':min(x['query_pool'] for x in counts.values()),'within_archive_duplicate_groups':duplicate_groups,'cross_class_conflicts':conflicts,'excluded_by_reason':dict(Counter(x['reason'] for x in excluded)),'ineligible_classes':ineligible,'reference_manifest_count':len(reference),'reference_unique_rgb_count':len(known),'prior_manifests':reference,'decoded_errors':errors,'non_image_members':non_images,'unpacked_jpeg_bytes':image_bytes,'six_view_float32_feature_bytes_all_images':(len(query)+len(gallery))*6*896*4,'no_target_accuracy_computed':True,'elapsed_seconds':time.time()-start,'free_gib':shutil.disk_usage(Q).free/2**30,'scope':'Exact RGB identity only within the eight listed local manifests; no perceptual/pretraining/global-history independence proof.'}
(HERE/'identity_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='prior_manifests'},indent=2))
