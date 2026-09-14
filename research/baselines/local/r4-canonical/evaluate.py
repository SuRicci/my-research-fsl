"""Frozen V0 comparison on an explicit new local semantic-retrieval contract."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='6'
from pathlib import Path
import argparse,hashlib,json,time,shutil
import numpy as np
import torch
from source.methods import unit
from source.lab_methods import FeatureMap,METHODS
from source.improved import kernel_residual_scores
from source.lab_metrics import evaluate_rank
from source.geometry import variants

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
OUT=ROOT/'outputs'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False))

def load_data():
 ar=Path(P['asset_root']);manifest=json.loads((ar/'manifest.json').read_text())
 assert sha(ar/'manifest.json')==P['source_feature_manifest_sha256']
 data={};identity_hashes={}
 for ds in P['datasets']:
  ip=ar/(ds+'_identities.json');ids=json.loads(ip.read_text());d={'identity':ids}
  identity_hashes[ds]=sha(ip)
  for role in ('query','gallery'):
   for enc in P['pair']:
    entry=manifest['datasets'][ds]['backbones'][enc+'_'+role];path=Path(entry['path'])
    assert sha(path)==entry['sha256']
    z=torch.load(path,map_location='cpu');assert z['ids'].tolist()==ids[role+'_ids']
    x=unit(z['features'].float().numpy());assert len(x)==len(ids[role+'_rgb'])
    d[enc+'_'+role]=x
   assert len(set(ids[role+'_rgb']))==len(ids[role+'_rgb'])
  assert not set(ids['query_rgb'])&set(ids['gallery_rgb'])
  labels=np.array(ids['query_labels']);rng=np.random.default_rng(P['identity_selection_seed'])
  qi=np.concatenate([rng.choice(np.flatnonzero(labels==c),P['queries_per_class'],replace=False) for c in np.unique(labels)])
  remaining=np.setdiff1d(np.arange(len(labels)),qi)
  classes=np.unique(labels)
  if ds=='dtd':classes=np.random.default_rng(P['split_seed']).permutation(classes)
  split={'dev':classes[:23],'eval':classes[23:]} if ds=='dtd' else {'eval':classes}
  d.update(query_labels=labels,gallery_labels=np.array(ids['gallery_labels']),query_indices=qi,remaining=remaining,splits=split)
  data[ds]=d
 a,b=[data[n]['identity'] for n in P['datasets']]
 assert not set(a['query_rgb']+a['gallery_rgb'])&set(b['query_rgb']+b['gallery_rgb'])
 return data,identity_hashes

def geometry_scores(go,qo):
 z,_,_=variants(go,qo,external=True)
 return {'old_centered':z['center'][1]@z['center'][0].T,'old_whiten':z['white'][1]@z['white'][0].T}

def scores(go,qo,qn,ao,an,geometry):
 result={}
 for method in METHODS:
  fm=FeatureMap(ao,an,method);result[method]=fm.query(qo,qn)@fm.gallery(go).T
 result.update(geometry)
 residual=result['v0_press']-result['old_old'];residual-=residual.mean(1,keepdims=True)
 ratio=result['old_centered'].std(1,keepdims=True)/np.maximum(result['old_old'].std(1,keepdims=True),1e-10)
 result['centered_residual_2']=result['old_centered']+2*ratio*residual
 return result

def ap_from_ranks(rank,qy,gy):
 hit=gy[rank]==qy[:,None]
 return ((np.cumsum(hit,axis=1)/np.arange(1,rank.shape[1]+1))*hit).sum(1)/hit.sum(1)

def validate(data,identity_hashes):
 d=data['dtd'];q=d['query_indices'][:3];a=d['remaining'][:32];old,new=P['pair']
 go=d[old+'_gallery'][:60];qo=d[old+'_query'][q];qn=d[new+'_query'][q];ao=d[old+'_query'][a];an=d[new+'_query'][a]
 geom=geometry_scores(go,qo);s=scores(go,qo,qn,ao,an,geom)
 direct,_=kernel_residual_scores(go,qo,qn,ao,an,'rbf_0.5',True)
 equivalence=float(np.max(abs(s['v0_press']-direct)));assert equivalence<1e-9
 error=0.
 for i in range(3):
  one=scores(go,qo[i:i+1],qn[i:i+1],ao,an,{k:v[i:i+1] for k,v in geom.items()})
  error=max(error,max(float(np.max(abs(s[k][i]-one[k][0]))) for k in one))
 assert error<1e-9
 rank=np.array([[2,0,3,1],[0,1,2,3]]);qy=np.array([1,2]);gy=np.array([1,2,1,2])
 ap=ap_from_ranks(rank,qy,gy)
 for i in range(2):assert abs(ap[i]-evaluate_rank(rank[i],{'positive':np.flatnonzero(gy==qy[i])})['ap'])<1e-14
 for n,h in P['source_sha256'].items():assert sha(ROOT/'source'/n)==h
 v={'passed':True,'v0_original_equivalence_max':equivalence,'query_independence_max':error,'all_eight_feature_hashes_match':True,'source_files':len(P['source_sha256']),'identity_hashes':identity_hashes,'rgb_role_and_domain_disjoint':True,'ap_reference_check':True,'torch':torch.__version__,'numpy':np.__version__,'protocol_sha256':sha(ROOT/'protocol.json'),'evaluator_sha256':sha(__file__)}
 dump(ROOT/'validation.json',v);print(json.dumps(v),flush=True)

def run(data,identity_hashes):
 v=json.loads((ROOT/'validation.json').read_text());assert v['passed'] and v['evaluator_sha256']==sha(__file__)
 assert v['protocol_sha256']==sha(ROOT/'protocol.json') and v['identity_hashes']==identity_hashes
 assert not (OUT/'completion.json').exists(),'Completed; do not rerun'
 start=time.time();dump(OUT/'manifest.json',{'protocol':P,'validation':v,'started_at':start,'cwd':str(ROOT),'command':'/opt/anaconda3/envs/torch/bin/python evaluate.py','new_image_encodings':0})
 total=0;old,new=P['pair']
 for ds,d in data.items():
  for phase,classes in d['splits'].items():
   qi=d['query_indices'][np.isin(d['query_labels'][d['query_indices']],classes)];qy=d['query_labels'][qi]
   go=d[old+'_gallery'];qo=d[old+'_query'][qi];qn=d[new+'_query'][qi]
   geom=geometry_scores(go,qo);oracle=qn@d[new+'_gallery'].T
   for policy in ('matched','cross'):
    bd=ds if policy=='matched' else next(n for n in data if n!=ds);other=data[bd]
    pool=other['remaining']
    if policy=='matched':pool=pool[np.isin(other['query_labels'][pool],classes)]
    for seed in P['bridge_seeds']:
     perm=np.random.default_rng(seed).permutation(pool)
     for m in P['budgets']:
      assert shutil.disk_usage(ROOT).free/2**30>=P['limits']['free_disk_min_gib']
      ai=perm[:m];assert len(ai)==m
      ao=other[old+'_query'][ai];an=other[new+'_query'][ai]
      result=scores(go,qo,qn,ao,an,geom);result['new_new_oracle']=oracle
      assert set(result)==set(P['methods'])
      ranks=[];aps=[]
      for name in P['methods']:
       assert np.isfinite(result[name]).all()
       rank=np.argsort(-result[name],axis=1,kind='stable').astype(np.uint16)
       ranks.append(rank);aps.append(ap_from_ranks(rank,qy,d['gallery_labels']))
      key=f'{ds}_{policy}_m{m}_s{seed}';dest=OUT/phase/(key+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
      np.savez_compressed(dest,methods=P['methods'],ranks=np.stack(ranks),ap=np.stack(aps),query_indices=qi,query_ids=np.array(d['identity']['query_ids'])[qi],query_labels=qy,gallery_ids=d['identity']['gallery_ids'],gallery_labels=d['gallery_labels'],bridge_indices=ai,bridge_ids=np.array(other['identity']['query_ids'])[ai],bridge_domain=bd,seed=seed,budget=m)
      dump(dest.with_suffix('.json'),{'dataset':ds,'phase':phase,'policy':policy,'budget':m,'seed':seed,'npz_sha256':sha(dest),'query_count':len(qi),'gallery_count':len(go),'bridge_count':len(ai)})
      total+=1;dump(OUT/'progress.json',{'completed_jobs':total,'latest':key,'phase':phase,'elapsed_seconds':time.time()-start});print('COMPLETE',phase,key,total,flush=True)
 assert total==60
 dump(OUT/'completion.json',{'complete':True,'job_count':total,'eval_jobs':40,'dev_jobs':20,'elapsed_seconds':time.time()-start})

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
 torch.set_num_threads(6);data,identity_hashes=load_data()
 if args.validate_only:validate(data,identity_hashes)
 else:run(data,identity_hashes)
