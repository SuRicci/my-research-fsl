"""Replay the frozen R2 5-shot DTD protocol using restored, unchanged features."""
from pathlib import Path
import json, hashlib, sys, time
import numpy as np
import torch
from methods import representation, evaluate_configs
from optimized import fit_optimized
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=HERE/'results';OUT.mkdir(exist_ok=True)
torch.set_num_threads(8)
paths=(HERE/'data/dtd/labels/test1.txt').read_text().splitlines()
classes=sorted(set(p.split('/')[0] for p in paths))
labels=np.array([classes.index(p.split('/')[0]) for p in paths])
assert len(labels)==1880 and len(classes)==47
drop=json.loads((HERE/'source/dtd_test_query_dedup.json').read_text())['excluded_indices']
ids=np.array([i for i in range(1880) if i not in drop])
assert len(ids)==1877
raw=[torch.load(HERE/'assets'/('dtd_test_'+b+'_query.pt'),weights_only=True)['features'].float() for b in ['clip_vitb16','dinov2_vits14']]
qf=representation(*raw,.5)
pools={int(c):ids[labels[ids]==c] for c in np.unique(labels)}
allowed=np.random.RandomState(120909).permutation(sorted(pools))[23:]
supports=[];queries=[];csall=[];seeds=[]
for seed in [129191,129192,129193]:
    rng=np.random.RandomState(seed+5)
    for _ in range(200):
        cs=rng.choice(allowed,5,replace=False)
        idx=np.stack([rng.choice(pools[int(c)],20,replace=False) for c in cs])
        supports.append(idx[:,:5]);queries.append(idx[:,5:]);csall.append(cs);seeds.append(seed)
si=np.stack(supports);xi=np.stack(queries);y=np.repeat(np.arange(5),15)
configs=[{'family':'raw_ridge','w':.5,'lam':1.,'name':'r2_overall'},
         {'family':'raw','w':.5,'name':'same_feature_prototype'}]
score={c['name']:[] for c in configs};started=time.time()
for st in range(0,600,24):
    S=qf[si[st:st+24]];X=qf[xi[st:st+24].reshape(-1,75)]
    _,scores,_=evaluate_configs(S,X,torch.empty(0,896),configs,return_scores=True)
    for name,values in scores.items():score[name].append(values.numpy())
scores=np.stack([np.concatenate(score[c['name']]) for c in configs])
pred=scores.argmax(-1);acc=(pred==y).mean(-1)
api_error=0.;api_flips=0
for e in [0,199,200,399,400,599]:
    model=fit_optimized(raw[0][si[e].ravel()],raw[1][si[e].ravel()],torch.arange(5).repeat_interleave(5))
    v=model.scores(raw[0][xi[e].ravel()],raw[1][xi[e].ravel()]).numpy()
    api_error=max(api_error,float(np.abs(v-scores[0,e]).max()))
    api_flips+=int((v.argmax(-1)!=pred[0,e]).sum())
assert api_error<1e-5 and api_flips==0
path=OUT/'dtd_k5_original_retest.npz'
np.savez_compressed(path,scores=scores,predictions=pred,accuracy=acc,yq=y,support_indices=si,query_indices=xi,class_ids=csall,seed=seeds,names=[c['name'] for c in configs],query_pool_ids=ids)
report={'status':'computed','dataset':'dtd','phase':'original_r2_retest','shot':5,'episodes':600,'query_pool':len(ids),'allowed_classes':allowed.tolist(),'seeds':[129191,129192,129193],
'accuracy':{c['name']:float(acc[i].mean()) for i,c in enumerate(configs)},
'seed_accuracy':{c['name']:[float(v.mean()) for v in acc[i].reshape(3,200)] for i,c in enumerate(configs)},
'packaged_api_max_abs_score_error':api_error,'packaged_api_prediction_flips':api_flips,
'elapsed_seconds':time.time()-started,'npz_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
'deviations':['CPU float32 evaluator versus historical CUDA; no gradient training or query batch statistics',
'Images not yet rehashed locally; original recorded exact-RGB exclusions used unchanged',
'Original per-query predictions absent locally; point estimate comparison does not prove bitwise historical prediction identity']}
(OUT/'dtd_k5_original_retest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
