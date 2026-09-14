"""Audit source-only feasibility without computing new classifier outcomes."""
from pathlib import Path
import ast, hashlib, json, shutil
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/'experiments/main/representation-scatter-20260913'
cfg=json.loads((BASE/'protocol.json').read_text())
ucfg=json.loads((ROOT/'experiments/main/utility-composition-qualification-20260913/protocol.json').read_text())
cfg.update({k:ucfg[k] for k in ['train_seed','train_episodes']})
sampler=ROOT/'experiments/main/utility-composition-qualification-20260913/run.py'
tree=ast.parse(sampler.read_text())
func=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='tasks')
ns={'np':np,'CFG':cfg}
exec(compile(ast.Module(body=[func],type_ignores=[]),str(sampler),'exec'),ns)
asset=Path(cfg['asset_root'])
features=json.loads((BASE/'assets/feature_manifest.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={}
assert features['status']=='completed'
for key,entry in features['files'].items():
 p=Path(entry['path']);assert p.exists(),p
 actual=sha(p);assert actual==entry['sha256'],p
 files[key]={'path':str(p),'bytes':p.stat().st_size,'sha256':actual}
domains={};identities={}
gamma=json.loads((ROOT/'experiments/main/scatter-centering-stack-20260913/protocol.json').read_text())['gamma_by_source']
for ds in cfg['source_domains']:
 ip=asset/(ds+'_identities.json');ident=json.loads(ip.read_text());identities[ds]=ident
 labels=np.array(ident['query_labels']);rgb=np.array(ident['query_rgb'])
 assert len(set(rgb))==len(rgb)
 t={phase:ns['tasks'](ident,1,phase) for phase in ['train','selection']}
 ids={phase:np.unique(np.concatenate([v['support_indices'].ravel(),v['query_indices'].ravel()])) for phase,v in t.items()}
 assert not set(rgb[ids['train']])&set(rgb[ids['selection']])
 assert not set(rgb)&set(ident['gallery_rgb'])
 for phase,z in t.items():
  assert z['query_indices'].shape==(100,5,15)
  for i in range(100):
   q=z['query_indices'][i];s=z['support_indices'][i]
   assert len(set(np.r_[q.ravel(),s.ravel()]))==80
   assert np.array_equal(labels[s].reshape(5),z['class_ids'][i])
   assert np.array_equal(labels[q],np.repeat(z['class_ids'][i,:,None],15,axis=1))
 # Verify class halves support the one-shot request; no dependence on sampled tasks.
 sizes=[np.count_nonzero(labels==c) for c in np.unique(labels)]
 gy=np.array(ident['gallery_labels'])
 max5=sum(sorted([np.count_nonzero(gy==c) for c in np.unique(gy)],reverse=True)[:5])
 remaining=len(gy)-max5
 assert remaining>=1024
 domains[ds]={'classes':len(sizes),'query_identities':len(rgb),'gallery_identities':len(gy),
  'min_class_images':min(sizes),'min_train_half':min(sizes)//2,'min_selection_half':(min(sizes)+1)//2,
  'train_tasks':100,'selection_tasks':100,'query_per_class':15,
  'train_used_rgb':len(ids['train']),'selection_used_rgb':len(ids['selection']),
  'train_selection_rgb_overlap':0,'query_gallery_rgb_overlap':0,
  'worst_case_task_excluded_gallery_size':remaining,
  'required_gamma':gamma[ds]['1'],'invalid_existing_reverse_bank_gamma':gamma['eurosat' if ds=='dtd' else 'dtd']['1'],
  'identity_sha256':sha(ip)}
allrgb={d:set(z['query_rgb'])|set(z['gallery_rgb']) for d,z in identities.items()}
assert not allrgb['dtd']&allrgb['eurosat']
out={'status':'passed','domains':domains,'feature_files':files,'feature_count':len(files),
 'cross_domain_rgb_overlap':0,'sampler_sha256':sha(sampler),
 'source_contract':'Use first identity half for fitting and second for source selection; inherited gamma is selected from that source only. Second half already served gamma selection and is not pristine validation. Source training/gallery may use source labels; target feature/label use forbidden during gate fitting.',
 'missing':'Full centered/raw/blend scores must be generated for SOURCE train/selection using source gamma. Existing reverse-direction scored banks are inadmissible as direct gate-training inputs.',
 'cost':'Reuse verified frozen features; no encoder run or downloads; source heads only.',
 'free_gib':shutil.disk_usage(ROOT).free/2**30}
(HERE/'outputs/source_feasibility.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['status','domains','feature_count','cross_domain_rgb_overlap','free_gib']},indent=2))
