"""Fixed backbone variant; reuse the retained equations and image permissions."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, os, shutil, sys
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; OUT=HERE/'outputs'; ASSET=HERE/'assets'
CFG=json.loads((HERE/'protocol.json').read_text()); REF=HERE.parent/'query-consistency-20260913/outputs'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
z=module('base_retained',REF.parent/'consistency_model.py')
ex=module('base_views',HERE.parent/'representation-scatter-20260913/extract_views.py')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()
def dump(path,x):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(x,indent=2)+'\n');os.replace(temp,path)
def guard(reserve=0):
    assert shutil.disk_usage(HERE).free>=CFG['min_free_gib']*2**30+reserve,'disk floor'
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline']),'deadline'
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<CFG['max_new_mib']*2**20,'output cap'
def backbone(device):
    manifest=json.loads((ASSET/'weight_manifest.json').read_text());p=ASSET/'dinov2_vitb14_pretrain.pth';assert p.stat().st_size==CFG['weights_bytes'] and sha(p)==manifest['sha256']
    repo=ex.enc.Q/'baselines/imported/r45-specialize/repository/research_common/extern/dinov2'
    m=torch.hub.load(str(repo),'dinov2_vitb14',source='local',pretrained=False)
    state=torch.load(p,map_location='cpu',weights_only=True);m.load_state_dict(state,strict=True)
    assert m.embed_dim==768
    return m.float().eval().requires_grad_(False).to(device)
def retained(s,q,g,gamma,eta):return z.evaluate(s,q,g,gamma*CFG['base_dimension']/s.shape[-1],[eta],[0.])
def ridge(x,q,weight,penalty=.1):
    y=np.eye(5).repeat(len(x)//5,0);xm=np.average(x,axis=0,weights=weight);ym=np.average(y,axis=0,weights=weight)
    a=(x-xm)*np.sqrt(weight)[:,None];b=(y-ym)*np.sqrt(weight)[:,None]
    return (q-xm)@a.T@np.linalg.solve(a@a.T+penalty*np.eye(len(x)),b)+ym
