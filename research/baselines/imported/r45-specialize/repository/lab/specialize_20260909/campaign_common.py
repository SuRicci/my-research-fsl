"""Isolated fourth campaign; immutable inputs and sequential candidate execution."""
from pathlib import Path
import hashlib,json,os,time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=Path('/data/liuhaoyu/r45-specialize-20260909')
ORDER=['r4_whiten','r4_rank','r4_diffusion','r4_ensemble','r4_top100',
       'r5_consensus','r5_median','r5_variance','r5_distribution','r5_support_cv']

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False));os.replace(temp,p)
def bind(p,x):
    if Path(p).exists():assert read(p)==json.loads(json.dumps(x)),str(p)
    else:dump(p,x)
def sources():
    files=list(HERE.glob('*.py'))+[HERE/'candidates.json',HERE/'PLAN.md',HERE/'ideas.md']
    files+=list((ROOT/'lab/refine_20260909').glob('*.py'))
    files+=[ROOT/'lab/server_scale_20260909/r5_scale.py']
    files+=list((ROOT/'研究4——少桥接样本的跨编码器检索/src').glob('*.py'))
    files+=list((ROOT/'研究5——少样本支持集的检索增强与可靠性加权/studies/r2_20260909').glob('methods.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files))}
def validate_sources():
    p=read(OUT/'protocol.json')
    for name,digest in p['sources'].items():assert sha(ROOT/name)==digest,name
    return p
