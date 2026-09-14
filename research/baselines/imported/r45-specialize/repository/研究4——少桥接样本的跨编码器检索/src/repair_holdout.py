"""Resolve exact pixel identity collisions without changing methods or query/gallery sets."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
from research_common.records import read_json,write_json,sha256
from .data import PROJECT
from .optimize import run_grid

def main():
    manifest_path=PROJECT/'data/optimization_holdout_v1.json';m=read_json(manifest_path)
    roles={i:k for k,v in m['roles'].items() for i in v}
    groups={}
    for r in m['records']:
        with Image.open(Path(m['source_directory'])/'images'/r['relative_path']) as im:a=np.asarray(im.convert('RGB'))
        h=hashlib.sha256(np.asarray(a.shape,np.int64).tobytes()+a.tobytes()).hexdigest()
        groups.setdefault(h,[]).append({'id':r['id'],'role':roles[r['id']]})
    duplicates=[v for v in groups.values() if len(v)>1]
    excluded=[]
    for group in duplicates:
        assert len(group)==2 and {r['role'] for r in group}=={'bridge','gallery'},'Unexpected collision; inspect before scoring'
        excluded += [r['id'] for r in group if r['role']=='bridge']
    affected=[]
    for f in (PROJECT/'runs/optimization_holdout_v1/jobs').glob('*/input_receipt.json'):
        if set(read_json(f)['anchors']) & set(excluded):affected.append(f.parent.name)
    receipt={'manifest_sha256':sha256(manifest_path),'selection_sha256':sha256(PROJECT/'runs/optimization_dev_v1/selection.json'),
        'exact_rgb_duplicate_groups':duplicates,'bridge_ids_to_exclude':sorted(excluded),'affected_original_jobs':sorted(affected),
        'discovered_after_initial_holdout_evaluation':True,'original_run_preserved':True,
        'correction_rule':'Keep all gallery and query identities. Remove exact duplicate bridge images from the original seed permutation before taking the same bridge budget.',
        'method_or_hyperparameter_changes':False,'new_independent_replication':False}
    path=PROJECT/'data/holdout_rgb_audit.json'
    if not path.exists():write_json(path,receipt)
    else:assert read_json(path)==receipt
    print(json.dumps(receipt),flush=True)
    run_grid(PROJECT/'runs/optimization_holdout_deduplicated_v1',holdout=True,deduplicate_bridge=True)

if __name__=='__main__':main()
