from pathlib import Path
import numpy as np
import torch
from PIL import Image
from research_common.records import read_json,write_json,sha256
from research_common.encoders import FrozenEncoder
from .data import PROJECT

def encode():
    selection=PROJECT/'runs/optimization_dev_v1/selection.json'
    if not selection.exists() or not read_json(selection)['frozen']:raise RuntimeError('Freeze selection first')
    manifest_file=PROJECT/'data/optimization_holdout_v1.json';m=read_json(manifest_file)
    if sha256(manifest_file)!=read_json(selection)['holdout_manifest_sha256']:raise ValueError('Holdout seal changed')
    folder=PROJECT/'cache/optimization_holdout';folder.mkdir(parents=True,exist_ok=True)
    for name in ['clip_b32','clip_b16','dino_s14']:
        path=folder/(name+'_all.npz')
        if path.exists():
            meta=read_json(path.with_suffix('.json'))
            if meta['npz_sha256']!=sha256(path):raise ValueError('Cache mismatch')
            continue
        model=FrozenEncoder(name);arrays=[];ids=[]
        for start in range(0,len(m['records']),32):
            rows=m['records'][start:start+32];images=[]
            for row in rows:
                image=Path(m['source_directory'])/'images'/row['relative_path']
                if sha256(image)!=row['file_sha256']:raise ValueError('Image changed')
                with Image.open(image) as im:images.append(im.convert('RGB').copy())
            arrays.append(model.encode_pil(images));ids.extend(r['id'] for r in rows)
            if start%256==0:print(f'heldout encode {name}: {start+len(rows)}/{len(m["records"])}',flush=True)
        np.savez_compressed(path,ids=np.array(ids),embeddings=np.concatenate(arrays))
        write_json(path.with_suffix('.json'),{'model':model.identity,'manifest_sha256':sha256(manifest_file),'selection_sha256':sha256(selection),'npz_sha256':sha256(path),'evaluation_only_new_gallery':True})
        model.close()

if __name__=='__main__':encode()
