"""Read-only asset/protocol audit; new evidence is written here only."""
import sys, json, hashlib, time
from pathlib import Path
from collections import Counter
import requests
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
E11 = next(ROOT.glob('实验11*'))
E12 = next(ROOT.glob('实验12*'))
sys.path.insert(0, str(E11))
sys.path.insert(0, str(E12))
from e11 import io
from e12 import r3bridge

def rgb_hash(image):
    a = np.asarray(image.convert('RGB'))
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()

def main():
    out = {'date': '2026-09-09', 'e11': {}, 'e12': {}}
    for split in ['train', 'test']:
        try:
            ps = io.load_problems(split, True, 4)
        except json.JSONDecodeError as e:
            out['e11'][split] = {'metadata_error':str(e)}
            continue
        counts = Counter(len(p.support_query(p.adaptive_k())[2]) for p in ps)
        out['e11'][split] = {'adaptive_problems': len(ps), 'query_count_hist': dict(counts),
                             'strict_complete_problems': len(io.load_problems(split, True, 7)),
                             'commonsense_codes': dict(Counter(p.common_sense for p in ps))}
    out['e11']['baseline_equivalence'] = 'contrast and knn_vote have identical signs by algebra'
    (OUT/'audit_initial.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    for qname, gname in [('cifar_fs','cifar100'), ('dtd','dtd'), ('miniimagenet','miniimagenet')]:
        q, g = r3bridge.load_cls_dataset(qname,'test'), r3bridge.load_cls_dataset(gname,'train')
        hq = [rgb_hash(q.get_image(i)) for i in range(len(q))]
        hg = [rgb_hash(g.get_image(i)) for i in range(len(g))]
        overlap = set(hq)&set(hg)
        entry = {'query_n':len(q), 'gallery_n':len(g), 'query_unique_rgb':len(set(hq)),
                 'gallery_unique_rgb':len(set(hg)), 'overlap_rgb_n':len(overlap),
                 'gallery_exclude_indices':[i for i,h in enumerate(hg) if h in overlap],
                 'shared_class_names':sorted(set(q.classnames)&set(g.classnames)),
                 'query_meta':q.meta, 'gallery_meta':g.meta}
        out['e12'][qname+':'+gname] = entry
        np.savez_compressed(OUT/f'identities_{qname}_{gname}.npz', query=np.array(hq), gallery=np.array(hg))
        print(qname, 'rgb_overlap', len(overlap), 'sizes', len(q),len(g), flush=True)
        (OUT/'audit_initial.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print('audit done', flush=True)

if __name__ == '__main__':
    main()
