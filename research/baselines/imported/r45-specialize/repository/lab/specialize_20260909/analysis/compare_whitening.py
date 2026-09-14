"""Post-run comparator audit; no method fitting or selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from campaign_common import OUT,ORDER,read,dump,sha

def main():
    rows=[]
    for method in ORDER[:5]:
        groups={}
        for p in sorted((OUT/method).glob('*/*/complete.json')):
            m=read(p);z=np.load(p.parent/'predictions.npz',allow_pickle=False)
            base=OUT/'r4_whiten'/m['dataset']/p.parent.name/'predictions.npz';b=np.load(base,allow_pickle=False)
            np.testing.assert_array_equal(z['query_ids'],b['query_ids']);np.testing.assert_array_equal(z['conditions'],b['conditions'])
            delta=z['ap']-b['ap'];key=(m['dataset'],*m['pair'],m['budget'],m['policy'])
            groups.setdefault(key,[]).append((m['seed'],delta,list(z['conditions'])))
        assert len(groups)==36
        for key,packs in sorted(groups.items()):
            packs.sort();assert len(packs)==5;d=np.stack([x[1] for x in packs])
            for ci,c in enumerate(packs[0][2]):
                a=d[:,:,ci];rng=np.random.default_rng(902197)
                boot=[np.nanmean(a[rng.integers(5,size=5)][:,rng.integers(a.shape[1],size=a.shape[1])]) for _ in range(1000)]
                rows.append(dict(method=method,dataset=key[0],pair=list(key[1:3]),budget=key[3],policy=key[4],condition=str(c),
                    gain_pp=float(np.nanmean(a)*100),ci95_pp=(100*np.nanquantile(boot,[.025,.975])).tolist(),seed_gain_pp=(100*np.nanmean(a,1)).tolist()))
    dump(OUT/'comparison_whitening.json',dict(complete=True,rows=rows,source_summaries={m:sha(OUT/m/'summary.json') for m in ORDER[:5]},
        note='New strong geometry comparator from the same campaign; descriptive comparison, no retuning or new independent confirmation',analysis_source_sha256=sha(__file__)))
    print('WHITENING_COMPARATOR_COMPLETE',len(rows))

if __name__=='__main__':main()
