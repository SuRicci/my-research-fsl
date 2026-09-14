"""Label-free geometry of all verified pooled descriptors; no classification."""
from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];ASSET=ROOT/'experiments/main/dino-local-20260913/assets'
def stats(a):return {'n':len(a),'mean':float(a.mean()),'q05_median_q95':np.quantile(a,[.05,.5,.95]).tolist()}
def main():
    manifest=json.loads((ASSET/'manifest.json').read_text());assert manifest['status']=='completed';out={'status':'completed','labels_used':False,'accuracy_evaluated':False,'domains':{}}
    for domain in ['dtd','eurosat']:
        path=ASSET/(domain+'_query.npy');h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==manifest['pools'][domain+'_query']['sha256'];x=np.load(path,mmap_mode='r');parts=[]
        for start in range(0,len(x),256):
            p=np.asarray(x[start:start+256],dtype=np.float64);p=p/np.linalg.norm(p,axis=-1,keepdims=True);mean=p.mean(1);energy=np.square(mean).sum(-1);g=p@p.transpose(0,2,1);off=(g.sum((1,2))-16)/240;rank=256/np.square(g).sum((1,2));res=np.square(p-mean[:,None]).sum(-1).mean(-1)
            assert np.max(abs(res-(1-energy)))<1e-12;assert np.max(abs(off-(16*energy-1)/15))<1e-12;assert ((rank>=1-1e-12)&(rank<=16+1e-12)).all();parts.append(np.stack([off,energy,res,rank],1))
        a=np.concatenate(parts);out['domains'][domain]={'source_sha256':h,'mean_pairwise_token_cosine':stats(a[:,0]),'image_mean_energy':stats(a[:,1]),'centered_residual_energy':stats(a[:,2]),'token_gram_participation_rank':stats(a[:,3])}
    out['boundary']='Image-local geometric redundancy is not semantic utility or a performance result. It does not prove residualization or reconstruction will improve classification.'
    (HERE/'RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
