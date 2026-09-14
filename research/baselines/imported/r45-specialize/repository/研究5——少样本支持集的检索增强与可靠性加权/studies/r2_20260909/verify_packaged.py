# -*- coding: utf-8 -*-
"""Check the usable single-episode API against frozen batched experiment scores."""
import json,time
import numpy as np
import torch
from assets_io import HERE,Assets,dump
from e12.optimized import fit_optimized,prepare_gallery

def main():
    store=Assets();selection=json.loads((HERE/'selection.json').read_text());rows=[]
    for path in sorted((HERE/'outputs').glob('*.npz')):
        if '_dev.' in path.name:continue
        d=np.load(path);meta=json.loads(path.with_suffix('.json').read_text());shot=meta['shot']
        q,g=store.pair(*meta['cell']);lut={int(v):i for i,v in enumerate(q['ids'])};name=selection[str(shot)]['overall']['name'];j=d['names'].tolist().index(name)
        tp=time.perf_counter();prepared=prepare_gallery(*g['features']) if shot==1 else None
        prepare_ms=1000*(time.perf_counter()-tp)
        for episode in [0,200,400]:
            si=[lut[int(i)] for i in d['support_indices'][episode].ravel()];qi=[lut[int(i)] for i in d['query_indices'][episode].ravel()]
            S=[f[si] for f in q['features']];X=[f[qi] for f in q['features']];labels=torch.arange(5).repeat_interleave(shot)
            t=time.perf_counter();model=fit_optimized(*S,labels,*g['features'])
            fit_ms=1000*(time.perf_counter()-t);t=time.perf_counter();scores=model.scores(*X);pred=model.predict(*X);pred_ms=1000*(time.perf_counter()-t)
            error=float(np.max(np.abs(scores.numpy()-d['scores'][j,episode])))
            assert error<2e-5,(path.name,error)
            assert np.array_equal(pred.numpy(),d['predictions'][j,episode]),path.name
            t=time.perf_counter();fast=fit_optimized(*S,labels,prepared_gallery=prepared)
            prepared_fit_ms=1000*(time.perf_counter()-t)
            assert torch.equal(fast.predict(*X),pred)
            assert torch.allclose(scores,torch.cat([model.scores(X[0][:11],X[1][:11]),model.scores(X[0][11:],X[1][11:])]),atol=1e-6)
            if shot==5:
                empty=fit_optimized(*S,labels,None,None)
                assert torch.equal(empty.predict(*X),pred) and empty.gallery_rows_read==0
            rows.append({'cell':meta['cell'],'shot':shot,'phase':meta['phase'],'episode':episode,'score_max_error':error,'fit_ms':fit_ms,'gallery_prepare_ms':prepare_ms,'prepared_gallery_fit_ms':prepared_fit_ms,'75_query_scores_and_predict_ms':pred_ms})
    report={'status':'passed','episodes_compared':len(rows),'queries_compared':len(rows)*75,'rows':rows,'scope':'CPU 4 Torch threads; input features already cached; scoring called twice for API comparison, not single-query end-to-end latency.'}
    dump(HERE/'packaged_api_verification.json',report);print('Packaged API verified',len(rows),'episodes',len(rows)*75,'queries',flush=True)

if __name__=='__main__':main()
