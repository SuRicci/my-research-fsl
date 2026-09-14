"""Synchronized local encoder timing; decoded image crops are preloaded."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import rv_study as m

def main():
    torch.set_num_threads(6);m.guard()
    enc=m.prior.metric  # Loader is reached through the already verified extraction entrypoint.
    extractor=sys.modules['extract_views'];enc=extractor.enc
    assert torch.backends.mps.is_available()
    rows=json.loads((m.PARENT/'assets/identities.json').read_text())
    paths=rows['dtd_query']['paths'][:16]+rows['eurosat_query']['paths'][:16]
    images=[extractor.views(p) for p in paths]
    cases=[[0],[0,1],[0,1,2],list(range(6))]
    choice=json.loads((m.OUT/'selection_lock.json').read_text())
    for v in choice.values():
        if v['subset'] not in cases:cases.append(v['subset'])
    records={};output={}
    for bi,backbone in enumerate(enc.BACKBONES):
        model=enc.model(backbone)
        for subset in cases:extractor.batch(model,backbone,bi,images,subset)
        torch.mps.synchronize()
        records[backbone]={str(v):[] for v in cases}
        rng=np.random.RandomState(26091392)
        for rep in range(5):
            for j in rng.permutation(len(cases)):
                m.guard();subset=cases[j]
                torch.mps.synchronize();start=time.perf_counter()
                feats=extractor.batch(model,backbone,bi,images,subset)
                torch.mps.synchronize();elapsed=time.perf_counter()-start
                assert feats.shape[:2]==(32,len(subset)) and torch.isfinite(feats).all()
                records[backbone][str(subset)].append(elapsed)
            print('ENCODER_TIMING',backbone,'repetition',rep+1,flush=True)
            m.dump(m.OUT/'encoding_timing_progress.json',records)
        output[backbone]={k:dict(seconds=v,median_seconds=float(np.median(v))) for k,v in records[backbone].items()}
        del model;torch.mps.empty_cache()
    full=str(list(range(6)));summaries={}
    for subset in cases:
        key=str(subset);paired=np.array([sum(records[b][key][i] for b in enc.BACKBONES) for i in range(5)])
        base=np.array([sum(records[b][full][i] for b in enc.BACKBONES) for i in range(5)])
        summaries[key]={'combined_median_seconds':float(np.median(paired)),
            'median_combined_speedup':float(np.median(base)/np.median(paired)),
            'per_repetition_speedup':(base/paired).tolist(),'encoder_application_ratio':len(subset)/6}
    result={'status':'measured','backbones':output,'combined':summaries,'images':paths,'batch_size':32,
        'warmups':1,'repetitions':5,'boundary':'Two encoders run sequentially on MPS, view-batched at32images. Decoding/cropping/model loading excluded; tensor normalization, transfers and forward included. Fixed preloaded source batch, not application end-to-end or universal throughput.'}
    m.dump(m.OUT/'encoding_timing.json',result);print('ENCODING_BENCHMARK_COMPLETE',summaries,flush=True)

if __name__=='__main__':main()
