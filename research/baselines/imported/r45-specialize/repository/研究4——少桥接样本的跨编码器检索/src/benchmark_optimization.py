"""Cached-feature CPU scoring cost, including bridge kernel setup per call."""
import time,json
import numpy as np
from research_common.records import write_json
from .data import PROJECT
from .improved import affinity_residual,kernel_residual_scores
from .methods import unit,local_weights

def main():
    p=PROJECT/'runs/optimization_holdout_deduplicated_v1/jobs/clip_b16__dino_s14__same_domain__m64__s50851/input.npz'
    with np.load(p,allow_pickle=False) as z:go,qo,qn,ao,an=[unit(z[k]) for k in ['go','qo','qn','ao','an']]
    def local(std,k):
        w,_=local_weights(go,ao,k,.05)
        return qo@go.T+affinity_residual(qo,qn,ao,an,std)@w.T
    functions={'candidate':lambda:kernel_residual_scores(go,qo,qn,ao,an,'rbf_0.5',True)[0],
        'tuned_local':lambda:local(True,16),'original_local':lambda:local(False,8)}
    results={}
    for name,f in functions.items():
        f();times=[]
        for _ in range(20):
            start=time.perf_counter();scores=f();times.append(time.perf_counter()-start)
        results[name]={'median_ms':1000*float(np.median(times)),'min_ms':1000*min(times),'max_ms':1000*max(times),'scores_shape':list(scores.shape)}
    result={'backend':'NumPy CPU; OPENBLAS_NUM_THREADS=4','queries':len(qo),'gallery':len(go),'bridges':len(ao),
        'includes_bridge_kernel_setup':True,'image_encoding_included':False,'timings':results,'repetitions':20,
        'model_forwards_per_live_query':2,'method_new_gallery_forwards':0}
    write_json(PROJECT/'reports/evidence/optimization_timing.json',result,overwrite=True)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
