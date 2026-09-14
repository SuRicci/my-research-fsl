"""Feature-only worker for the new bounded study; legacy worker remains immutable."""
from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.nextstage import score_bundle

def main():
    inp,cfg,out,probe=[Path(s).resolve() for s in sys.argv[1:]]
    config=json.loads(cfg.read_text(encoding='utf-8'))
    with np.load(inp,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    events={'blocked_probe':False,'unexpected_reads':[]}
    roots=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),Path(__file__).parent]
    def audit(event,args):
        if event!='open' or not args or isinstance(args[0],int):return
        p=Path(args[0]).resolve()
        if p in [inp,out,out.with_suffix('.json')] or any(p==r or r in p.parents for r in roots):return
        events['unexpected_reads'].append(str(p));raise PermissionError(p.name)
    sys.addaudithook(audit)
    try:probe.read_bytes()
    except PermissionError:events['blocked_probe']=True
    if not events['blocked_probe']:raise RuntimeError('Data access isolation failed')
    start=time.perf_counter();scores,info=score_bundle(data,config)
    np.savez_compressed(out,**{k:np.argsort(-v,axis=1,kind='stable').astype(np.int32) for k,v in scores.items()})
    info.update(seconds=time.perf_counter()-start,access_audit=events)
    out.with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')

if __name__=='__main__':main()
