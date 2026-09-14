"""Separate feature-only inference process; Python file reads are audited."""
from pathlib import Path
import sys,json,time
import numpy as np
from methods import score_all

def main():
    input_path,config_path,output_path,probe_path=map(lambda s:Path(s).resolve(),sys.argv[1:])
    config=json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('worker_family')=='optimization_menu':
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from src.improved import score_menu
        scorer=score_menu
    else:
        scorer=score_all
    with np.load(input_path,allow_pickle=False) as z:
        inputs={k:z[k] for k in z.files}
    events={'blocked_probe':False,'allowed_data_reads':[str(input_path)],'unexpected_data_reads':[]}
    code_root=Path(__file__).resolve().parent
    env_roots=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
    def audit(event,args):
        if event!='open' or not args or isinstance(args[0],int):return
        p=Path(args[0]).resolve()
        if p in (input_path,output_path,output_path.with_suffix('.json')):return
        if code_root in p.parents or any(r==p or r in p.parents for r in env_roots):return
        events['unexpected_data_reads'].append(str(p))
        raise PermissionError(f'Feature-only worker denied filesystem access: {p.name}')
    sys.addaudithook(audit)
    try:
        with open(probe_path,'rb') as f:f.read(1)
    except PermissionError:
        events['blocked_probe']=True
    if not events['blocked_probe']:raise RuntimeError('Read isolation self-check failed')
    start=time.perf_counter()
    scores,diagnostics=scorer(inputs,config)
    rankings={k:np.argsort(-v,axis=1,kind='stable').astype(np.int32) for k,v in scores.items()}
    np.savez_compressed(output_path,**rankings)
    diagnostics.update(worker_seconds=time.perf_counter()-start,access_audit=events,
                       method_keys=sorted(inputs),gallery_image_reads=0,gallery_new_embedding_reads=0)
    output_path.with_suffix('.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')

if __name__=='__main__':main()
