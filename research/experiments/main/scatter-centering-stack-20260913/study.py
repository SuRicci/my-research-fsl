"""One fixed cumulative-composition development run, no outcome-selected settings."""
from pathlib import Path
import json,sys,time,hashlib,shutil,datetime
import numpy as np
import torch
import model as m
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'
CFG=json.loads((HERE/'protocol.json').read_text())

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
    assert shutil.disk_usage(HERE).free/2**30>=CFG['resources']['free_gib_min']
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['resources']['deadline'])
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def main():
    started=time.time();torch.set_num_threads(CFG['resources']['threads']);guard()
    assert json.loads((OUT/'numeric_validation.json').read_text())['status']=='passed'
    locks=json.loads((HERE/'code_lock.json').read_text());assert all(sha(k)==v for k,v in locks.items())
    data=m.old.load();gm={d:data[d]['gallery'].double().mean(-2) for d in CFG['domains']}
    dump(OUT/'manifest.json',{'argv':[sys.executable,*sys.argv],'config':CFG,'code_hashes':locks,'torch':torch.__version__,'numpy':np.__version__,'threads':torch.get_num_threads(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for shot in CFG['shots']:
            t=m.old.sampler.tasks(data[target]['ident'],shot,'eval');gamma=CFG['gamma_by_source'][source][str(shot)]
            assert gamma==json.loads((m.old.OUT/f'{source}_k{shot}_selection.json').read_text())['gamma']
            galleries=[target,source];preds={g:[] for g in galleries};sample_scores={g:[] for g in galleries};sample_neighbors={g:[] for g in galleries}
            oldpaths={g:m.old.OUT/'cells'/f'{source}_to_{target}_{g}_k{shot}.npz' for g in galleries}
            for g,f in oldpaths.items():
                with np.load(f) as z:
                    for key in t:assert np.array_equal(t[key],z[key]),(f,key)
                assert not set(data[target]['ident']['query_rgb'])&set(data[g]['ident']['gallery_rgb'])
            for i in range(len(t['seeds'])):
                guard();s=data[target]['query'][t['support_indices'][i]];q=data[target]['query'][t['query_indices'][i].reshape(-1)]
                prep=m.prepare(s,q,gamma);cached=None
                for g in galleries:
                    if shot==5 and cached is not None:sc,ix=cached
                    else:sc,ix=m.packages(prep,gm[g]);cached=(sc,ix)
                    assert torch.isfinite(sc).all()
                    preds[g].append(sc.argmax(-1).numpy().astype(np.uint8))
                    if i in CFG['audit_task_indices']:
                        sample_scores[g].append(sc.numpy());sample_neighbors[g].append(ix.numpy().astype(np.int32))
                if i%100==0:
                    progress={'target':target,'shot':shot,'tasks_done':i+1,'elapsed_seconds':time.time()-started}
                    dump(OUT/'progress.json',progress);print('PROGRESS',json.dumps(progress),flush=True)
            y=np.repeat(np.arange(5),15)
            for g in galleries:
                pred=np.stack(preds[g],axis=1);name=f'{source}_to_{target}_{g}_k{shot}'
                np.savez_compressed(OUT/(name+'.npz'),predictions=pred,accuracy=(pred==y).mean(-1),names=m.NAMES,yq=y,chosen_gamma=gamma,sample_scores=np.stack(sample_scores[g],1),sample_neighbors=np.stack(sample_neighbors[g],1),sample_indices=CFG['audit_task_indices'],**t)
                print('CELL_COMPLETE',name,dict(zip(m.NAMES,((pred==y).mean((1,2))*100).tolist())),flush=True)
    dump(OUT/'complete.json',{'status':'completed','task_conditions':4000,'methods':len(m.NAMES),'elapsed_seconds':time.time()-started,'code_hashes':locks,'output_bytes':sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())})
    print('COMPLETE',time.time()-started,flush=True)
if __name__=='__main__':main()
