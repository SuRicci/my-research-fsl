"""Complete inherited one-shot selection precision audit; old banks stay immutable."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, shutil, sys, time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; OUT=HERE/'outputs'
SRC=ROOT/'experiments/main/representation-scatter-20260913'
sys.path.insert(0,str(SRC));import evaluate as e
sys.path.insert(0,str(SRC));from verify import dense_metric, numpy_r2
CFG=json.loads((HERE/'protocol.json').read_text());TOL=CFG['score_atol']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def guard():
    assert shutil.disk_usage(HERE).free>CFG['min_free_gib']*2**30,'disk floor'
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline']),'deadline'
    assert sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<CFG['max_new_mib']*2**20,'output cap'
def inputs():
    paths=[HERE/'audit.py',HERE/'protocol.json',SRC/'outputs/output_audit_diagnostics.json',SRC/'evaluation_lock.json']
    paths += [SRC/'outputs'/f'{d}_k1_selection{ext}' for d in CFG['domains'] for ext in ['.json','.npz']]
    paths += [Path(p) for p in json.loads((SRC/'evaluation_lock.json').read_text())]
    return {str(p):sha(p) for p in paths}
def load():
    frozen=json.loads((SRC/'evaluation_lock.json').read_text())
    assert all(sha(p)==h for p,h in frozen.items())
    prior=json.loads((SRC/'outputs/output_audit_diagnostics.json').read_text())
    for d in CFG['domains']:
        p=SRC/'outputs'/f'{d}_k1_selection.npz';assert sha(p)==prior['input_sha256'][str(p)]
    return e.load()
def one(data,ds,bank,i,c,gamma):
    s=data[ds]['query'][bank['support_indices'][i]]
    q=data[ds]['query'][bank['query_indices'][i].reshape(-1)]
    g=data[ds]['gallery'][bank['gallery_indices'][i,c]]
    dense,_=dense_metric(s,q,g,gamma);ind=numpy_r2(*dense)
    sd,qd,gd=s.double(),q.double(),g.double()
    factor=e.metric.scatter_factor(sd,gamma)
    ss,qq,gg=[e.metric.transform(x.mean(-2),factor) for x in [sd,qd,gd]]
    proto=e.metric.norm(ss.mean(1));ix=torch.argsort(-(proto@gg.T),dim=-1,stable=True)[:,:64]
    x=e.metric.norm(.5*proto+.5*e.metric.norm(gg[ix].mean(1)))
    y=torch.eye(5,dtype=torch.float64);xc=x-x.mean(0);yc=y-y.mean(0)
    coef=torch.linalg.solve(xc@xc.T+.1*torch.eye(5,dtype=torch.float64),yc)
    double=((qq-x.mean(0))@xc.T@coef+y.mean(0)).numpy()
    return ind,double

def check():
    data=load();rows=[]
    for ds in CFG['domains']:
        z=np.load(SRC/'outputs'/f'{ds}_k1_selection.npz');a,b=one(data,ds,z,99,1,.1)
        error=float(np.max(abs(a-b)));assert error<TOL and np.array_equal(a.argmax(-1),b.argmax(-1))
        rows.append({'domain':ds,'episode':99,'condition':1,'gamma':.1,'double_independent_error':error,'historical_score_error':float(np.max(abs(a-z['scores'][99,1,0])))})
    dump(OUT/'precheck.json',{'status':'passed','rows':rows,'input_hashes':inputs(),'numpy':np.__version__,'torch':torch.__version__})
    print('PRECHECK',json.dumps(rows),flush=True)
def run():
    assert not (OUT/'manifest.json').exists(),'Adopt existing execution'
    checkrec=json.loads((OUT/'precheck.json').read_text());lock=inputs();assert lock==checkrec['input_hashes']
    data=load();started=time.time();dump(OUT/'manifest.json',{'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],'input_hashes':lock})
    rows=[];summary={};done=0
    for ds in CFG['domains']:
        z=np.load(SRC/'outputs'/f'{ds}_k1_selection.npz');tasks=e.sampler.tasks(data[ds]['ident'],1,'selection')
        for k,v in tasks.items():assert np.array_equal(z[k],v),(ds,k)
        assert z['scores'].shape==(100,2,3,75,5)
        scores=np.empty(z['scores'].shape,dtype=np.float64);y=np.repeat(np.arange(5),15)
        for i in range(CFG['episodes']):
            for c in range(CFG['gallery_conditions']):
                for j,gamma in enumerate(CFG['gamma']):
                    guard();a,b=one(data,ds,z,i,c,gamma);assert np.isfinite(a).all() and np.isfinite(b).all()
                    scores[i,c,j]=a;old=z['scores'][i,c,j]
                    newerr=float(np.max(abs(a-b)));olderr=float(np.max(abs(a-old)))
                    rows.append({'domain':ds,'episode':i,'condition':c,'gamma':gamma,'float64_score_error':newerr,'float64_prediction_changes':int(np.count_nonzero(a.argmax(-1)!=b.argmax(-1))),'historical_score_error':olderr,'historical_prediction_changes':int(np.count_nonzero(a.argmax(-1)!=old.argmax(-1)))})
                    done+=1
            if i%10==0:
                dump(OUT/'progress.json',{'completed':done,'total':CFG['comparisons'],'domain':ds,'episode':i,'elapsed_seconds':time.time()-started})
                print('PROGRESS',done,CFG['comparisons'],round(time.time()-started,1),flush=True)
        correct=(scores.argmax(-1)==y);oldcorrect=(z['scores'].argmax(-1)==y)
        counts=correct.sum((0,1,3));oldcounts=oldcorrect.sum((0,1,3))
        choose=lambda v:min(g for g,n in zip(CFG['gamma'],v) if n==v.max())
        saved=json.loads((SRC/'outputs'/f'{ds}_k1_selection.json').read_text());assert choose(oldcounts)==saved['gamma']
        np.savez_compressed(OUT/f'{ds}.npz',scores=scores,predictions=scores.argmax(-1),correct_counts=counts,yq=y,**{k:z[k] for k in ['support_indices','query_indices','gallery_indices','class_ids','seeds']})
        summary[ds]={'old_counts':oldcounts.tolist(),'float64_counts':counts.tolist(),'old_gamma':saved['gamma'],'float64_gamma':choose(counts),'winner_unchanged':choose(counts)==saved['gamma'],'total_prediction_changes':int(np.count_nonzero(scores.argmax(-1)!=z['scores'].argmax(-1)))}
        print('DOMAIN_COMPLETE',ds,json.dumps(summary[ds]),flush=True)
    assert done==CFG['comparisons'];assert inputs()==lock
    result={'status':'completed','comparison_count':done,'unique_source_episodes':200,'query_prediction_comparisons':done*75,'domains':summary,'all_winners_unchanged':all(v['winner_unchanged'] for v in summary.values()),'float64_validation_passed':all(v['float64_score_error']<TOL and v['float64_prediction_changes']==0 for v in rows),'maximum_float64_score_error':max(v['float64_score_error'] for v in rows),'historical_score_failures':sum(v['historical_score_error']>=TOL or v['historical_prediction_changes']!=0 for v in rows),'historical_prediction_changes':sum(v['historical_prediction_changes'] for v in rows),'rows':rows,'input_hashes':lock,'output_hashes':{str(OUT/f'{d}.npz'):sha(OUT/f'{d}.npz') for d in CFG['domains']},'elapsed_seconds':time.time()-started,'boundary':CFG['canonical_metric_deviation'],'old_frozen_score_audit_status':'failed; not superseded'}
    dump(OUT/'RESULT.json',result);print('COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ['rows','input_hashes','output_hashes']}),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','run'],required=True);args=ap.parse_args()
    torch.set_num_threads(CFG['threads']);guard();check() if args.phase=='check' else run()
