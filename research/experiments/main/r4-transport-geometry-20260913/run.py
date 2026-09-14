"""Paired test of residual interpolation geometry; fixed canonical image permissions."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='6'
from pathlib import Path
import argparse,sys,json,time,hashlib,shutil
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[2]/'baselines/local/r4-canonical'
sys.path.insert(0,str(BASE))
import evaluate as baseline
from source.methods import unit
from source.improved import affinity_residual,kernel_matrices,kernel_residual_scores
P=json.loads((ROOT/'protocol.json').read_text())
OUT=ROOT/'outputs'
ALPHAS=P['alphas']
NAMES=[f'{g}_a{a:g}' for g in ['raw','center'] for a in ALPHAS]+['raw_fixed001_a1','center_fixed001_a1']

def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Transport:
    def __init__(self,go,ao,an):
        self.go,self.ao,self.an=map(unit,(go,ao,an))
        self.mu=self.go.mean(0);self.gc=unit(self.go-self.mu)
        self.fits={}
        for geom,g,a in [('raw',self.go,self.ao),('center',self.gc,unit(self.ao-self.mu))]:
            k,c=kernel_matrices(g,a,'rbf_0.5')
            e,v=np.linalg.eigh((k+k.T)/2);e=np.maximum(e,0)
            scale=max(float(np.trace(k))/len(k),1e-8)
            self.fits[geom]=(e,v,scale,c@v)
    def predict(self,qo,qn):
        qo,qn=map(unit,(qo,qn));raw=qo@self.go.T;bc=unit(qo-self.mu)@self.gc.T
        delta=affinity_residual(qo,qn,self.ao,self.an,True)
        rho=bc.std(1,keepdims=True)/np.maximum(raw.std(1,keepdims=True),1e-10)
        scores={};diagnostics={}
        for geom,(e,v,scale,cv) in self.fits.items():
            proj=delta@v;errors=[]
            for coefficient in P['press_grid']:
                factors=1/(e+coefficient*scale)
                fitted=(proj*(e*factors)[None,:])@v.T
                diagonal=(v*v)@(e*factors)+1/len(e)
                press=(delta-fitted)/np.maximum(1-diagonal[None,:],1e-8)
                errors.append(np.mean(press**2,axis=1))
            choices=np.argmin(np.stack(errors,1),axis=1)
            coef=np.asarray(P['press_grid'])[choices]
            correction=(proj/(e[None,:]+scale*coef[:,None]))@cv.T
            correction-=correction.mean(1,keepdims=True)
            for alpha in ALPHAS:scores[f'{geom}_a{alpha:g}']=bc+alpha*rho*correction
            fixed=(proj/(e[None,:]+scale*.001))@cv.T
            fixed-=fixed.mean(1,keepdims=True)
            scores[geom+'_fixed001_a1']=bc+rho*fixed
            diagnostics[geom]={'coefficients':coef,'press_mse':np.stack(errors,1)[np.arange(len(qo)),choices]}
        return scores,diagnostics

def read_job(path,data):
    meta=json.loads(path.with_suffix('.json').read_text())
    assert sha(path)==meta['npz_sha256']
    with np.load(path,allow_pickle=False) as z:
        keys=['methods','ap','query_indices','query_ids','query_labels','gallery_ids','gallery_labels','bridge_indices','bridge_ids','bridge_domain','seed','budget']
        job={k:z[k] for k in keys}
    ds=meta['dataset'];d=data[ds];bd=str(job['bridge_domain']);qi=job['query_indices'];ai=job['bridge_indices']
    assert np.array_equal(np.asarray(d['identity']['query_ids'])[qi],job['query_ids'])
    assert np.array_equal(np.asarray(data[bd]['identity']['query_ids'])[ai],job['bridge_ids'])
    assert np.array_equal(d['identity']['gallery_ids'],job['gallery_ids'])
    assert not set(job['query_ids'])&set(job['bridge_ids']) if ds==bd else True
    old,new=P['baseline_protocol']['pair']
    inputs=[d[old+'_gallery'],d[old+'_query'][qi],d[new+'_query'][qi],data[bd][old+'_query'][ai],data[bd][new+'_query'][ai]]
    return meta,job,inputs

def validate(data,hashes):
    p=sorted((BASE/'outputs/dev').glob('*.npz'))[0];meta,job,(go,qo,qn,ao,an)=read_job(p,data)
    qo,qn=qo[:3],qn[:3];model=Transport(go,ao,an);s,_=model.predict(qo,qn)
    legacy=baseline.scores(go,qo,qn,ao,an,baseline.geometry_scores(go,qo))
    err=max(float(np.max(abs(s['raw_a2']-legacy['centered_residual_2']))),float(np.max(abs(s['raw_a0']-legacy['old_centered']))))
    assert err<1e-8,err
    one=max(float(np.max(abs(model.predict(qo[i:i+1],qn[i:i+1])[0][n][0]-s[n][i]))) for n in NAMES for i in range(3))
    assert one<1e-9,one
    same=Transport(go,ao,ao).predict(qo,qo)[0]
    constant=Transport(go,ao,np.repeat(an[:1],len(an),axis=0)).predict(qo,qn)[0]
    null=max(float(np.max(abs(z[n]-s['raw_a0']))) for z in [same,constant] for n in NAMES)
    assert null<1e-9,null
    # Independent direct solve at fixed lambda validates centered-kernel spectral factorization.
    gc=unit(unit(go)-model.mu);ac=unit(unit(ao)-model.mu);k,c=kernel_matrices(gc,ac,'rbf_0.5')
    delta=affinity_residual(unit(qo),unit(qn),unit(ao),unit(an),True)
    scale=max(float(np.trace(k))/len(k),1e-8)
    direct=delta@np.linalg.solve(k+.001*scale*np.eye(len(k)),c.T)
    direct-=direct.mean(1,keepdims=True)
    rho=s['raw_a0'].std(1,keepdims=True)/(unit(qo)@unit(go).T).std(1,keepdims=True)
    factor=float(np.max(abs(s['center_fixed001_a1']-(s['raw_a0']+rho*direct))))
    assert factor<1e-8,factor
    result={'passed':True,'legacy_score_max_error':err,'query_independence_max_error':one,'identity_and_constant_residual_max_error':null,'direct_solve_max_error':factor,'numpy':np.__version__,'torch':torch.__version__,'threads':torch.get_num_threads(),'source_hashes':{str(BASE/'evaluate.py'):sha(BASE/'evaluate.py'),str(BASE/'source/improved.py'):sha(BASE/'source/improved.py')},'identity_hashes':hashes,'code_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'protocol.json'),'input_schema':['go','qo','qn','ao','an'],'gallery_new_to_method':False,'source_job':str(p)}
    dump(ROOT/'validation.json',result);print(json.dumps(result),flush=True)

def run(data,hashes):
    v=json.loads((ROOT/'validation.json').read_text())
    assert v['passed'] and v['code_sha256']==sha(__file__) and v['protocol_sha256']==sha(ROOT/'protocol.json')
    assert v['identity_hashes']==hashes
    assert not (OUT/'completion.json').exists(),'Completed run must not be repeated'
    assert not list(OUT.glob('*/*.npz')),'Do not overwrite partial outputs; recover first'
    start=time.time();count=0;dev={n:[] for n in NAMES}
    dump(OUT/'manifest.json',{'protocol':P,'validation':v,'command':'/opt/anaconda3/envs/torch/bin/python run.py','started_at':start,'new_encoding_calls':0,'download_bytes':0})
    for phase in ['dev','eval']:
        if phase=='eval':assert (OUT/'selection.json').exists()
        paths=sorted((BASE/'outputs'/phase).glob('*.npz'));assert len(paths)==(20 if phase=='dev' else 40)
        for path in paths:
            assert shutil.disk_usage(ROOT).free>=10*2**30
            meta,job,(go,qo,qn,ao,an)=read_job(path,data)
            scores,diag=Transport(go,ao,an).predict(qo,qn)
            rank=np.stack([np.argsort(-scores[n],axis=1,kind='stable').astype(np.uint16) for n in NAMES])
            ap=np.stack([baseline.ap_from_ranks(r,job['query_labels'],job['gallery_labels']) for r in rank])
            assert np.isfinite(ap).all()
            for n,old in [('raw_a0','old_centered'),('raw_a2','centered_residual_2')]:
                err=np.max(abs(ap[NAMES.index(n)]-job['ap'][list(job['methods']).index(old)]));assert err<1e-10,(path,n,err)
            dest=OUT/phase/path.name;dest.parent.mkdir(exist_ok=True,parents=True)
            np.savez_compressed(dest,methods=NAMES,ranks=rank,ap=ap,**{k:z for k,z in job.items() if k not in ['methods','ap']},raw_coefficients=diag['raw']['coefficients'],center_coefficients=diag['center']['coefficients'])
            dump(dest.with_suffix('.json'),{**meta,'npz_sha256':sha(dest),'baseline_npz':str(path),'baseline_npz_sha256':sha(path)})
            if phase=='dev':
                for n,a in zip(NAMES,ap):dev[n].append(float(a.mean()*100))
            count+=1;progress={'completed_jobs':count,'phase':phase,'latest':path.name,'elapsed_seconds':time.time()-start}
            dump(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
        if phase=='dev':
            means={n:float(np.mean(x)) for n,x in dev.items()};selected={g:min(ALPHAS,key=lambda a:(-means[f'{g}_a{a:g}'],a)) for g in ['raw','center']}
            selection={'alphas':selected,'development_map':means,'selection':'one global alpha per geometry; DTD four cells/five seeds; smaller-alpha exact tie break','frozen_before_evaluation':True,'written_at':time.time()}
            dump(OUT/'selection.json',selection);print('SELECTION',json.dumps(selection),flush=True)
    assert count==60
    dump(OUT/'completion.json',{'complete':True,'jobs':count,'elapsed_seconds':time.time()-start})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(6);data,hashes=baseline.load_data()
    if args.validate_only:validate(data,hashes)
    else:run(data,hashes)
