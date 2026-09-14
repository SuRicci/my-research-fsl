"""Independent algebra/null checks and saved prediction/statistic validation."""
from pathlib import Path
import argparse, importlib.util, json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('depth_study',HERE/'study.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
def check():
    torch.set_num_threads(s.CFG['threads']);torch.manual_seed(26091460)
    # Dense independent covariance inverse square root on a small matrix.
    v=torch.randn(5,1,6,11,dtype=torch.double);q=torch.randn(75,6,11,dtype=torch.double);g=torch.randn(80,11,dtype=torch.double)
    v=torch.nn.functional.normalize(v,dim=-1);q=torch.nn.functional.normalize(q,dim=-1);g=torch.nn.functional.normalize(g,dim=-1)
    d=(v-v.mean(-2,keepdim=True)).numpy().reshape(-1,11);cov=d.T@d/len(d);gamma=1.7
    val,vec=np.linalg.eigh(np.eye(11)+gamma*11*cov/np.trace(cov));t=(vec*val**-.5)@vec.T
    expected=s.z.m.n(torch.from_numpy(q.mean(-2).numpy()@t)).numpy()
    actual=s.z.m.old.metric.transform(q.mean(-2),s.z.m.old.metric.scatter_factor(v,gamma)).numpy()
    scatter_error=float(np.max(abs(expected-actual)));assert scatter_error<1e-10
    # Weighted primal/dual ridge consistency on small geometry.
    x=np.random.RandomState(1).normal(size=(30,11));qq=np.random.RandomState(2).normal(size=(75,11));w=np.full(30,1/6);y=np.eye(5).repeat(6,0)
    xm=np.average(x,axis=0,weights=w);ym=np.average(y,axis=0,weights=w);xc=x-xm;yc=y-ym
    primal=(qq-xm)@np.linalg.solve(xc.T@(w[:,None]*xc)+.1*np.eye(11),xc.T@(w[:,None]*yc))+ym
    ridge_error=float(np.max(abs(primal-s.independent_ridge(x,qq,w,.1))));assert ridge_error<1e-10
    data=s.z.m.old.load();checks=[]
    for target in s.CFG['domains']:
        source=next(d for d in s.CFG['domains'] if d!=target)
        for gal in s.CFG['domains']:
            ref=np.load(s.REF/f'{target}_{gal}.npz');i=0;sv=data[target]['query'][ref['support_indices'][i]].double();qv=data[target]['query'][ref['query_indices'][i].reshape(-1)].double();gm=data[gal]['gallery'].double().mean(-2)
            a=s.retained(sv,qv,gm,s.CFG['source_gamma'][source],s.CFG['source_eta'][source]);b=s.retained(s.duplicate(sv),s.duplicate(qv),s.duplicate(gm),s.CFG['source_gamma'][source],s.CFG['source_eta'][source])
            error=float((a['consistency']-b['consistency']).abs().max());assert error<1e-10
            expected=ref['scores'][ref['names'].tolist().index('consistency'),i];old_error=float(abs(a['consistency'][0].numpy()-expected).max());assert old_error<1e-10
            assert torch.equal(a['consistency'].argmax(-1),b['consistency'].argmax(-1))
            checks.append({'target':target,'gallery':gal,'duplicate_error':error,'old_error':old_error})
    result={'status':'passed','dense_scatter_error':scatter_error,'primal_ridge_error':ridge_error,'real_duplicate_cases':checks,'scope':'exact dimension-adjusted null on4fixedrealcases plus independent small-matrix formulas; full extraction parity and final result audit remain'}
    s.dump(s.OUT/'precheck.json',result);print('PRECHECK_PASSED',json.dumps(result),flush=True)
def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
def independent_features(data,maps,ds,side,ids):
    v=data[ds][side][ids].double().numpy();e=unit(np.array(maps[ds+'_'+side][ids],dtype=np.float64))
    e=e*np.linalg.norm(v[...,512:],axis=-1,keepdims=True)[...,None]
    return np.concatenate([v[...,:512],np.concatenate([e.reshape(*v.shape[:-1],-1),v[...,512:]],-1)/2],-1)
def independent_primary(sv,qv,gm,gamma,eta):
    d=(sv-sv.mean(-2,keepdims=True)).reshape(-1,sv.shape[-1]);trace=np.square(d).sum()/len(d)
    _,sigma,v=np.linalg.svd(d,full_matrices=False)
    coef=(1+gamma*896*np.square(sigma)/len(d)/trace)**-.5-1
    def trans(x):return unit(x+((x@v.T)*coef)@v)
    ss=trans(sv.mean(-2));qq=trans(qv.mean(-2));gg=trans(gm);vv=trans(qv);mu=ss.reshape(-1,ss.shape[-1]).mean(0);scores=[]
    for centered in [False,True]:
        sh,qh,gh,vh=[unit(x-mu) for x in [ss,qq,gg,vv]] if centered else [ss,qq,gg,vv]
        proto=unit(sh.mean(1));ix=np.argsort(-(proto@gh.T),axis=-1,kind='stable')[:,:64]
        x=unit(.5*proto+.5*unit(gh[ix].mean(1)));xm=x.mean(0);a=x-xm;b=np.eye(5)-.2
        f=np.linalg.solve(a@a.T+.1*np.eye(5),np.eye(5));w=a.T@f@b;qc=qh-xm
        r=(vh-vh.mean(1,keepdims=True))/np.sqrt(6);rt=r.swapaxes(-1,-2)
        u=(rt-a.T@f@(a@rt))/.1
        correction=eta*(qc[:,None,:]@u)@np.linalg.solve(np.eye(6)+eta*(r@u),r@w)
        scores.append(qc@w+.2-correction[:,0])
    return np.mean(scores,0)
def independent_interval(delta,groups):
    rng=np.random.RandomState(s.CFG['bootstrap']['seed']);sample=np.zeros(s.CFG['bootstrap']['replicates'])
    for seed in np.unique(groups):
        values=delta[groups==seed];indices=rng.randint(0,len(values),size=(len(sample),len(values)))
        sample+=values[indices].sum(axis=1)/len(delta)
    return np.array([delta.mean()*100,*np.percentile(sample*100,[2.5,97.5])])
def full():
    torch.set_num_threads(s.CFG['threads']);data,maps=s.load();summary=json.loads((s.OUT/'analysis.json').read_text());count=0;stats=0;errors=[];domain_arrays=[];domain_seeds=[];cells={}
    for target in s.CFG['domains']:
        source=next(d for d in s.CFG['domains'] if d!=target);arrays=[]
        for gal in s.CFG['domains']:
            f=np.load(s.OUT/f'{target}_{gal}.npz');ref=np.load(s.REF/f'{target}_{gal}.npz')
            assert f['predictions'].shape==(len(s.NAMES),500,75) and f['names'].tolist()==s.NAMES
            for key in ['support_indices','query_indices','class_ids','seeds']:assert np.array_equal(f[key],ref[key])
            labels=np.asarray(data[target]['ident']['query_labels'])
            assert np.array_equal(labels[f['support_indices']],f['class_ids'][:,:,None])
            assert np.array_equal(labels[f['query_indices']],np.repeat(f['class_ids'][:,:,None],15,2))
            assert np.array_equal(f['audit_scores'].argmax(-1),f['predictions'][:,s.CFG['audit_indices']])
            assert np.array_equal(f['predictions'][1],ref['predictions'][ref['names'].tolist().index('consistency')])
            assert np.array_equal(f['predictions'][2],ref['predictions'][ref['names'].tolist().index('incumbent')])
            acc=(f['predictions']==s.Y).mean(-1);count+=f['predictions'].size;arrays.append(acc);row=summary['cells'][target+'_'+gal]
            assert np.max(abs(np.array([row['accuracy_pct'][n] for n in s.NAMES])-100*acc.mean(-1)))<1e-10
            actual=independent_interval(acc[0]-acc[1],f['seeds']);expected=np.array([row['vs_final']['delta_pp'],*row['vs_final']['ci95_pp']]);assert np.max(abs(actual-expected))<1e-10;stats+=1
            cells[target+'_'+gal]=actual
            i=99;sv=independent_features(data,maps,target,'query',f['support_indices'][i]);qv=independent_features(data,maps,target,'query',f['query_indices'][i].reshape(-1))
            gm=np.concatenate([independent_features(data,maps,gal,'gallery',slice(j,j+128)).mean(-2) for j in range(0,len(data[gal]['gallery']),128)])
            sc=independent_primary(sv,qv,gm,s.CFG['source_gamma'][source],s.CFG['source_eta'][source])
            error=float(np.max(abs(sc-f['audit_scores'][0,s.CFG['audit_indices'].index(i)])));assert error<1e-9
            assert np.array_equal(sc.argmax(-1),f['predictions'][0,i]);errors.append({'target':target,'gallery':gal,'task':i,'max_error':error})
        acc=np.mean(arrays,0);domain_arrays.append(acc);domain_seeds.append(f['seeds']+len(domain_seeds)*100000000)
        for name,row in summary['domains'][target]['comparisons'].items():
            actual=independent_interval(acc[0]-acc[s.NAMES.index(name)],f['seeds'])
            assert np.max(abs(actual-np.array([row['delta_pp'],*row['ci95_pp']])))<1e-10;stats+=1
        assert all(abs(summary['domains'][target]['accuracy_pct'][name]-100*acc[j].mean())<1e-10 for j,name in enumerate(s.NAMES))
    allacc=np.concatenate(domain_arrays,1);groups=np.concatenate(domain_seeds);passed=True
    for name,row in summary['pooled'].items():
        actual=independent_interval(allacc[0]-allacc[s.NAMES.index(name)],groups)
        assert np.max(abs(actual-np.array([row['delta_pp'],*row['ci95_pp']])))<1e-10;stats+=1
        passed &= actual[0]>=.5 and actual[1]>0
    passed &= all(d['comparisons'][n]['delta_pp']>=0 for d in summary['domains'].values() for n in s.CFG['gate_controls'])
    passed &= all(row[1]>=-.5 for row in cells.values())
    assert bool(passed)==summary['gate_passed']
    assert s.immutable()==json.loads((s.OUT/'code_lock.json').read_text())
    result={'status':'passed','prediction_arithmetic_checks':count,'independent_intervals':stats,'independent_primary_cases':errors,'gate_passed':bool(passed),'scope':'all saved prediction/statistic arithmetic and task labels; independent NumPy extraction, covariance, retrieval, ridge and consistency on four prespecified task cases. Logistic solver not independently reimplemented.'}
    s.dump(s.OUT/'audit.json',result);print('AUDIT_PASSED',json.dumps(result),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['check','full'],required=True);a=ap.parse_args()
    check() if a.phase=='check' else full()
