"""Fixed source-trained query competence; immutable expert heads and tasks."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, sys, time, warnings, shutil
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import sklearn

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; OUT=HERE/'outputs'; OUT.mkdir(exist_ok=True)
CFG=json.loads((HERE/'protocol.json').read_text())
spec=importlib.util.spec_from_file_location('frozen_fusion',ROOT/'experiments/main/scatter-score-fusion-20260913/study.py')
fusion=importlib.util.module_from_spec(spec);spec.loader.exec_module(fusion)
a=fusion.a; m=fusion.m
NAMES=['stack','raw','blend','margin','learned']
ORDER=np.array([2,0,1])
Y=np.repeat(np.arange(5),15)

def dump(name,value): (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
    assert shutil.disk_usage(HERE).free/2**30>=10
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline'])

def features(sc):
    # sc: expert, task, query, class. No labels or other query rows are inputs.
    ordered=np.sort(sc,axis=-1)
    profiles=np.moveaxis(ordered,0,-2).reshape(-1,15)
    pred=sc.argmax(-1)
    agreement=np.stack([pred[0]==pred[1],pred[0]==pred[2],pred[1]==pred[2]],axis=-1).reshape(-1,3)
    return np.concatenate([profiles,agreement],axis=1)

def classify(sc,model):
    x=features(sc);z=(x-np.array(model['mean']))/np.array(model['scale'])
    logits=z@np.array(model['coef']).T+np.array(model['intercept'])
    # Sigmoid is monotone, so logit argmax exactly preserves competence selection.
    chosen=ORDER[np.argmax(logits[:,ORDER],axis=1)]
    ordered=np.sort(sc,axis=-1);margin=(ordered[...,-1]-ordered[...,-2]).reshape(3,-1).T
    simple=ORDER[np.argmax(margin[:,ORDER],axis=1)]
    pred=sc.argmax(-1);flat=pred.reshape(3,-1)
    selected=flat[chosen,np.arange(len(chosen))].reshape(pred.shape[1:])
    margin_pred=flat[simple,np.arange(len(simple))].reshape(pred.shape[1:])
    return np.concatenate([pred,margin_pred[None],selected[None]],0),chosen.reshape(pred.shape[1:]),logits

def fit(sc):
    x=features(sc);scaler=StandardScaler().fit(x);z=scaler.transform(x)
    correct=(sc.argmax(-1)==Y).reshape(3,-1)
    coef=[];intercept=[];iterations=[]
    for j in range(3):
        assert len(np.unique(correct[j]))==2
        clf=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',max_iter=1000,tol=1e-8,random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            clf.fit(z,correct[j])
        assert clf.classes_.tolist()==[False,True]
        coef.append(clf.coef_[0].tolist());intercept.append(float(clf.intercept_[0]));iterations.append(int(clf.n_iter_[0]))
        manual=1/(1+np.exp(-(z@clf.coef_[0]+clf.intercept_[0])))
        assert np.max(abs(manual-clf.predict_proba(z)[:,1]))<1e-12
    return {'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coef':coef,'intercept':intercept,'iterations':iterations,'train_rows':len(x),'training_correct_fraction':correct.mean(1).tolist()}

def save_bank(name,sc,t,model,gallery_indices=None):
    pred,choices,logits=classify(sc,model)
    assert np.isfinite(sc).all() and np.isfinite(logits).all()
    # Class and query permutation invariance of the selection interface.
    permutation=np.array([2,4,0,1,3])
    assert np.array_equal(features(sc[...,permutation]),features(sc))
    pp,cc,_=classify(sc[:,:,::-1],model)
    assert np.array_equal(pp[:,:,::-1],pred) and np.array_equal(cc[:,::-1],choices)
    np.savez_compressed(OUT/(name+'.npz'),scores=sc,predictions=pred,accuracy=(pred==Y).mean(-1),names=NAMES,yq=Y,choices=choices,
                        gallery_indices=np.array([]) if gallery_indices is None else gallery_indices,**t)
    return pred

@torch.no_grad()
def source_scores(domain,phase,data):
    ident=data[domain]['ident']; t=m.old.sampler.tasks(ident,1,phase)
    source=data[domain];gm=source['gallery'].double().mean(-2);gy=np.array(ident['gallery_labels'])
    banks=[[],[]];indices=[[],[]];gamma=CFG['gamma_by_source'][domain]
    for i in range(len(t['seeds'])):
        guard();sv=source['query'][t['support_indices'][i]];qv=source['query'][t['query_indices'][i].reshape(-1)]
        _,(s,q),factor=m.prepare(sv,qv,gamma)
        for j,excluded in enumerate([False,True]):
            rng=np.random.RandomState(CFG['source_gallery_seed']+10000*j+i)
            pool=np.flatnonzero(~np.isin(gy,t['class_ids'][i])) if excluded else np.arange(len(gy))
            gi=rng.choice(pool,1024,replace=False)
            if excluded: assert not np.isin(gy[gi],t['class_ids'][i]).any()
            scores,_=a.heads(s,q,m.old.metric.transform(gm[gi],factor))
            sc=torch.stack([scores[3],scores[0],.5*(scores[3]+scores[0])]).numpy()
            banks[j].append(sc);indices[j].append(gi)
            if phase=='train' and i==0 and j==0:
                dense,_=a.independent(sv,qv,gm[gi],gamma)
                expected=np.stack([dense[3],dense[0],.5*(dense[3]+dense[0])])
                err=float(abs(sc-expected).max())
                assert err<2e-5 and np.array_equal(sc.argmax(-1),expected.argmax(-1))
                print('SOURCE_DENSE_AUDIT',domain,err,flush=True)
        if i%50==0: print('SOURCE',domain,phase,i,flush=True)
    return [np.stack(x,axis=1) for x in banks],t,[np.stack(x) for x in indices]

def summarize():
    cells={};domains={};vectors={}
    for domain in CFG['domains']:
        accs=[]
        for gallery in CFG['domains']:
            name=domain+'_'+gallery;z=np.load(OUT/(name+'.npz'))
            acc=(z['predictions']==z['yq']).mean(-1);assert np.array_equal(acc,z['accuracy'])
            cells[name]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),
                         'comparisons':{ref:a.interval(acc[4]-acc[j],z['seeds']) for j,ref in enumerate(NAMES[:4])},
                         'selection_counts':np.bincount(z['choices'].ravel(),minlength=3).tolist()}
            accs.append(acc);seeds=z['seeds']
        acc=np.mean(accs,axis=0)
        domains[domain]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'comparisons':{}}
        for j,ref in enumerate(NAMES[:4]):
            delta=acc[4]-acc[j];domains[domain]['comparisons'][ref]=a.interval(delta,seeds);vectors[domain+'/'+ref]=(delta,seeds)
    pooled={}
    for ref in NAMES[:4]:
        delta=np.concatenate([vectors[d+'/'+ref][0] for d in CFG['domains']])
        seeds=np.concatenate([vectors[d+'/'+ref][1]+i*100000000 for i,d in enumerate(CFG['domains'])])
        pooled[ref]=a.interval(delta,seeds)
    tests={'increment':pooled['blend']['delta_pp']>=.5,'blend_ci':pooled['blend']['ci95_pp'][0]>0,
           'domains':all(d['comparisons']['blend']['delta_pp']>=0 for d in domains.values()),
           'cells':all(c['comparisons']['blend']['ci95_pp'][0]>=-.5 for c in cells.values()),
           'simple_controls':all(pooled[n]['delta_pp']>0 and pooled[n]['ci95_pp'][0]>0 for n in ['stack','raw','margin'])}
    return {'cells':cells,'domains':domains,'pooled':pooled,'gate':{'passed':all(tests.values()),'tests':tests},
            'pooled_accuracy_pct':{n:float(np.mean([d['accuracy_pct'][n] for d in domains.values()])) for n in NAMES},
            'scope':'Auxiliary, repeatedly exposed DTD/EuroSAT; conditional fixed-pool intervals. No new canonical Pets or Caltech results.'}

def main():
    start=time.time();torch.set_num_threads(6);guard()
    lock=json.loads((HERE/'code_lock.json').read_text());assert all(sha(Path(p))==h for p,h in lock.items())
    dump('manifest.json',{'argv':[sys.executable,*sys.argv],'config':CFG,'sources':lock,'sklearn':sklearn.__version__,'torch':torch.__version__,'numpy':np.__version__,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    m.old.sampler.CFG=dict(m.old.CFG,train_seed=26091361,train_episodes=100)
    data=m.old.load();models={};source_metrics={}
    for source in CFG['domains']:
        train,tt,tg=source_scores(source,'train',data)
        selection,vt,vg=source_scores(source,'selection',data)
        tr=np.unique(np.r_[tt['support_indices'].ravel(),tt['query_indices'].ravel()])
        va=np.unique(np.r_[vt['support_indices'].ravel(),vt['query_indices'].ravel()])
        rgb=np.array(data[source]['ident']['query_rgb']);assert not set(rgb[tr])&set(rgb[va])
        model=fit(np.concatenate(train,axis=1));model.update(source=source,gamma=CFG['gamma_by_source'][source])
        models[source]=model;dump(source+'_model.json',model)
        for phase,banks,t,gi in [('train',train,tt,tg),('selection',selection,vt,vg)]:
            accs=[]
            for j,sc in enumerate(banks):
                name=source+'_'+phase+('_excluded' if j else '_ordinary')
                pred=save_bank(name,sc,t,model,gi[j]);accs.append((pred==Y).mean(-1))
            source_metrics[source+'/'+phase]=dict(zip(NAMES,(np.mean(accs,axis=0).mean(-1)*100).tolist()))
        print('MODEL_FROZEN',source,source_metrics[source+'/selection'],flush=True)
    # No target features or labels are used by fit(), and both model files now exist.
    dump('source_metrics.json',source_metrics)
    model_hashes={d:sha(OUT/(d+'_model.json')) for d in CFG['domains']}
    parity=0;score_error=0.
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        t=m.old.sampler.tasks(data[target]['ident'],1,'eval')
        gm={g:data[g]['gallery'].double().mean(-2) for g in CFG['domains']}
        old={g:np.load(fusion.OUT/(target+'_'+g+'.npz')) for g in CFG['domains']}
        for g,z in old.items():
            assert all(np.array_equal(t[k],z[k]) for k in t)
            assert not set(data[target]['ident']['query_rgb'])&set(data[g]['ident']['gallery_rgb'])
        banks={g:[] for g in CFG['domains']}
        for i in range(500):
            guard();sv=data[target]['query'][t['support_indices'][i]];qv=data[target]['query'][t['query_indices'][i].reshape(-1)]
            _,(s,q),factor=m.prepare(sv,qv,CFG['gamma_by_source'][source])
            for g in CFG['domains']:
                sc,_=a.heads(s,q,m.old.metric.transform(gm[g],factor))
                scores=torch.stack([sc[3],sc[0],.5*(sc[3]+sc[0])]).numpy();banks[g].append(scores)
                assert np.array_equal(scores.argmax(-1),old[g]['predictions'][:3,i]);parity+=3
                if i in old[g]['sample_indices']:
                    j=old[g]['sample_indices'].tolist().index(i);err=float(abs(scores-old[g]['sample_scores'][:,j]).max());score_error=max(score_error,err);assert err<2e-5
            if i%100==0:print('TARGET',target,i,'seconds',time.time()-start,flush=True)
        for g in CFG['domains']:
            save_bank(target+'_'+g,np.stack(banks[g],axis=1),t,models[source]);print('CELL_DONE',target,g,flush=True)
    assert all(sha(OUT/(d+'_model.json'))==h for d,h in model_hashes.items())
    result=summarize();dump('analysis.json',result)
    dump('validation.json',{'status':'passed','parent_endpoint_task_checks':parity,'parent_sample_score_max_error':score_error,'source_models_unchanged_during_target':True,'all_bank_class_and_query_permutation_invariance':True,'convergence_warnings':0,'independent_audit':'separate audit pending'})
    dump('complete.json',{'status':'completed','elapsed_seconds':time.time()-start,'source_task_conditions':800,'target_task_conditions':2000,'gate':result['gate']['passed']})
    print('COMPLETE',json.dumps({'accuracy':result['pooled_accuracy_pct'],'pooled':result['pooled'],'gate':result['gate'],'seconds':time.time()-start}),flush=True)
if __name__=='__main__':main()
