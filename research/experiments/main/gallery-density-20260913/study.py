"""A fixed paired development comparison and independent numerical audit."""
from pathlib import Path
import argparse,datetime,hashlib,json,shutil,sys,time
import numpy as np
import torch
import density as m
HERE=m.HERE;OUT=HERE/'outputs';ROOT=m.ROOT;PARENT=HERE.parent/'query-consistency-20260913/outputs'
CFG=json.loads((HERE/'protocol.json').read_text());KEYS=['support_indices','query_indices','class_ids','seeds'];NAMES=m.NAMES

def dump(name,obj):(OUT/name).write_text(json.dumps(obj,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
    assert shutil.disk_usage(HERE).free/2**30>=CFG['min_free_gib']
    assert datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(CFG['deadline'])
def inputs():
    data=m.z.m.old.load();gallery={d:data[d]['gallery'].double().mean(-2) for d in CFG['domains']}
    ids={d:m.identities(data[d]['ident']['gallery_rgb']) for d in CFG['domains']}
    return data,gallery,ids

def validate_case(sv,qv,g,gamma,eta,ids):
    scores,ix,b=m.evaluate(sv,qv,g,gamma,eta,*ids)
    a,ni,nb=m.independent(sv,qv,g,gamma,eta,*ids)
    err=float(np.max(abs(a-scores.numpy())));berr=float(np.max(abs(nb-b.numpy())))
    assert err<2e-5 and berr<2e-5,(err,berr)
    assert np.array_equal(ix.numpy(),ni),'independent neighbor mismatch'
    assert np.array_equal(scores.argmax(-1).numpy(),a.argmax(-1)),'independent prediction mismatch'
    parent=m.z.evaluate(sv,qv,g,gamma,[eta],[0.])['consistency'][0]
    perr=float((parent-scores[0]).abs().max());assert perr<1e-10
    return {'score_error':err,'density_error':berr,'parent_error':perr},scores,ix,b

def check():
    torch.manual_seed(26091398)
    def rand(*shape):return torch.nn.functional.normalize(torch.randn(*shape,dtype=torch.float64),dim=-1)
    sv=rand(5,1,6,32);qv=rand(7,6,32);g=rand(180,32);rgb=[f'{i:064x}' for i in range(180)];ids=m.identities(rgb)
    checks=[]
    for gamma,eta in [(1,10),(10,0)]:
        r,sc,ix,b=validate_case(sv,qv,g,gamma,eta,ids);checks.append({'kind':'synthetic',**r})
        qp=torch.tensor([3,0,6,2,1,5,4]);perm=np.random.RandomState(10).permutation(180);cp=torch.tensor([2,4,0,3,1])
        assert torch.allclose(m.evaluate(sv,qv[qp],g,gamma,eta,*ids)[0],sc[:,qp],atol=1e-10,rtol=0)
        assert torch.allclose(m.evaluate(sv[cp],qv,g,gamma,eta,*ids)[0],sc[:,:,cp],atol=1e-10,rtol=0)
        gi=m.identities([rgb[i] for i in perm]);ss,ii,_=m.evaluate(sv,qv,g[perm],gamma,eta,*gi)
        assert torch.allclose(ss,sc,atol=1e-10,rtol=0) and np.array_equal(perm[ii.numpy()],ix.numpy())
    data,gs,refs=inputs()
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for gallery in CFG['domains']:
            old=np.load(PARENT/(target+'_'+gallery+'.npz'));i=0
            sv=data[target]['query'][old['support_indices'][i]];qv=data[target]['query'][old['query_indices'][i].reshape(-1)]
            r,sc,_,_=validate_case(sv,qv,gs[gallery],CFG['gamma_by_source'][source],CFG['eta_by_source'][source],refs[gallery]);checks.append({'kind':target+'_'+gallery,**r})
            assert np.max(abs(sc[0].numpy()-old['scores'][1,i]))<1e-10
    paths=[HERE/'density.py',HERE/'study.py',HERE/'protocol.json',Path(m.z.__file__),PARENT/'selection_lock.json']
    dump('check.json',{'status':'passed','checks':checks,'query_class_gallery_permutation':'passed','torch':torch.__version__,'numpy':np.__version__})
    (HERE/'code_lock.json').write_text(json.dumps({str(p):sha(p) for p in paths},indent=2)+'\n')
    print('CHECK_PASSED',json.dumps(checks),flush=True)

def interval(delta,groups):
    rng=np.random.RandomState(CFG['bootstrap_seed']);v=np.zeros(CFG['bootstrap_replicates']);unique=np.unique(groups)
    for group in unique:
        d=delta[groups==group];v+=d[rng.randint(len(d),size=(len(v),len(d)))].mean(1)/len(unique)
    return {'delta_pp':float(delta.mean()*100),'ci95_pp':(np.quantile(v,[.025,.975])*100).tolist()}

def summarize():
    cells={};domains={};vectors={}
    for d in CFG['domains']:
        aa=[]
        for g in CFG['domains']:
            bank=np.load(OUT/(d+'_'+g+'.npz'));pr=bank['predictions'];acc=(pr==bank['yq']).mean(-1);assert np.array_equal(acc,bank['accuracy']);aa.append(acc)
            cells[d+'_'+g]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'comparisons':{NAMES[j]:interval(acc[1]-acc[j],bank['seeds']) for j in [0,2,3]},'repairs':int(((pr[1]==bank['yq'])&(pr[0]!=bank['yq'])).sum()),'spoils':int(((pr[1]!=bank['yq'])&(pr[0]==bank['yq'])).sum()),'mean_unique_neighbors':bank['unique_neighbors'].mean(0).tolist(),'mean_selected_penalty':bank['selected_penalty'].mean(0).tolist()}
        acc=np.mean(aa,0);domains[d]={'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'comparisons':{NAMES[j]:interval(acc[1]-acc[j],bank['seeds']) for j in [0,2,3]}}
        vectors[d]=(acc,bank['seeds'])
    groups=np.concatenate([vectors[d][1]+i*100000000 for i,d in enumerate(CFG['domains'])]);acc=np.concatenate([vectors[d][0] for d in CFG['domains']],1)
    pooled={NAMES[j]:interval(acc[1]-acc[j],groups) for j in [0,2,3]}
    tests={'increment':pooled['parent']['delta_pp']>=.5,'positive_ci':pooled['parent']['ci95_pp'][0]>0,'domains':all(x['comparisons']['parent']['delta_pp']>=0 for x in domains.values()),'cells':all(x['comparisons']['parent']['ci95_pp'][0]>=-.5 for x in cells.values()),'controls':all(pooled[n]['ci95_pp'][0]>0 for n in ['global','shuffled'])}
    return {'scope':CFG['metric_deviation'],'cells':cells,'domains':domains,'pooled':pooled,'accuracy_pct':dict(zip(NAMES,(acc.mean(-1)*100).tolist())),'gate':{'passed':all(tests.values()),'tests':tests}}

def run():
    start=time.time();guard();assert json.loads((OUT/'check.json').read_text())['status']=='passed'
    lock=json.loads((HERE/'code_lock.json').read_text());assert all(sha(p)==h for p,h in lock.items())
    data,gs,refs=inputs();dump('manifest.json',{'argv':[sys.executable,*sys.argv],'config':CFG,'code_hashes':lock,'parent_bank_hashes':{str(p):sha(p) for p in PARENT.glob('*.npz') if 'selection' not in p.name},'versions':{'torch':torch.__version__,'numpy':np.__version__},'reference_indices':{k:v[0].tolist() for k,v in refs.items()},'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    count=0;maxerr=0.
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for gallery in CFG['domains']:
            name=target+'_'+gallery;old=np.load(PARENT/(name+'.npz'));rows=[];uniq=[];pen=[]
            assert not set(data[target]['ident']['query_rgb'])&set(data[gallery]['ident']['gallery_rgb'])
            for i in range(len(old['seeds'])):
                sv=data[target]['query'][old['support_indices'][i]];qv=data[target]['query'][old['query_indices'][i].reshape(-1)]
                sc,ix,b=m.evaluate(sv,qv,gs[gallery],CFG['gamma_by_source'][source],CFG['eta_by_source'][source],*refs[gallery]);arr=sc.numpy()
                err=float(abs(arr[0]-old['scores'][1,i]).max());maxerr=max(maxerr,err);assert err<1e-10
                assert np.array_equal(arr[0].argmax(-1),old['predictions'][1,i]);count+=1
                assert np.isfinite(arr).all() and np.allclose(arr.sum(-1),1,atol=1e-10,rtol=0)
                rows.append(arr);uniq.append([[len(torch.unique(h)) for h in head] for head in ix]);pen.append([[float(b[h,1][ix[h,j]].mean()) for j in range(4)] for h in range(2)])
                if i%100==0:
                    guard();progress={'cell':name,'tasks':i+1,'task_conditions_done':count,'elapsed_seconds':time.time()-start};dump('progress.json',progress);print('PROGRESS',json.dumps(progress),flush=True)
            scores=np.stack(rows,1);pred=scores.argmax(-1).astype(np.uint8);y=old['yq']
            np.savez_compressed(OUT/(name+'.npz'),scores=scores,predictions=pred,accuracy=(pred==y).mean(-1),yq=y,names=NAMES,reference_indices=refs[gallery][0],shuffle_indices=refs[gallery][1],unique_neighbors=np.array(uniq),selected_penalty=np.array(pen),**{k:old[k] for k in KEYS})
            print('CELL_COMPLETE',name,flush=True)
    assert all(sha(p)==h for p,h in lock.items());result=summarize();dump('analysis.json',result);dump('complete.json',{'status':'completed','seconds':time.time()-start,'task_conditions':count,'parent_max_error':maxerr,'independent_audit':'pending'})
    print('COMPLETE',json.dumps({'accuracy':result['accuracy_pct'],'pooled':result['pooled'],'gate':result['gate']}),flush=True)

def audit():
    data,gs,refs=inputs();checks=[];count=0
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for gallery in CFG['domains']:
            name=target+'_'+gallery;b=np.load(OUT/(name+'.npz'));old=np.load(PARENT/(name+'.npz'));assert all(np.array_equal(b[k],old[k]) for k in KEYS)
            assert np.array_equal(b['predictions'],b['scores'].argmax(-1)) and np.array_equal(b['accuracy'],(b['predictions']==b['yq']).mean(-1));count+=b['predictions'].size
            assert np.max(abs(b['scores'][0]-old['scores'][1]))<1e-10
            for i in CFG['audit_indices']:
                sv=data[target]['query'][b['support_indices'][i]];qv=data[target]['query'][b['query_indices'][i].reshape(-1)]
                r,sc,ix,pen=validate_case(sv,qv,gs[gallery],CFG['gamma_by_source'][source],CFG['eta_by_source'][source],refs[gallery]);assert np.max(abs(sc.numpy()-b['scores'][:,i]))<1e-10;checks.append({'cell':name,'task':i,**r})
    assert summarize()==json.loads((OUT/'analysis.json').read_text())
    dump('audit.json',{'status':'passed','checked_prediction_entries':count,'independent_cases':checks,'max_score_error':max(x['score_error'] for x in checks),'all_parent_scores_equal':True,'all_task_identities_equal':True,'statistics':'Recomputed from all saved predictions; paired-bootstrap helper deterministic, independent scalar interval pending.'})
    print('AUDIT_PASSED',json.dumps({'entries':count,'cases':len(checks),'max_error':max(x['score_error'] for x in checks)}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['check','run','audit'],required=True);args=p.parse_args();torch.set_num_threads(CFG['threads']);guard();globals()[args.phase]()
