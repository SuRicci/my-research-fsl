"""Fixed paired source evaluation; never fit or select on query labels."""
import time
from sklearn.linear_model import LogisticRegression
from common import *
NAMES=CFG['methods'];Y=np.repeat(np.arange(5),15)
def load():
    data=z.m.old.load();manifest=json.loads((ASSET/'manifest.json').read_text());assert manifest['status']=='completed' and manifest['images']==CFG['expected_images'];maps={}
    for key,row in manifest['pools'].items():
        p=ASSET/(key+'.npy');ids=json.loads((ASSET/(key+'_identities.json')).read_text());assert sha(p)==row['sha256'] and sha(ASSET/(key+'_identities.json'))==row['identity_sha256'];maps[key]=np.load(p,mmap_mode='r');ds,side=key.split('_');assert ids['ids']==data[ds]['ident'][side+'_ids']
    return data,maps
def take(data,maps,ds,side,ix):
    old=data[ds][side][ix].double();new=F.normalize(torch.from_numpy(np.array(maps[ds+'_'+side][ix])).double(),dim=-1)
    return torch.cat([old[...,:512],new*old[...,512:].norm(dim=-1,keepdim=True)],-1)
def gallery(data,maps,ds):return torch.cat([take(data,maps,ds,'gallery',slice(i,i+128)).mean(-2) for i in range(0,len(data[ds]['gallery']),128)])
def score(s,q,g,gamma,eta):
    ret=retained(s,q,g,gamma,eta);sm=F.normalize(s.mean(-2),dim=-1);qm=F.normalize(q.mean(-2),dim=-1);gm=F.normalize(g,dim=-1)
    mr,_=z.m.head(sm,qm,gm,False,.1);mc,_=z.m.head(sm,qm,gm,True,.1);x=sm.reshape(5,-1).numpy();qn=qm.numpy();support=ridge(x,qn,np.ones(5));aug=ridge(s.flatten(0,2).numpy(),qn,np.full(30,1/6));lr=[]
    for C in [1,10]:
        clf=LogisticRegression(C=C,solver='lbfgs',multi_class='multinomial',max_iter=2000,tol=1e-8,random_state=26091275);clf.fit(x,np.arange(5));assert clf.n_iter_.max()<2000;lr.append(clf.decision_function(qn))
    return np.stack([ret['consistency'][0].numpy(),ret['incumbent'].numpy(),mr.numpy(),mc.numpy(),support,*lr,aug])
def interval(delta,groups):
    rng=np.random.RandomState(CFG['bootstrap']['seed']);out=np.zeros(CFG['bootstrap']['replicates'])
    for seed in np.unique(groups):
        v=delta[groups==seed];out+=v[rng.randint(len(v),size=(len(out),len(v)))].sum(1)/len(delta)
    return {'delta_pp':float(delta.mean()*100),'ci95_pp':(np.percentile(out,[2.5,97.5])*100).tolist()}
def analyze():
    domains={};cells={};allacc=[];allgroups=[];interactions=[]
    for ds in CFG['domains']:
        aa=[];old=[]
        for gal in CFG['domains']:
            f=np.load(OUT/f'{ds}_{gal}.npz');a=(f['predictions']==Y).mean(-1);aa.append(a);old.append((f['small_control_scores'].argmax(-1)==Y).mean(-1))
            cells[ds+'_'+gal]={'accuracy_pct':dict(zip(NAMES,(a.mean(-1)*100).tolist())),'vs_small':interval(a[0]-a[-1],f['seeds'])}
        a=np.mean(aa,0);old=np.mean(old,0);allacc.append(a);allgroups.append(f['seeds']+len(allgroups)*100000000)
        inter=np.stack([(a[0]-a[2])-(a[-1]-old[0]),(a[0]-a[3])-(a[-1]-old[1])]);interactions.append(inter)
        domains[ds]={'accuracy_pct':dict(zip(NAMES,(a.mean(-1)*100).tolist())),'comparisons':{n:interval(a[0]-a[NAMES.index(n)],f['seeds']) for n in CFG['gate_controls']},'parent_increment':interval(a[0]-a[1],f['seeds']),'recipe_interaction':{n:interval(inter[j],f['seeds']) for j,n in enumerate(['mean_r2','mean_CS'])}}
    a=np.concatenate(allacc,1);groups=np.concatenate(allgroups);pooled={n:interval(a[0]-a[NAMES.index(n)],groups) for n in CFG['gate_controls']};inter=np.concatenate(interactions,1)
    gate=all(x['delta_pp']>=.5 and x['ci95_pp'][0]>0 for x in pooled.values()) and all(x['delta_pp']>=0 for d in domains.values() for x in d['comparisons'].values()) and all(x['vs_small']['ci95_pp'][0]>=-.5 for x in cells.values())
    result={'status':'computed','gate_passed':bool(gate),'domains':domains,'cells':cells,'pooled':pooled,'parent_increment':interval(a[0]-a[1],groups),'recipe_interaction':{n:interval(inter[j],groups) for j,n in enumerate(['mean_r2','mean_CS'])},'unique_tasks':1000,'task_gallery_conditions':2000,'methods':NAMES,'scope':CFG['scope'],'audit_pending':True};dump(OUT/'analysis.json',result);return result
def immutable():
    files=[HERE/n for n in ['protocol.json','common.py','study.py','audit.py','encode.py']]+[ASSET/'manifest.json',ROOT/'baselines/local/r2-pets-transfer/json/metric_contract.json']
    files += [Path(p) for p in json.loads((ASSET/'manifest.json').read_text())['lock']]
    files += list((ROOT/'experiments').rglob('*.py'))
    files += [REF/f'{d}_{g}.npz' for d in CFG['domains'] for g in CFG['domains']]
    for m in list(sys.modules.values()):
        f=getattr(m,'__file__',None)
        if f and str(ROOT/'experiments') in str(f) and str(f).endswith('.py'):files.append(Path(f))
    return {str(p):sha(p) for p in sorted(set(files))}
def run():
    guard();assert json.loads((OUT/'precheck.json').read_text())['status']=='passed' and json.loads((OUT/'encoding_validation.json').read_text())['status']=='passed';assert not (OUT/'run_manifest.json').exists(),'adopt existing run';torch.set_num_threads(CFG['threads']);data,maps=load();locked=immutable();dump(OUT/'run_manifest.json',{'config':CFG,'lock':locked,'command':[sys.executable,*sys.argv],'started':datetime.datetime.now(datetime.timezone.utc).isoformat()});start=time.time()
    for ds in CFG['domains']:
        source=next(x for x in CFG['domains'] if x!=ds);gamma=CFG['source_gamma'][source];eta=CFG['source_eta'][source]
        for gal in CFG['domains']:
            bank=np.load(REF/f'{ds}_{gal}.npz');g=gallery(data,maps,gal);gs=data[gal]['gallery'].double().mean(-2);scores=[];smallcontrols=[]
            for i in range(CFG['eval_tasks']):
                guard();si=bank['support_indices'][i];qi=bank['query_indices'][i].reshape(-1);s=take(data,maps,ds,'query',si);q=take(data,maps,ds,'query',qi);actual=score(s,q,g,gamma,eta)
                sv=data[ds]['query'][si].double();qv=data[ds]['query'][qi].double();small=retained(sv,qv,gs,gamma,eta)['consistency'][0].numpy();expected=bank['scores'][bank['names'].tolist().index('consistency'),i];assert np.max(abs(small-expected))<1e-10
                sm=F.normalize(sv.mean(-2),dim=-1);qm=F.normalize(qv.mean(-2),dim=-1);gm=F.normalize(gs,dim=-1);smallcontrols.append(np.stack([z.m.head(sm,qm,gm,b,.1)[0].numpy() for b in [False,True]]));v=np.concatenate([actual,small[None]]);assert np.isfinite(v).all();scores.append(v)
                if i%50==0:
                    progress={'domain':ds,'gallery':gal,'tasks_done':i+1,'seconds':time.time()-start};dump(OUT/'classification_progress.json',progress);print('EVAL',json.dumps(progress),flush=True)
            scores=np.stack(scores,1);np.savez_compressed(OUT/f'{ds}_{gal}.npz',scores=scores,predictions=scores.argmax(-1).astype(np.uint8),small_control_scores=np.stack(smallcontrols,1),names=np.array(NAMES),yq=Y,**{k:bank[k] for k in ['support_indices','query_indices','class_ids','seeds']});print('CELL_COMPLETE',ds,gal,flush=True)
    result=analyze();assert immutable()==locked;dump(OUT/'complete.json',{'status':'computed','seconds':time.time()-start,'gate_passed':result['gate_passed'],'audit_pending':True});print('COMPUTE_COMPLETE',result['gate_passed'],flush=True)
if __name__=='__main__':run()
