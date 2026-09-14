"""Separate NumPy algebra and complete saved-data checks for the fixed control."""
import argparse
from common import *
s=module('base_study_audit',HERE/'study.py')
ind=module('base_prior_numpy_audit',HERE.parent/'dino-depth-readout-20260914/audit.py')
def check():
    torch.set_num_threads(CFG['threads']);torch.manual_seed(26091460)
    v=F.normalize(torch.randn(5,1,6,11,dtype=torch.double),dim=-1);q=F.normalize(torch.randn(75,6,11,dtype=torch.double),dim=-1)
    d=(v-v.mean(-2,keepdim=True)).numpy().reshape(-1,11);cov=d.T@d/len(d);gamma=1.7
    val,vec=np.linalg.eigh(np.eye(11)+gamma*11*cov/np.trace(cov));t=(vec*val**-.5)@vec.T
    expected=ind.unit(q.mean(-2).numpy()@t);actual=z.m.old.metric.transform(q.mean(-2),z.m.old.metric.scatter_factor(v,gamma)).numpy()
    scatter_error=float(abs(expected-actual).max());assert scatter_error<1e-10
    x=np.random.RandomState(1).normal(size=(30,11));qq=np.random.RandomState(2).normal(size=(75,11));w=np.full(30,1/6);y=np.eye(5).repeat(6,0)
    xm=np.average(x,axis=0,weights=w);ym=np.average(y,axis=0,weights=w);xc=x-xm;yc=y-ym
    primal=(qq-xm)@np.linalg.solve(xc.T@(w[:,None]*xc)+.1*np.eye(11),xc.T@(w[:,None]*yc))+ym
    ridge_error=float(abs(primal-ridge(x,qq,w)).max());assert ridge_error<1e-10
    data=z.m.old.load();checks=[]
    for target in CFG['domains']:
        source=next(d for d in CFG['domains'] if d!=target)
        for gal in CFG['domains']:
            ref=np.load(REF/f'{target}_{gal}.npz');i=0;sv=data[target]['query'][ref['support_indices'][i]].double();qv=data[target]['query'][ref['query_indices'][i].reshape(-1)].double();gm=data[gal]['gallery'].double().mean(-2)
            gamma=CFG['source_gamma'][source];eta=CFG['source_eta'][source];a=retained(sv,qv,gm,gamma,eta);padded=[F.pad(x,(0,384)) for x in [sv,qv,gm]];b=retained(*padded,gamma,eta)
            error=float(abs(a['consistency']-b['consistency']).max());expected=ref['scores'][ref['names'].tolist().index('consistency'),i];old_error=float(abs(a['consistency'][0].numpy()-expected).max())
            assert max(error,old_error)<1e-10 and torch.equal(a['consistency'].argmax(-1),b['consistency'].argmax(-1))
            # All eight endpoints must remain equivalent when only empty coordinates are added.
            sa=s.score(sv,qv,gm,gamma,eta);sb=s.score(*padded,gamma,eta);control_error=float(abs(sa-sb).max());assert control_error<1e-8 and np.array_equal(sa.argmax(-1),sb.argmax(-1))
            checks.append({'target':target,'gallery':gal,'padding_error':error,'old_error':old_error,'all_controls_padding_error':control_error})
    result={'status':'passed','dense_scatter_error':scatter_error,'primal_ridge_error':ridge_error,'real_padding_cases':checks,'scope':'four fixed real null tasks and independent small matrix formulas; does not establish base accuracy'}
    dump(OUT/'precheck.json',result);print('PRECHECK_PASSED',json.dumps(result),flush=True)
def features():
    rec=json.loads((ASSET/'manifest.json').read_text());assert rec['status']=='completed' and rec['images']==CFG['expected_images'];rows=ex.source_rows();assert set(rec['pools'])==set(rows);norm_errors={};count=0
    for p,h in rec['lock'].items():assert sha(p)==h,p
    assert sha(ASSET/'dinov2_vitb14_pretrain.pth')==json.loads((ASSET/'weight_manifest.json').read_text())['sha256']
    for key,row in rows.items():
        p=ASSET/(key+'.npy');meta=rec['pools'][key];ids=ASSET/(key+'_identities.json');assert sha(p)==meta['sha256'] and sha(ids)==meta['identity_sha256'];assert json.loads(ids.read_text())=={'ids':row['ids'],'rgb':row['rgb']}
        a=np.load(p,mmap_mode='r');assert a.shape==(len(row['ids']),6,768) and a.dtype==np.float16;err=0.
        for start in range(0,len(a),256):
            v=np.asarray(a[start:start+256],dtype=np.float32);assert np.isfinite(v).all();err=max(err,float(abs(np.linalg.norm(v,axis=-1)-1).max()))
        assert err<CFG['encoding_checks']['unit_norm_error'];norm_errors[key]=err;count+=len(a)
    assert count==14153
    dump(OUT/'encoding_validation.json',{'status':'passed','images':count,'all_pool_hashes_identities_shapes_and_values_checked':True,'unit_norm_max_errors':norm_errors,'manifest_sha256':sha(ASSET/'manifest.json')});print('FEATURE_VALIDATION_PASSED',count,flush=True)
def full():
    torch.set_num_threads(CFG['threads']);data,maps=s.load();summary=json.loads((OUT/'analysis.json').read_text());total=0;stats=[];errors=[];domain_a=[];domain_g=[];domain_inter=[];cell_lowers=[];domain_points=[]
    def interval(d,groups,row,label):
        rng=np.random.RandomState(CFG['bootstrap']['seed']);v=np.zeros(CFG['bootstrap']['replicates'])
        for group in np.unique(groups):
            values=d[groups==group];ix=rng.randint(len(values),size=(len(v),len(values)))
            counts=np.array([np.bincount(a,minlength=len(values)) for a in ix]);v+=counts@values/len(d)
        actual=np.array([100*d.mean(),*np.percentile(100*v,[2.5,97.5])]);assert abs(actual-np.array([row['delta_pp'],*row['ci95_pp']])).max()<1e-10;stats.append(label);return actual
    def features_np(ds,side,ix):
        old=data[ds][side][ix].double().numpy();new=ind.unit(np.array(maps[ds+'_'+side][ix],dtype=np.float64));return np.concatenate([old[...,:512],new*np.linalg.norm(old[...,512:],axis=-1,keepdims=True)],-1)
    for ds in CFG['domains']:
        source=next(x for x in CFG['domains'] if x!=ds);aa=[];old=[]
        for gal in CFG['domains']:
            f=np.load(OUT/f'{ds}_{gal}.npz');ref=np.load(REF/f'{ds}_{gal}.npz');pred=f['predictions'];assert pred.shape==(9,500,75) and f['scores'].shape==(9,500,75,5) and f['names'].tolist()==s.NAMES and np.array_equal(f['yq'],s.Y)
            assert np.isfinite(f['scores']).all() and np.array_equal(pred,f['scores'].argmax(-1));assert f['small_control_scores'].shape==(2,500,75,5) and np.isfinite(f['small_control_scores']).all()
            for k in ['support_indices','query_indices','class_ids','seeds']:assert np.array_equal(f[k],ref[k]),k
            labels=np.asarray(data[ds]['ident']['query_labels']);assert np.array_equal(labels[f['support_indices']],f['class_ids'][:,:,None]);assert np.array_equal(labels[f['query_indices']],np.repeat(f['class_ids'][:,:,None],15,2))
            for si,qi in zip(f['support_indices'],f['query_indices']):assert not np.intersect1d(si,qi).size
            reference=ref['scores'][ref['names'].tolist().index('consistency')];assert abs(f['scores'][-1]-reference).max()<1e-10 and np.array_equal(pred[-1],reference.argmax(-1));total+=pred.size
            a=(pred==s.Y).mean(-1);aa.append(a);old.append((f['small_control_scores'].argmax(-1)==s.Y).mean(-1));row=summary['cells'][ds+'_'+gal];assert max(abs(row['accuracy_pct'][n]-a[j].mean()*100) for j,n in enumerate(s.NAMES))<1e-10
            cell_lowers.append(interval(a[0]-a[-1],f['seeds'],row['vs_small'],ds+'_'+gal)[1]);i=99;sv=features_np(ds,'query',f['support_indices'][i]);qv=features_np(ds,'query',f['query_indices'][i].reshape(-1));gm=np.concatenate([features_np(gal,'gallery',slice(j,j+128)).mean(-2) for j in range(0,len(data[gal]['gallery']),128)])
            for method,eta in [(0,CFG['source_eta'][source]),(1,0)]:
                sc=ind.independent_primary(sv,qv,gm,CFG['source_gamma'][source],eta);err=float(abs(sc-f['scores'][method,i]).max());assert err<CFG['encoding_checks']['independent_scores_abs'] and np.array_equal(sc.argmax(-1),pred[method,i]);errors.append({'domain':ds,'gallery':gal,'task':i,'method':s.NAMES[method],'max_abs':err})
        a=np.mean(aa,0);oc=np.mean(old,0);inter=np.stack([(a[0]-a[2])-(a[-1]-oc[0]),(a[0]-a[3])-(a[-1]-oc[1])]);row=summary['domains'][ds]
        assert max(abs(row['accuracy_pct'][n]-a[j].mean()*100) for j,n in enumerate(s.NAMES))<1e-10
        for n in CFG['gate_controls']:domain_points.append(interval(a[0]-a[s.NAMES.index(n)],f['seeds'],row['comparisons'][n],ds+n)[0])
        interval(a[0]-a[1],f['seeds'],row['parent_increment'],ds+'parent')
        for j,n in enumerate(['mean_r2','mean_CS']):interval(inter[j],f['seeds'],row['recipe_interaction'][n],ds+n+'interaction')
        domain_a.append(a);domain_g.append(f['seeds']+len(domain_g)*100000000);domain_inter.append(inter)
    a=np.concatenate(domain_a,1);groups=np.concatenate(domain_g);inter=np.concatenate(domain_inter,1);passed=True
    for n in CFG['gate_controls']:
        r=interval(a[0]-a[s.NAMES.index(n)],groups,summary['pooled'][n],'pooled'+n);passed &= r[0]>=.5 and r[1]>0
    interval(a[0]-a[1],groups,summary['parent_increment'],'pooledparent')
    for j,n in enumerate(['mean_r2','mean_CS']):interval(inter[j],groups,summary['recipe_interaction'][n],'pooled'+n+'interaction')
    passed &= min(domain_points)>=0 and min(cell_lowers)>=-.5;assert bool(passed)==summary['gate_passed']
    for p,h in json.loads((OUT/'run_manifest.json').read_text())['lock'].items():assert sha(p)==h,p
    result={'status':'passed','prediction_arithmetic_checks':total,'independent_intervals':len(stats),'independent_primary_cases':errors,'gate_passed':bool(passed),'bank_hashes':{f'{d}_{g}':sha(OUT/f'{d}_{g}.npz') for d in CFG['domains'] for g in CFG['domains']},'scope':'Complete saved arithmetic/task integrity; independent NumPy primary and parent on four fixed tasks; logistic optimizer not independently implemented.'};dump(OUT/'audit.json',result);print('AUDIT_PASSED',json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['check','features','full'],required=True);a=p.parse_args();{'check':check,'features':features,'full':full}[a.phase]()
