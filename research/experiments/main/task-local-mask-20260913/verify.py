"""Independent explicit-feature mask optimizer and NumPy64 score reconstruction."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
import torch.nn.functional as F
import study as s
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def mask(sc,sd):
    sc=F.normalize(sc.double(),dim=-1);sd=F.normalize(sd.double(),dim=-1);k=sc.shape[1]
    a=torch.zeros(2,dtype=torch.float64,requires_grad=True);eg=np.zeros(2);ed=np.zeros(2)
    for _ in range(40):
        lam=a.sigmoid();x=torch.cat([lam[0]*sc,lam[1]*sd],-1);p=F.normalize(x.mean(1),dim=-1)
        sim=F.normalize(x.flatten(0,1),dim=-1)@p.T;y=torch.arange(5).repeat_interleave(k)
        loss=F.cross_entropy(sim,y);g=torch.autograd.grad(loss,a)[0].numpy()
        eg=.9*eg+.1*g*g;delta=-np.sqrt(ed+1e-6)/np.sqrt(eg+1e-6)*g;ed=.9*ed+.1*delta*delta
        a=torch.tensor(a.detach().numpy()+100*delta,requires_grad=True)
    lam=a.detach().sigmoid().numpy();return lam[0]**2/np.sum(lam**2)
def score(sc,sd,qc,qd,gc,gd,w,origin):
    def join(a,b):return np.concatenate([np.sqrt(w)*norm(a),np.sqrt(1-w)*norm(b)],axis=-1)
    sc,sd,qc,qd,gc,gd=[v.astype(np.float64) for v in [sc,sd,qc,qd,gc,gd]]
    support=join(sc,sd);query=join(qc,qd);k=support.shape[1];gallery=join(gc,gd)
    if origin=='support':
        m=support.reshape(5*k,-1).mean(0);support=norm(support-m);query=norm(query-m);gallery=norm(gallery-m)
    proto=norm(support.mean(1))
    if origin=='ncc':return query@proto.T
    if k==1:
        idx=np.argsort(-(proto@gallery.T),axis=-1,kind='stable')[:,:64]
        train=norm(.5*proto+.5*norm(gallery[idx].mean(1)));y=np.eye(5)
    else:train=support.reshape(5*k,-1);y=np.eye(5).repeat(k,axis=0)
    lam=.1 if origin=='support' or k==1 else 1.
    a=train-train.mean(0);b=y-y.mean(0)
    return (query-train.mean(0))@a.T@np.linalg.solve(a@a.T+lam*np.eye(len(a)),b)+y.mean(0)
def stats(x,seeds):
    rng=np.random.RandomState(s.CFG['bootstrap']['seed']);draws=np.zeros(s.CFG['bootstrap']['n'])
    for seed in np.unique(seeds):
        v=x[seeds==seed];draws+=v[rng.randint(len(v),size=(len(draws),len(v)))].mean(1)/len(np.unique(seeds))
    return {'mean_pp':float(x.mean()*100),'ci95_pp':(np.percentile(draws,[2.5,97.5])*100).tolist(),'by_seed_pp':{str(v):float(x[seeds==v].mean()*100) for v in np.unique(seeds)}}
def main():
    started=time.time();assert json.loads((OUT/'completion.json').read_text())['status']=='completed'
    lock=json.loads((HERE/'lock.json').read_text())
    for p,h in lock['files'].items():assert s.sha(p)==h,p
    data,manifest=s.assets_module.assets();assert manifest==lock['features']
    cells={};arrays={};audit=[];mask_audit=[];refcheck=[];diagnostics={}
    for ds in s.CFG['domains']:
        for k in s.CFG['shots']:
            z=np.load(OUT/f'masks_{ds}_k{k}.npz');t=s.task(ds,k)
            for key in t:assert np.array_equal(z[key],t[key])
            assert np.isfinite(z['weights']).all() and ((z['weights']>=0)&(z['weights']<=1)).all()
            idx=[0,100,499];args=s.take(data,ds,t,np.array(idx))
            for i,task_id in enumerate(idx):
                w=mask(args[0][i],args[1][i]);err=abs(w-z['weights'][task_id]);assert err<3e-4,(ds,k,err)
                mask_audit.append({'ds':ds,'shot':k,'task':task_id,'error':float(err)})
            diagnostics[f'{ds}_k{k}']={'weight_mean':float(z['weights'].mean()),'weight_quantiles':np.quantile(z['weights'],[0,.1,.5,.9,1]).tolist(),'fraction_clip_dominant':float((z['weights']>.5).mean()),'support_loss_change':float((z['after']-z['before']).mean())}
            for gallery in s.CFG['domains']:
                cell=f'{ds}_{gallery}_k{k}';a=np.load(OUT/f'{cell}.npz');assert a['names'].tolist()==s.NAMES
                pred=a['predictions'];assert pred.shape==(8,500,75) and ((pred>=0)&(pred<5)).all()
                acc=(pred==a['yq']).mean(-1);arrays[cell]=acc
                comp={name:stats(acc[0]-acc[j],t['seeds']) for j,name in enumerate(s.NAMES) if j!=0}
                cells[cell]={'accuracy_pct':dict(zip(s.NAMES,(acc.mean(-1)*100).tolist())),'comparisons':comp}
                g,keep=s.gal(data,ds,gallery);assert np.array_equal(keep,a['gallery_pool_indices'])
                for pos,task_id in enumerate(a['audit_indices']):
                    aa=s.take(data,ds,t,np.array([task_id]));raw=[v[0].numpy() for v in aa];gg=[v.numpy() for v in g];w=float(z['weights'][task_id])
                    spec=[(w,'support'),(.5,'support'),(1.,'support'),(0.,'support'),(.5,'none'),(w,'none'),(.5,'ncc'),(w,'ncc')]
                    for j,(ww,origin) in enumerate(spec):
                        value=score(*raw,*gg,ww,origin);err=np.max(np.abs(value-a['audit_scores'][j,pos]));assert err<3e-5,(cell,j,err)
                        assert np.array_equal(value.argmax(-1),pred[j,task_id]),(cell,j,task_id)
                        audit.append({'cell':cell,'method':s.NAMES[j],'task':int(task_id),'max_error':float(err)})
                old=np.load(s.BANK/'cells'/f'eval_{ds}_k{k}_{gallery}_v00.npz')
                for name,j in [('r2',4),('CS_l2',1)]:
                    ix=['r2','CS_l2','linear_mean'].index(name);diff=int(np.sum(old['predictions'][ix]!=pred[j]));assert diff==0,(cell,name,diff)
                    refcheck.append({'cell':cell,'method':name,'disagreements':diff})
    macro={};passes=[]
    for ds in s.CFG['domains']:
        acc=np.mean([arrays[f'{ds}_{g}_k1'] for g in s.CFG['domains']],axis=0);seeds=s.task(ds,1)['seeds']
        co={m:stats(acc[0]-acc[j],seeds) for j,m in enumerate(s.NAMES) if j!=0}
        macro[ds]={'accuracy_pct':dict(zip(s.NAMES,(acc.mean(-1)*100).tolist())),'comparisons':co}
        for m in ['equal_cs','clip_cs','dino_cs']:passes.append(co[m]['mean_pp']>=.5 and co[m]['ci95_pp'][0]>0)
    for cell,d in cells.items():passes.append(d['comparisons']['equal_cs']['ci95_pp'][0]>=-.5)
    result={'status':'verified','gate_passed':bool(all(passes)),'scope':'auxiliary exposeddevelopment; no prospectivePetsresult','cells':cells,'one_shot_domains':macro,'mask_diagnostics':diagnostics,'uncertainty':'5000pairedtaskbootstrapstratifiedby5fixedseeds; galleryconditions averagedwithin task, notindependentreplicates; conditional on reusedimages','audit_scope':'NumPy64independent192scorecases and12explicitfeaturemaskfits; all2.4millionpredictionvalues schema checked; all600000fixedR2/CSreferencepredictions exactlymatchpriorbank','runtime_seconds':json.loads((OUT/'completion.json').read_text())['elapsed_seconds']}
    s.dump(OUT/'analysis.json',result)
    s.dump(OUT/'independent_audit.json',{'status':'passed','score_cases':len(audit),'max_score_error':max(x['max_error'] for x in audit),'mask_cases':len(mask_audit),'max_mask_error':max(x['error'] for x in mask_audit),'reference_checks':refcheck,'code_input_lock_verified':True,'score_checks':audit,'mask_checks':mask_audit,'verifier_sha256':s.sha(__file__),'analysis_sha256':s.sha(OUT/'analysis.json'),'seconds':time.time()-started})
    print('VERIFIED',json.dumps({'gate':result['gate_passed'],'domain_deltas':{ds:macro[ds]['comparisons']['equal_cs'] for ds in macro},'diagnostics':diagnostics}),flush=True)
if __name__=='__main__':main()
