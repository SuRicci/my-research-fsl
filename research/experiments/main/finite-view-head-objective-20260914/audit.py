"""Independent NumPy equations, saved selection/prediction and paired-statistic audit."""
from pathlib import Path
import importlib.util,json,sys
import numpy as np
import torch

def normalized(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
def independent(net,p):
    state=net.state_dict();down=state['down.weight'].numpy();up=state['up.weight'].numpy()
    def mapped(x):
        a=x.numpy();return normalized(a+np.maximum(a@down.T,0)@up.T)
    x,q=[mapped(a) for a in p];values=[]
    for xx,qq in zip(x,q):
        heads=[]
        for a,b in zip(xx,qq):
            xm=a.mean(0);xc=a-xm;y=np.eye(5);yc=y-y.mean(0)
            w=xc.T@np.linalg.solve(xc@xc.T+.1*np.eye(5),yc)
            heads.append((b-xm)@w+y.mean(0))
        values.append(np.mean(heads,0))
    return np.array(values)
def ce(sc,y):
    logits=10*sc;mx=logits.max(-1,keepdims=True)
    logz=np.log(np.exp(logits-mx).sum(-1))+mx.squeeze(-1)
    return logz-np.take_along_axis(logits,np.broadcast_to(y,logits.shape[:-1])[...,None],-1).squeeze(-1)
def precheck(s,data):
    torch.manual_seed(2614);net=s.Map(7,3)
    with torch.no_grad():net.up.weight.normal_(0,.02)
    p=(torch.randn(2,2,5,7,dtype=torch.double),torch.randn(2,2,525,7,dtype=torch.double))
    actual=s.predict(net,p);expected=independent(net,p);error=float(abs(actual.detach().numpy()-expected).max());assert error<1e-10
    mm=ce(expected[:,:75],s.Y);vv=ce(expected[:,75:].reshape(2,75,6,5),s.Y[:,None])
    manual=[mm.mean(),vv.mean(),vv.max(-1).mean()];gradients=[]
    for arm,value in zip(s.ARMS,manual):
        loss=s.objective(actual,arm);assert abs(float(loss)-value)<1e-10
        f=lambda scores:s.objective(scores,arm)
        # Compare one directional derivative on scores; max arm is away from ties.
        score=actual.detach().requires_grad_(True);direction=torch.randn_like(score);direction/=direction.norm()
        derivative=(torch.autograd.grad(f(score),score)[0]*direction).sum()
        eps=1e-5;fd=(f(score+eps*direction)-f(score-eps*direction))/(2*eps)
        err=float(abs(derivative-fd));assert err<1e-7
        net.zero_grad();s.objective(s.predict(net,p),arm).backward()
        assert all(torch.isfinite(x.grad).all() and x.grad.norm()>0 for x in net.parameters())
        gradients.append({'arm':arm,'finite_difference_error':err})
    # Independent primal normal equations on the small nonidentity network.
    xx=net(p[0]).detach().numpy();qq=net(p[1]).detach().numpy();dense=[]
    for a,b in zip(xx,qq):
        h=[]
        for x,q in zip(a,b):
            xc=x-x.mean(0);yc=np.eye(5)-.2
            w=np.linalg.solve(xc.T@xc+.1*np.eye(7),xc.T@yc);h.append((q-x.mean(0))@w+.2)
        dense.append(np.mean(h,0))
    primal_error=float(abs(np.array(dense)-expected).max());assert primal_error<1e-10
    checks=[]
    for ds in s.CFG['domains']:
        for role in ['train','selection']:
            banks=s.source_banks(data,ds,role);gm=data[ds]['gallery'].double().mean(-2);x=data[ds]['query']
            for j,b in enumerate(banks):
                i=0;pp=tuple(v[None] for v in s.frozen_pack(x[b['support_indices'][i]],x[b['query_indices'][i].reshape(-1)],gm[b['gallery_indices'][i]],s.CFG['gamma'][ds]))
                null=s.Map();calc=s.predict(null,pp).detach().numpy()[0];old=b['scores'][b['names'].tolist().index('blend'),i]
                err=float(abs(calc[:75]-old).max());assert err<1e-10 and np.array_equal(calc[:75].argmax(-1),old.argmax(-1))
                np_score=independent(null,pp)[0];ie=float(abs(np_score-calc).max());assert ie<1e-10
                # Query rows cannot affect each other; view/class permutation laws.
                perm=torch.arange(525-1,-1,-1);reverse=s.predict(null,(pp[0],pp[1][:,:,perm])).detach().numpy()[0]
                assert np.allclose(reverse[::-1],calc,atol=1e-12,rtol=0)
                cp=torch.tensor([2,4,0,1,3]);cl=s.predict(null,(pp[0][:,:,cp],pp[1])).detach().numpy()[0]
                assert np.allclose(cl,calc[:,cp],atol=1e-12,rtol=0)
                checks.append({'source':ds,'role':role,'gallery':j,'null_error':err,'independent_error':ie})
    return {'status':'passed','synthetic_nonidentity_dual_error':error,'synthetic_primal_error':primal_error,'objective_gradients':gradients,'real_checks':checks,'train_selection_disjoint':True,'torch':torch.__version__,'numpy':np.__version__,'threads':torch.get_num_threads(),'device':'cpu','dtype':'float64'}

def intervals(delta,seeds,cfg):
    rng=np.random.RandomState(cfg['seed']);n=cfg['replicates'];sample=np.zeros(n)
    for seed in np.unique(seeds):
        x=delta[seeds==seed];ix=rng.randint(0,len(x),size=(n,len(x)))
        sample+=x[ix].sum(axis=1)/len(delta)
    return [float(delta.mean()*100),*np.percentile(sample*100,[2.5,97.5]).tolist()]
def main():
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('finite_study',here/'study.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
    torch.set_num_threads(s.CFG['resources']['threads']);choices=json.loads((s.OUT/'selection_lock.json').read_text())
    assert len(choices)==18 and json.loads((s.OUT/'complete.json').read_text())['status']=='computed'
    hist=0
    for e in choices.values():
        rows=e['history'];rank=[]
        for row in rows:
            f=np.load(row['path']);correct=int((f['predictions']==s.Y).sum());loss=float(f['query_ce'].mean())
            assert correct==row['correct'] and abs(loss-row['ce'])<1e-12
            rank.append((correct,-loss,-row['step']));hist+=1
        assert e['step']==rows[max(range(len(rank)),key=lambda i:rank[i])]['step']
        assert s.sha(e['path'])==e['sha256']
    data=s.z.m.old.load();reconstruction=[];full_selected_predictions=0
    for ds in s.CFG['domains']:
        p=s.prepare_source(data,ds,'selection')
        for key,e in choices.items():
            if e['source']!=ds:continue
            net=s.Map();net.load_state_dict(torch.load(e['path'],weights_only=True));sc=s.batched(net,p)
            row=next(r for r in e['history'] if r['step']==e['step']);f=np.load(row['path'])
            assert np.array_equal(sc.argmax(-1),f['predictions']);full_selected_predictions+=f['predictions'].size
            assert np.max(abs(ce(sc,s.Y)-f['query_ce']))<1e-10
            pp=p.get([0,99,199]);np_sc=independent(net,pp);torch_sc=s.predict(net,pp).detach().numpy()
            err=float(abs(np_sc-torch_sc).max());assert err<1e-10
            reconstruction.append({'configuration':key,'max_error':err})
        del p
    stats=json.loads((s.OUT/'analysis.json').read_text());statchecks=0;prediction_count=0;targetchecks=[]
    for target in s.CFG['domains']:
        source=next(d for d in s.CFG['domains'] if d!=target);aa=[];rr=[]
        for gallery in s.CFG['domains']:
            f=np.load(s.OUT/f'{target}_{gallery}.npz');ref=np.load(s.REF/f'{target}_{gallery}.npz');prediction_count+=f['predictions'].size
            assert f['predictions'].shape==(3,3,500,75)
            for k in s.KEYS:assert np.array_equal(f[k],ref[k])
            assert np.array_equal(f['audit_scores'].argmax(-1),f['predictions'][:,:,s.CFG['audit_indices']])
            a=(f['predictions']==s.Y).mean(-1);zero=(f['identity_predictions']==s.Y).mean(-1)
            consistency=(ref['predictions'][ref['names'].tolist().index('consistency')]==s.Y).mean(-1)
            aa.append(a);rr.append(np.stack([zero,consistency]))
            row=stats['cells'][target+'_'+gallery];assert np.max(abs(np.asarray(row['accuracy_pct'])-100*a.mean(-1)))<1e-10
            inp=data[target]['query'];gm=data[gallery]['gallery'].double().mean(-2)
            i=99;j=s.CFG['audit_indices'].index(i)
            pp=tuple(v[None] for v in s.frozen_pack(inp[f['support_indices'][i]],inp[f['query_indices'][i].reshape(-1)],gm,s.CFG['gamma'][source],False))
            for ia,arm in enumerate(s.ARMS):
                for iz,seed in enumerate(s.CFG['seeds']):
                    e=choices[f'{source}_{seed}_{arm}'];net=s.Map();net.load_state_dict(torch.load(e['path'],weights_only=True))
                    ns=independent(net,pp)[0];err=float(abs(ns-f['audit_scores'][ia,iz,j]).max());assert err<1e-10
                    assert np.array_equal(ns.argmax(-1),f['predictions'][ia,iz,i]);targetchecks.append(err)
        a=np.mean(aa,0);r=np.mean(rr,0);avg=a.mean(1)
        comparisons={s.ARMS[k]:avg[2]-avg[k] for k in [0,1]};comparisons.update({n:avg[2]-r[k] for k,n in enumerate(['scatter','consistency'])})
        for name,delta in comparisons.items():
            expected=intervals(delta,f['seeds'],s.CFG['bootstrap']);actual=stats['domains'][target]['comparisons'][name]
            assert np.max(abs(np.array(expected)-np.array([actual['delta_pp'],*actual['ci95_pp']])))<1e-10;statchecks+=1
    assert s.immutable()==json.loads((s.OUT/'code_lock.json').read_text())
    result={'status':'passed','checkpoint_rank_checks':hist,'full_selected_source_prediction_checks':full_selected_predictions,'eval_prediction_arithmetic_checks':prediction_count,'independent_selected_model_checks':reconstruction,'independent_target_model_checks':len(targetchecks),'max_target_error':max(targetchecks),'independent_primary_intervals':statchecks,'scope':'all saved predictions/statistics; independent equations on selected source/target samples; conditional development only'}
    s.dump('audit.json',result);print('AUDIT_PASSED',json.dumps(result),flush=True)
if __name__=='__main__':main()
