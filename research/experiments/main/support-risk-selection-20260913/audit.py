"""NumPy64 reconstruction; does not call the measured fold/head functions."""
import json, numpy as np
import study as s

def unit(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)
def fuse(a,b,w): return unit(np.concatenate([a*np.sqrt(w),b*np.sqrt(1-w)],-1))
def predict(S,Q,G):
    c,k,d=S.shape
    mean=S.reshape(-1,d).mean(0)
    X=unit(S.reshape(-1,d)-mean); Q=unit(Q-mean)
    if k==1:
        G=unit(G-mean);P=unit(X.reshape(c,k,d).mean(1))
        ix=np.argsort(-(P@G.T),axis=-1,kind='stable')[:,:64]
        X=unit(.5*P+.5*unit(G[ix].mean(1)));Y=np.eye(c)
    else:Y=np.eye(c)[np.repeat(np.arange(c),k)]
    mx=X.mean(0);my=Y.mean(0);A=X-mx
    return (Q-mx)@A.T@np.linalg.solve(A@A.T+.1*np.eye(len(A)),Y-my)+my

def run():
    data,lock=s.load();assert lock==json.loads((s.HERE/'lock.json').read_text())
    cases=[];refchecks=[];scoremax=0.;foldmax=0.;values=0
    for ds in s.CFG['domains']:
        source=next(d for d in s.CFG['domains'] if d!=ds)
        for k in s.CFG['shots']:
            t=s.task(ds,k)
            for gal in s.CFG['domains']:
                z=np.load(s.OUT/f'{ds}_{gal}_k{k}.npz')
                pred=z['predictions'];assert pred.shape==(5,500,75) and pred.min()>=0 and pred.max()<5
                assert np.array_equal(z['selected'],z['risk'].mean(-1).argmin(0))
                assert np.array_equal(z['view_selected'],z['view_risk'].mean(-1).argmin(0))
                assert np.array_equal(pred[3],pred[z['selected'],np.arange(500)])
                assert np.array_equal(pred[4],pred[z['view_selected'],np.arange(500)])
                values+=pred.size
                old=np.load(s.MAIN/'representation-scatter-20260913/outputs/cells'/f'{source}_to_{ds}_{gal}_k{k}.npz')
                for key in ['support_indices','query_indices','class_ids','seeds']:assert np.array_equal(t[key],old[key]),key
                j=old['names'].tolist().index('mean_CS_l2')
                mismatch=int(np.count_nonzero(pred[0]!=old['predictions'][j]))
                refchecks.append({'cell':f'{ds}_{gal}_k{k}','sixview_equal_prediction_mismatches':mismatch,'total':37500})
                assert mismatch==0
                gg,keep=s.gallery(data,ds,gal);assert np.array_equal(keep,z['gallery_pool_indices'])
                gg=[v.double().numpy() for v in gg]
                for ai,index in enumerate(z['audit_indices']):
                    aa=[v[0].double().numpy() for v in s.args(data,ds,t,np.array([index]))]
                    sc,sd,qc,qd=aa
                    for wi,w in enumerate(s.CFG['weights']):
                        gallery=fuse(gg[0].mean(-2),gg[1].mean(-2),w)
                        score=predict(fuse(sc.mean(-2),sd.mean(-2),w),fuse(qc.mean(-2),qd.mean(-2),w),gallery)
                        e=float(np.max(np.abs(score-z['audit_scores'][wi,ai])));scoremax=max(scoremax,e)
                        assert e<3e-5 and np.array_equal(score.argmax(-1),pred[wi,index])
                        fold=[]
                        for f in range(k if k==5 else 6):
                            if k==5:
                                keep=[j for j in range(k) if j!=f]
                                sa,sb=sc[:,keep].mean(-2),sd[:,keep].mean(-2)
                                qa,qb=sc[:,f].mean(-2),sd[:,f].mean(-2)
                            else:
                                keep=[j for j in range(6) if j!=f]
                                sa,sb=sc[:,:,keep].mean(-2),sd[:,:,keep].mean(-2)
                                qa,qb=sc[:,:,f].reshape(5,-1),sd[:,:,f].reshape(5,-1)
                            fs=predict(fuse(sa,sb,w),fuse(qa,qb,w),gallery);fold.append(fs)
                        fold=np.stack(fold);fe=float(np.max(np.abs(fold-z['audit_fold_scores'][wi,ai])));foldmax=max(foldmax,fe)
                        assert fe<3e-5
                        risk=((fold-np.eye(5))**2).mean((-2,-1))
                        assert np.max(np.abs(risk-z['risk'][wi,index]))<3e-6
                        cases.append({'cell':f'{ds}_{gal}_k{k}','task':int(index),'weight':w,'score_error':e,'fold_error':fe})
    report={'status':'passed','prediction_entries':values,'score_cases':len(cases),'score_max_error':scoremax,'fold_max_error':foldmax,'reference_checks':refchecks,'cases':cases,'all_selections_recomputed':True,'scope':'numpy64 reconstruction of72fixed query cases and396fold score matrices; shared audited assetloader; independent head/fold formula'}
    s.dump(s.OUT/'audit.json',report);print('AUDIT_PASS',len(cases),scoremax,foldmax,flush=True)
if __name__=='__main__':run()
