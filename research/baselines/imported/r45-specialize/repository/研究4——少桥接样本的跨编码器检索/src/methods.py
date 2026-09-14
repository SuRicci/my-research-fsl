"""Feature-only compatibility estimators. No file, image, label or oracle access."""
from __future__ import annotations
import time
import numpy as np


def unit(x):
    x=np.asarray(x,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('Non-finite features')
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def local_weights(points,anchors,k=8,temperature=.05):
    similarity=points@anchors.T
    k=min(k,len(anchors))
    indices=np.argsort(-similarity,axis=1,kind='stable')[:,:k]
    values=np.take_along_axis(similarity,indices,axis=1)
    values=np.exp((values-values.max(1,keepdims=True))/temperature)
    values/=values.sum(1,keepdims=True)
    weights=np.zeros_like(similarity)
    np.put_along_axis(weights,indices,values,axis=1)
    return weights,similarity.max(1)


def procrustes_scores(go,qn,ao,an):
    d=max(ao.shape[1],an.shape[1])
    old=np.pad(ao,((0,0),(0,d-ao.shape[1])))
    new=np.pad(an,((0,0),(0,d-an.shape[1])))
    u,_,vt=np.linalg.svd(new.T@old,full_matrices=False)
    transformed=np.pad(qn,((0,0),(0,d-qn.shape[1])))@(u@vt)
    return transformed@np.pad(go,((0,0),(0,d-go.shape[1]))).T


def ridge_scores(go,qn,ao,an,lambdas):
    mean_x,mean_y=an.mean(0),ao.mean(0)
    x,y=an-mean_x,ao-mean_y
    kernel=x@x.T
    eye=np.eye(len(x))
    trials=[]
    for reg in lambdas:
        inverse=np.linalg.inv(kernel+reg*eye)
        hat=kernel@inverse+np.ones_like(kernel)/len(x)
        fitted=hat@ao
        loo=(ao-fitted)/np.maximum(1-np.diag(hat)[:,None],1e-10)
        trials.append((float(np.mean(loo**2)),float(reg)))
    loss,reg=min(trials)
    beta=np.linalg.solve(kernel+reg*eye,y)
    mapped=(qn-mean_x)@x.T@beta+mean_y
    return unit(mapped)@go.T,{'lambda':reg,'loo_feature_mse':loss,'all_bridge_labels_used':False}


def score_all(inputs,config):
    if set(inputs)!={'go','qo','qn','ao','an','fit_count'}:
        raise ValueError('Unexpected method input keys (oracle/labels are forbidden)')
    go,qo,qn,ao,an=[unit(inputs[k]) for k in ('go','qo','qn','ao','an')]
    m=len(ao); fit_count=int(np.asarray(inputs['fit_count']).item())
    if len(an)!=m or m not in config['bridge_budgets'] or not 2<=fit_count<m:
        raise ValueError('Invalid bridge budget/calibration split')
    if len(qo)!=len(qn) or go.shape[1]!=qo.shape[1] or ao.shape[1]!=go.shape[1] or an.shape[1]!=qn.shape[1]:
        raise ValueError('Dimension mismatch')
    result={};diagnostics={'bridge_total':m,'fit':fit_count,'calibration':m-fit_count,'gallery_new_calls':0}
    timings={}
    start=time.perf_counter();old=qo@go.T;result['old_old']=old
    timings['old_old']=time.perf_counter()-start
    start=time.perf_counter();result['procrustes']=procrustes_scores(go,qn,ao,an)
    timings['procrustes']=time.perf_counter()-start
    start=time.perf_counter();result['ridge'],diagnostics['ridge']=ridge_scores(go,qn,ao,an,config['ridge_lambdas'])
    timings['ridge']=time.perf_counter()-start
    start=time.perf_counter()
    gr=go@ao.T; qr=qn@an.T
    result['relative']=unit(qr)@unit(gr).T
    result['relative_row_centered']=unit(qr-qr.mean(1,keepdims=True))@unit(gr-gr.mean(1,keepdims=True)).T
    timings['relative_family']=time.perf_counter()-start
    start=time.perf_counter()
    weights,_=local_weights(go,ao,config['local_k'],config['temperature'])
    delta=qn@an.T-qo@ao.T
    if config.get('center_residual',False):
        # Ranking is invariant to a query-wise additive constant. Gallery-dependent
        # coverage must not turn that unidentifiable offset into a ranking signal.
        delta=delta-delta[:,:fit_count].mean(axis=1,keepdims=True)
    result['local_kernel']=qr@weights.T
    result['residual_all']=old+config['residual_lambda']*(delta@weights.T)
    fit_o,fit_n=ao[:fit_count],an[:fit_count]
    w_fit,gallery_max=local_weights(go,fit_o,config['local_k'],config['temperature'])
    residual=delta[:,:fit_count]@w_fit.T
    result['residual_matched_fit']=old+config['residual_lambda']*residual
    # Coverage is calibrated from leave-one-fit-anchor neighborhood distances.
    aa=fit_o@fit_o.T
    np.fill_diagonal(aa,-np.inf)
    radius=float(np.quantile(1-aa.max(1),config['coverage_quantile']))
    distance=1-gallery_max
    coverage=np.clip((2*max(radius,1e-8)-distance)/max(radius,1e-8),0,1)
    cal_weights,_=local_weights(ao[fit_count:],fit_o,config['local_k'],config['temperature'])
    cal_pred=delta[:,:fit_count]@cal_weights.T
    cal_error=np.mean(np.abs(delta[:,fit_count:]-cal_pred),axis=1)
    # Query rows never interact: each shrinkage uses that query's allowed calibration anchors only.
    shrink=np.maximum(0,np.abs(residual)-config['error_multiplier']*cal_error[:,None])/np.maximum(np.abs(residual),1e-12)
    correction=config['residual_lambda']*coverage[None,:]*shrink*residual
    result['coverage_only']=old+config['residual_lambda']*coverage[None,:]*residual
    result['calibration_only']=old+config['residual_lambda']*shrink*residual
    result['candidate']=old+correction
    timings['local_residual_family']=time.perf_counter()-start
    singular=np.linalg.svd(fit_o-fit_o.mean(0),compute_uv=False)
    positive=singular[singular>max(singular[0]*1e-8,1e-12)]
    diagnostics.update(coverage_radius=radius,mean_gallery_coverage=float(coverage.mean()),
        mean_calibration_error=float(cal_error.mean()),mean_abs_correction=float(np.abs(correction).mean()),
        corrected_fraction=float((np.abs(correction)>1e-8).mean()),effective_anchor_rank=int(len(positive)),
        anchor_condition_number=float(positive[0]/positive[-1]) if len(positive) else None,
        per_query_calibration_error=cal_error.tolist(),per_query_mean_abs_correction=np.abs(correction).mean(1).tolist(),
        timings_seconds=timings)
    for name,scores in result.items():
        if not np.isfinite(scores).all():raise ValueError(f'Non-finite score: {name}')
    return result,diagnostics
