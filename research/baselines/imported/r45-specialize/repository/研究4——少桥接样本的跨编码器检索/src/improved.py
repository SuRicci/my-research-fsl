"""All-anchor residual interpolation with old-space complement preservation."""
import numpy as np
from .methods import unit,local_weights


def affinity_residual(qo,qn,ao,an,standardize=False):
    old=qo@ao.T;new=qn@an.T
    old_center=old-old.mean(1,keepdims=True)
    new_center=new-new.mean(1,keepdims=True)
    if standardize:
        os=old_center.std(1,keepdims=True);ns=new_center.std(1,keepdims=True)
        new_center=new_center*os/np.maximum(ns,1e-10)
        # No variation in new anchor affinities is not evidence to erase old information.
        new_center=np.where(ns>1e-10,new_center,old_center)
    return new_center-old_center


def kernel_matrices(go,ao,kind):
    aa=ao@ao.T;ga=go@ao.T
    if kind=='linear':raw_a,raw_g=aa,ga
    elif kind=='poly2':raw_a,raw_g=(1+aa)**2,(1+ga)**2
    elif kind.startswith('rbf'):
        multiplier=float(kind.split('_')[1]);dist=np.maximum(0,2-2*aa)
        bandwidth=max(float(np.median(dist[np.triu_indices(len(ao),1)]))*multiplier,1e-8)
        raw_a=np.exp(-dist/bandwidth);raw_g=np.exp(-np.maximum(0,2-2*ga)/bandwidth)
    else:raise ValueError(kind)
    # Kernel centering includes an unregularized intercept in PRESS; ranking drops the intercept.
    mean_a=raw_a.mean(0);grand=float(raw_a.mean())
    ka=raw_a-mean_a[None,:]-mean_a[:,None]+grand
    kg=raw_g-raw_g.mean(1,keepdims=True)-mean_a[None,:]+grand
    return ka,kg


def kernel_residual_scores(go,qo,qn,ao,an,kind='linear',standardize=False,regs=(.001,.01,.1,1.,10.),direct=False):
    old=qo@go.T
    delta=affinity_residual(qo,qn,ao,an,standardize)
    if direct:
        old_anchor=qo@ao.T
        delta=delta+old_anchor-old_anchor.mean(axis=1,keepdims=True)
    ka,kg=kernel_matrices(go,ao,kind);m=len(ao)
    eigen,u=np.linalg.eigh((ka+ka.T)/2);eigen=np.maximum(eigen,0)
    scale=max(float(np.trace(ka))/m,1e-8)
    projected=delta@u
    errors=[];predictions=[]
    for coefficient in regs:
        reg=coefficient*scale
        factors=1/(eigen+reg)
        fitted=(projected*(eigen*factors)[None,:])@u.T
        diagonal=(u*u)@(eigen*factors)+1/m
        press=(delta-fitted)/np.maximum(1-diagonal[None,:],1e-8)
        errors.append(np.mean(press**2,axis=1))
        predictions.append((0 if direct else old)+(projected*factors[None,:])@(kg@u).T)
    choices=np.argmin(np.stack(errors,axis=1),axis=1)
    scores=np.stack(predictions,axis=1)[np.arange(len(qo)),choices]
    return scores,{'regularization_choices':np.asarray(regs)[choices].tolist(),
        'press_error':np.stack(errors,axis=1)[np.arange(len(qo)),choices].tolist(),
        'uses_all_bridge_samples':True,'calibration':'analytic leave-one-out of fixed observed residual field; not a confidence guarantee',
        'standardized':standardize,'kernel':kind,'scale':scale,'direct_without_old_complement':direct}


def score_menu(inputs,config):
    if set(inputs)!={'go','qo','qn','ao','an','fit_count'}:raise ValueError('Forbidden input')
    go,qo,qn,ao,an=[unit(inputs[k]) for k in ('go','qo','qn','ao','an')]
    if len(ao)!=len(an) or len(ao) not in config.get('budgets',[32,64]):raise ValueError('Bridge budget mismatch')
    requested=set(config['selected_methods']) if config.get('selected_methods') else None
    old=qo@go.T;results={'old_old':old};diagnostics={'bridge_total':len(ao),'gallery_new_calls':0,'details':{}}
    for normalized in [False,True]:
        suffix='std' if normalized else 'raw'
        delta=affinity_residual(qo,qn,ao,an,normalized)
        for k in [4,8,16,32]:
            for tau in [.03,.05,.1]:
                name=f'control_local_{suffix}_k{k}_t{tau}'
                if requested is not None and name not in requested:continue
                w,_=local_weights(go,ao,k,tau)
                results[name]=old+delta@w.T
        for kind in ['linear','poly2','rbf_0.5','rbf_1.0','rbf_2.0']:
            name=f'candidate_{kind}_{suffix}'
            if requested is None or name in requested:
                results[name],diagnostics['details'][name]=kernel_residual_scores(go,qo,qn,ao,an,kind,normalized,config['ridge_lambdas'])
            direct_name=f'control_direct_{kind}_{suffix}'
            if requested is None or direct_name in requested:
                results[direct_name],diagnostics['details'][direct_name]=kernel_residual_scores(go,qo,qn,ao,an,kind,normalized,config['ridge_lambdas'],direct=True)
    if requested is not None and set(results)!=requested:raise ValueError('Selected method set mismatch')
    return results,diagnostics
