"""Bounded, label-free scale and cross-fitting experiments on paired bridges."""
import numpy as np
from .methods import unit,local_weights,ridge_scores,procrustes_scores
from .improved import kernel_matrices,kernel_residual_scores

REGS=(.001,.01,.1,1.,10.)
CANDIDATES={
 'candidate_global_press':('global','press'),
 'candidate_shrink16_press':('shrink16','press'),
 'candidate_shrink64_press':('shrink64','press'),
 'candidate_mad_press':('mad','press'),
 'candidate_std_cv_mse':('std','cv_mse'),
 'candidate_std_cv_rank':('std','cv_rank'),
 'candidate_shrink16_cv_mse':('shrink16','cv_mse'),
 'candidate_shrink16_cv_rank':('shrink16','cv_rank'),
}

def prior_variance(a):
    sim=a@a.T
    rows=sim[~np.eye(len(a),dtype=bool)].reshape(len(a),len(a)-1)
    return float(np.median(rows.var(axis=1)))

def residual_field(qo,qn,ao,an,mode='std'):
    old=qo@ao.T;new=qn@an.T
    mo=old.mean(1,keepdims=True);mn=new.mean(1,keepdims=True)
    vo=old.var(1,keepdims=True);vn=new.var(1,keepdims=True)
    if mode=='std':ratio=np.sqrt(vo/np.maximum(vn,1e-20))
    elif mode=='raw':ratio=np.ones_like(vo)
    elif mode=='global':ratio=np.full_like(vo,np.sqrt(prior_variance(ao)/max(prior_variance(an),1e-20)))
    elif mode.startswith('shrink'):
        strength=float(mode.removeprefix('shrink'));n=len(ao)-1
        ratio=np.sqrt((n*vo+strength*prior_variance(ao))/np.maximum(n*vn+strength*prior_variance(an),1e-20))
    elif mode=='mad':
        so=np.median(np.abs(old-np.median(old,1,keepdims=True)),1,keepdims=True)
        sn=np.median(np.abs(new-np.median(new,1,keepdims=True)),1,keepdims=True)
        ratio=np.where((so>1e-10)&(sn>1e-10),so/np.maximum(sn,1e-10),np.sqrt(vo/np.maximum(vn,1e-20)))
    else:raise ValueError(mode)
    active=vn>1e-20
    delta=np.where(active,ratio*(new-mn)-(old-mo),0.)
    return delta,{'old_mean':mo,'new_mean':mn,'ratio':ratio,'active':active}

def eigensystem(ka):
    e,u=np.linalg.eigh((ka+ka.T)/2)
    return np.maximum(e,0),u,max(float(np.trace(ka))/len(ka),1e-8)

def press_choices(delta,ka):
    e,u,scale=eigensystem(ka);p=delta@u;errors=[]
    for reg in REGS:
        h=e/(e+reg*scale);fitted=(p*h)@u.T;diag=(u*u)@h+1/len(ka)
        errors.append(np.mean(((delta-fitted)/np.maximum(1-diag,1e-8))**2,axis=1))
    errors=np.stack(errors,1)
    return np.asarray(REGS)[np.argmin(errors,1)],errors

def canonical_folds(ao,an):
    order=np.lexsort(np.concatenate([ao,an],axis=1).T[::-1])
    return [order[i::4] for i in range(4)]

def crossfit_choices(qo,qn,ao,an,mode,loss):
    errors=np.zeros((len(qo),len(REGS)));fold_records=[]
    for held in canonical_folds(ao,an):
        fit=np.setdiff1d(np.arange(len(ao)),held)
        delta,state=residual_field(qo,qn,ao[fit],an[fit],mode)
        ka,kg=kernel_matrices(ao[held],ao[fit],'rbf_0.5');e,u,scale=eigensystem(ka)
        old=qo@ao[held].T
        teacher=np.where(state['active'],state['old_mean']+state['ratio']*(qn@an[held].T-state['new_mean']),old)
        denom=np.maximum(teacher.var(1),1e-12)
        ii,jj=np.triu_indices(len(held),1)
        target_diff=teacher[:,ii]-teacher[:,jj]
        # Fixed top-quarter emphasis; teacher is bridge-new similarity, never class relevance.
        top=np.argsort(-teacher,axis=1,kind='stable')[:,:max(1,len(held)//4)]
        focus=np.zeros_like(teacher,dtype=bool);np.put_along_axis(focus,top,True,axis=1)
        weights=np.abs(target_diff)*np.where(focus[:,ii]|focus[:,jj],1.,.1)
        for j,reg in enumerate(REGS):
            pred=old+((delta@u)/(e+reg*scale))@(kg@u).T
            mse=np.mean((pred-teacher)**2,axis=1)/denom
            if loss=='cv_mse':error=mse
            else:
                margin=(pred[:,ii]-pred[:,jj])*np.sign(target_diff)
                wrong=np.where(margin<0,1.,np.where(margin==0,.5,0.))
                error=np.sum(weights*wrong,1)/np.maximum(weights.sum(1),1e-12)+1e-6*mse
            errors[:,j]+=error/4
        fold_records.append({'fit':len(fit),'held':len(held)})
    return np.asarray(REGS)[np.argmin(errors,1)],errors,fold_records

class ResidualIndex:
    """Gallery preprocessing uses old vectors only; query-specific lambda stays query-side."""
    def __init__(self,go,ao,an):
        self.go=unit(go);self.ao=unit(ao);self.an=unit(an)
        self.ka,kg=kernel_matrices(self.go,self.ao,'rbf_0.5')
        self.e,self.u,self.scale=eigensystem(self.ka)
        self.gtail=kg@self.u

    def query(self,qo,qn,mode='std',selector='press'):
        qo,qn=unit(qo),unit(qn)
        delta,state=residual_field(qo,qn,self.ao,self.an,mode)
        if selector=='fixed':coeff=np.full(len(qo),REGS[0]);errors=None;folds=[]
        elif selector=='press':coeff,errors=press_choices(delta,self.ka);folds=[]
        else:coeff,errors,folds=crossfit_choices(qo,qn,self.ao,self.an,mode,selector)
        tail=(delta@self.u)/(self.e[None,:]+self.scale*coeff[:,None])
        return np.concatenate([qo,tail],1),{'regularization_choices':coeff.tolist(),'scale_ratio':state['ratio'][:,0].tolist(),
            'scale_mode':mode,'selector':selector,'fold_sizes':folds,'gallery_new_calls':0,'queries_independent':True}

    def gallery(self):return np.concatenate([self.go,self.gtail],1)

    def scores(self,qo,qn,mode='std',selector='press'):
        query,info=self.query(qo,qn,mode,selector)
        return query@self.gallery().T,info

def wip_restricted(go,qn,ao,an,ridge=.1):
    """Restricted WIP variant: moments fitted on paired bridges only, including self entries.

    This is not the original large-metric-set or trained-PARAM implementation.
    """
    def whiten(anchors,points):
        rows=anchors@anchors.T;mean=rows.mean(0);centered=rows-mean
        cov=centered.T@centered/max(len(anchors)-1,1)
        e,u=np.linalg.eigh((cov+cov.T)/2);scale=max(np.trace(cov)/len(cov),1e-8)
        transform=(u/np.sqrt(np.maximum(e,0)+ridge*scale))@u.T
        return (points@anchors.T-mean)@transform
    return unit(whiten(an,qn))@unit(whiten(ao,go)).T

def score_bundle(inputs,config):
    if set(inputs)!={'go','qo','qn','ao','an','fit_count'}:raise ValueError('Forbidden worker input')
    go,qo,qn,ao,an=[unit(inputs[k]) for k in ['go','qo','qn','ao','an']]
    if len(ao)!=len(an) or int(inputs['fit_count'])!=len(ao):raise ValueError('All-bridge budget violated')
    old=qo@go.T;index=ResidualIndex(go,ao,an);results={'old_old':old};details={}
    selected=config.get('selected_methods')
    menu={'v0_press':('std','press'),'control_fixed_min':('std','fixed'),**CANDIDATES}
    for name,(mode,selector) in menu.items():
        if selected is None or name in selected:results[name],details[name]=index.scores(qo,qn,mode,selector)
    for name,mode in [('control_local_std','std'),('control_local_shrink16','shrink16')]:
        if selected is None or name in selected:
            w,_=local_weights(go,ao,16,.05);delta,_=residual_field(qo,qn,ao,an,mode);results[name]=old+delta@w.T
    if selected is None or 'control_linear_std' in selected:results['control_linear_std'],_=kernel_residual_scores(go,qo,qn,ao,an,'linear',True)
    if selected is None or 'control_wip_restricted' in selected:results['control_wip_restricted']=wip_restricted(go,qn,ao,an)
    return results,{'bridge_total':len(ao),'gallery_new_calls':0,'details':details,'new_natural_confirmation':False}
