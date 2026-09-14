import numpy as np

def unit(x):
    x=np.asarray(x);return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def variants(go,qo,external=False):
    go,qo=unit(go),unit(qo);mu=go.mean(0)
    x=(go-mu).astype('float64');q=(qo-mu).astype('float64')
    cov=x.T@x/(len(x)-1);e,u=np.linalg.eigh(cov);e=np.maximum(e,0)
    ridge=.1*max(e.mean(),1e-10);w=(e+ridge)**(-.25)
    w=w/np.exp(np.log(w).mean());cut=len(e)-int(np.ceil(.1*len(e)))
    weights={'center':np.ones_like(w),'white':w}
    if not external:
        high=np.ones_like(w);high[cut:]=w[cut:]
        rest=np.ones_like(w);rest[:cut]=w[:cut]
        weights.update(high10=high,rest90=rest)
    zs={};parts={};bands=[np.arange(cut,len(e)),np.arange(0,cut)]
    for n,v in weights.items():
        gg=unit(x@(u*v));qq=unit(q@(u*v));zs[n]=(gg,qq)
        parts[n]=[(qq[:,b],gg[:,b]) for b in bands]
    zs={'raw':(go,qo),**zs}
    return zs,parts,dict(eigenvalues=e.tolist(),scales=w.tolist(),cut=cut,
        high10_variance_fraction=float(e[cut:].sum()/e.sum()),ridge=ridge,
        fitted_on='old gallery only',high_rest_product_equals_white=True)

def rank_all(gg,qq,batch=256):
    rows=[]
    for st in range(0,len(qq),batch):
        scores=np.asarray(qq[st:st+batch]@gg.T,dtype='float32')
        rows.append(np.argsort(-scores,axis=1,kind='stable').astype('uint16'))
    return np.concatenate(rows)
