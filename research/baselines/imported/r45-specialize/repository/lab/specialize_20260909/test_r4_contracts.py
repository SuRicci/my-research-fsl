import numpy as np
from campaign_common import HERE,read
from r4_candidates import run,prepare,unit

def inputs():
    rng=np.random.default_rng(197);return [unit(rng.normal(size=s)) for s in [(140,12),(3,12),(3,8),(20,12),(20,8)]]

def test_query_independence_and_finite():
    go,qo,qn,ao,an=inputs();cfgs=read(HERE/'candidates.json')
    for method,cfg in cfgs.items():
        if not method.startswith('r4_'):continue
        state=prepare(go,qo,method,cfg,'cpu');full,ranks,_,_=run(go,qo,qn,ao,an,method,cfg,state)
        for i in range(len(qo)):
            one=prepare(go,qo[i:i+1],method,cfg,'cpu') if method=='r4_whiten' else state
            scores,r,_,_=run(go,qo[i:i+1],qn[i:i+1],ao,an,method,cfg,one)
            if ranks is None:np.testing.assert_allclose(full[i],scores[0],atol=1e-7);assert np.isfinite(full).all()
            else:np.testing.assert_array_equal(ranks[i],r[0])

def test_zero_bridge_residual_and_fixed_reranking_scope():
    go,qo,_,ao,_=inputs();cfgs=read(HERE/'candidates.json')
    for method in ['r4_rank','r4_diffusion','r4_ensemble','r4_top100']:
        cfg=cfgs[method];state=prepare(go,qo,method,cfg,'cpu');scores,rank,base,_=run(go,qo,qo,ao,ao,method,cfg,state)
        if scores is not None:np.testing.assert_allclose(scores,base['old_centered'],atol=1e-8)
        else:np.testing.assert_array_equal(rank,np.argsort(-base['old_centered'].astype('float32'),axis=1,kind='stable'))
    go,qo,qn,ao,an=inputs();_,rank,base,_=run(go,qo,qn,ao,an,'r4_top100',cfgs['r4_top100'])
    old=np.argsort(-base['old_centered'].astype('float32'),axis=1,kind='stable')
    np.testing.assert_array_equal(rank[:,100:],old[:,100:]);np.testing.assert_array_equal(np.sort(rank[:,:100]),np.sort(old[:,:100]))

def test_ensemble_anchor_order_and_graph_weights():
    x=inputs();go,qo,qn,ao,an=x;cfgs=read(HERE/'candidates.json');cfg=cfgs['r4_ensemble']
    a,_,_,_=run(*x,'r4_ensemble',cfg);b,_,_,_=run(go,qo,qn,ao[::-1],an[::-1],'r4_ensemble',cfg)
    np.testing.assert_allclose(a,b,atol=1e-7)
    p=prepare(go,qo,'r4_diffusion',cfgs['r4_diffusion'],'cpu')['graph'];np.testing.assert_allclose(np.asarray(p.sum(1)),1,atol=2e-7)
