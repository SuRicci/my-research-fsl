"""Independent dense covariance, eigensolve and primal ridge validation."""
from pathlib import Path
import json, time
import numpy as np
import torch
import class_metric as m
HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE/'protocol.json').read_text())

def norm(x):
    return x/np.linalg.norm(x, axis=-1, keepdims=True)

def dense_factor(sv, gamma, c, rho):
    r = sv-sv.mean(axis=-2, keepdims=True)
    per = np.array([x.reshape(-1, x.shape[-1]).T@x.reshape(-1, x.shape[-1])/np.prod(x.shape[:-1]) for x in r])
    pooled = per.mean(0)
    trace = np.trace(pooled)
    covariance = (1-rho)*pooled+rho*per[c]
    matrix = np.eye(sv.shape[-1])
    if gamma and trace > 1e-12:
        matrix += gamma*sv.shape[-1]*covariance/trace
    ev, vectors = np.linalg.eigh(matrix)
    return (vectors/np.sqrt(ev))@vectors.T

def independent_head(s, q, g, center):
    if center:
        origin = s.reshape(-1, s.shape[-1]).mean(0)
        s, q, g = [norm(x-origin) for x in [s, q, g]]
    p = norm(s.mean(1))
    ix = np.argsort(-(p@g.T), axis=-1, kind='stable')[:, :64]
    x = norm(0.5*p+0.5*norm(g[ix].mean(1)))
    y = np.eye(len(x))
    xm, ym = x.mean(0), y.mean(0)
    xc = x-xm
    w = np.linalg.solve(xc.T@xc+0.1*np.eye(x.shape[-1]), xc.T@(y-ym))
    return (q-xm)@w+ym, ix

def independent(sv, qv, galleries, gamma, rho):
    result = {g: [] for g in galleries}
    neighbors = {g: [] for g in galleries}
    for c in range(len(sv)):
        transform = dense_factor(sv, gamma, c, rho)
        s, q = norm(sv.mean(-2)@transform), norm(qv.mean(-2)@transform)
        for g, gm in galleries.items():
            gal = norm(gm@transform)
            raw, ri = independent_head(s, q, gal, False)
            cent, ci = independent_head(s, q, gal, True)
            result[g].append((raw+cent)/2)
            neighbors[g].append(np.stack([ri, ci]))
    return {g: (np.array(result[g]), np.array(neighbors[g])) for g in galleries}

def main():
    start = time.time()
    torch.set_num_threads(CFG['resources']['threads'])
    rng = np.random.RandomState(26091337)
    sv = torch.tensor(norm(rng.normal(size=(5,1,6,19))))
    qv = torch.tensor(norm(rng.normal(size=(11,6,19))))
    gal = {'synthetic': torch.tensor(norm(rng.normal(size=(80,19))))}
    cases = [('regular', sv, 2., .5), ('gamma_zero', sv, 0., .5),
             ('shared', sv, 2., 0.), ('zero_scatter', sv[:,:,:1].expand(-1,-1,6,-1), 2., .5)]
    checks = []
    for name, s, gamma, rho in cases:
        actual = m.evaluate(s, qv, gal, gamma, rho)['synthetic']
        expected, ix = independent(s.numpy(), qv.numpy(), {g:x.numpy() for g,x in gal.items()}, gamma, rho)['synthetic']
        error = float(abs(actual[1].numpy()-expected).max())
        assert error < CFG['numeric_tolerance']
        assert np.array_equal(actual[2].numpy(), ix)
        if name != 'regular':
            assert torch.allclose(actual[0][0], actual[0][1], atol=1e-10, rtol=0)
        checks.append({'case':name, 'max_error':error})
    original = m.evaluate(sv, qv, gal, 2.)['synthetic'][0]
    partition = torch.cat([m.evaluate(sv, q, gal, 2.)['synthetic'][0] for q in qv.split(4)], dim=1)
    assert torch.allclose(original, partition, atol=1e-10, rtol=0)
    perm = torch.tensor([2,0,4,1,3])
    renamed = m.evaluate(sv[perm], qv, gal, 2.)['synthetic'][0]
    assert torch.allclose(renamed, original[:,:,perm], atol=1e-9, rtol=0)
    residual = torch.tensor(rng.normal(size=(1,1,6,19))*.1)
    residual -= residual.mean(-2, keepdim=True)
    common = torch.tensor(rng.normal(size=(5,1,1,19)))+residual
    identical = m.evaluate(common, qv, gal, 2.)['synthetic'][0]
    assert torch.allclose(identical[0], identical[1], atol=1e-10, rtol=0)
    data = m.base.old.load()
    gm = {d:data[d]['gallery'].double().mean(-2) for d in CFG['domains']}
    for target in CFG['domains']:
        t = m.base.old.sampler.tasks(data[target]['ident'],1,'eval')
        for i in CFG['audit_indices']:
            s = data[target]['query'][t['support_indices'][i]].double()
            q = data[target]['query'][t['query_indices'][i].reshape(-1)].double()
            gamma = CFG['gamma_by_target'][target]
            actual = m.evaluate(s, q, gm, gamma)
            expected = independent(s.numpy(), q.numpy(), {g:x.numpy() for g,x in gm.items()}, gamma, .5)
            for g in gm:
                bank, ix = expected[g]
                error = float(abs(actual[g][1].numpy()-bank).max())
                assert error < CFG['numeric_tolerance'], (target,i,g,error)
                assert np.array_equal(actual[g][2].numpy(), ix), (target,i,g,'neighbors')
                assoc = np.stack([bank[c,:,c] for c in range(5)],axis=-1)
                ensemble = bank.mean(0)
                foreign = np.stack([np.delete(bank[:, :, c],c,axis=0).mean(0) for c in range(5)],axis=-1)
                for j, score in enumerate([assoc,ensemble,foreign],1):
                    assert np.array_equal(actual[g][0][j].argmax(-1).numpy(), score.argmax(-1))
                checks.append({'target':target,'task':i,'gallery':g,'full_metric_bank_max_error':error})
            print('REAL_CASE_VALIDATED',target,i,flush=True)
    report={'status':'passed','checks':checks,'class_equivariance':True,'query_partition':True,
            'identical_class_scatter_null':True,'numpy':np.__version__,'torch':torch.__version__,
            'elapsed_seconds':time.time()-start,'tolerance':CFG['numeric_tolerance'],
            'scope':'4 synthetic dense cases and4 realtasks x2 galleries x5 metrics, exact neighbors and all3 prediction aggregations'}
    (HERE/'outputs/numeric_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NUMERIC_VALIDATION',json.dumps(report),flush=True)
if __name__ == '__main__':
    main()
