"""Independent numerical and source-identity checks on saved experimental evidence."""
import json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_e11 as e11
import run_e12 as e12

def main():
    passed=[]
    rng=np.random.default_rng(99)
    S=rng.normal(size=(12,20));X=rng.normal(size=(2,20));y=e11.Y
    Ks=S@S.T;Kq=X@S.T
    v=e11.ridge(Ks,Kq,y,.1)
    assert np.allclose(v,-e11.ridge(Ks,Kq,-y,.1));passed.append('ridge_label_reversal')
    perm=rng.permutation(12)
    assert np.allclose(v,e11.ridge(Ks[np.ix_(perm,perm)],Kq[:,perm],y[perm],.1));passed.append('ridge_support_permutation')
    assert np.allclose(v,np.r_[e11.ridge(Ks,Kq[:1],y,.1),e11.ridge(Ks,Kq[1:],y,.1)]);passed.append('ridge_query_batch_independence')
    for phase in ['dev','retest']:
        files=list((HERE/'e12').glob('*_'+phase+'.npz'))
        for path in files:
            d=np.load(path)
            name=next(n for n in ['cifar_fs','miniimagenet','dtd'] if path.name.startswith(n+'_'))
            hg=np.load(HERE/f'identities_{name}_{"cifar100" if name=="cifar_fs" else name}.npz')['query']
            for s,q,t in zip(d['support_indices'],d['query_indices'],d['true25_indices']):
                sh=set(hg[s.ravel()]);qh=set(hg[q.ravel()]);th=set(hg[t[t>=0]])
                assert not sh&qh and not th&qh and sh<=th
                assert len(sh)==s.size and len(qh)==q.size
            yq=d['yq'];pred=d['predictions'];means=(pred==yq).mean((1,2))
            report=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))['means']
            for n,v in zip(d['config_names'],means):assert abs(report[str(n)]-v)<1e-12
        if files:passed.append(f'{phase}_all_episode_identity_and_reaggregation_{len(files)}_files')
    devpath=HERE/'e12'/'manifest_dev.json';testpath=HERE/'e12'/'manifest_retest.json'
    if devpath.exists() and testpath.exists():
        dev=json.loads(devpath.read_text());test=json.loads(testpath.read_text())
        assert dev.keys()==test.keys()
        for k in dev:assert not set(dev[k]['classes'])&set(test[k]['classes'])
        passed.append('e12_dev_retest_class_disjoint')
    # Compare saved vectorized historical reference against the independently maintained pilot implementation.
    for b in e12.BACKBONES:
        for q,g in [('cifar_fs','cifar100'),('dtd','dtd'),('miniimagenet','miniimagenet')]:
            qs=e12.load_query(q,b); gs,gl,gn=e12.load_gallery(g,b)
            d=np.load(HERE/'e12'/f'{q}_{g}_{b}_k1_dev.npz')
            names=d['config_names'].tolist();si=d['support_indices'][0].ravel();qi=d['query_indices'][0].ravel()
            S=qs[0][[qs[4][int(i)] for i in si]];X=qs[0][[qs[4][int(i)] for i in qi]];Y=torch.arange(5)
            old=json.loads((next(HERE.parent.glob('实验12*'))/'results'/'pilot_main.json').read_text(encoding='utf-8'))
            m=next(m for m in old['metas'] if m['query']==q+'_test' and m['gallery']==g+'_train' and m['backbone']==b and m['shot']==1)
            cfg=e12.augment.AugConfig(**m['dev_best_cfg'])
            for name,fun in [('raw',e12.augment.raw),('legacy_uniform',e12.augment.uniform_aug),('legacy_weighted',e12.augment.weighted_aug)]:
                if name=='raw':v=fun(S,Y,X,5)
                else:v=fun(S,Y,X,5,gs['features'],cfg,hub_vec=gs['hub'])
                assert np.array_equal(v.numpy(),d['predictions'][names.index(name),0]),(q,g,b,name)
    passed.append('18_saved_reference_predictions_match_original_pilot_implementation')
    for split in ['val','test']:
        path=HERE/'e11'/(split+'_predictions.npz')
        if path.exists():
            d=np.load(path);a=((d['scores']>0)==d['yq']).mean(-1);mask=d['clean']
            rows=json.loads((HERE/'e11'/(split+'_summary.json')).read_text())['rows']
            for j,row in enumerate(rows):assert abs(a[mask,j].mean()-row['clean_accuracy'])<1e-12
            passed.append('e11_'+split+'_reaggregation')
    result={'passed':passed,'n_checks':len(passed),'failed':0}
    (HERE/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
