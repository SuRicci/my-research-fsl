"""Independent scalar feature/decision reconstruction and paired statistic check."""
from pathlib import Path
import json, hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'
CFG=json.loads((HERE/'protocol.json').read_text())
result=json.loads((OUT/'analysis.json').read_text())
ROOT=HERE.parents[2]
Y=np.repeat(np.arange(5),15)
models={d:json.loads((OUT/(d+'_model.json')).read_text()) for d in CFG['domains']}
def scalar_features(scores):
    out=[]
    for task in range(scores.shape[1]):
        for query in range(75):
            s=scores[:,task,query,:]
            pred=[max(range(5),key=lambda c:float(s[e,c])) for e in range(3)]
            out.append([v for e in range(3) for v in sorted(s[e].tolist())]+[int(pred[0]==pred[1]),int(pred[0]==pred[2]),int(pred[1]==pred[2])])
    return np.array(out)
def audit_bank(path,source):
    z=np.load(path);s=z['scores'];m=models[source];x=scalar_features(s)
    std=(x-np.array(m['mean']))/np.array(m['scale'])
    p=1/(1+np.exp(-(std@np.array(m['coef']).T+np.array(m['intercept']))))
    # Stable explicit scalar tie policy, independent of production advanced indexing.
    choices=np.array([max([2,0,1],key=lambda j:row[j]) for row in p])
    raw=s.argmax(-1);flat=raw.reshape(3,-1)
    selected=np.array([flat[j,i] for i,j in enumerate(choices)]).reshape(raw.shape[1:])
    simple=[]
    for i in range(len(x)):
        margins=[x[i,5*j+4]-x[i,5*j+3] for j in range(3)]
        j=max([2,0,1],key=lambda j:margins[j]);simple.append(flat[j,i])
    assert np.array_equal(selected,z['predictions'][4])
    assert np.array_equal(np.array(simple).reshape(selected.shape),z['predictions'][3])
    assert np.array_equal(choices.reshape(selected.shape),z['choices'])
    assert np.array_equal(raw,z['predictions'][:3])
    accuracy=np.array([[sum(int(v)==int(y) for v,y in zip(row,Y))/75 for row in expert] for expert in z['predictions']])
    assert np.array_equal(accuracy,z['accuracy'])
    return x,accuracy,z['seeds'],int(selected.size)
cases=0;sourcechecks={}
for source in CFG['domains']:
    train=[]
    for phase in ['train','selection']:
        for gallery in ['ordinary','excluded']:
            x,acc,seeds,n=audit_bank(OUT/(source+'_'+phase+'_'+gallery+'.npz'),source);cases+=n
            if phase=='train':train.append(x)
    x=np.concatenate(train);m=models[source]
    assert np.allclose(x.mean(0),m['mean'],atol=1e-10,rtol=0)
    scale=x.std(0);scale[scale==0]=1
    assert np.allclose(scale,m['scale'],atol=1e-10,rtol=0)
    sourcechecks[source]={'scaler_matches_only_source_train':True,'train_rows':len(x)}
vectors=[];groups=[];cellchecks=0
for di,target in enumerate(CFG['domains']):
    source='eurosat' if target=='dtd' else 'dtd';deltas=[]
    for gallery in CFG['domains']:
        name=target+'_'+gallery;x,acc,seeds,n=audit_bank(OUT/(name+'.npz'),source);cases+=n
        for j,method in enumerate(['stack','raw','blend','margin','learned']):
            assert abs(acc[j].mean()*100-result['cells'][name]['accuracy_pct'][method])<1e-10
        deltas.append(acc[4]-acc[2]);cellchecks+=1
    vectors.extend((deltas[0]+deltas[1])/2);groups.extend(seeds+di*100000000)
d=np.array(vectors);g=np.array(groups);rng=np.random.RandomState(26091399);draws=[]
for group in np.unique(g):
    v=d[g==group];draw=[]
    for b in range(5000):
        ids=rng.randint(len(v),size=len(v));draw.append(sum(float(v[i]) for i in ids)/len(v))
    draws.append(draw)
ci=np.percentile(np.mean(draws,axis=0)*100,[2.5,97.5])
assert np.allclose(ci,result['pooled']['blend']['ci95_pp'],atol=1e-10,rtol=0)
assert abs(d.mean()*100-result['pooled']['blend']['delta_pp'])<1e-10
checks={'status':'passed','query_occurrences_independently_reconstructed':cases,'target_cells':cellchecks,
 'source_scalers':sourcechecks,'independent_pooled_delta_pp':float(d.mean()*100),'independent_ci95_pp':ci.tolist(),
 'scope':'All12saved banks; scalar sorted features and tie policy, sigmoid correctness scores, unchanged raw endpoints, every accuracy and independent main paired interval.'}
(OUT/'independent_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks))
