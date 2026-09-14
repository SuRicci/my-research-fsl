"""Predeclared paired bootstrap and cumulative qualification gates."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs';CFG=json.loads((HERE/'protocol.json').read_text())

def interval(d,groups):
    rng=np.random.RandomState(CFG['bootstrap']['seed']);values=np.unique(groups);bs=np.zeros(CFG['bootstrap']['replicates'])
    for group in values:
        x=d[groups==group];bs+=x[rng.randint(len(x),size=(len(bs),len(x)))].mean(1)/len(values)
    return {'delta_pp':float(d.mean()*100),'ci95_pp':(np.quantile(bs,[.025,.975])*100).tolist()}

def collect():
    cells={};domains={};vectors={}
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        for shot in [1,5]:
            pieces=[]
            for gallery in [target,source]:
                name=f'{source}_to_{target}_{gallery}_k{shot}';z=np.load(OUT/(name+'.npz'));names=z['names'].tolist();a=(z['predictions']==z['yq']).mean(-1);assert np.array_equal(a,z['accuracy'])
                cell={'accuracy_pct':dict(zip(names,(100*a.mean(-1)).tolist())),'comparisons':{}}
                pairs=[('scatter_cs','mean_cs'),('scatter_cs','scatter_r2'),('scatter_cs','scatter_raw_lam01'),('mean_cs','mean_raw_lam01')]
                for left,right in pairs:
                    key=left+'_minus_'+right;delta=a[names.index(left)]-a[names.index(right)]
                    cell['comparisons'][key]=interval(delta,z['seeds']);vectors['cell/'+name+'/'+key]=(delta,z['seeds'])
                cells[name]=cell;pieces.append((a,z['seeds']))
            a=(pieces[0][0]+pieces[1][0])/2;seeds=pieces[0][1];assert np.array_equal(seeds,pieces[1][1])
            key=target+'_k'+str(shot);comparisons={}
            for left,right in pairs:
                name=left+'_minus_'+right;delta=a[names.index(left)]-a[names.index(right)]
                comparisons[name]=interval(delta,seeds);vectors['domain/'+key+'/'+name]=(delta,seeds)
            # Package interaction at1shot; use sharedlambda0.1 reference at5shot.
            raw='r2' if shot==1 else 'raw_lam01'
            delta=(a[names.index('scatter_cs')]-a[names.index('scatter_'+raw)])-(a[names.index('mean_cs')]-a[names.index('mean_'+raw)])
            comparisons['interaction']=interval(delta,seeds);vectors['domain/'+key+'/interaction']=(delta,seeds)
            domains[key]={'accuracy_pct':dict(zip(names,(100*a.mean(-1)).tolist())),'comparisons':comparisons}
    pooled={}
    for shot in [1,5]:
        groups=[]
        for j,ds in enumerate(['dtd','eurosat']):groups.append(vectors['domain/'+ds+'_k'+str(shot)+'/interaction'][1]+j*100000000)
        for key in domains['dtd_k'+str(shot)]['comparisons']:
            delta=np.concatenate([vectors['domain/'+ds+'_k'+str(shot)+'/'+key][0] for ds in ['dtd','eurosat']]);group=np.concatenate(groups)
            pooled['k'+str(shot)+'/'+key]=interval(delta,group);vectors['pooled/k'+str(shot)+'/'+key]=(delta,group)
    return cells,domains,pooled,vectors

def verdict(cells,domains,pooled):
    gate=CFG['gate'];out={}
    for ref in ['mean_cs','scatter_r2']:
        key='scatter_cs_minus_'+ref;p=pooled['k1/'+key]
        floor=gate['incumbent_pooled_gain_pp'] if ref=='mean_cs' else 0
        tests={'pooled_gain':p['delta_pp']>=floor,'pooled_ci':p['ci95_pp'][0]>0,
          'one_shot_domains':all(domains[d+'_k1']['comparisons'][key]['delta_pp']>=0 and domains[d+'_k1']['comparisons'][key]['ci95_pp'][0]>=-.5 for d in ['dtd','eurosat']),
          'all_cells':all(c['comparisons'][key]['ci95_pp'][0]>=-.5 for c in cells.values())}
        out[ref]={'passed':all(tests.values()),'tests':tests}
    return {'promote_stack':all(x['passed'] for x in out.values()),'by_reference':out}

def main():
    assert json.loads((OUT/'complete.json').read_text())['status']=='completed'
    cells,domains,pooled,_=collect();result={'tier':'auxiliary/dev','task_conditions':4000,'methods':6,'cells':cells,'domains':domains,'pooled':pooled,'gate':verdict(cells,domains,pooled),'scope':'conditional fixed pools and exposeddevelopment; no newPetsresults'}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'domains':domains,'pooled':pooled,'gate':result['gate']},indent=2),flush=True)
if __name__=='__main__':main()
