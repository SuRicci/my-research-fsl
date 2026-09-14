"""Audit saved outcomes, source choice, paired statistics and direct-primal scores."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
import consistency_model as z
import importlib.util
spec=importlib.util.spec_from_file_location("query_consistency_verify",z.HERE/"verify.py")
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference);dense=reference.dense
HERE=z.HERE;OUT=HERE/'outputs';CFG=json.loads((HERE/'protocol.json').read_text());Y=np.repeat(np.arange(5),15)
NAMES=['incumbent','consistency','isotropic','query_view_mean']
BOOT=json.loads((z.a.PARENT/'protocol.json').read_text())['bootstrap']

def independent_interval(delta,groups):
    rng=np.random.RandomState(BOOT['seed']);values=np.unique(groups);means=np.zeros(BOOT['replicates'])
    for g in values:
        d=delta[groups==g];indices=rng.randint(len(d),size=(len(means),len(d)))
        counts=np.zeros((len(means),len(d)),dtype=np.int32)
        np.add.at(counts,(np.arange(len(means))[:,None],indices),1)
        means+=(counts@d)/len(d)/len(values)
    return np.r_[delta.mean()*100,np.quantile(means,[.025,.975])*100]

def main():
    start=time.time();torch.set_num_threads(6);analysis=json.loads((OUT/'analysis.json').read_text());lock=json.loads((OUT/'selection_lock.json').read_text());source=json.loads((OUT/'source_selection.json').read_text())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in json.loads((HERE/'code_lock.json').read_text()).items())
    rows=0;banks=0
    for p in OUT.glob('*.npz'):
        f=np.load(p);pred=f['scores'].argmax(-1);assert np.array_equal(pred,f['predictions']);assert np.array_equal((pred==Y).mean(-1),f['accuracy']);rows+=pred.size;banks+=1
    assert banks==8
    for d in CFG['domains']:
        counts=[]
        for cond in ['ordinary','excluded']:
            name=d+'_selection_'+cond;f=np.load(OUT/(name+'.npz'));old=z.ROOT/'experiments/main/query-competence-20260913/outputs'/(name+'.npz')
            assert hashlib.sha256(old.read_bytes()).hexdigest()==lock['source_bank_hashes'][name]
            counts.append((f['scores'].argmax(-1)==Y).sum((1,2)))
        total=sum(counts);assert total.tolist()==list(source[d]['correct_counts'].values())
        for family,offset in [('consistency',0),('isotropic',4)]:
            best=int(np.argmax(total[offset:offset+4]));assert CFG['eta_grid'][best]==lock['choices'][d][family+'_eta']
    max_stat=0;checks=0;vectors={}
    for d in CFG['domains']:
        aa=[]
        for g in CFG['domains']:
            f=np.load(OUT/(d+'_'+g+'.npz'));acc=(f['scores'].argmax(-1)==Y).mean(-1);aa.append(acc)
            assert np.max(abs(acc.mean(-1)*100-np.array(list(analysis['cells'][d+'_'+g]['accuracy_pct'].values()))))<1e-10
            vectors['cells/'+d+'_'+g]=(acc,f['seeds'])
        acc=np.mean(aa,0);vectors['domains/'+d]=(acc,f['seeds'])
    acc=np.concatenate([vectors['domains/'+d][0] for d in CFG['domains']],axis=1);groups=np.concatenate([vectors['domains/'+d][1]+i*100000000 for i,d in enumerate(CFG['domains'])]);vectors['pooled']=(acc,groups)
    for key,(acc,groups) in vectors.items():
        reference=analysis['pooled'] if key=='pooled' else analysis[key.split('/')[0]][key.split('/')[1]]['comparisons']
        for j in [0,2,3]:
            got=independent_interval(acc[1]-acc[j],groups);want=reference[NAMES[j]];expected=np.r_[want['delta_pp'],want['ci95_pp']];err=float(abs(got-expected).max());max_stat=max(max_stat,err);assert err<1e-10;checks+=1
    data=z.m.old.load();gm={d:data[d]['gallery'].double().mean(-2) for d in CFG['domains']};scorechecks=[]
    for source,target in [('dtd','eurosat'),('eurosat','dtd')]:
        choice=lock['choices'][source]
        for gallery in CFG['domains']:
            f=np.load(OUT/(target+'_'+gallery+'.npz'))
            for i in CFG['audit_indices']:
                sv=data[target]['query'][f['support_indices'][i]];qv=data[target]['query'][f['query_indices'][i].reshape(-1)];parts=[]
                for x,q,v in z.packs(sv,qv,gm[gallery],CFG['gamma_by_source'][source]):
                    ix=[0,17,74];parts.append(np.stack([dense(x,q[ix],v[ix],0),dense(x,q[ix],v[ix],choice['consistency_eta']),dense(x,q[ix],v[ix],choice['isotropic_eta'],True),dense(x,v[ix].mean(1),v[ix],0)]))
                expected=np.mean(parts,0);actual=f['scores'][:,i][:,[0,17,74]];err=float(abs(actual-expected).max());assert err<1e-9 and np.array_equal(actual.argmax(-1),expected.argmax(-1));scorechecks.append({'target':target,'gallery':gallery,'task':i,'max_error':err})
    result={'status':'passed','all_saved_bank_count':banks,'all_saved_prediction_count':rows,'independent_interval_checks':checks,'max_interval_error_pp':max_stat,'source_integer_selection_verified':True,'direct_primal_task_checks':len(scorechecks),'direct_primal_query_method_scores':len(scorechecks)*3*4,'max_direct_primal_error':max(x['max_error'] for x in scorechecks),'score_checks':scorechecks,'boundary':'Independent NumPy direct-primal head solves on existing verified frozen preprocessing; every saved score-to-prediction/accuracy and all source choices audited. No independent image generalization claim.','seconds':time.time()-start}
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
