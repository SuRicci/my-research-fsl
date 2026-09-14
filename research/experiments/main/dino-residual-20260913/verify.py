"""Independent count-weighted reconstruction of all fixed paired intervals."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent; OUT=HERE/'outputs'; CFG=json.loads((HERE/'protocol.json').read_text())

def reference(delta,seeds):
    rng=np.random.RandomState(CFG['bootstrap_seed']); values=np.zeros(CFG['bootstrap_replicates']); groups=np.unique(seeds)
    for g in groups:
        d=delta[seeds==g]; indices=rng.randint(len(d),size=(len(values),len(d))); counts=np.zeros((len(values),len(d)),dtype=np.int32)
        np.add.at(counts,(np.arange(len(values))[:,None],indices),1); values+=counts@d/len(d)/len(groups)
    return np.r_[delta.mean()*100,np.percentile(values,[2.5,97.5])*100]

def main():
    result=json.loads((OUT/'analysis.json').read_text()); audit=json.loads((OUT/'audit.json').read_text()); banks={}; errors=[]; n=0
    for p,h in audit['output_hashes'].items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    def compare(delta,seeds,want):
        error=float(abs(reference(delta,seeds)-np.r_[want['delta_pp'],want['ci95_pp']]).max()); errors.append(error); assert error<1e-10
    for domain in CFG['domains']:
        arrays=[]; prior=None
        for gallery in CFG['domains']:
            name=domain+'_'+gallery; b=np.load(OUT/(name+'.npz')); scores=b['scores']; assert scores.shape==(7,500,75,5) and np.isfinite(scores).all()
            pred=scores.argmax(-1); acc=(pred==b['yq']).mean(-1); arrays.append(acc); n+=pred.size
            assert b['names'].tolist()==CFG['methods']; assert np.array_equal(pred,b['predictions']) and np.array_equal(acc,b['accuracy'])
            if prior is not None:
                assert all(np.array_equal(prior[k],b[k]) for k in ['support_indices','query_indices','class_ids','seeds','yq'])
            prior=b
            for j in CFG['comparison_indices']: compare(acc[2]-acc[j],b['seeds'],result['cells'][name]['comparisons'][CFG['methods'][j]])
            assert np.max(abs(acc.mean(-1)*100-np.array(list(result['cells'][name]['accuracy_pct'].values()))))<1e-10
            assert result['cells'][name]['repairs']==int(((pred[2]==b['yq'])&(pred[0]!=b['yq'])).sum())
            assert result['cells'][name]['spoils']==int(((pred[2]!=b['yq'])&(pred[0]==b['yq'])).sum())
        avg=np.mean(arrays,0); banks[domain]=(avg,b['seeds'])
        assert np.max(abs(avg.mean(-1)*100-np.array(list(result['domains'][domain]['accuracy_pct'].values()))))<1e-10
        for j in CFG['comparison_indices']: compare(avg[2]-avg[j],b['seeds'],result['domains'][domain]['comparisons'][CFG['methods'][j]])
    acc=np.concatenate([banks[d][0] for d in CFG['domains']],1); groups=np.concatenate([banks[d][1]+i*100000000 for i,d in enumerate(CFG['domains'])])
    assert np.max(abs(acc.mean(-1)*100-np.array(list(result['accuracy_pct'].values()))))<1e-10
    for j in CFG['comparison_indices']: compare(acc[2]-acc[j],groups,result['pooled'][CFG['methods'][j]])
    out={'status':'passed','independent_interval_count':len(errors),'max_interval_error_pp':max(errors),'checked_prediction_entries':n,'bootstrap':'count-weighted; domains+seeds stratified; galleries averaged within episode; conditional development intervals'}
    (OUT/'statistics_audit.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out))
if __name__=='__main__': main()
