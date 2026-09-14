"""Independently reproduce every reported paired interval using bootstrap counts."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs';CFG=json.loads((HERE/'protocol.json').read_text())

def reference(delta,seeds):
    rng=np.random.RandomState(CFG['bootstrap_seed']);values=np.zeros(CFG['bootstrap_replicates']);groups=np.unique(seeds)
    for g in groups:
        d=delta[seeds==g];indices=rng.randint(len(d),size=(len(values),len(d)));counts=np.zeros((len(values),len(d)),dtype=np.int32)
        np.add.at(counts,(np.arange(len(values))[:,None],indices),1)
        values+=counts@d/len(d)/len(groups)
    return np.r_[delta.mean()*100,np.percentile(values,[2.5,97.5])*100]

def main():
    result=json.loads((OUT/'analysis.json').read_text());banks={};errors=[];n=0
    def compare(delta,seeds,want):
        error=float(abs(reference(delta,seeds)-np.r_[want['delta_pp'],want['ci95_pp']]).max());errors.append(error);assert error<1e-10
    for domain in CFG['domains']:
        arrays=[]
        for gallery in CFG['domains']:
            name=domain+'_'+gallery;b=np.load(OUT/(name+'.npz'));scores=b['scores'];assert scores.shape==(5,500,75,5) and np.isfinite(scores).all()
            pred=scores.argmax(-1);acc=(pred==b['yq']).mean(-1);arrays.append(acc);n+=pred.size
            assert b['names'].tolist()==['parent','patch','fusion','mean_fusion','shuffled_fusion']
            for j,label in [(0,'parent'),(3,'mean_fusion'),(4,'shuffled_fusion')]:compare(acc[2]-acc[j],b['seeds'],result['cells'][name]['comparisons'][label])
            assert np.max(abs(acc.mean(-1)*100-np.array(list(result['cells'][name]['accuracy_pct'].values()))))<1e-10
        avg=np.mean(arrays,0);banks[domain]=(avg,b['seeds'])
        for j,label in [(0,'parent'),(3,'mean_fusion'),(4,'shuffled_fusion')]:compare(avg[2]-avg[j],b['seeds'],result['domains'][domain]['comparisons'][label])
    acc=np.concatenate([banks[d][0] for d in CFG['domains']],1);groups=np.concatenate([banks[d][1]+i*100000000 for i,d in enumerate(CFG['domains'])])
    for j,label in [(0,'parent'),(3,'mean_fusion'),(4,'shuffled_fusion')]:compare(acc[2]-acc[j],groups,result['pooled'][label])
    out={'status':'passed','independent_interval_count':len(errors),'max_interval_error_pp':max(errors),'checked_prediction_entries':n,'bootstrap':'independent count-weighted reconstruction, fixed-pool descriptive95percent intervals'}
    (OUT/'statistics_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
