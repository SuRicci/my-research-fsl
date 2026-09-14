"""Count-weighted independent paired interval and full prediction check."""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs';CFG=json.loads((HERE/'protocol.json').read_text())
def interval(d,seeds):
    rng=np.random.RandomState(CFG['bootstrap_seed']);v=np.zeros(CFG['bootstrap_replicates']);groups=np.unique(seeds)
    for group in groups:
        x=d[seeds==group];ix=rng.randint(len(x),size=(len(v),len(x)));counts=np.zeros((len(v),len(x)),dtype=int)
        np.add.at(counts,(np.arange(len(v))[:,None],ix),1);v+=counts@x/len(x)/len(groups)
    return np.r_[d.mean()*100,np.percentile(v,[2.5,97.5])*100]
def main():
    want=json.loads((OUT/'analysis.json').read_text());audit=json.loads((OUT/'audit.json').read_text());errs=[];vectors=[];primary={};n=0
    for f,h in audit['output_hashes'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
    def check(d,seeds,x):
        err=float(abs(interval(d,seeds)-np.r_[x['delta_pp'],x['ci95_pp']]).max());errs.append(err);assert err<1e-10
    for gallery in ['caltech101','dtd','eurosat']:
        b=np.load(OUT/(gallery+'.npz'));pred=b['scores'].argmax(-1);n+=pred.size;assert np.array_equal(pred,b['predictions'])
        acc=(pred==b['yq']).mean(-1);old=(b['old_predictions']==b['yq']).mean(-1);assert np.array_equal(acc,b['accuracy'])
        cell=want['cells'][gallery];assert abs(acc.mean()*100-cell['accuracy_pct'])<1e-10
        assert cell['prediction_changes']==int((pred!=b['old_predictions']).sum())
        vectors.append(acc-old);check(acc-old,b['seeds'],cell['comparisons']['repair_minus_old'])
        for source in ['dtd','eurosat']:
            d=(b[source+'_primary_predictions']==b['yq']).mean(-1)-acc;primary.setdefault(source,[]).append(d)
            check(d,b['seeds'],cell['comparisons'][source+'_primary_minus_repair'])
    check(np.mean(vectors,0),b['seeds'],want['repair_vs_old'])
    for source,d in primary.items():check(np.mean(d,0),b['seeds'],want['sources'][source])
    result={'status':'passed','prediction_entries':n,'interval_count':len(errs),'max_error_pp':max(errs)}
    (OUT/'statistics_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
