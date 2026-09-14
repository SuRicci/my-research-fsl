"""Descriptive error overlap; no fitted selector, prediction change or accuracy rescue."""
from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];BANK=ROOT/'experiments/main/dino-local-20260913/outputs'
def auc(x,y):
    vals,inv,n=np.unique(x,return_inverse=True,return_counts=True);ranks=(np.cumsum(n)-.5*n+.5)[inv];pos=int(y.sum());neg=len(y)-pos
    return float((ranks[y].sum()-pos*(pos+1)/2)/(pos*neg)) if pos and neg else None
def margin(x):
    z=(x-x.mean(-1,keepdims=True))/np.maximum(x.std(-1,keepdims=True),1e-12);a=np.sort(z,axis=-1);return a[:,-1]-a[:,-2]
def main():
    cells={};inputs={}
    for path in sorted(BANK.glob('*.npz')):
        b=np.load(path);assert b['scores'].shape==(5,500,75,5);assert np.array_equal(b['predictions'],b['scores'].argmax(-1));inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        pred=b['predictions'].reshape(5,-1);truth=np.tile(b['yq'],500);pc=pred[0]==truth;lc=pred[1]==truth;different=pred[0]!=pred[1];eligible=pc^lc;pm=margin(b['scores'][0].reshape(-1,5));lm=margin(b['scores'][1].reshape(-1,5));edges=np.quantile(pm,[0,.25,.5,.75,1]);bins=np.searchsorted(edges[1:-1],pm,side='right');rows=[]
        for i in range(4):
            take=bins==i;rows.append({'parent_margin_quartile':i+1,'n':int(take.sum()),'patch_only_correct':int((take&lc&~pc).sum()),'parent_only_correct':int((take&pc&~lc).sum())})
        assert int((pc&lc).sum()+(~pc&lc).sum()+(pc&~lc).sum()+(~pc&~lc).sum())==len(pc)
        cells[path.stem]={'n':len(pc),'parent_accuracy_pct':float(pc.mean()*100),'patch_accuracy_pct':float(lc.mean()*100),'both_correct':int((pc&lc).sum()),'both_wrong':int((~pc&~lc).sum()),'patch_only_correct':int((~pc&lc).sum()),'parent_only_correct':int((pc&~lc).sum()),'prediction_disagreements':int(different.sum()),'label_oracle_upper_bound_pct':float((pc|lc).mean()*100),'unfitted_margin_difference_auc_on_exactly_one_correct':auc((lm-pm)[eligible],lc[eligible]),'margin_quartiles':rows}
    sums={k:sum(x[k] for x in cells.values()) for k in ['n','both_correct','both_wrong','patch_only_correct','parent_only_correct','prediction_disagreements']}
    result={'status':'completed','parent_report':'report-6eb4cbf9','scope':'post-hoc descriptive diagnostic on repeatedly exposed source predictions','prediction_changes':0,'fitted_parameters':0,'cells':cells,'totals':sums,'input_hashes':inputs,'interpretation_boundary':'Oracle uses ground truth and is not achievable algorithm accuracy. AUC is descriptive only, computed after results on cases whose correctness is known; it does not validate a deployable selector. Shared episodes across galleries are not independent.'}
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
