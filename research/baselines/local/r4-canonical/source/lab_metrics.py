"""Official revisited AP convention and class retrieval metrics; evaluator-only truth."""
import numpy as np

def revisited_ap(positive_ranks,npositive):
    r=np.sort(np.asarray(positive_ranks,dtype=int))
    if npositive==0:return float('nan')
    if not len(r):return 0.
    ap=0.
    for j,rank in enumerate(r):
        before=1. if rank==0 else j/float(rank)
        after=(j+1)/float(rank+1)
        ap+=(before+after)/2/npositive
    return float(ap)

def evaluate_rank(rank,truth,protocol='class'):
    rank=np.asarray(rank);positive=np.asarray(truth['positive'],dtype=int);ignore=np.asarray(truth.get('ignore',[]),dtype=int)
    keep=rank[~np.isin(rank,ignore)];positive_positions=np.flatnonzero(np.isin(keep,positive))
    if protocol=='revisited':ap=revisited_ap(positive_positions,len(positive))
    else:ap=float(np.sum(np.arange(1,len(positive_positions)+1)/(positive_positions+1))/len(positive)) if len(positive) else float('nan')
    result={'ap':ap,'positive_ranks':positive_positions.tolist(),'positive_count':len(positive),
        'ignore_ranks':np.flatnonzero(np.isin(rank,ignore)).tolist()}
    for k in [1,5,10,100]:result[f'recall_at_{k}']=float(np.any(positive_positions<k))
    # RevisitOP mP@k uses min(last-positive-rank+1,k), not a universal k denominator.
    for k in [1,5,10]:
        if not len(positive_positions):result[f'precision_at_{k}']=float('nan') if not len(positive) else 0.
        else:
            effective=min(int(positive_positions[-1])+1,k)
            result[f'precision_at_{k}']=float(np.sum(positive_positions<effective)/effective)
    return result
