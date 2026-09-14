import numpy as np

def per_query_ap(rankings,query_labels,gallery_labels):
    rankings=np.asarray(rankings)
    relevance=np.asarray(gallery_labels)[rankings]==np.asarray(query_labels)[:,None]
    denominator=relevance.sum(1)
    if np.any(denominator==0):raise ValueError('Query has no relevant gallery item')
    precision=np.cumsum(relevance,axis=1)/np.arange(1,rankings.shape[1]+1)
    ap=(precision*relevance).sum(1)/denominator
    return ap,relevance[:,0].astype(float)

def ranks(scores):
    return np.argsort(-np.asarray(scores),axis=1,kind='stable').astype(np.int32)
