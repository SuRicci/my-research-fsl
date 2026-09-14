# -*- coding: utf-8 -*-
"""E12 R2 validated 5-way profiles, with no encoder training or query coupling.

Input features must refer to the same images across encoders. The gallery must
already exclude support/query source identities, as in the experiment manifests.
1-shot: equal CLIP/DINO fusion, top-64 retrieval, 0.5 prototype blend, ridge 0.1.
5-shot: equal CLIP/DINO fusion, ridge 1.0 over real support; gallery is not read.
"""
import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F

def _fuse(clip: torch.Tensor,dino: torch.Tensor) -> torch.Tensor:
    if clip.ndim!=2 or dino.ndim!=2 or len(clip)!=len(dino):
        raise ValueError('CLIP and DINO features must be aligned two-dimensional image matrices.')
    if clip.device!=dino.device:raise ValueError('Both encoders must use the same device.')
    if not torch.isfinite(clip).all() or not torch.isfinite(dino).all():raise ValueError('Features must be finite.')
    return F.normalize(torch.cat([F.normalize(clip.float(),dim=-1)/math.sqrt(2),F.normalize(dino.float(),dim=-1)/math.sqrt(2)],dim=-1),dim=-1)

@dataclass
class PreparedGallery:
    features: torch.Tensor

@torch.no_grad()
def prepare_gallery(gallery_clip:torch.Tensor,gallery_dino:torch.Tensor)->PreparedGallery:
    """Fuse and normalize a fixed gallery once, for reuse across many episodes."""
    G=_fuse(gallery_clip,gallery_dino)
    if not len(G):raise ValueError('The gallery must be nonempty.')
    return PreparedGallery(G)

@dataclass
class FittedReadout:
    weights: torch.Tensor
    bias: torch.Tensor
    classes: torch.Tensor
    shot: int
    gallery_rows_read: int
    retrieved_per_class: int

    @torch.no_grad()
    def scores(self,query_clip:torch.Tensor,query_dino:torch.Tensor)->torch.Tensor:
        X=_fuse(query_clip,query_dino)
        if X.device!=self.weights.device:raise ValueError('Queries must use the fitted readout device.')
        return X@self.weights+self.bias

    @torch.no_grad()
    def predict(self,query_clip:torch.Tensor,query_dino:torch.Tensor)->torch.Tensor:
        return self.classes[self.scores(query_clip,query_dino).argmax(-1)]

@torch.no_grad()
def fit_optimized(support_clip:torch.Tensor,support_dino:torch.Tensor,
                  support_labels:torch.Tensor,gallery_clip:Optional[torch.Tensor]=None,
                  gallery_dino:Optional[torch.Tensor]=None,*,
                  prepared_gallery:Optional[PreparedGallery]=None)->FittedReadout:
    """Fit the frozen R2 configuration; no gallery labels or queries are accepted.

    Support labels may be arbitrary integer class IDs. Supported evaluated
    protocols are balanced 5-way 1-shot and balanced 5-way 5-shot.
    """
    S=_fuse(support_clip,support_dino)
    if support_labels.ndim!=1 or len(support_labels)!=len(S):raise ValueError('One label is required per support image.')
    classes,y,counts=torch.unique(support_labels.to(S.device),sorted=True,return_inverse=True,return_counts=True)
    if len(classes)!=5 or not bool((counts==counts[0]).all()) or int(counts[0]) not in (1,5):
        raise ValueError('The validated profile requires balanced 5-way 1-shot or 5-shot supports.')
    shot=int(counts[0]);gallery_rows=0;retrieved=0
    if shot==1:
        if prepared_gallery is not None:
            if gallery_clip is not None or gallery_dino is not None:raise ValueError('Pass a prepared gallery or the two raw gallery matrices, not both.')
            G=prepared_gallery.features
        else:
            if gallery_clip is None or gallery_dino is None:raise ValueError('The 1-shot profile requires both gallery feature matrices.')
            G=prepare_gallery(gallery_clip,gallery_dino).features
        if G.device!=S.device:raise ValueError('Gallery and support must use the same device.')
        if not len(G):raise ValueError('The 1-shot gallery must be nonempty.')
        P=F.normalize(torch.stack([S[y==c].mean(0) for c in range(5)]),dim=-1)
        retrieved=min(64,len(G));idx=(P@G.T).topk(retrieved,dim=-1).indices
        U=F.normalize(G[idx].mean(1),dim=-1)
        A=F.normalize(.5*P+.5*U,dim=-1);Y=torch.eye(5,device=S.device);lam=.1;gallery_rows=len(G)
    else:
        A=S;Y=F.one_hot(y,5).float();lam=1.
    mu=A.mean(0,keepdim=True);ymu=Y.mean(0);centered=A-mu
    coef=torch.linalg.solve(centered@centered.T+lam*torch.eye(len(A),device=A.device),Y-ymu)
    weights=centered.T@coef;bias=ymu-mu[0]@weights
    return FittedReadout(weights,bias,classes,shot,gallery_rows,retrieved)
