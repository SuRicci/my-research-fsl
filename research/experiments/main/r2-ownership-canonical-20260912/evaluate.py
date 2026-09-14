"""One-pass ownership of original gallery neighbors; no query/gallery labels enter adaptation."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import torch
import torch.nn.functional as F
import reference_eval as ref
import geometry_reference as geo
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/"protocol.json").read_text());ref.CFG=CFG
torch.set_num_threads(6)
LAMBDAS=[.1,1.,.01,.001]
HEADS=["ownership","count_nearest","count_random","mass_only","exclusive_mass","centered_ownership","r2_tuned","centered_r2","support_tuned"]
def dump(path,obj):path.write_text(json.dumps(obj,indent=2))
def pool(P,candidates,mask,mix):
    n=mask.sum(-1)
    mu=F.normalize((candidates*mask[...,None]).sum(2)/n.clamp_min(1)[...,None],dim=-1)
    alpha=torch.full_like(n,.5,dtype=P.dtype) if mix=="fixed" else .5*n/64
    aug=F.normalize((1-alpha[...,None])*P+alpha[...,None]*mu,dim=-1)
    return torch.where((n>0)[...,None],aug,P)
def allocation(S,Q,G,center=False):
    e,c,k,d=S.shape
    m=S.flatten(1,2).mean(1,keepdim=True)
    A=F.normalize(S.flatten(1,2)-m,dim=-1).reshape_as(S) if center else S
    X=F.normalize(Q-m,dim=-1) if center else Q
    H=F.normalize(G[None]-m,dim=-1) if center else G[None].expand(e,-1,-1)
    P=F.normalize(A.mean(2),dim=-1)
    sim=P@H.transpose(1,2)
    idx=sim.topk(64,dim=-1).indices
    owner=sim.argmax(1)
    mask=owner.gather(1,idx.reshape(e,-1)).reshape_as(idx)==torch.arange(c)[None,:,None]
    cand=H[torch.arange(e)[:,None,None],idx]
    n=mask.sum(-1)
    nearest=torch.arange(64)[None,None,:]<n[...,None]
    priority=torch.randperm(len(G),generator=torch.Generator().manual_seed(CFG["random_control_seed"]))
    ranks=priority[idx].argsort(-1).argsort(-1)
    random=ranks<n[...,None]
    assert torch.equal(nearest.sum(-1),n) and torch.equal(random.sum(-1),n)
    full=torch.ones_like(mask)
    original=pool(P,cand,full,"fixed")
    mass=.5*n[...,None]/64
    mu=F.normalize(cand.mean(2),dim=-1)
    trains={"ownership":pool(P,cand,mask,"fixed"),"count_nearest":pool(P,cand,nearest,"fixed"),
            "count_random":pool(P,cand,random,"fixed"),"mass_only":F.normalize((1-mass)*P+mass*mu,dim=-1),
            "exclusive_mass":pool(P,cand,mask,"count"),"r2_tuned":original,"support_tuned":P}
    return trains,X,{"indices":idx,"mask":mask,"counts":n,"owners":owner}
def scores_for(S,Q,G,lambdas):
    raw,X,info=allocation(S,Q,G)
    centered,Z,ci=allocation(S,Q,G,True)
    raw["centered_r2"]=centered["r2_tuned"];raw["centered_ownership"]=centered["ownership"]
    Y=torch.eye(5)[None].expand(len(S),-1,-1)
    scores={"r2":ref.ridge_scores(raw["r2_tuned"],Y,X,.1),"support_proto":X@raw["support_tuned"].transpose(1,2)}
    for method in HEADS:
        target=Z if method.startswith("centered_") else X
        for lam in lambdas[method]:
            key=method+"_"+str(lam) if len(lambdas[method])>1 else method
            scores[key]=ref.ridge_scores(raw[method],Y,target,lam)
    return scores,info,ci
def run():
    data,manifest=ref.assets();started=time.time();selection={};all_times={}
    for phase in ["dev","eval"]:
        out=HERE/"outputs"/phase;out.mkdir(parents=True,exist_ok=True);start=time.time();summary={};dev={}
        dump(out/"run_manifest.json",{"config":CFG,"phase":phase,"feature_manifest":manifest,
             "code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")},
             "command_argv":[sys.executable,*sys.argv],"cwd":str(Path.cwd()),"torch":torch.__version__,"numpy":np.__version__})
        spec=CFG[phase]
        for name in spec["datasets"]:
            for shot in ([1] if phase=="dev" else [1,5]):
                si,qi,cs,seeds=ref.tasks(data[name]["ident"],name,shot,phase);y=np.repeat(np.arange(5),15)
                for gallery in (spec["gallery_datasets"] if shot==1 else [name]):
                    qhash=set(data[name]["ident"]["query_rgb"])
                    keep=[i for i,h in enumerate(data[gallery]["ident"]["gallery_rgb"]) if h not in qhash]
                    G=data[gallery]["gallery"][keep];cell=name+"_"+gallery+"_k"+str(shot)
                    values={};counts=[];c_counts=[];owners=[];indices=[];masks=[]
                    for st in range(0,len(si),CFG["batch_size"]):
                        stop=st+CFG["batch_size"];S=data[name]["query"][si[st:stop]];Q=data[name]["query"][qi[st:stop].reshape(-1,75)]
                        if shot==1:
                            ll={m:LAMBDAS if phase=="dev" else [selection[m]] for m in HEADS}
                            v,info,ci=scores_for(S,Q,G,ll)
                            counts.append(info["counts"].numpy());c_counts.append(ci["counts"].numpy())
                            indices.append(info["indices"].numpy());masks.append(info["mask"].numpy())
                            if phase=="eval":owners.append(info["owners"].numpy().astype(np.int8))
                        else:
                            base=ref.r2_scores(S,Q,G)
                            v={m:base for m in HEADS};v.update({"r2":base,"support_proto":Q@F.normalize(S.mean(2),dim=-1).transpose(1,2)})
                            v["centered_r2"]=geo.head(geo.prepare(S,Q,G,"support"),1.)
                            v["centered_ownership"]=v["centered_r2"]
                        for method,vv in v.items():
                            assert torch.isfinite(vv).all()
                            values.setdefault(method,[]).append(vv.numpy())
                    names=list(values);sc=np.stack([np.concatenate(values[m]) for m in names])
                    pred=sc.argmax(-1);acc=(pred==y).mean(-1)
                    extra={}
                    if shot==1:
                        extra={"retained_counts":np.concatenate(counts),"centered_retained_counts":np.concatenate(c_counts),
                               "neighbor_indices":np.concatenate(indices),"ownership_mask":np.concatenate(masks)}
                        if owners:extra["gallery_owners"]=np.concatenate(owners)
                    np.savez_compressed(out/(cell+".npz"),scores=sc,predictions=pred,accuracy=acc,names=names,yq=y,
                          support_indices=si,query_indices=qi,class_ids=cs,seeds=seeds,gallery_pool_indices=keep,**extra)
                    summary[ref.metric_key(name,gallery,shot)]={m:100*float(acc[i].mean()) for i,m in enumerate(names)}
                    for i,m in enumerate(names):dev.setdefault(m,[]).append(float(acc[i].mean()))
                    dump(out/"summary.json",summary);print("CELL_COMPLETE",phase,cell,summary[ref.metric_key(name,gallery,shot)],flush=True)
        if phase=="dev":
            means={m:float(np.mean(v)) for m,v in dev.items()}
            selection={m:max(LAMBDAS,key=lambda x:means[m+"_"+str(x)]) for m in HEADS}
            dump(HERE/"selection.json",{"lambdas":selection,"dev_accuracy":means,"locked_before_eval":True,"tie_order":LAMBDAS,
                 "rule":"equal original DTD development-cell accuracy; same4lambda budget per readout"})
            print("SELECTION_FROZEN",selection,flush=True)
        all_times[phase]=time.time()-start
        dump(out/"completion.json",{"status":"completed","cells":len(summary),"elapsed_seconds":all_times[phase]})
    dump(HERE/"outputs/run_completion.json",{"status":"completed","elapsed_seconds":time.time()-started,"phase_seconds":all_times})
    print("RUN_COMPLETE",all_times,flush=True)
if __name__=="__main__":run()
