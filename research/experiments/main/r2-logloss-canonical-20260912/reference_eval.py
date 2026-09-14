"""Paired inductive FSL evaluation. No adaptation function receives query/gallery labels."""
from pathlib import Path
import argparse,hashlib,json,time,sys,os
import numpy as np
import torch
import torch.nn.functional as F
from baseline_methods import representation,ridge_scores,evaluate_configs,retrieve
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/"protocol.json").read_text())
torch.set_num_threads(6)

def save_json(path,obj):path.write_text(json.dumps(obj,indent=2))
def weighted_ridge(S,Y,Q,w,lam):
    if w.ndim==1:w=w[None].expand(len(S),-1)
    mass=w.sum(1)[:,None,None];sm=(S*w[:,:,None]).sum(1,keepdim=True)/mass
    ym=(Y*w[:,:,None]).sum(1,keepdim=True)/mass
    A=(S-sm)*w.sqrt()[:,:,None];B=(Y-ym)*w.sqrt()[:,:,None]
    coeff=torch.linalg.solve(A@A.transpose(1,2)+lam*torch.eye(S.shape[1])[None],B)
    return (Q-sm)@A.transpose(1,2)@coeff+ym
def fit_pseudo(S,Q,P,lam):
    e,c,k,d=S.shape;n=P.shape[2]
    X=torch.cat([S,P],dim=2).flatten(1,2)
    Y=F.one_hot(torch.arange(c).repeat_interleave(k+n),c).float()[None].expand(e,-1,-1)
    w=torch.cat([torch.ones(k),torch.full((n,),1/n)]).repeat(c)
    return weighted_ridge(X,Y,Q,w,lam)
def r2_scores(S,Q,G):
    k=S.shape[2]
    config={"name":"r2","family":"retrieval_ridge","r":64,"mix":.5,"lam":.1} if k==1 else {"name":"r2","family":"raw_ridge","lam":1.}
    return evaluate_configs(S,Q,G,[config],return_scores=True)[1]["r2"]
def support_scores(S,Q,lam,mass_match=False):
    e,c,k,d=S.shape;Y=F.one_hot(torch.arange(c).repeat_interleave(k),c).float()[None].expand(e,-1,-1)
    if not mass_match:return ridge_scores(S.flatten(1,2),Y,Q,lam)
    return weighted_ridge(S.flatten(1,2),Y,Q,torch.full((c*k,),(k+1)/k),lam)
def candidates(S,G):
    proto=F.normalize(S.mean(2),dim=-1)
    idx=retrieve(proto,G,64)
    ranks=torch.linspace(0,63,16).long()
    return G[idx[:,:,ranks]]
def residual_scores(S,Q,C,lam):
    centered=C-C.mean(2,keepdim=True)
    return fit_pseudo(S,Q,S.mean(2,keepdim=True)+centered,lam)
def controls(S,Q,G,C,lam,support_lam,rand_ids):
    base=r2_scores(S,Q,G);raw=Q@F.normalize(S.mean(2),dim=-1).transpose(1,2)
    if S.shape[2]==5:return {"r2":base,"residual":base,"support_proto":raw}
    mu=C.mean(2,keepdim=True);Z=C-mu
    trace=(Z.square().sum(-1).mean(2).sum(1))/S.shape[-1]
    iso=torch.cat([support_scores(S[e:e+1],Q[e:e+1],lam+float(trace[e]),True) for e in range(len(S))])
    return {"r2":base,"residual":residual_scores(S,Q,C,lam),
            "support_proto":raw,"support_ridge":support_scores(S,Q,.1),
            "support_tuned":support_scores(S,Q,support_lam),
            "support_mass_matched":support_scores(S,Q,lam,True),
            "isotropic":iso,"mean_only":fit_pseudo(S,Q,mu,lam),
            "distribution":fit_pseudo(S,Q,C,lam),
            "random_residual":residual_scores(S,Q,G[rand_ids],lam)}
def assets():
    root=Path(CFG["asset_root"]);manifest=json.loads((root/"manifest.json").read_text())
    assert "elapsed_seconds" in manifest,"Canonical feature run has not completed"
    data={}
    for name in ["dtd","eurosat"]:
        ident=json.loads((root/(name+"_identities.json")).read_text());data[name]={"ident":ident}
        for side in ["query","gallery"]:
            packs=[]
            for b in ["clip_vitb16","dinov2_vits14"]:
                path=root/(name+"_"+b+"_"+side+".pt")
                assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest["datasets"][name]["backbones"][b+"_"+side]["sha256"]
                p=torch.load(path,weights_only=True)
                assert p["ids"].tolist()==ident[side+"_ids"] and torch.isfinite(p["features"]).all()
                packs.append(p["features"].float())
            data[name][side]=representation(*packs,.5)
    return data,manifest
def tasks(ident,name,shot,phase):
    spec=CFG["dev" if phase=="dev" else "eval"];labels=np.array(ident["query_labels"]);classes=np.unique(labels)
    if name=="dtd":
        perm=np.random.RandomState(CFG["class_split_seed"]).permutation(classes);cut=len(perm)//2
        classes=perm[:cut] if phase=="dev" else perm[cut:]
    pools={int(c):np.flatnonzero(labels==c) for c in classes};si=[];qi=[];cs=[];ss=[]
    for seed in spec["seeds"]:
        rng=np.random.RandomState(seed+shot)
        for _ in range(spec["episodes_per_seed"]):
            selected=rng.choice(classes,5,replace=False)
            idx=np.stack([rng.choice(pools[int(c)],shot+15,replace=False) for c in selected])
            si.append(idx[:,:shot]);qi.append(idx[:,shot:]);cs.append(selected);ss.append(seed)
    return np.stack(si),np.stack(qi),np.stack(cs),np.array(ss)
def metric_key(name,gallery,shot):
    return name+("_5shot_accuracy" if shot==5 else "_"+("matched" if name==gallery else "mismatched")+"_1shot_accuracy")
def run(phase):
    data,manifest=assets();out=HERE/"outputs"/phase;out.mkdir(parents=True,exist_ok=True);started=time.time()
    save_json(out/"run_manifest.json",{"phase":phase,"config":CFG,"feature_manifest":manifest,
        "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"torch":torch.__version__,"numpy":np.__version__,"command_argv":[sys.executable,*sys.argv],"cwd":str(Path.cwd()),"environment":{"PYTORCH_ENABLE_MPS_FALLBACK":os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}})
    summary={};dev_acc={};spec=CFG["dev" if phase=="dev" else "eval"]
    choice=json.loads((HERE/"selection.json").read_text()) if phase=="eval" else None
    for name in spec["datasets"]:
        for shot in ([1] if phase=="dev" else [1,5]):
            si,qi,cs,seeds=tasks(data[name]["ident"],name,shot,phase);y=np.repeat(np.arange(5),15)
            for gallery in (spec["gallery_datasets"] if shot==1 else [name]):
                G=data[gallery]["gallery"];qhash=set(data[name]["ident"]["query_rgb"])
                keep=[i for i,h in enumerate(data[gallery]["ident"]["gallery_rgb"]) if h not in qhash]
                G=G[keep];scores={};cell=name+"_"+gallery+"_k"+str(shot)
                for st in range(0,len(si),CFG["batch_size"]):
                    S=data[name]["query"][si[st:st+CFG["batch_size"]]]
                    Q=data[name]["query"][qi[st:st+CFG["batch_size"]].reshape(-1,75)]
                    C=candidates(S,G) if shot==1 and phase!="baseline" else None
                    if phase=="baseline":v={"r2":r2_scores(S,Q,G),"support_proto":Q@F.normalize(S.mean(2),dim=-1).transpose(1,2)}
                    elif phase=="dev":
                        v={}
                        for lam in CFG["lambda_grid"]:
                            v["residual_"+str(lam)]=residual_scores(S,Q,C,lam)
                            v["support_"+str(lam)]=support_scores(S,Q,lam)
                    else:
                        rng=np.random.RandomState(CFG["bootstrap_seed"]+st)
                        rnd=np.array([[rng.choice(len(G),16,replace=False) for _ in range(5)] for _ in range(len(S))])
                        v=controls(S,Q,G,C,choice["residual_lambda"],choice["support_lambda"],torch.tensor(rnd))
                    for method,values in v.items():scores.setdefault(method,[]).append(values.numpy())
                names=list(scores);sc=np.stack([np.concatenate(scores[m]) for m in names]);pred=sc.argmax(-1);acc=(pred==y).mean(-1)
                np.savez_compressed(out/(cell+".npz"),scores=sc,predictions=pred,accuracy=acc,names=names,yq=y,
                    support_indices=si,query_indices=qi,class_ids=cs,seeds=seeds,query_pool_ids=data[name]["ident"]["query_ids"])
                key=metric_key(name,gallery,shot);summary[key]={m:100*float(acc[i].mean()) for i,m in enumerate(names)}
                if phase=="dev":
                    for i,m in enumerate(names):dev_acc.setdefault(m,[]).append(float(acc[i].mean()))
                print("CELL_COMPLETE",phase,cell,summary[key],flush=True)
    save_json(out/"summary.json",summary)
    if phase=="dev":
        means={m:float(np.mean(v)) for m,v in dev_acc.items()}
        # Deterministic tie break prefers the incumbent regularization strength.
        order=[.1,.03,1]
        chosen={fam+"_lambda":max(order,key=lambda x:means[fam+"_"+str(x)]) for fam in ["residual","support"]}
        chosen.update({"development_accuracy":means,"selection_rule":"maximum equal-cell DTD development mean; ties prefer0.1","locked_before_main":True})
        save_json(HERE/"selection.json",chosen);print("SELECTION_FROZEN",chosen,flush=True)
    if phase=="baseline":
        vals={k:v["r2"] for k,v in summary.items()}
        vals["macro_1shot_accuracy"]=float(np.mean([v for k,v in vals.items() if "_1shot_" in k]))
        save_json(out/"metrics_summary.json",vals)
    print("PHASE_COMPLETE",phase,"seconds",time.time()-started,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=["baseline","dev","eval"],required=True);run(p.parse_args().phase)
