"""Known support-centering transform as a strong comparator; no query-batch adaptation."""
from pathlib import Path
import argparse,json,hashlib,sys,time
import numpy as np
import torch
import torch.nn.functional as F
import reference_eval as ref
HERE=Path(__file__).resolve().parent;CFG=json.loads((HERE/"protocol.json").read_text());ref.CFG=CFG
torch.set_num_threads(6)
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def normalized(x):
    assert float(x.norm(dim=-1).min())>1e-6,"degenerate centered feature"
    return F.normalize(x,dim=-1)
def prepare(S,Q,G,origin):
    e,c,k,d=S.shape
    if origin=="none":A,X,H=S,Q,G[None].expand(e,-1,-1)
    else:
        m=S.flatten(1,2).mean(1,keepdim=True) if origin=="support" else G.mean(0)[None,None].expand(e,-1,-1)
        A=normalized(S.flatten(1,2)-m).reshape_as(S);X=normalized(Q-m)
        H=normalized(G[None]-m) if k==1 else None
    P=normalized(A.mean(2));idx=None;aug=None
    if k==1:
        idx=ref.retrieve(P,G,64) if origin=="none" else (P@H.transpose(1,2)).topk(64,dim=-1).indices
        cand=H[torch.arange(e)[:,None,None],idx]
        mu=normalized(cand.mean(2));aug=normalized(.5*P+.5*mu)
    return A,X,P,aug,idx
def head(pack,lam,retrieval=True):
    A,X,P,aug,idx=pack;e,c,k,d=A.shape
    if retrieval and k==1:Y=torch.eye(c)[None].expand(e,-1,-1);train=aug
    else:Y=F.one_hot(torch.arange(c).repeat_interleave(k),c).float()[None].expand(e,-1,-1);train=A.flatten(1,2)
    return ref.ridge_scores(train,Y,X,lam)
def run(phase):
    data,manifest=ref.assets();spec=CFG["dev" if phase=="dev" else "eval"]
    out=HERE/"outputs"/phase;out.mkdir(parents=True,exist_ok=True);start=time.time();summary={};dev={}
    selection=json.loads((HERE/"selection.json").read_text()) if phase=="eval" else None
    dump(out/"run_manifest.json",{"phase":phase,"config":CFG,"feature_manifest":manifest,
      "code_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"evaluate.py",HERE/"reference_eval.py",HERE/"baseline_methods.py"]},
      "command_argv":[sys.executable,*sys.argv],"cwd":str(Path.cwd()),"torch":torch.__version__,"numpy":np.__version__})
    for name in spec["datasets"]:
        for shot in [1,5]:
            si,qi,cs,seeds=ref.tasks(data[name]["ident"],name,shot,phase);y=np.repeat(np.arange(5),15)
            for gallery in (spec["gallery_datasets"] if shot==1 else [name]):
                G=data[gallery]["gallery"];qhash=set(data[name]["ident"]["query_rgb"])
                keep=[i for i,h in enumerate(data[gallery]["ident"]["gallery_rgb"]) if h not in qhash];G=G[keep]
                all_scores={};retrieval={};cell=name+"_"+gallery+"_k"+str(shot)
                for st in range(0,len(si),CFG["batch_size"]):
                    stop=st+CFG["batch_size"];S=data[name]["query"][si[st:stop]];Q=data[name]["query"][qi[st:stop].reshape(-1,75)]
                    raw=prepare(S,Q,G,"none");center=prepare(S,Q,G,"support")
                    packs={"r2_tuned":(raw,True),"centered_r2":(center,True),"centered_support":(center,False),"support_ridge":(raw,False)}
                    if shot==1:packs["gallery_r2"]=(prepare(S,Q,G,"gallery"),True)
                    v={"r2":ref.r2_scores(S,Q,G),"support_proto":raw[1]@raw[2].transpose(1,2),
                       "centered_proto":center[1]@center[2].transpose(1,2)}
                    assert torch.allclose(head(raw,.1 if shot==1 else 1.),v["r2"],atol=1e-5)
                    for method,(pack,retr) in packs.items():
                        for lam in (CFG["lambda_grid"] if phase=="dev" else [selection["lambdas"][str(shot)][method]]):
                            key=method+"_"+str(lam) if phase=="dev" else method;v[key]=head(pack,lam,retr)
                        if phase=="eval" and pack[4] is not None and method in ["r2_tuned","centered_r2","gallery_r2"]:
                            retrieval.setdefault(method,[]).append(pack[4].numpy())
                    for m,s in v.items():
                        assert torch.isfinite(s).all();all_scores.setdefault(m,[]).append(s.numpy())
                    if st%(10*CFG["batch_size"])==0:print("PROGRESS",phase,cell,st,"/",len(si),flush=True)
                names=list(all_scores);sc=np.stack([np.concatenate(all_scores[m]) for m in names]);pred=sc.argmax(-1);acc=(pred==y).mean(-1)
                extras={"neighbors_"+m:np.concatenate(v) for m,v in retrieval.items()}
                np.savez_compressed(out/(cell+".npz"),scores=sc,predictions=pred,accuracy=acc,names=names,yq=y,
                    support_indices=si,query_indices=qi,class_ids=cs,seeds=seeds,query_pool_ids=data[name]["ident"]["query_ids"],gallery_pool_indices=keep,**extras)
                key=ref.metric_key(name,gallery,shot);summary[key]={m:100*float(acc[i].mean()) for i,m in enumerate(names)}
                for i,m in enumerate(names):dev.setdefault(str(shot),{}).setdefault(m,[]).append(float(acc[i].mean()))
                dump(out/"summary.json",summary);print("CELL_COMPLETE",phase,cell,summary[key],flush=True)
    if phase=="dev":
        means={k:{m:float(np.mean(v)) for m,v in records.items()} for k,records in dev.items()};order=[.1,1,.01,.001];chosen={}
        for shot in ["1","5"]:
            methods=["r2_tuned","centered_r2","centered_support","support_ridge"]+(["gallery_r2"] if shot=="1" else [])
            chosen[shot]={m:max(order,key=lambda lam:means[shot][m+"_"+str(lam)]) for m in methods}
        dump(HERE/"selection.json",{"lambdas":chosen,"dev_accuracy":means,"rule":"per-shot equal DTD development-cell mean, fixed4lambda grid; tie0.1,1,0.01,0.001","locked_before_eval":True})
        print("SELECTION_FROZEN",chosen,flush=True)
    dump(out/"completion.json",{"elapsed_seconds":time.time()-start,"cells":len(summary),"status":"completed"})
    print("PHASE_COMPLETE",phase,"seconds",time.time()-start,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=["dev","eval"],required=True);run(p.parse_args().phase)
