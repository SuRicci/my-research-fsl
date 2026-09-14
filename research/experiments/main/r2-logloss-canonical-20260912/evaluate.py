"""Loss-by-neighborhood comparison using established multinomial logistic regression."""
from pathlib import Path
import argparse,json,hashlib,sys,os,time,warnings
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
import sklearn,threadpoolctl
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import reference_eval as ref
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/"protocol.json").read_text())
ref.CFG=CFG
torch.set_num_threads(1)
METHODS=["distribution_logistic","mean_logistic","support_logistic",
         "distribution_ridge","mean_ridge","support_ridge","r2_tuned"]
NITERS=[]
warnings.simplefilter("error",ConvergenceWarning)
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def training(S,C,kind):
    e,c,k,d=S.shape
    if kind=="support":X=S;weights=torch.ones(c*k)
    else:
        P=C if kind=="distribution" else C.mean(2,keepdim=True)
        n=P.shape[2];X=torch.cat([S,P],dim=2)
        weights=torch.cat([torch.ones(k),torch.full((n,),1/n)]).repeat(c)
    X=X.flatten(1,2)
    labels=np.arange(c).repeat(X.shape[1]//c)
    return X,labels,weights
def logistic_one(args):
    X,y,w,Q,lam=args
    clf=LogisticRegression(C=1/lam,solver="lbfgs",multi_class="multinomial",
                           tol=CFG["solver"]["tol"],max_iter=CFG["solver"]["max_iter"],random_state=0)
    clf.fit(np.asarray(X,dtype=np.float64),y,sample_weight=np.asarray(w,dtype=np.float64))
    scores=clf.decision_function(np.asarray(Q,dtype=np.float64))
    return scores,int(clf.n_iter_.max())
def scores(S,Q,G,C,method,lam):
    if method=="r2_tuned":
        config={"name":"r2_tuned","family":"retrieval_ridge","r":64,"mix":.5,"lam":lam}
        return ref.evaluate_configs(S,Q,G,[config],return_scores=True)[1]["r2_tuned"].numpy()
    kind,loss=method.rsplit("_",1);X,y,w=training(S,C,kind)
    if loss=="ridge":
        Y=F.one_hot(torch.tensor(y),5).float()[None].expand(len(S),-1,-1)
        return ref.weighted_ridge(X,Y,Q,w,lam).numpy()
    args=[(X[i].numpy(),y,w.numpy(),Q[i].numpy(),lam) for i in range(len(S))]
    with ThreadPoolExecutor(max_workers=CFG["solver"]["workers"]) as pool:ans=list(pool.map(logistic_one,args))
    NITERS.extend(n for _,n in ans)
    return np.stack([s for s,_ in ans])
def run(phase):
    data,manifest=ref.assets();out=HERE/"outputs"/phase;out.mkdir(parents=True,exist_ok=True)
    choice=json.loads((HERE/"selection.json").read_text()) if phase=="eval" else None
    spec=CFG["dev" if phase=="dev" else "eval"];started=time.time();summary={};dev={}
    dump(out/"run_manifest.json",{"phase":phase,"config":CFG,"feature_manifest":manifest,
        "code_hashes":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"evaluate.py",HERE/"reference_eval.py",HERE/"baseline_methods.py"]},
        "command_argv":[sys.executable,*sys.argv],"cwd":str(Path.cwd()),"torch":torch.__version__,
        "sklearn":sklearn.__version__,"numpy":np.__version__,"threadpoolctl":threadpoolctl.__version__,"pythonpath":os.environ.get("PYTHONPATH")})
    with threadpool_limits(limits=1):
        for name in spec["datasets"]:
            for shot in ([1] if phase=="dev" else [1,5]):
                si,qi,cs,seeds=ref.tasks(data[name]["ident"],name,shot,phase);y=np.repeat(np.arange(5),15)
                for gallery in (spec["gallery_datasets"] if shot==1 else [name]):
                    G=data[gallery]["gallery"];qhash=set(data[name]["ident"]["query_rgb"])
                    keep=[i for i,h in enumerate(data[gallery]["ident"]["gallery_rgb"]) if h not in qhash];G=G[keep]
                    cell=name+"_"+gallery+"_k"+str(shot);all_scores={}
                    for st in range(0,len(si),CFG["batch_size"]):
                        stop=st+CFG["batch_size"]
                        S=data[name]["query"][si[st:stop]];Q=data[name]["query"][qi[st:stop].reshape(-1,75)]
                        base=ref.r2_scores(S,Q,G).numpy()
                        v={"r2":base,"support_proto":(Q@F.normalize(S.mean(2),dim=-1).transpose(1,2)).numpy()}
                        if shot==5:v["distribution_logistic"]=base
                        else:
                            C=ref.candidates(S,G)
                            for method in METHODS:
                                for lam in (CFG["lambda_grid"] if phase=="dev" else [choice["lambdas"][method]]):
                                    key=method+"_"+str(lam) if phase=="dev" else method
                                    v[key]=scores(S,Q,G,C,method,lam)
                        for m,s in v.items():
                            assert np.isfinite(s).all()
                            all_scores.setdefault(m,[]).append(s.astype(np.float32))
                        if (st//CFG["batch_size"])%10==0:print("PROGRESS",phase,cell,st,"/",len(si),flush=True)
                    names=list(all_scores);sc=np.stack([np.concatenate(all_scores[m]) for m in names])
                    pred=sc.argmax(-1);acc=(pred==y).mean(-1)
                    np.savez_compressed(out/(cell+".npz"),scores=sc,predictions=pred,accuracy=acc,names=names,yq=y,
                        support_indices=si,query_indices=qi,class_ids=cs,seeds=seeds,query_pool_ids=data[name]["ident"]["query_ids"])
                    summary[ref.metric_key(name,gallery,shot)]={m:100*float(acc[i].mean()) for i,m in enumerate(names)}
                    for i,m in enumerate(names):dev.setdefault(m,[]).append(float(acc[i].mean()))
                    print("CELL_COMPLETE",phase,cell,summary[ref.metric_key(name,gallery,shot)],flush=True)
                    dump(out/"summary.json",summary)
    if phase=="dev":
        means={m:float(np.mean(v)) for m,v in dev.items()};order=[.1,1,.01,.001]
        lambdas={m:max(order,key=lambda v:means[m+"_"+str(v)]) for m in METHODS}
        dump(HERE/"selection.json",{"lambdas":lambdas,"dev_mean_accuracy":means,
            "rule":"equal DTD cell mean; identical four-value grid; ties prefer0.1,1,0.01,0.001",
            "locked_before_evaluation":True})
        print("SELECTION_FROZEN",lambdas,flush=True)
    dump(out/"solver_summary.json",{"fits":len(NITERS),"max_iterations_used":max(NITERS,default=0),
        "min_iterations_used":min(NITERS,default=0),"convergence_warnings":0,
        "elapsed_seconds":time.time()-started})
    print("PHASE_COMPLETE",phase,"seconds",time.time()-started,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--phase",choices=["dev","eval"],required=True);run(p.parse_args().phase)
