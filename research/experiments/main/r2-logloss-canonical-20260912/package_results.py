"""Validate paired records and build the baseline/result contracts from measured outputs."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
CFG=json.loads((HERE/"protocol.json").read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def cell_key(cell):
    name,gallery,k=cell.split("_");shot=int(k[1:])
    return name+("_5shot_accuracy" if shot==5 else "_"+("matched" if name==gallery else "mismatched")+"_1shot_accuracy")
def validate(path):
    a=np.load(path);s=a["support_indices"];q=a["query_indices"]
    assert len(s)==1000 and a["scores"].shape[1:]==(1000,75,5)
    assert np.isfinite(a["scores"]).all()
    for si,qi in zip(s,q):assert not np.intersect1d(si,qi).size
    pred=a["scores"].argmax(-1);acc=(pred==a["yq"]).mean(-1)
    assert np.array_equal(pred,a["predictions"]) and np.allclose(acc,a["accuracy"],atol=1e-12)
    return a
def paired_bootstrap(datasets,seed=26091299):
    rng=np.random.RandomState(seed);samples=np.zeros(CFG["bootstrap_replicates"])
    for values,seeds in datasets:
        result=np.zeros(len(samples))
        for seed in np.unique(seeds):
            x=values[seeds==seed]
            idx=rng.randint(len(x),size=(len(samples),len(x)));result+=x[idx].mean(1)/len(np.unique(seeds))
        samples+=result/len(datasets)
    return [float(x) for x in np.percentile(samples,[2.5,97.5])]
def result():
    out=HERE/"outputs/eval";base=Path('/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/outputs/baseline');tables={};raw={};values={}
    for path in sorted(out.glob("*.npz")):
        a=validate(path);b=validate(base/path.name)
        assert np.array_equal(a["support_indices"],b["support_indices"]) and np.array_equal(a["query_indices"],b["query_indices"])
        names=list(a["names"]);bn=list(b["names"]);i=names.index("distribution_logistic");j=names.index("r2")
        assert np.allclose(a["scores"][j],b["scores"][bn.index("r2")],atol=1e-5)
        key=cell_key(path.stem);acc=a["accuracy"]*100
        values[key]=float(acc[i].mean())
        tables[key]={"accuracy_percent":{name:float(acc[k].mean()) for k,name in enumerate(names)},
          "delta_vs_r2_pp":float((acc[i]-acc[j]).mean()),"paired_ci95_pp":paired_bootstrap([(acc[i]-acc[j],a["seeds"])])}
        if "_1shot_" in key:raw[path.stem]=(a,acc)
    # Same query/support tasks are shared by both gallery conditions within each dataset.
    macro={};per_dataset={}
    for dataset in ["dtd","eurosat"]:
        pair=[raw[dataset+"_"+g+"_k1"] for g in ["dtd","eurosat"]]
        a0,a1=pair[0][0],pair[1][0];assert np.array_equal(a0["query_indices"],a1["query_indices"])
        names=list(a0["names"]);assert names==list(a1["names"])
        mean=(pair[0][1]+pair[1][1])/2;per_dataset[dataset]=(mean,a0["seeds"],names)
    names=per_dataset["dtd"][2];ci=names.index("distribution_logistic")
    for method in names:
        mi=names.index(method);parts=[(x[0][ci]-x[0][mi],x[1]) for x in per_dataset.values()]
        macro[method]={"delta_pp":float(np.mean([x.mean() for x,_ in parts])),"paired_ci95_pp":paired_bootstrap(parts)}
    values["macro_1shot_accuracy"]=float(np.mean([v for k,v in values.items() if "_1shot_" in k]))
    main=macro["r2"];worst=min(v["delta_vs_r2_pp"] for k,v in tables.items() if "_1shot_" in k)
    strong={m:macro[m]["paired_ci95_pp"][0]>0 for m in ["r2_tuned","support_logistic","mean_logistic","support_ridge","mean_ridge","support_proto","distribution_ridge"]}
    gate={"positive_r2_macro_ci":main["paired_ci95_pp"][0]>0,"no_cell_loss_above_0_5pp":worst>=-.5,
          "beats_simple_controls":all(strong.values()),"simple_controls":strong}
    gate["passed"]=all(gate[k] for k in ["positive_r2_macro_ci","no_cell_loss_above_0_5pp","beats_simple_controls"])
    report={"per_metric":tables,"macro_vs_controls":macro,"gate":gate,"metrics_summary":values,
       "scope":"Paired locked task draws within historically viewed image pools; seed-stratified bootstrap preserves shared-gallery pairing. No independent-image or SOTA claim."}
    parts=[]
    for mean,seeds,names in per_dataset.values():
        contrast=mean[names.index("distribution_logistic")]-mean[names.index("mean_logistic")]-mean[names.index("distribution_ridge")]+mean[names.index("mean_ridge")]
        parts.append((contrast,seeds))
    report["loss_neighborhood_interaction"]={"delta_pp":float(np.mean([x.mean() for x,_ in parts])),
        "paired_ci95_pp":paired_bootstrap(parts),"interpretation":"Operational contrast between separately development-tuned heads, not a fixed-parameter causal effect."}
    dump(out/"comparison.json",report);dump(out/"metrics_summary.json",values)
    lines=["# Paired experiment result","",report["scope"],"","| Condition | R2 % | Neighborhood logistic % | Delta pp | 95% paired CI pp |","|---|---:|---:|---:|---|"]
    for k,v in tables.items():lines.append("| %s | %.3f | %.3f | %+.3f | [%.3f, %.3f] |"%(k,v["accuracy_percent"]["r2"],v["accuracy_percent"]["distribution_logistic"],v["delta_vs_r2_pp"],*v["paired_ci95_pp"]))
    lines+=["","Decision gate: "+str(gate),"","Macro control comparison:",json.dumps(macro,indent=2)]
    (out/"RESULT.md").write_text("\n".join(lines))
    print(json.dumps({"loss_neighborhood_interaction":report["loss_neighborhood_interaction"],"gate":gate,"macro_vs_controls":macro,"metrics_summary":values},indent=2))

if __name__=="__main__":result()
