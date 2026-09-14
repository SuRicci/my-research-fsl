"""Recompute paired outputs against immutable endpoints and predeclared ownership gates."""
from pathlib import Path
import json
import numpy as np
from paired_reference import validate,paired_bootstrap
HERE=Path(__file__).resolve().parent
ROOT=Path("/Users/decoqwq/DeepScientist/quests/012")
PARENT=ROOT/".ds/worktrees/r2-geometry-canonical-20260912/experiments/main/r2-geometry-canonical-20260912/outputs/eval"
def dump(p,x):p.write_text(json.dumps(x,indent=2))
def key(cell):
 n,g,k=cell.split("_")
 return n+("_5shot_accuracy" if k=="k5" else "_"+("matched" if n==g else "mismatched")+"_1shot_accuracy")
out=HERE/"outputs/eval";tables={};raw={};metrics={};endpoint={};support_checks=[]
for p in sorted(out.glob("*.npz")):
 a=validate(p);b=validate(PARENT/p.name);names=list(a["names"]);bn=list(b["names"]);acc=a["accuracy"]*100;pi=names.index("ownership")
 for v in ["support_indices","query_indices","class_ids","seeds"]:assert np.array_equal(a[v],b[v])
 for m in ["r2","centered_r2"]:
  err=float(np.max(np.abs(a["scores"][names.index(m)]-b["scores"][bn.index(m)])))
  endpoint[p.stem+"_"+m]=err;assert err<2e-5
  assert np.array_equal(a["predictions"][names.index(m)],b["predictions"][bn.index(m)])
 metrics[key(p.stem)]=float(acc[pi].mean())
 rows={m:{"accuracy_percent":float(acc[i].mean()),"ownership_minus_control_pp":float((acc[pi]-acc[i]).mean()),
       "paired_ci95_pp":paired_bootstrap([(acc[pi]-acc[i],a["seeds"])])} for i,m in enumerate(names)}
 tables[key(p.stem)]={"methods":rows}
 if p.stem.endswith("k1"):
  assert np.array_equal(a["ownership_mask"].sum(-1),a["retained_counts"])
  expected=np.take_along_axis(a["gallery_owners"],a["neighbor_indices"].reshape(1000,-1),axis=1).reshape(1000,5,64)==np.arange(5)[None,:,None]
  assert np.array_equal(expected,a["ownership_mask"])
  tables[key(p.stem)]["retained_count_mean"]=float(a["retained_counts"].mean())
  tables[key(p.stem)]["empty_pool_fraction"]=float((a["retained_counts"]==0).mean())
  raw[p.stem]=(a,acc)
 else:assert np.array_equal(a["scores"][pi],a["scores"][names.index("r2")])
parts={}
for dataset in ["dtd","eurosat"]:
 a0,ac0=raw[dataset+"_dtd_k1"];a1,ac1=raw[dataset+"_eurosat_k1"]
 assert np.array_equal(a0["query_indices"],a1["query_indices"])
 assert list(a0["names"])==list(a1["names"])
 parts[dataset]=((ac0+ac1)/2,a0["seeds"],list(a0["names"]))
names=parts["dtd"][2];pi=names.index("ownership");macro={}
for m in names:
 mi=names.index(m);ds=[(v[0][pi]-v[0][mi],v[1]) for v in parts.values()]
 macro[m]={"delta_pp":float(np.mean([d.mean() for d,_ in ds])),"paired_ci95_pp":paired_bootstrap(ds)}
metrics["macro_1shot_accuracy"]=float(np.mean([v for k,v in metrics.items() if "_1shot_" in k]))
strong={m:macro[m]["delta_pp"]>=.2 and macro[m]["paired_ci95_pp"][0]>0 for m in ["r2","centered_r2"]}
cell_gate={m:min(t["methods"][m]["ownership_minus_control_pp"] for k,t in tables.items() if "_1shot_" in k)>=-.5 for m in ["r2","centered_r2"]}
attribution={m:macro[m]["paired_ci95_pp"][0]>0 for m in ["count_nearest","count_random","mass_only"]}
gate={"strong_macro":strong,"cell_loss":cell_gate,"ownership_attribution":attribution,
      "passed":all(strong.values()) and all(cell_gate.values()) and all(attribution.values())}
baseline=json.loads((ROOT/"baselines/local/r2-canonical/json/metrics_summary.json").read_text())
assert set(metrics)==set(baseline) and all(np.isfinite(v) for v in metrics.values())
report={"metrics_summary":metrics,"per_metric":tables,"macro_vs_controls":macro,"gate":gate,
        "scope":"Exploratory paired task inference on viewed fixed image pools; shared-gallery pairing preserved. Five-shot primary is inherited R2; no new-method or SOTA inference.",
        "validation":{"reference_endpoint_max_error":max(endpoint.values()),"reference_predictions_equal":True,"task_identity_equal":True,"ownership_recomputed":True,"five_shot_inherited_exact":True,"required_metrics":len(metrics)}}
dump(out/"metrics_summary.json",metrics);dump(out/"comparison.json",report);dump(out/"validation.json",report["validation"])
lines=["# Ownership paired main result","",report["scope"],"","| Condition | Ownership % | R2 % | CS_l2 % |","|---|---:|---:|---:|"]
for k,v in tables.items():lines.append("| %s | %.3f | %.3f | %.3f |"%(k,v["methods"]["ownership"]["accuracy_percent"],v["methods"]["r2"]["accuracy_percent"],v["methods"]["centered_r2"]["accuracy_percent"]))
lines+=["","Gate: "+json.dumps(gate),"","Macro comparisons:",json.dumps(macro,indent=2)]
(out/"RESULT.md").write_text("\n".join(lines))
print("VALIDATED_RESULT",json.dumps({"metrics_summary":metrics,"macro_vs_controls":macro,"gate":gate,"validation":report["validation"]}),flush=True)
