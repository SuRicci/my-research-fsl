"""Measure new five-shot selector and explicitly inherit locked one-shot predictions."""
from pathlib import Path
import json, hashlib, time, sys
import numpy as np
import torch
import prevalidated as method
import assets_helper as assets
import reference_eval as ref
import statistics_helper as stats

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "protocol.json").read_text())
OUT = HERE / "outputs/eval"
OUT.mkdir(parents=True, exist_ok=True)
assert json.loads((HERE / "validation.json").read_text())["status"] == "passed"
data, manifest = assets.load_pool("original")
start = time.time()
records, metrics, parts = {}, {}, {}
for path in sorted(Path(CFG["geometry_parent_output"]).glob("*.npz")):
    parent = np.load(path)
    dataset, gallery, k = path.stem.split("_")
    shot = int(k[1:])
    names = ["r2", "parent_centered", "raw_tuned", "radius", "preval", "press"]
    pi = list(parent["names"])
    scores = {"r2": parent["scores"][pi.index("r2")], "parent_centered": parent["scores"][pi.index("centered_r2")]}
    extras = {}
    if shot == 1:
        scores.update(raw_tuned=parent["scores"][pi.index("r2_tuned")], radius=scores["parent_centered"],
                      preval=scores["parent_centered"], press=scores["parent_centered"])
        provenance = "One-shot parent policy inherited unchanged; radius/PRESS/PreVal do not operate in this cell."
    else:
        controls = np.load(Path(CFG["scale_parent_output"]) / ("original__" + dataset + "__fusion.npz"))
        assert np.array_equal(parent["support_indices"], controls["support_indices"]) and np.array_equal(parent["query_indices"], controls["query_indices"])
        cn = list(controls["names"])
        scores.update(raw_tuned=controls["scores"][cn.index("raw_tuned")], radius=controls["scores"][cn.index("radius_tuned")])
        collected, diagnostic = {}, {}
        for st in range(0, len(parent["seeds"]), CFG["batch_size"]):
            en = st + CFG["batch_size"]
            S = data[dataset]["fusion"][parent["support_indices"][st:en]]
            Q = data[dataset]["fusion"][parent["query_indices"][st:en].reshape(-1, 75)]
            model = method.fit(S)
            predictions = method.predict(model, Q)
            assert np.max(np.abs(predictions["raw_fixed_check"].numpy() - scores["r2"][st:en])) < 1e-5
            for name in ["preval", "press"]:
                collected.setdefault(name, []).append(predictions[name].numpy())
            for name in ["preval_index", "press_index", "scale", "nll", "mse"]:
                diagnostic.setdefault(name, []).append(model[name].numpy())
        scores.update({name: np.concatenate(v) for name, v in collected.items()})
        extras = {name: np.concatenate(v, axis=1 if name in ["scale", "nll", "mse"] else 0) for name, v in diagnostic.items()}
        provenance = "New support-only fits; raw/global/radius controls from paired scale campaign."
    sc = np.stack([scores[name] for name in names])
    pred = sc.argmax(-1)
    acc = (pred == parent["yq"]).mean(-1) * 100
    key = ref.metric_key(dataset, gallery, shot)
    metrics[key] = float(acc[names.index("preval")].mean())
    contrasts = {}
    for control in ["r2", "parent_centered", "raw_tuned", "radius", "press"]:
        delta = acc[names.index("preval")] - acc[names.index(control)]
        contrasts[control] = {"delta_pp": float(delta.mean()), "paired_ci95_pp": stats.paired_bootstrap([(delta, parent["seeds"])])}
        if shot == 5:
            parts.setdefault(control, []).append((delta, parent["seeds"]))
    record = {"accuracy_percent": {name: float(acc[i].mean()) for i, name in enumerate(names)}, "candidate_vs": contrasts, "provenance": provenance}
    if shot == 5:
        chosen = extras["scale"][extras["preval_index"], np.arange(len(parent["seeds"]))]
        record["selection"] = {"preval_counts": np.bincount(extras["preval_index"], minlength=4).tolist(),
           "press_counts": np.bincount(extras["press_index"], minlength=4).tolist(),
           "scale_zero_count": int((chosen == 0).sum()), "scale_upper_count": int((chosen == 100).sum())}
    records[key] = record
    np.savez_compressed(OUT / path.name, scores=sc, predictions=pred, accuracy_percent=acc, names=names,
        support_indices=parent["support_indices"], query_indices=parent["query_indices"], class_ids=parent["class_ids"], seeds=parent["seeds"], yq=parent["yq"], **extras)
    print("CELL_COMPLETE", key, json.dumps(record), flush=True)
metrics["macro_1shot_accuracy"] = float(np.mean([v for k, v in metrics.items() if "_1shot_" in k]))
macro = {name: {"delta_pp": float(np.mean([d.mean() for d, s in rows])), "paired_ci95_pp": stats.paired_bootstrap(rows)} for name, rows in parts.items()}
gate = {"beats_tuned_raw": macro["raw_tuned"]["paired_ci95_pp"][0] > 0, "beats_radius": macro["radius"]["paired_ci95_pp"][0] > 0,
        "no_five_shot_loss_over_0_5pp": min(v["candidate_vs"]["r2"]["delta_pp"] for k, v in records.items() if "_5shot_" in k) >= -.5}
gate["passed"] = all(gate.values())
result = {"per_metric": records, "metrics_summary": metrics, "macro_5shot_candidate_vs": macro, "gate": gate,
          "elapsed_seconds": time.time() - start, "scope": "Known PreVal-inspired bounded adaptation; old fixed image pools; one-shot gains inherited, not caused by this selector."}
for filename, value in [("comparison.json", result), ("metrics_summary.json", metrics), ("run_manifest.json",
    {"protocol": CFG, "feature_manifest": manifest, "command": [sys.executable, *sys.argv], "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")}, "validation": json.loads((HERE / "validation.json").read_text())})]:
    (OUT / filename).write_text(json.dumps(value, indent=2))
print("PHASE_COMPLETE", json.dumps({"macro_5shot": macro, "gate": gate, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
