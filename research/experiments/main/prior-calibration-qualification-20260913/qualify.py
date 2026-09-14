"""Paired auxiliary qualification on existing exposed source domains."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import calibrate

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "protocol.json").read_text())
OUT = HERE / "outputs"
BASE = HERE.parent / "r2-geometry-canonical-20260912"
sys.path.insert(0, str(BASE))
import reference_eval as ref
# Geometry is a read-only established comparator. Restore our asset root after import.
sys.path.insert(0, str(HERE.parent / "r2-fusion-stage-canonical-20260912"))
import geometry_reference as geo
ref.CFG = {"asset_root": CFG["asset_root"]}
torch.set_num_threads(CFG["resources"]["cpu_threads"])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    os.replace(temp, path)


def guard():
    import shutil
    assert shutil.disk_usage(HERE).free >= 10 * 2**30
    assert datetime.now(timezone.utc) < datetime.fromisoformat(CFG["resources"]["deadline_utc"])


def tasks(ident, shot, phase):
    labels = np.array(ident["query_labels"])
    pools = {}
    rng = np.random.RandomState(CFG["image_split_seed"])
    for c in np.unique(labels):
        ids = rng.permutation(np.flatnonzero(labels == c))
        if phase == "train":
            ids = ids[:len(ids)//2]
        elif phase == "selection":
            ids = ids[len(ids)//2:]
        pools[int(c)] = ids
    count = shot if phase == "train" else shot + 15
    assert all(len(v) >= count for v in pools.values())
    if phase == "train":
        seeds, n = [CFG["train_seed"]], CFG["train_episodes"]
    elif phase == "selection":
        seeds, n = [CFG["selection_seed"]], CFG["selection_episodes"]
    else:
        seeds, n = CFG["eval_seeds"], CFG["episodes_per_seed"]
    support, query, classes, ss = [], [], [], []
    for seed in seeds:
        rng = np.random.RandomState(seed + shot)
        for _ in range(n):
            cs = rng.choice(sorted(pools), 5, replace=False)
            ix = np.stack([rng.choice(pools[int(c)], count, replace=False) for c in cs])
            support.append(ix[:, :shot]); query.append(ix[:, shot:]); classes.append(cs); ss.append(seed)
    return dict(support_indices=np.stack(support), query_indices=np.stack(query),
                class_ids=np.stack(classes), seeds=np.array(ss))


def logistic(S, Q, C):
    y = np.repeat(np.arange(5), S.shape[2])
    outputs = []
    for s, q in zip(S, Q):
        model = LogisticRegression(C=C, solver="lbfgs", multi_class="multinomial",
                                   max_iter=2000, tol=1e-8, random_state=26091275)
        model.fit(s.flatten(0, 1).numpy(), y)
        outputs.append(model.decision_function(q.numpy()))
    return np.stack(outputs)


def controls(S, Q, G):
    """Every source-selected configuration defines its five-shot fallback in advance."""
    result = {"r2": ref.r2_scores(S, Q, G).numpy()}
    center = geo.prepare(S, Q, G, "support")
    for lam in CFG["source_control_grid"]["lambda"]:
        result["cs_" + str(lam)] = geo.head(center, lam).numpy()
    if S.shape[2] == 1:
        configs = [dict(name="grid_" + str(m) + "_" + str(l), family="retrieval_ridge",
                        r=64, mix=m, lam=l) for m in CFG["source_control_grid"]["mix"]
                   for l in CFG["source_control_grid"]["lambda"]]
        _, scores, _ = ref.evaluate_configs(S, Q, G, configs, return_scores=True)
        result.update({k: v.numpy() for k, v in scores.items()})
    else:
        for m in CFG["source_control_grid"]["mix"]:
            for lam in CFG["source_control_grid"]["lambda"]:
                result["grid_" + str(m) + "_" + str(lam)] = result["r2"]
    for C in CFG["source_control_grid"]["logistic_C"]:
        result["logistic_" + str(C)] = logistic(S, Q, C)
    return result


def train_source(name, data):
    ident = data["ident"]; t = tasks(ident, 1, "train")
    G = data["gallery"]; gy = np.array(ident["gallery_labels"])
    xs, ys = [], []
    for start in range(0, len(t["seeds"]), 8):
        S = data["query"][t["support_indices"][start:start+8]]
        xs.append(calibrate.features(S, G).reshape(-1, 7))
        ys.append(np.stack([np.isin(gy, cs) for cs in t["class_ids"][start:start+8]]).reshape(-1))
    model = calibrate.fit_calibrator(np.concatenate(xs), np.concatenate(ys), CFG["calibration"])
    dump(OUT / ("calibrator_" + name + ".json"), model)
    np.savez_compressed(OUT / ("train_tasks_" + name + ".npz"), **t)
    t = tasks(ident, 1, "selection"); acc = {}
    for start in range(0, len(t["seeds"]), 8):
        S = data["query"][t["support_indices"][start:start+8]]
        Q = data["query"][t["query_indices"][start:start+8].reshape(-1, 75)]
        for key, score in controls(S, Q, G).items():
            acc.setdefault(key, []).extend((score.argmax(-1) == np.repeat(np.arange(5), 15)).mean(-1).tolist())
    means = {key: float(np.mean(value)) for key, value in acc.items()}
    winner = max(means, key=means.get)
    dump(OUT / ("selection_" + name + ".json"), dict(winner=winner, means=means, rule=CFG["source_control_grid"]["selection"]))
    np.savez_compressed(OUT / ("selection_tasks_" + name + ".npz"), **t)
    print("SOURCE_FROZEN", name, "rows", model["train_rows"], "control", winner, flush=True)
    return model, winner


def evaluate_cell(source, target, gallery, shot, data, model, winner):
    t = tasks(data[target]["ident"], shot, "eval")
    X, G = data[target]["query"], data[gallery]["gallery"]
    ident, gident = data[target]["ident"], data[gallery]["ident"]
    assert not set(ident["query_rgb"]) & set(gident["gallery_rgb"])
    name = source + "_to_" + target + "_" + gallery + "_k" + str(shot)
    scores, diagnostic = {}, []
    for start in range(0, len(t["seeds"]), CFG["resources"]["batch_size"]):
        guard(); sl = slice(start, start + CFG["resources"]["batch_size"])
        S = X[t["support_indices"][sl]]
        Q = X[t["query_indices"][sl].reshape(-1, 75)]
        values = controls(S, Q, G)
        if shot == 1:
            prob = calibrate.predict_calibrator(model, calibrate.features(S, G))
            posterior, prior, iterations = calibrate.adapt_prior(prob, CFG["calibration"])
            for key, weights, mode in [("calibrated", posterior, "adaptive"),
                                      ("same_mass", posterior, "same_mass"),
                                      ("composition_only", posterior, "composition"),
                                      ("fixed_prior", prob, "adaptive")]:
                values[key] = calibrate.retrieval_scores(S, Q, G, weights, ref.ridge_scores, ref.retrieve, mode).numpy()
            # Labels enter only this scoring block, never the fitted posterior or classifier.
            actual = np.stack([np.isin(gident["gallery_labels"], cs) if gallery == target
                               else np.zeros(len(G), dtype=bool) for cs in t["class_ids"][sl]])
            for j in range(len(S)):
                y = actual[j]; pp = posterior[j]
                diagnostic.append(dict(prior=float(prior[j]), actual_prior=float(y.mean()),
                    brier=float(np.mean((pp-y)**2)), raw_brier=float(np.mean((prob[j]-y)**2)),
                    auc=float(roc_auc_score(y, pp)) if y.any() and not y.all() else None,
                    mean_posterior=float(pp.mean()), iterations=iterations))
        else:
            for key in ["calibrated", "same_mass", "composition_only", "fixed_prior"]:
                values[key] = values["r2"]
        for key, v in values.items():
            assert np.isfinite(v).all()
            scores.setdefault(key, []).append(v)
        if start % 80 == 0:
            print("EVAL_PROGRESS", name, start, "/", len(t["seeds"]), flush=True)
    keys = list(scores); sc = np.stack([np.concatenate(scores[key]) for key in keys])
    pred = sc.argmax(-1); yq = np.repeat(np.arange(5), 15); acc = (pred == yq).mean(-1)
    path = OUT / "cells" / (name + ".npz"); path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, names=keys, scores=sc, predictions=pred, accuracy=acc, yq=yq,
                        source_winner=winner, **t)
    dump(OUT / "diagnostics" / (name + ".json"), diagnostic)
    print("CELL_COMPLETE", name, "tasks", len(t["seeds"]), flush=True)


def interval(delta, seeds):
    rng = np.random.RandomState(CFG["gate"]["bootstrap_seed"])
    draws = np.zeros(CFG["gate"]["bootstrap_replicates"])
    for seed in np.unique(seeds):
        d = delta[seeds == seed]
        draws += d[rng.randint(len(d), size=(len(draws), len(d)))].mean(1) / len(np.unique(seeds))
    return dict(delta_pp=float(delta.mean()*100), ci95_pp=(np.quantile(draws,[.025,.975])*100).tolist())


def analyze():
    cells = {}; bundles = {}; passed = True
    for path in sorted((OUT/"cells").glob("*.npz")):
        with np.load(path) as z:
            names=z["names"].tolist(); acc=z["accuracy"]; seeds=z["seeds"]; win=str(z["source_winner"])
            assert np.array_equal(z["predictions"],z["scores"].argmax(-1))
            assert np.allclose(acc,(z["predictions"]==z["yq"]).mean(-1))
            assert len(seeds)==500
            current={k:acc[i] for i,k in enumerate(names)}; candidate=current["calibrated"]
            comparisons={key:interval(candidate-value,seeds) for key,value in current.items() if key!="calibrated"}
            cells[path.stem]=dict(accuracy_pct={k:float(v.mean()*100) for k,v in current.items()},
                                  comparisons=comparisons, source_winner=win,sha256=sha(path))
            source=path.stem.split("_to_")[0]
            if path.stem.endswith("_k1"):
                bundles.setdefault(source,[]).append((current,seeds,win))
            for key in ["r2","cs_0.1","logistic_1","logistic_10",win]:
                if comparisons[key]["delta_pp"] < -CFG["gate"]["max_cell_loss_pp"]:
                    passed=False
    directions={}
    for source,rows in bundles.items():
        assert len(rows)==2 and np.array_equal(rows[0][1],rows[1][1])
        seeds=rows[0][1]; winner=rows[0][2]
        comps={}
        for key in [winner,"same_mass","r2","cs_0.1","logistic_1","logistic_10"]:
            delta=sum((r[0]["calibrated"]-r[0][key]) for r in rows)/len(rows)
            comps[key]=interval(delta,seeds)
            if comps[key]["ci95_pp"][0]<=0: passed=False
        directions[source]=comps
    assert len(cells)==8 and len(directions)==2
    result=dict(status="passed" if passed else "refuted_source_gate",cell_count=len(cells),
                task_condition_evaluations=4000,unique_sampled_tasks=2000,
                directions=directions,cells=cells,
                scope="Exposed source-domain folds; no independent target or novel-method claim.",
                next_action="freeze complete Pets main/test" if passed else "record decision; keep Caltech unconsumed")
    dump(OUT/"analysis.json",result)
    print("QUALIFICATION_VERDICT",result["status"],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["check","run"],required=True)
    args=ap.parse_args();OUT.mkdir(exist_ok=True);guard()
    sources=[Path(__file__),HERE/"calibrate.py",HERE/"protocol.json",BASE/"reference_eval.py",
             BASE/"baseline_methods.py",HERE.parent/"r2-fusion-stage-canonical-20260912/geometry_reference.py"]
    fingerprint={str(p):sha(p) for p in sources}
    if args.phase=="check":
        check=calibrate.self_check(CFG["calibration"],ref)
        dump(OUT/"implementation_validation.json",check)
        dump(HERE/"locked_sources.json",fingerprint)
        print(json.dumps(check),flush=True);return
    assert fingerprint==json.loads((HERE/"locked_sources.json").read_text())
    assert json.loads((OUT/"implementation_validation.json").read_text())["status"]=="passed"
    assert not (OUT/"complete.json").exists(),"completed run must not be repeated"
    import sklearn, scipy
    dump(OUT/"run_manifest.json",dict(command=[sys.executable,*sys.argv],config=CFG,
        source_hashes=fingerprint,started_at=datetime.now(timezone.utc).isoformat(),
        python=sys.version,torch=torch.__version__,numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__))
    data,manifest=ref.assets()
    dump(OUT/"asset_manifest.json",manifest)
    trained={name:train_source(name,data[name]) for name in CFG["source_domains"]}
    for source,target in [("dtd","eurosat"),("eurosat","dtd")]:
        model,winner=trained[source]
        for shot in CFG["shots"]:
            for gallery in [target,source]:
                evaluate_cell(source,target,gallery,shot,data,model,winner)
    analyze();guard()
    dump(OUT/"complete.json",dict(status="success",finished_at=datetime.now(timezone.utc).isoformat(),
                                 source_hashes=fingerprint))
    print("RUN_COMPLETE",flush=True)


if __name__ == "__main__":
    main()
