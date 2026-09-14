"""Five-shot ridge scale controls with equal development-only tuning."""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
import reference_eval as ref
import statistics_helper as stats

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "protocol.json").read_text())
ref.CFG = CFG
torch.set_num_threads(6)

def dump(path, value):
    path.write_text(json.dumps(value, indent=2))

def load_pool(pool):
    root = Path(CFG["asset_roots"][pool])
    manifest = json.loads((root / "manifest.json").read_text())
    assert "elapsed_seconds" in manifest
    data = {}
    for dataset in ["dtd", "eurosat"]:
        identity = json.loads((root / (dataset + "_identities.json")).read_text())
        features = []
        for backbone in ["clip_vitb16", "dinov2_vits14"]:
            path = root / (dataset + "_" + backbone + "_query.pt")
            expected = manifest["datasets"][dataset]["backbones"][backbone + "_query"]["sha256"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
            pack = torch.load(path, weights_only=True)
            assert pack["ids"].tolist() == identity["query_ids"]
            features.append(pack["features"].float())
        data[dataset] = {"ident": identity, "clip": F.normalize(features[0], dim=-1), "dino": F.normalize(features[1], dim=-1),
                         "fusion": ref.representation(*features, .5)}
    return data, manifest

def heads(S, Q, choices):
    A = S.flatten(1, 2)
    mean = A.mean(1, keepdim=True)
    radius = (A - mean).square().sum(-1).mean(1)[:, None, None]
    assert float(radius.min()) > 1e-6
    Y = F.one_hot(torch.arange(5).repeat_interleave(5), 5).float()[None].expand(len(S), -1, -1)
    centered_A = F.normalize(A - mean, dim=-1)
    centered_Q = F.normalize(Q - mean, dim=-1)
    scores = {}
    for name, (family, coefficient) in choices.items():
        if family == "centered":
            scores[name] = ref.ridge_scores(centered_A, Y, centered_Q, coefficient)
        else:
            penalty = coefficient * radius if family == "radius" else coefficient
            scores[name] = ref.ridge_scores(A, Y, Q, penalty)
    scaled = ref.ridge_scores((A - mean) / radius.sqrt(), Y, (Q - mean) / radius.sqrt(), 1.)
    equivalent = ref.ridge_scores(A, Y, Q, radius)
    error = float((scaled - equivalent).abs().max())
    assert error < 1e-5
    scores["prototype"] = Q @ F.normalize(S.mean(2), dim=-1).transpose(1, 2)
    return scores, error

def run(phase):
    start = time.time()
    out = HERE / "outputs" / phase
    out.mkdir(parents=True, exist_ok=True)
    summaries, records, dev_scores = {}, {}, {}
    selection = json.loads((HERE / "selection.json").read_text()) if phase == "eval" else None
    max_error = {"radius_identity": 0., "fused_parent": 0.}
    manifest_record = {"phase": phase, "protocol": CFG, "selection": selection,
       "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob("*.py")},
       "command_argv": [sys.executable, *sys.argv], "torch": torch.__version__, "numpy": np.__version__, "features": {}}
    for pool in (["original"] if phase == "dev" else ["original", "new_images"]):
        data, manifest = load_pool(pool)
        manifest_record["features"][pool] = manifest
        ref.CFG["eval"]["seeds"] = CFG["pool_seeds"][pool]
        for dataset in (["dtd"] if phase == "dev" else ["dtd", "eurosat"]):
            si, qi, classes, seeds = ref.tasks(data[dataset]["ident"], dataset, 5, phase)
            y = np.repeat(np.arange(5), 15)
            assert all(not np.intersect1d(a, b).size for a, b in zip(si, qi))
            for representation in ["clip", "dino", "fusion"]:
                if phase == "dev":
                    choices = {family + "_" + str(c): (family, c) for family in ["raw", "radius", "centered"] for c in map(float, CFG["lambda_grid"])}
                else:
                    choices = {"raw_fixed": ("raw", 1.), "radius_fixed": ("radius", 1.)}
                    choices.update({family + "_tuned": (family, selection["coefficients"][representation][family]) for family in ["raw", "radius", "centered"]})
                features = data[dataset][representation]
                bags = {}
                for st in range(0, len(si), CFG["batch_size"]):
                    en = st + CFG["batch_size"]
                    scores, error = heads(features[si[st:en]], features[qi[st:en].reshape(-1, 75)], choices)
                    max_error["radius_identity"] = max(max_error["radius_identity"], error)
                    for name, value in scores.items():
                        assert torch.isfinite(value).all()
                        bags.setdefault(name, []).append(value.numpy())
                names = list(bags)
                scores = np.stack([np.concatenate(bags[n]) for n in names])
                predictions = scores.argmax(-1)
                accuracy = (predictions == y).mean(-1) * 100
                cell = pool + "__" + dataset + "__" + representation
                if phase == "eval" and representation == "fusion":
                    parent = np.load(Path(CFG["parent_outputs"][pool]) / (dataset + "_" + dataset + "_k5.npz"))
                    assert np.array_equal(si, parent["support_indices"]) and np.array_equal(qi, parent["query_indices"])
                    error = float(np.max(np.abs(scores[names.index("raw_fixed")] - parent["scores"][list(parent["names"]).index("r2")])))
                    max_error["fused_parent"] = max(max_error["fused_parent"], error)
                    assert error < 1e-5
                np.savez_compressed(out / (cell + ".npz"), scores=scores, predictions=predictions, accuracy_percent=accuracy,
                    names=names, support_indices=si, query_indices=qi, class_ids=classes, seeds=seeds, yq=y)
                summaries[cell] = {name: float(accuracy[i].mean()) for i, name in enumerate(names)}
                if phase == "dev":
                    dev_scores[representation] = summaries[cell]
                else:
                    contrasts = {}
                    for a, b in [("radius_fixed", "raw_fixed"), ("radius_tuned", "raw_tuned"), ("centered_tuned", "raw_tuned"), ("radius_tuned", "centered_tuned"), ("raw_tuned", "raw_fixed")]:
                        delta = accuracy[names.index(a)] - accuracy[names.index(b)]
                        contrasts[a + "_vs_" + b] = {"delta_pp": float(delta.mean()),
                              "paired_ci95_pp": stats.paired_bootstrap([(delta, seeds)])}
                    records[cell] = {"accuracy_percent": summaries[cell], "contrasts": contrasts}
                print("CELL_COMPLETE", phase, cell, json.dumps(summaries[cell]), flush=True)
    if phase == "dev":
        order = [1., .1, .01, .001]
        selected = {rep: {family: max(order, key=lambda c: values[family + "_" + str(c)])
                    for family in ["raw", "radius", "centered"]} for rep, values in dev_scores.items()}
        dump(HERE / "selection.json", {"coefficients": selected, "development_scores": dev_scores,
             "rule": "Equal four-value grid, original DTD development classes only, ties prefer1,.1,.01,.001. Frozen before all evaluation.", "locked_before_eval": True})
        print("SELECTION_FROZEN", json.dumps(selected), flush=True)
    else:
        dump(out / "comparison.json", {"cells": records, "validation_max_abs_score_error": max_error,
             "scope": "Fixed two-dataset pools and three feature representations. Equal development-only tuning. Paired task intervals are conditional, not independent-domain evidence; new pool was previously inspected for another method."})
    dump(out / "summary.json", summaries)
    dump(out / "run_manifest.json", manifest_record)
    dump(out / "completion.json", {"phase": phase, "elapsed_seconds": time.time() - start, "cell_count": len(summaries), "validation": max_error})
    print("PHASE_COMPLETE", phase, time.time() - start, json.dumps(max_error), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["dev", "eval"], required=True)
    run(parser.parse_args().phase)
