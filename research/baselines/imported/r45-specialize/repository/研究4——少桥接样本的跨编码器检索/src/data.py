from __future__ import annotations
import hashlib
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from research_common.records import WORKSPACE, read_json, write_json, object_hash, sha256
from research_common.encoders import FrozenEncoder

PROJECT = Path(__file__).resolve().parents[1]
CUB = WORKSPACE / "研究3——少样本学习测试时计算扩展/data/CUB_200_2011"


def prepare():
    destination = PROJECT / "data/manifest_v1.json"
    if destination.exists():
        manifest = read_json(destination)
        if manifest["config_sha256"] != sha256(PROJECT / "configs/pilot_v1.json"):
            raise ValueError("Existing manifest binds a different config")
        return manifest
    config = read_json(PROJECT / "configs/pilot_v1.json")
    names = {int(line.split()[0]): line.split(maxsplit=1)[1] for line in (CUB / "images.txt").read_text().splitlines()}
    labels = {int(a): int(b) for a,b in (line.split() for line in (CUB / "image_class_labels.txt").read_text().splitlines())}
    training = {int(a): int(b) for a,b in (line.split() for line in (CUB / "train_test_split.txt").read_text().splitlines())}
    rng = np.random.default_rng(config["seed"])
    classes = list(map(int, rng.permutation(sorted(set(labels.values())))))
    target = classes[:config["target_classes"]]
    stress = classes[config["target_classes"]:config["target_classes"] + config["stress_classes"]]
    roles = {"gallery": [], "query": [], "bridge": [], "stress_bridge": []}
    for c in target + stress:
        train_ids = rng.permutation(sorted(i for i in labels if labels[i] == c and training[i]))
        test_ids = rng.permutation(sorted(i for i in labels if labels[i] == c and not training[i]))
        if c in target:
            n = config["gallery_per_class"]
            if len(train_ids) < n + config["bridge_pool_per_class"] or len(test_ids) < config["query_per_class"]:
                raise ValueError("Insufficient declared role samples")
            roles["gallery"].extend(map(int, train_ids[:n]))
            roles["bridge"].extend(map(int, train_ids[n:n + config["bridge_pool_per_class"]]))
            roles["query"].extend(map(int, test_ids[:config["query_per_class"]]))
        else:
            roles["stress_bridge"].extend(map(int, train_ids[:config["bridge_pool_per_class"]]))
    for key in roles:
        roles[key] = list(map(int, rng.permutation(roles[key])))
    ids = sum(roles.values(), [])
    assert len(ids) == len(set(ids))
    records, seen_rgb = [], {}
    for image_id in sorted(ids):
        path = CUB / "images" / names[image_id]
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        rgb_hash = hashlib.sha256(np.asarray(rgb.shape, np.int64).tobytes() + rgb.tobytes()).hexdigest()
        if rgb_hash in seen_rgb:
            raise ValueError(f"Duplicate RGB identity in selected roles: {image_id}, {seen_rgb[rgb_hash]}")
        seen_rgb[rgb_hash] = image_id
        records.append({"id": image_id, "relative_path": names[image_id], "label_evaluator_only": labels[image_id],
                        "file_sha256": sha256(path), "rgb_sha256": rgb_hash})
    manifest = {"version": 1, "dataset": "CUB200", "stage": "development_only", "source_directory": str(CUB),
                "config_sha256": sha256(PROJECT / "configs/pilot_v1.json"),
                "metadata_sha256": {n:sha256(CUB/n) for n in ("images.txt","image_class_labels.txt","train_test_split.txt")},
                "target_classes": target, "stress_classes": stress, "roles": roles, "records": records,
                "exact_rgb_duplicates": [], "near_duplicate_audit": "not_completed; all outcomes remain development",
                "query_labels_available_to_method": False, "gallery_new_available_to_method": False,
                "confirmation_used": False}
    manifest["content_sha256"] = object_hash(manifest)
    write_json(destination, manifest)
    print({"manifest": str(destination), "roles": {k:len(v) for k,v in roles.items()}}, flush=True)
    return manifest


def encode_all():
    manifest = prepare()
    config = read_json(PROJECT / "configs/pilot_v1.json")
    for name in config["encoders"]:
        path = PROJECT / "cache" / (name + "_all.npz")
        meta_path = path.with_suffix(".json")
        if path.exists() and meta_path.exists():
            meta = read_json(meta_path)
            if meta["manifest_sha256"] != sha256(PROJECT / "data/manifest_v1.json") or meta["npz_sha256"] != sha256(path):
                raise ValueError("Cache identity mismatch")
            print(f"Verified existing cache {name}", flush=True)
            continue
        model = FrozenEncoder(name)
        output, ids = [], []
        times, batch = [], config["batch_size"]
        torch.cuda.reset_peak_memory_stats()
        for start in range(0, len(manifest["records"]), batch):
            rows = manifest["records"][start:start+batch]
            images = []
            for row in rows:
                source = CUB / "images" / row["relative_path"]
                if sha256(source) != row["file_sha256"]:
                    raise ValueError("Image changed after manifest freeze")
                with Image.open(source) as im:
                    images.append(im.convert("RGB").copy())
            torch.cuda.synchronize()
            before = time.monotonic()
            features = model.encode_pil(images)
            torch.cuda.synchronize()
            times.append(time.monotonic() - before)
            output.append(features)
            ids.extend(r["id"] for r in rows)
            if start % (batch * 8) == 0:
                print(f"encode {name} {min(start+batch,len(manifest['records']))}/{len(manifest['records'])}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=np.asarray(ids), embeddings=np.concatenate(output))
        record = {"model": model.identity, "manifest_sha256": sha256(PROJECT / "data/manifest_v1.json"),
                  "npz_sha256": sha256(path), "source_code_sha256": sha256(WORKSPACE / "research_common/encoders.py"),
                  "n_images": len(ids), "dimension": output[0].shape[1], "batch_size": batch,
                  "total_batch_seconds": float(sum(times)), "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                  "roles_encoded": {k:len(v) for k,v in manifest["roles"].items()},
                  "gallery_new_role": "oracle evaluator preparation only; never forwarded to inference worker"}
        write_json(meta_path, record)
        print({"encoded": name, "seconds": sum(times), "shape": [len(ids),output[0].shape[1]]}, flush=True)
        model.close()


def load_features(name):
    path = PROJECT / "cache" / (name + "_all.npz")
    metadata = read_json(path.with_suffix(".json"))
    if sha256(path) != metadata["npz_sha256"]:
        raise ValueError("Corrupt cache")
    with np.load(path, allow_pickle=False) as z:
        return {int(i): x for i,x in zip(z["ids"], z["embeddings"])}
