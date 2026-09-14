# -*- coding: utf-8 -*-
"""实验12 pilot：图库编码 + 检索增强少样本分类。

两个阶段：
1) --encode：把图库（train split，无标签使用）编码进本目录 cache/gallery/；
   幂等，已缓存跳过。
2) 评估（默认）：在 (查询数据集 × 图库) cell 网格上跑 episode 协议。
   - 开发 episode（seed 12000）只用于选择 AugConfig 超参与 τ_cov；
   - 评估 episode（seed 34000）用冻结配置出主结果；
   - true25 为超预算诊断参照，不进主表排名；
   - 特征按 (查询集, backbone) 只加载一次，dev 选参与评估共享；
   - 污染率经类名映射计算（cifar_fs 的标签 id 空间与 cifar100 不同，
     直接比 id 会得到伪污染）；类不相交图库（miniIN）只报诊断量。

用法示例：
  python run_pilot.py --encode --backbones clip_vitb16
  python run_pilot.py --episodes 200
  python run_pilot.py --cells cifar_fs_test:cifar100 --backbones clip_vitb16
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e12 import augment, r3bridge  # noqa: E402

# (查询数据集, 图库, 图库与查询是否共享类别)
CELLS = [
    ("miniimagenet_test", "miniimagenet", False),
    ("cifar_fs_test", "cifar100", True),
    ("dtd_test", "dtd", True),
    ("dtd_test", "miniimagenet", False),   # 跨域压力
]
N_WAY, Q_PER_CLS = 5, 15
DEV_SEED, EVAL_SEED = 12000, 34000

DEV_GRID = {
    "r": [16, 64],
    "alpha": [2.0, 8.0],
    "beta": [0.0, 2.0],
    "gamma": [0.0, 1.0],
}
TAU_GRID = [0.0, 0.05, 0.15]


def encode_gallery(ds_name: str, backbone: str, device: str, batch: int = 128):
    ds = r3bridge.load_cls_dataset(ds_name, "train")
    scope = f"{ds_name}_train"
    fc = r3bridge.gallery_cache(scope, backbone)
    tfm = r3bridge.center_transform(backbone)
    todo = [i for i in range(len(ds))
            if not fc.has(r3bridge.gallery_key(backbone, scope, i))]
    t0 = time.time()
    for st in range(0, len(todo), batch):
        chunk = todo[st:st + batch]
        xs = torch.stack([tfm(ds.get_image(i)) for i in chunk])
        feats = r3bridge.encode(backbone, xs, device=device)
        for i, f in zip(chunk, feats):
            fc.put(r3bridge.gallery_key(backbone, scope, i), f)
        if st % (batch * 20) == 0 and st > 0:
            rate = st / (time.time() - t0)
            print(f"    [{scope}|{backbone}] {st}/{len(todo)} "
                  f"({rate:.0f} img/s)", flush=True)
    fc.flush()
    print(f"[encode] {scope}|{backbone}: 新编码 {len(todo)}/{len(ds)} "
          f"({time.time() - t0:.0f}s)")


def load_gallery(ds_name: str, backbone: str):
    ds = r3bridge.load_cls_dataset(ds_name, "train")
    scope = f"{ds_name}_train"
    fc = r3bridge.gallery_cache(scope, backbone)
    keys = [r3bridge.gallery_key(backbone, scope, i) for i in range(len(ds))]
    missing = fc.missing(keys)
    if missing:
        raise RuntimeError(f"图库缓存缺 {len(missing)} 条，先 --encode: {scope}")
    G = fc.get_many(keys)
    return G, np.asarray(ds.labels), scope, list(ds.classnames)


def sample_episode(pools, n_way, k_shot, q_per_cls, rng, true_k=25):
    """按缓存池采 episode；额外采 true_k 诊断支持（超预算参照）。"""
    cand = np.array([c for c, idx in pools.items()
                     if len(idx) >= k_shot + q_per_cls + 1], dtype=np.int64)
    if len(cand) < n_way:
        raise ValueError(f"缓存覆盖不足: {len(cand)} < {n_way}")
    classes = rng.choice(cand, size=n_way, replace=False)
    s_idx, q_idx, t_idx, y_t = [], [], [], []
    for li, c in enumerate(classes):
        pool = pools[int(c)]
        n_t = min(true_k, len(pool) - k_shot - q_per_cls)
        take = rng.choice(pool, size=k_shot + q_per_cls + n_t, replace=False)
        s_idx.extend(take[:k_shot])
        q_idx.extend(take[k_shot:k_shot + q_per_cls])
        t_idx.extend(take[k_shot + q_per_cls:])
        y_t.extend([li] * n_t)
    return (np.array(s_idx), np.repeat(np.arange(n_way), k_shot),
            np.array(q_idx), np.repeat(np.arange(n_way), q_per_cls),
            np.array(t_idx), np.array(y_t))


class QuerySide:
    """(查询集, backbone) 的特征与池，加载一次复用。"""

    def __init__(self, query_ds: str, backbone: str):
        self.labels, self.classnames = r3bridge.load_classnames(query_ds)
        self.pools = r3bridge.query_cached_pools(self.labels, query_ds,
                                                 backbone)

    def load_feats(self, episodes):
        all_idx = sorted({int(i) for e in episodes for i in
                          np.concatenate([e[0], e[2], e[4]])})
        self.lut = {int(v): r for r, v in enumerate(all_idx)}
        return None  # feats 由 runner 按 backbone 缓存后传入

    def name2label(self):
        return {n: i for i, n in enumerate(self.classnames)}


def eval_episodes(qs: QuerySide, feats, G, g_labels, g_name2label,
                  gallery_shared, k_shot, episodes, cfg, methods,
                  hub_vec=None):
    acc = {m: [] for m in methods}
    contam, fb_rate = [], []
    for (s_idx, y_s, q_idx, y_q, t_idx, y_t) in episodes:
        S = feats[[qs.lut[int(i)] for i in s_idx]]
        yS = torch.tensor(y_s)
        X = feats[[qs.lut[int(i)] for i in q_idx]]
        yQ = torch.tensor(y_q)
        T = feats[[qs.lut[int(i)] for i in t_idx]]
        yT = torch.tensor(y_t)
        for m in methods:
            if m == "true25_diag":
                pred = augment.true_k_diag(T, yT, X, N_WAY)
            else:
                fn = getattr(augment, m)
                pred = fn(S, yS, X, N_WAY, G, cfg, hub_vec=hub_vec) \
                    if m != "raw" else fn(S, yS, X, N_WAY)
            acc[m].append(float((pred == yQ).float().mean()))
        if gallery_shared:
            # 真实污染率（类名映射：episode 类名 → 图库标签 id）
            protos = augment.proto_from(S, yS, N_WAY)
            sim = protos @ augment._normalize(G).T
            top = sim.topk(min(cfg.r, G.shape[0]), dim=1).indices.cpu().numpy()
            q_name2label = qs.name2label()
            ep_glabels = []
            for c in range(N_WAY):
                qname = qs.classnames[int(qs.labels[int(q_idx[c * Q_PER_CLS])])]
                ep_glabels.append(g_name2label.get(qname, -1))
            for c in range(N_WAY):
                gl = g_labels[top[c]]
                contam.append(float((gl != ep_glabels[c]).mean()))
        diag = augment.diagnostics(S, yS, N_WAY, G, cfg, hub_vec=hub_vec)
        fb_rate.append(diag["fallback_rate"])
    return acc, float(np.mean(contam)) if contam else None, \
        float(np.mean(fb_rate))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--galleries", nargs="+",
                    default=["miniimagenet", "cifar100", "dtd"])
    ap.add_argument("--backbones", nargs="+",
                    default=["clip_vitb16", "dinov2_vits14"])
    ap.add_argument("--cells", nargs="*", default=None,
                    help="形如 cifar_fs_test:cifar100，默认全部")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--shots", type=int, nargs="+", default=[1, 5])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="pilot_main")
    args = ap.parse_args()

    if args.encode:
        for g in args.galleries:
            for b in args.backbones:
                encode_gallery(g, b, args.device)
        return

    cells = CELLS
    if args.cells:
        wanted = {tuple(c.split(":")) for c in args.cells}
        cells = [c for c in CELLS if (c[0], c[1]) in wanted]

    methods = ["raw", "uniform_aug", "weighted_aug", "weighted_fallback",
               "true25_diag"]
    all_rows, metas = [], []
    qside_cache = {}
    for query_ds, gallery_ds, shared in cells:
        for backbone in args.backbones:
            try:
                G, g_labels, gscope, g_names = load_gallery(gallery_ds,
                                                            backbone)
            except (RuntimeError, FileNotFoundError) as e:
                print(f"[跳过] {gallery_ds}|{backbone}: {e}")
                continue
            g_name2label = {n: i for i, n in enumerate(g_names)}
            hub_path = (Path(__file__).resolve().parent / "cache"
                        / f"hub_{gscope}_{backbone}.pt")
            if hub_path.exists():
                hub_vec = torch.load(hub_path, weights_only=True)
            else:
                t_hub = time.time()
                hub_vec = augment.precompute_hub(G, k=16)
                torch.save(hub_vec, hub_path)
                print(f"[hub] {gscope}|{backbone}: {time.time() - t_hub:.0f}s")
            key = (query_ds, backbone)
            if key not in qside_cache:
                qs = QuerySide(query_ds, backbone)
                qside_cache[key] = qs
            qs = qside_cache[key]
            for shot in args.shots:
                rng = np.random.RandomState(DEV_SEED + shot)
                dev_eps = [sample_episode(qs.pools, N_WAY, shot, Q_PER_CLS,
                                          rng)
                           for _ in range(args.episodes)]
                rng = np.random.RandomState(EVAL_SEED + shot)
                eval_eps = [sample_episode(qs.pools, N_WAY, shot, Q_PER_CLS,
                                           rng)
                            for _ in range(args.episodes)]
                qs.load_feats(dev_eps + eval_eps)
                feats = r3bridge.load_query_features(
                    query_ds, backbone, sorted(qs.lut.keys()))

                # ---- 开发集选超参（只评 weighted_aug）----
                best = None
                t0 = time.time()
                for r in DEV_GRID["r"]:
                    for a in DEV_GRID["alpha"]:
                        for b_ in DEV_GRID["beta"]:
                            for g_ in DEV_GRID["gamma"]:
                                cfg = augment.AugConfig(r=r, alpha=a, beta=b_,
                                                        gamma=g_)
                                acc, _, _ = eval_episodes(
                                    qs, feats, G, g_labels, g_name2label,
                                    shared, shot, dev_eps, cfg,
                                    ["weighted_aug"], hub_vec=hub_vec)
                                m = float(np.mean(acc["weighted_aug"]))
                                if best is None or m > best[0]:
                                    best = (m, cfg)
                cfg = best[1]
                # τ_cov 单独为 fallback 选
                best_tau, best_m = TAU_GRID[0], -1.0
                for tau in TAU_GRID:
                    cfg_t = augment.AugConfig(**{**cfg.__dict__,
                                                 "tau_cov": tau})
                    acc, _, _ = eval_episodes(
                        qs, feats, G, g_labels, g_name2label,
                        shared, shot, dev_eps, cfg_t,
                        ["weighted_fallback"], hub_vec=hub_vec)
                    m = float(np.mean(acc["weighted_fallback"]))
                    if m > best_m:
                        best_m, best_tau = m, tau
                cfg = augment.AugConfig(**{**cfg.__dict__, "tau_cov": best_tau})

                # ---- 评估集冻结出数 ----
                acc, contam, fb = eval_episodes(
                    qs, feats, G, g_labels, g_name2label,
                    shared, shot, eval_eps, cfg, methods, hub_vec=hub_vec)
                row = {"query": query_ds, "gallery": gscope,
                       "shared_classes": shared, "backbone": backbone,
                       "shot": shot, "cfg": str(cfg.__dict__),
                       "contamination": contam, "fallback_rate": fb}
                for m in methods:
                    v = np.asarray(acc[m])
                    row[m] = float(v.mean())
                    row[f"{m}_ci95"] = float(1.96 * v.std(ddof=1)
                                             / len(v) ** 0.5)
                for ref in ("uniform_aug", "raw"):
                    d = np.asarray(acc["weighted_aug"]) - np.asarray(acc[ref])
                    rb = np.random.RandomState(0)
                    idx = rb.randint(0, len(d), size=(2000, len(d)))
                    lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
                    row[f"w_minus_{ref}"] = float(d.mean())
                    row[f"w_minus_{ref}_ci"] = f"[{lo:.4f},{hi:.4f}]"
                all_rows.append(row)
                metas.append({"query": query_ds, "gallery": gscope,
                              "backbone": backbone, "shot": shot,
                              "dev_best_cfg": cfg.__dict__,
                              "dev_time_s": round(time.time() - t0, 1)})
                print(f"[{query_ds}|{gscope}|{backbone}|{shot}shot] "
                      + " ".join(f"{m}={row[m]:.4f}" for m in methods)
                      + f" contam={contam} fb={fb:.2f}", flush=True)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.out}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows, "metas": metas}, f, indent=2,
                  ensure_ascii=False)
    if all_rows:
        with open(out.with_suffix(".csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
