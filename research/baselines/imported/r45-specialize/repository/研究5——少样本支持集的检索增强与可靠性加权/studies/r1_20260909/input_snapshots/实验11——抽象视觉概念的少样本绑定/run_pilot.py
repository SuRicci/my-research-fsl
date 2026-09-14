# -*- coding: utf-8 -*-
"""实验11 pilot：Bongard-OpenWorld 上的闭式概念读出。

协议：每题 6 正 6 负支持 + 每侧末 1 张查询（2 判定/题）。
- --encode：编码某 split 全部已下载图像（缺图跳过，幂等）。
- 评估（默认）：逐题 4 个闭式方法，报总体与各 commonSense 代码分项；
  与官方已发表基线（SNAIL+OpenCLIP 64.0 avg）只作数字对照，声明为
  "官方论文报告值"，不做同条件复现声称。
前提裁决（README）：
- P1 落差：闭式读出 vs 官方基线/随机（50%）的差距方向与幅度；
- P2 概念族结构：按 commonSense 代码分组的组间方差 vs 组内方差。
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
from e11 import io, r3bridge, readout  # noqa: E402


def encode_split(split: str, backbone: str, device: str, batch: int = 128,
                 min_side: int = 4):
    problems = io.load_problems(split, require_images=True,
                                min_per_side=min_side)
    fc = r3bridge.bow_cache(backbone)
    tfm = r3bridge.center_transform(backbone)
    mapping = io._crawl_map()
    rels = sorted({rel for p in problems for rel in p.pos + p.neg})
    todo = [r for r in rels if not fc.has(r3bridge.bow_key(backbone, r))]
    print(f"[encode] {split}|{backbone}: {len(todo)}/{len(rels)} 待编码")
    t0 = time.time()
    from PIL import Image
    for st in range(0, len(todo), batch):
        chunk = todo[st:st + batch]
        xs = []
        keep = []
        for rel in chunk:
            try:
                img = Image.open(io.resolve(rel, mapping)).convert("RGB")
                xs.append(tfm(img))
                keep.append(rel)
            except Exception as e:  # noqa: BLE001
                print(f"    [坏图跳过] {rel}: {type(e).__name__}")
        if not xs:
            continue
        feats = r3bridge.encode(backbone, torch.stack(xs), device=device)
        for rel, f in zip(keep, feats):
            fc.put(r3bridge.bow_key(backbone, rel), f)
    fc.flush()
    print(f"[encode] 完成 ({time.time() - t0:.0f}s)")


def eval_split(split: str, backbone: str, min_side: int = 4):
    problems = io.load_problems(split, require_images=True,
                                min_per_side=min_side)
    fc = r3bridge.bow_cache(backbone)

    def feat(rel):
        return fc.get(r3bridge.bow_key(backbone, rel))

    per_method = {m: [] for m in readout.METHODS}
    per_group = {}
    skipped = 0
    ks = []
    for p in problems:
        k = p.adaptive_k(6)   # 死链宽松协议：k=min(6, 每侧可用-1)
        sp, sl, qp, ql = p.support_query(k_shot=k)
        try:
            S = torch.stack([feat(r) for r in sp])
            X = torch.stack([feat(r) for r in qp])
        except KeyError:
            skipped += 1
            continue
        yS = torch.tensor(sl)
        yQ = torch.tensor(ql)
        ks.append(k)
        for m in readout.METHODS:
            scores = readout.run_method(m, S, yS, X)
            acc = float(((scores > 0).long() == yQ).float().mean())
            per_method[m].append(acc)
            per_group.setdefault(p.common_sense, {}).setdefault(m, []).append(acc)

    rows = []
    for m in readout.METHODS:
        v = np.asarray(per_method[m])
        rows.append({"split": split, "backbone": backbone, "method": m,
                     "acc": float(v.mean()),
                     "ci95": float(1.96 * v.std(ddof=1) / len(v) ** 0.5)})
        print(f"  [{split}|{backbone}] {m:<14s} acc={v.mean():.4f} "
              f"(±{rows[-1]['ci95']:.4f})")
    groups = {g: {m: float(np.mean(d[m])) for m in readout.METHODS}
              for g, d in sorted(per_group.items())}
    # P2：组间方差 vs 组内方差（用 contrast 的逐题 acc）
    means = np.array([g["contrast"] for g in groups.values()])
    within = np.array([float(np.var(d["contrast"]))
                       for d in per_group.values() if len(d["contrast"]) > 1])
    meta = {"split": split, "backbone": backbone, "problems": len(problems),
            "skipped": skipped, "groups": groups,
            "adaptive_k_mean": float(np.mean(ks)) if ks else None,
            "adaptive_k_hist": {int(k): int((np.array(ks) == k).sum())
                                for k in sorted(set(ks))},
            "between_group_var": float(means.var()),
            "within_group_var_mean": float(within.mean()) if len(within) else None}
    group_str = ", ".join(f"{g}: {v['contrast']:.3f}" for g, v in groups.items())
    print(f"  分组均值: {{ {group_str} }}")
    print(f"  P2 结构: 组间var={meta['between_group_var']:.4f} "
          f"组内var={meta['within_group_var_mean']}")
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--backbones", nargs="+",
                    default=["clip_vitb16", "dinov2_vits14"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="pilot_main")
    ap.add_argument("--min-side", type=int, default=4)
    args = ap.parse_args()

    if args.encode:
        for b in args.backbones:
            encode_split(args.split, b, args.device, min_side=args.min_side)
        return

    all_rows, metas = [], []
    for b in args.backbones:
        rows, meta = eval_split(args.split, b, min_side=args.min_side)
        all_rows.extend(rows)
        metas.append(meta)
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{args.out}_{args.split}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": all_rows, "metas": metas}, f, indent=2,
                  ensure_ascii=False)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
