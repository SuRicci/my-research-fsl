# -*- coding: utf-8 -*-
"""vov job 真实执行器（二十轮 R3-P0-4/10）：替代 vov_jobs.py 旧的
_demo_executor。单个 job = 一次 dump + 在锁定预算集 {1.25,1.5,1.75,2.5,3,6}
上逐预算重放（一次 dump、六行输出——预算形式按 R3-P0-10 锁死为单 job 内
重放，job 不拆成单预算）。

执行步骤（每步 fail closed）：
1. canonical dataset → ClsDataset → EpisodeGenerator → CachedViewProvider →
   run_dump（query log + .meta.json 指纹侧车）；
2. job.models_index_sha256 给定时先校验模型索引 hash；
3. 逐 budget_mult 调 run_eval（血缘/manifest/hash 链在 run_eval 内验证）；
4. 六行预算汇总落 result_rows.jsonl；返回含 result_files 清单的结果 dict
   （run_jobs 落 .done 前逐文件重算校验）。
"""
from pathlib import Path
from typing import Callable, Optional

from .pipeline import atomic_write_jsonl, file_sha256


def make_vov_executor(configs_dir: Optional[Path] = None,
                      cache_root: Optional[Path] = None,
                      device: str = "cuda") -> Callable[[dict], dict]:
    """构造真实执行器。job 必需字段见 pipeline.JOB_SCHEMA。"""
    configs = Path(configs_dir) if configs_dir else None

    def executor(job: dict) -> dict:
        from ..data.splits import EpisodeGenerator
        from ..run.run_vov import CachedViewProvider, run_dump, run_eval
        from .datasets import load_dataset

        out_dir = Path(job["output"])
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(job["dataset"])
        gen = EpisodeGenerator(ds.labels, n_way=int(job["n_way"]),
                               k_shot=int(job["shot"]),
                               q_per_cls=int(job["q_per_cls"]),
                               seed=int(job["seed"]))
        episodes = gen.generate(int(job["n_episodes"]))
        provider = CachedViewProvider(job["dataset"], job["backbone"],
                                      job["method"], cache_root=cache_root,
                                      dataset_obj=ds, device=device,
                                      method_params=job.get("method_params")
                                      or None)
        log_path = out_dir / "query_log.jsonl"
        run_dump(provider, episodes, job["dataset"], job["method"],
                 job["backbone"], int(job["seed"]), job.get("split", "test"),
                 log_path, configs_dir=configs)
        model_dir = job.get("model_dir")
        if not model_dir:
            raise RuntimeError(f"job {job['job_id']} 缺 model_dir——fail closed")
        rows = []
        result_files = [{"path": str(log_path),
                         "sha256": file_sha256(log_path)},
                        {"path": str(log_path) + ".meta.json",
                         "sha256": file_sha256(str(log_path) + ".meta.json")}]
        for b in job["budget_mults"]:
            rp = out_dir / f"eval_B{float(b):g}.json"
            r = run_eval(log_path, Path(model_dir), float(b), rp,
                         random_seed=int(job["seed"]), configs_dir=configs,
                         expect_index_sha256=job.get("models_index_sha256"))
            rows.append({"budget_mult": float(b), "acc_vov": r["acc_vov"],
                         "acc_hard_first": r["acc_hard_first"],
                         "acc_random_matched": r["acc_random_matched"],
                         "delta_vs_hard": r["delta_vs_hard"],
                         "delta_vs_random": r["delta_vs_random"],
                         "oracle_gap_matched": r["oracle"]["gap_matched_cost"]})
            for suffix in ("", ".trace.jsonl", ".audit.jsonl",
                           ".assignments.jsonl"):
                p = Path(str(rp) + suffix)
                result_files.append({"path": str(p), "sha256": file_sha256(p)})
        rows_path = out_dir / "result_rows.jsonl"
        atomic_write_jsonl(rows_path, rows)
        result_files.append({"path": str(rows_path),
                             "sha256": file_sha256(rows_path)})
        return {"job_id": job["job_id"], "dataset": job["dataset"],
                "method": job["method"], "backbone": job["backbone"],
                "seed": job["seed"], "n_budgets": len(rows),
                "budget_rows": rows, "result_files": result_files,
                "output_schema_version": job.get("output_schema_version")}

    return executor
