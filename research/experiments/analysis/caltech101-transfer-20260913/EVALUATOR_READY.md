# Caltech evaluator readiness — 2026-09-13T11:43:15.121868+00:00

The frozen ten-method evaluator and independent audit are ready. No Caltech classification result has been produced. Existing encoding remains live as bash-c4ad1066 / science-1704eee7; latest saved progress: {'status': 'running', 'backbone': 'clip_vitb16', 'images_done': 4608, 'image_count': 8677, 'elapsed_seconds': 1459.9002647399902, 'free_gib': 10.98532485961914}.

The prospective design (decision-dedee53d) is unchanged: 500 tasks, 3 galleries, both fixed source configurations, all 10 methods. This is a new-dataset supporting analysis, not a replacement for the five canonical Pets metrics. Both source configurations must pass the previously fixed comparisons; no target-driven source choice or parameter search is permitted.

Validation: 36 inherited source score comparisons were exactly equal, including predictions. All 48 independent dense NumPy ridge/scatter comparisons passed with exact argmax agreement; maximum score discrepancy 3.68008956364e-07. Maximum discrepancy for float64 stack methods is below 7e-15. Query order, query partition, class permutation and eta-zero invariances passed. All 500 task identities and 3072 selected gallery identities passed label/role/disjointness checks without loading target feature tensors. The two entrypoint provenance maps agree on 16 resolved source paths.

Failures preserved: first preflight numerical work passed but Python3.8 lacked Path.is_relative_to during provenance capture (bash-6c1e160a). Compatibility was repaired and one numerical rerun passed (bash-14c2e66e). A relative __main__ path was then normalized; all numerical functions remained AST-identical and a dedicated launch/identity verification passed (bash-5e92e5a0). No method, parameter, numerical tolerance, or target task changed. Historical source numerical and incremental gates remain failed; this preflight does not recertify those historical experiments.

Numerical audit scope: dense covariance eigendecomposition and primal ridge cover eight ridge-based methods. Logistic controls reuse frozen sklearn with convergence warnings treated as failures and source score parity; no independent logistic optimizer is claimed. Full target output integrity, conditional paired intervals, controls, all cell gates and fixed target score samples remain to be audited after classification.

Next execution, only after encoding completion and feature integrity checks:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 /opt/anaconda3/envs/torch/bin/python experiments/analysis/caltech101-transfer-20260913/caltech_eval.py
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/torch/bin/python experiments/analysis/caltech101-transfer-20260913/caltech_audit.py --phase audit

Use managed bash_exec detach sessions. Inspect existing managed sessions before launching either command. Do not launch a duplicate encoder or duplicate evaluation. caltech_eval.py validates the completed feature hashes/order/norms, frozen code/design and image identities before computing scores; it reuses complete gallery cells only with matching code/task locks. Both configurations share tasks and cannot be counted as independent replications. Review every launched outcome, update science state, and route through decision. One-domain evidence does not establish broad generalization, pretraining nonexposure, or submission readiness.
