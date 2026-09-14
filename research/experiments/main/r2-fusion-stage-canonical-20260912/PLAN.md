# Fusion-stage main experiment contract

Selected idea: idea-a921652c; decision-4b97a92f. Test separate ridge solves on identical R2 augmented prototypes versus joint fits. Full hypotheses, literature and gates: artifacts/idea/selected_idea.md.

Research type: controlled model/ensemble intervention; development then main/test on original repeated pools. Minimum: verified method, identities and complete seven metrics. Solid: >=0.5pp macro versus R2 with paired CI lower>0, <=0.5pp cell loss, interior weight and positive CI versus development-selected strongest simple control. Maximum evidence: new robustness/attribution only if solid gate survives; no advance promise of a paper.

Code map: copied reference_eval.py, baseline_methods.py, geometry_reference.py unchanged; new fusion.py for split/early/late kernels, evaluate.py for raw-view loading/grid selection and original task loop, validate.py for numerical/source equivalence, package_results.py for paired statistics and seven-metric audit. No edits to source baseline.

Fixed: original feature manifest, class split, gallery RGB exclusions, tasks and permissions; 20 parameter settings per family on original DTD development only. Five-shot inherited. Bootstraps use 5000 paired resamples, same task index jointly across galleries within a dataset and stratified across five seeds; report conditional intervals and seed effects. Report every control, no evaluation winner substitution.

Resources: local torch Conda interpreter from source manifest; cached features, batch16 CPU,6 threads initial chosen for established throughput, not a user cap. Memory/disk govern. No downloads. Abort on nonfinite output, hash/task mismatch or free disk<10GiB. Deadline2026-09-15 09:00UTC. One numerical validation before real detached run; no benchmark pilots unless a concrete failure changes the path.

Monitor: bash_exec detach/read/list; outputs/run_manifest.json, progress.json, selection.json, outputs/eval/*.npz and outputs/completion.json. First review after actual development checkpoint. Completion -> independent packaging/validation -> artifact.record_main_experiment -> decision; launching is not completion.

## Exit state
Completed and independently validated. run-eda8c43a / decision-079e924e. Small gain over R2, failed >=0.5pp gate and below known CS_l2. Retain measured effect; reject robust-method promotion and further tuning on these pools. Next anchor is R4 trust/reuse audit; no paper contribution established.
