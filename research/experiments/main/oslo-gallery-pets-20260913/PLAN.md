# OSLO_G main experiment contract

Selected idea-aca445b7; decision-00f1a4d2. Test the frozen source-default open-set objective against closed-set/zero-step ablations and all four Pets controls. Null: no strong-control gain or unacceptable harm; alternative: >=0.5pp matched1shot over R2/CS_l2 with positive paired bounds in both exposed and fresh scopes, other-cell lower bounds>=-0.5pp. Main/test transfer experiment; source defaults fixed, no calibration.

Inputs: unchanged five-metric Pets baseline contract; exact parent features and tasks;1804unusedidentities preaudited. Same37classes and exposed domain; fresh images supply prospective confirmation only. No gallery labels, query fitting, text information, encoder fine-tuning or newdata/model download. Candidate parameters:2steps,lambda_s.05,lambda_z.1,EMA1. Mean centering from S+G; all Q independent. Compare identical preprocessing and source updates on/off/zero.

Minimal changes: oslo.py fit/predict; validate_source.py source parity+query boundary; run_eval.py reuse parent task/feature/control logic and encode only fresh images; analyze.py complete output audit and paired intervals. Outputs: protocol/source hashes, validation, feature manifest, eight cells, metrics and report. Run in this dedicatedrunworkspace only. Use CPUbatches16/4threads and existing MPS encoder batches32. Free>=10GiB, incremental<=.15GiB, deadline2026-09-15 09:00UTC. Before additional worktree creation reserve its measured checkout cost; current copy consolidation recovered reserve after transient9.60GiB (no experiment ran there).

Minimum: all comparable cells finite with source parity; solid: fixed gates pass on newimages; maximum: only subsequent justified analysis. Failure records complete negative outcome, closes source-default family; never switch variant after results. Paired bootstrap5000stratifiedbyseed is conditional on fixed pools. Individualbreed/seed and gallerymass diagnostics descriptive. No publication-readiness claim.

Next: one source/inductiveness validation; freeze hashes; detached encoding+full eval; all-output audit; record_main_experiment and decision. Experiment is not complete at launch.

Exit2026-09-12T22:25Z: audited and registered run-2dac3539; all gates fail in both scopes. Follow decision-5094098f into one existing-fit influence audit. No source-default variant retry.
