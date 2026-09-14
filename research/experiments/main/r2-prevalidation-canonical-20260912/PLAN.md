# Support prevalidation main contract
Idea-8c090fcf. Run r2-prevalidation-canonical-20260912; baseline-2da45f14.
Question: can25labeled support examples select a ridge penalty better than a DTD-development constant or radius scaling?
H0: no positive five-shot mean improvement over either strong control; H1: positive paired95%CI over both, no cell loss>.5pp.
Tier: main/test after one integrated mathematical validation. Minimum: exact LOO/refit and scalar-solver agreement. Solid: complete original six cells and all seven metric ids. Maximum later: only justified new-pool/encoder confirmation.
Method: baseline-compatible bounded PreVal-inspired exact-intercept LOO, lambda[1,.1,.01,.001], scale[0,100], support cross-entropy; PRESS-MSE selector control. No algorithm novelty. Every1shot prediction inherited from locked parent CS_l2 R2, explicitly no new selector benefit.
Assets and pairing: original canonical feature manifests; parentgeometry six-cell tasks/scores; scale-campaign fused5shot controls. Full1000pairedtasks percell. No query-label adaptation, gallery labels, feature training or new assets.
Code map: prevalidated.py implements fit/predict; validate.py compares actual/rank-deficient tasks with all25 explicit refits and independent SciPy minimization; evaluate.py executes2new5shot cells and auditable4cell reuse, writes raw npz, metrics and manifest. Reuse existing asset/ridge/bootstrap helpers.
Numerics: double precision for exact LOO and near-unit hat diagonal. Validate raw float64 endpoint against canonical float32 to1e-5; record precision rather than implybitidentity.
Abandon: validation mismatch, invalid finite scores, hypothesisgatefailure. A failure is recorded then routed; no retuninglambda/scale bounds fromeval. No whole-run repeat withoutrealcodefix.
Budget: cached local CPU,minutes expected; no downloads/paidservices/remote. First round cap2026-09-15 09UTC.
Next: validation -> main -> record_main_experiment -> decision.
