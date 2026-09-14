# Complete one-shot source selection precision audit
Decision decision-bc6f816a; review report-904dfac3/R14-3. Auxiliary numerical evidence for representation-scatter-20260913; no manuscript C1-C4 claim or canonical Pets metric update.

All 1200 fixed one-shot source configurations completed:200source episodes across DTD and EuroSAT, two saved galleries, three original gamma values,90,000query-prediction comparisons. No prediction changed between the saved float32 computation and independent dense float64 evaluation. Correct-count vectors and the tie-broken source choices are identical.

| Source | Gamma0.1 count | Gamma1 count | Gamma10 count | Selected gamma, old/new |
|---|---:|---:|---:|---|
| DTD |11809|11817|11819|10/10|
| EuroSAT |11257|11282|11282|1/1|

Independent float64 dense covariance eigendecomposition/primal regression and low-rank covariance/dual regression agree in all1200cases: maximum score error 2.03170813506e-14, below the unchanged2e-5criterion, with identical argmax. The original float32-versus-float64 full-score criterion still fails exactly one case, DTD/source1shot/task99/excluded/gamma0.1, with error0.0030342261087177302. That failure is preserved, not overwritten or relabeled passed. No other historical full-score discrepancy emerged in this complete one-shot selection population.

Full-bank output audit independently reconstructed every integer count and chosen gamma, checked all task/gallery identities and 90,000predictions, matched per-case discrepancy rows, verified immutable input/output hashes and confirmed the historical failure remains failed. Validation output: outputs/validation.json. Source five-shot selection, all target results, model-selection sampling uncertainty, and accumulated-target performance are outside this audit. Numerical invariance does not show that a two-count lead or a statistical tie is robust to new source episodes.

Execution: bash-612daaa5 completed in 172.77s; postcheck bash-0001bf0c passed. Science computation science-ce3c92f3; validation science-15117a13. Environment NumPy1.24.4/Torch2.3.0 on local CPU,2threads. Reused frozen normalized float32 feature inputs and existing cache loader; no encoder calls or downloads. Outputs remain within50MiBcap and10GiBfloor. Initial check bash-3b366f61 failed because the legacy head constructs float32 targets; fixed only in the local audit branch with explicit float64targets, and precheck bash-b9e93492 passed. Original implementation, tolerances, choices and metrics were never changed.

Implication: R14-3's inherited one-shot gamma selection ambiguity is closed for numerical precision. A target rerun or a precision rescue of the original failed gate is not justified. Retain the conditional source incumbent and failed Caltech/general-improvement gates. Return to evidence-grounded idea selection, excluding another gamma/layer/weight resweep and already-tested source error-overlap analysis. Current OSLO technical note remains four direct groups/seven tables; this audit is appended to its existing reference-only scatter history.
