# Active experiment contract
Parent: idea-ca651025; run/r2-residual-canonical-20260912. One first-round mechanism-transfer experiment.
Question: can gallery residual covariance help while avoiding biased retrieved class means?
H0: residual transfer has no robust benefit over same-feature R2 and simple controls.
H1: positive paired macro gain across matched/mismatched galleries, no cell loss>0.5pp, and evidence beyond isotropic or support-only regularization.
Type: controlled computational experiment; currently no candidate result.

Baseline: accepted legacy r2-frozen-local is cached DTD5 numerical replay only. The new canonical-openai-quickgelu-native-dinov2 comparator must be separately measured/confirmed before candidate main evaluation. Never compare candidate canonical values with legacy90.1533%.
Datasets: existing DTD and EuroSAT images, original folds/splits/RGB dedup. DTD development first23 and evaluation remaining24 classes via seed120909; EuroSAT all10 evaluation classes. Main1shot2x2 dataset/gallery panel. Swapped gallery pairs are explicit new stress slices, not historical16-cell reproduction.
Locked task seeds, query count, pseudo mass, top-k/ranks and tuning set are in protocol.json. Development uses DTD only, two seeds100tasks each per gallery; choose one global candidate lambda from0.03,0.1,1 by mean dev accuracy, ties prefer0.1. Main uses5fresh seeds200tasks each; no retuning after it opens. All image pools historically viewed.
5shot candidate preserves R2 support-only behavior; report DTD and EuroSAT same-gallery metrics as protocol safeguards, not evidence of a new5shot method.

Minimal code map: copy read-only original methods.py for R2; implement one weighted ridge function, residual construction and diagnostic controls; evaluate.py owns fixed episode generation and paired outputs. No unrelated source edits.
Controls: R2, support prototype, support ridge, support ridge with matched total weight, pseudo mean-only, original pseudo distribution, residual-only, trace-matched isotropic, random-gallery residual. Lambda-tuned support ridge uses the same dev budget as candidate. Original pseudo distribution remains historical-inspired local control, not a paper reproduction.
Attribution: no normalization after recentering. Verify weighted-loss mean/scatter identity and query-independence, plus reference R2 API parity on fixed examples.
Statistics: raw scores/predictions/indices/seeds stored; paired, seed-stratified bootstrap of fixed-pool episode differences. Correlate matched and swapped tasks within each dataset by using identical draws. Report cell metrics and macro; do not reduce to one selected scalar.
Efficiency: frozen cachedfeatures, CPU batched low-rank ridge, common retrieval, no GPU competition with active encoding. Small batch16.
Minimum: valid paired outputs. Solid: gate satisfied against simple controls with mechanism-specific evidence. Maximum: independent extension and broader analyses only if solid.
Stop/abandon: provenance mismatch, leakage, nonfinite outputs, disk<10GiB,72hroundcap. Statistical failure is a valid negative result and routes to decision rather than endless tuning.

Process: root asset run bash-7935853a was launched before branch creation and continues there to avoid duplicate encoding; experiment code is in current run worktree and consumes only completed, hash-verified root caches. This exception is explicit and does not move the result's lineage.
Next: implement and validate algebra/permission wiring -> finish assets -> baseline measurement/confirmation -> dev selection -> locked main -> record_main_experiment -> decision.

## Exit 2026-09-12 10:13 UTC
Completed: canonical assets, accepted baseline-2da45f14, development selection, 6000-cell-episode comparison, validation and main artifact run-856a6e62. Residual-only route rejected: -1.6403 pp vs R2; all promotion gates false. R2 remains incumbent. Next anchor idea: structurally distinct objective/representation route, decision-9fd18783. Do not rerun feature recovery or tune this residual formulation on observed evaluation cells. All outputs remain immutable evidence.
