# View-distribution kernels: completed auxiliary source test

Selected idea-900e7c72, decision-0c5b874b. Current result branch run/view-distribution-kernel-20260913 derives from semantic idea/view-distribution-kernel-20260913 and prior run/representation-scatter-20260913. Same physical workspace avoids redundant multi-GiB materialization. The candidate brief is the selected artifact; the branch transition is recorded through artifact.git.

## Verdict

The fixed known-kernel transfer recipe is rejected for robust promotion. It adds only0.0747pp on EuroSAT and0.0667pp on DTD over an equally source-tuned mean-RBF classifier in1shot, far below the prespecified0.5ppboth-direction threshold. The DTD interval includes zero. Relative to mean-view R2 it loses1.9960pp and1.2853pp respectively. It therefore does not establish useful transferable information beyond the mean under this recipe, and cannot justify Pets/Caltech expansion. This is not a proof that every distribution-aware method fails.

| Source selection → evaluation | Distribution minus mean-RBF1shot(pp) |95%paired interval| Distribution minus mean-R2(pp)|95%paired interval|
|---|---:|---|---:|---|
|dtd_to_eurosat|0.0747|[0.010666666666666647, 0.13866666666666672]|-1.9960|[-2.3413333333333335, -1.658666666666667]|
|eurosat_to_dtd|0.0667|[-0.03733333333333333, 0.1706666666666668]|-1.2853|[-1.5707, -0.9986666666666668]|

## Design and complete evidence

Both domains and both shots are retained: eight gallery-conditioned cells,14methods,2000unique evaluation episodes,4000gallery-task conditions. Gallery-independent methods deliberately produce identical outputs across gallery conditions; these are not independent replications. All ten previous controls are reused byte-for-byte in their original float32 values, with file hashes and exact support/query identities checked before incorporation. No prior model is retuned.

Four new methods: integrated six-view RBF kernel, RBF on normalized view mean, original-view RBF, normalized linear mean ridge. Candidate and RBF controls each receive three inverse bandwidths1/4/16 and three penalties1/0.1/0.01. Source-only selection uses100existing episodes per domain/shot, integer correct counts and smaller-beta/larger-penalty tie order. Every source choice is locked before evaluation. Every nonlinear method selected beta1. No evaluation-driven grid expansion is permitted.

Integrated kernel principle is established prior art (Dao et al.,ICML2019); this test has empirical discrimination value, not method novelty. No new image/model downloads, encoder training, query-batch adaptation, gallery target labels or text labels. Source identity pools are development-exposed, with intervals conditional on those fixed pools.

## Verification

Prevalidation:180independent NumPy/Torch comparisons, max3.353e-14; query partition/order invariance,PSD,repeated-view limit and linear-mean collapse verified. Actual saved-output audit:440independent score comparisons, max2.820e-14below frozen1e-8; predicted classes agree and all source integer choices verify. All130paired means/intervals independently reconstructed via multinomial bootstrap weights, max3.553e-15pp.

Parent scatter strict audit remains failed1/88 and is not relabelled by this new-method audit. The new recipe already fails versus its new independently validated mean-RBF comparator and versus mean-R2, so rejection does not depend on treating the failed scatter audit as trustworthy. No canonical Pets result is submitted: all five required Pets metric ids remain unchanged and unmeasured here. This auxiliary report must not replace a canonical main experiment.

## Interpretation and next action

The comparison weakens the hypothesis that preserving all-pairs crop information is the missing improvement under the current frozen pooled embeddings and source tuning budget. It does not identify the causal reason for the earlier EuroSAT scatter gain. Larger beta is not an untested rescue: all three scales were included and source selection favored1. Ordinary averaging remains a strong information comparator, not a newly accepted target baseline.

Next: return to problem-framed idea selection beyond another fixed view-pooling or gamma grid. Check whether a source-learned representation/head trained on real views is permitted and materially different from the already failed utility/coverage models; challenge it against the cost-aware reduced-view alternative. Inspect existing source-learning contracts and FroFA/closest task-adaptation equations first. Do not implement or select an adapter until a concrete non-duplicate mechanism and equal-budget controls survive the draft gate. Existing paper requires utility/coverage/scatter/kernel mapping before further writing; no submission claim.

## Files and process

protocol.json and lock.json freeze parameters/code/control inputs. outputs/selection_lock.json fixes all16method/domain/shot selections. outputs/cells contains exact scores; outputs/analysis.json includes every cell and control. outputs/output_audit.json and statistical_audit.json record independent checks. Compute bash-c0b2dd84 finished in29.49seconds; saved-output audit bash-07745883 and statistics bash-eb7a3e34 completed.

Storage: current output directory 143023892bytes; free 10.645GiB. Scoped tmp scan found43files with none over50MiB, and prior redundant extraction chunks remain absent. No new deletion in this pass; preserve unique inputs and failure evidence. This scan is not an exhaustive disk inventory.
