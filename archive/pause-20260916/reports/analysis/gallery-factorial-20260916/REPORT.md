# Gallery amount and composition: fixed-state diagnostic

The prespecified four-test gate passed. This is a mechanism diagnostic, not a deployable method claim.

## Primary tests
| Pool | Contrast | Mean pp | Nominal95% interval | One-sided98.75% lower | Pass |
|---|---|---:|---|---:|---|
| canonical | original_size_penalty | 2.2460 | [2.009333333333334, 2.4820166666666674] | 1.9807 | True |
| canonical | balance_relief_interaction | 2.2353 | [2.0113333333333334, 2.462] | 1.9813 | True |
| fresh | original_size_penalty | 2.0040 | [1.7820000000000003, 2.2306833333333334] | 1.7513 | True |
| fresh | balance_relief_interaction | 2.0067 | [1.7940000000000005, 2.22135] | 1.7640 | True |

## All cells: accuracy (%)
| Pool | N | Fraction | Distractor | OSLO | Balanced | Oracle | Oracle balanced | Zero | R2 | CS_l2 | Support C10 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical | 100 | 0 | pets_nontask | 90.907 | 94.752 | 94.629 | 94.629 | 94.629 | 94.523 | 94.448 | 94.333 |
| canonical | 100 | 0 | dtd | 90.963 | 94.304 | 94.421 | 94.421 | 94.421 | 94.251 | 94.112 | 94.333 |
| canonical | 100 | 0.1 | pets_nontask | 94.141 | 95.379 | 96.395 | 96.637 | 94.707 | 94.573 | 94.467 | 94.333 |
| canonical | 100 | 0.1 | dtd | 94.275 | 94.971 | 96.309 | 96.637 | 94.515 | 94.368 | 94.173 | 94.333 |
| canonical | 100 | 0.5 | pets_nontask | 97.437 | 96.717 | 97.717 | 97.157 | 94.987 | 94.712 | 94.621 | 94.333 |
| canonical | 100 | 0.5 | dtd | 97.797 | 97.131 | 97.808 | 97.179 | 94.877 | 94.800 | 94.376 | 94.333 |
| canonical | 300 | 0 | pets_nontask | 89.237 | 94.883 | 94.611 | 94.611 | 94.611 | 94.720 | 94.664 | 94.333 |
| canonical | 300 | 0 | dtd | 84.613 | 94.235 | 94.405 | 94.405 | 94.405 | 94.115 | 94.069 | 94.333 |
| canonical | 300 | 0.1 | pets_nontask | 94.120 | 95.424 | 97.245 | 96.843 | 94.712 | 94.917 | 94.872 | 94.333 |
| canonical | 300 | 0.1 | dtd | 93.331 | 94.821 | 97.208 | 96.813 | 94.491 | 94.496 | 94.165 | 94.333 |
| canonical | 300 | 0.5 | pets_nontask | 97.653 | 96.787 | 97.987 | 97.219 | 94.984 | 95.557 | 95.976 | 94.333 |
| canonical | 300 | 0.5 | dtd | 98.067 | 97.203 | 98.075 | 97.251 | 94.875 | 94.907 | 95.405 | 94.333 |
| fresh | 100 | 0 | pets_nontask | 90.979 | 94.424 | 94.085 | 94.085 | 94.085 | 94.112 | 94.085 | 93.744 |
| fresh | 100 | 0 | dtd | 90.475 | 93.661 | 93.744 | 93.744 | 93.744 | 93.723 | 93.667 | 93.744 |
| fresh | 100 | 0.1 | pets_nontask | 94.240 | 95.064 | 96.187 | 96.309 | 94.152 | 94.195 | 94.093 | 93.744 |
| fresh | 100 | 0.1 | dtd | 94.219 | 94.547 | 95.989 | 96.243 | 93.835 | 93.883 | 93.736 | 93.744 |
| fresh | 100 | 0.5 | pets_nontask | 97.293 | 96.363 | 97.504 | 96.832 | 94.469 | 94.331 | 94.264 | 93.744 |
| fresh | 100 | 0.5 | dtd | 97.648 | 96.821 | 97.664 | 96.856 | 94.323 | 94.349 | 93.821 | 93.744 |
| fresh | 300 | 0 | pets_nontask | 90.067 | 94.480 | 94.099 | 94.099 | 94.099 | 94.355 | 94.208 | 93.744 |
| fresh | 300 | 0 | dtd | 83.811 | 93.637 | 93.739 | 93.739 | 93.739 | 93.605 | 93.605 | 93.744 |
| fresh | 300 | 0.1 | pets_nontask | 94.256 | 95.104 | 96.989 | 96.563 | 94.200 | 94.597 | 94.533 | 93.744 |
| fresh | 300 | 0.1 | dtd | 93.763 | 94.485 | 96.928 | 96.445 | 93.864 | 94.021 | 93.763 | 93.744 |
| fresh | 300 | 0.5 | pets_nontask | 97.603 | 96.464 | 97.949 | 96.941 | 94.525 | 95.501 | 95.859 | 93.744 |
| fresh | 300 | 0.5 | dtd | 98.005 | 96.923 | 98.000 | 96.965 | 94.336 | 94.704 | 95.235 | 93.744 |

## Evidence and limits
Two historical exposed Pets query pools,500 fixed five-way one-shot tasks each; same supports and queries across12gallery variants per pool. N100/300, in-task fraction0/0.1/0.5, Pets non-task or DTD distractors. Nested deterministic sampling, balanced inlier classes. Sampling labels are not prediction inputs.
All numerical results are recomputed from stored float32 scores. Source bridges have maximum score error0 at prespecified audit tasks; all24gallery constructions and output identities are validated. DTD RGB provenance is inherited from the historical audit.
Mass balancing intervenes only at the final update with fixed centering and latent weights. It cannot repair upstream contamination. The oracle removes final weights of non-task members and is not deployable. N changes finite-sample content along with count; this is not a pure multiplicity intervention.
Primary tests average low fractions and both distractor sources within task, separately by pool; four one-sided tests use Bonferroni alpha0.0125,10000seed-stratified bootstrap draws. Descriptive cell intervals are unadjusted; neither task bootstrapping nor two pools establishes new-domain generalization.
The original OSLO paper is query-transductive; this archived gallery-only inductive transfer changes its protocol. Conventional masking, soft clustering and safe weighting are prior art. No novel algorithm or safety theorem is claimed.
Five-shot sixview-geometry-dev91.56% is a separate unchanged comparison contract. This analysis does not report a main result against it.

## Reproduce
Existing Conda torch environment; evaluate.py then analyze.py. protocol.json records frozen controls and primary tests; completion.json records output hashes; output_audit.json records verification. Literature map: literature/gallery-size-composition-20260916/REPORT.md.

## Interpretation and remaining comparison gap
The low-fraction average original-size penalty is2.246pp and2.004pp in the two pools; balancing changes it by2.2353pp and2.0067pp respectively. This supports a conditional final-mass effect, not monotone harm in every condition. At50%inliers balancing loses0.667–1.139pp across all eight cells.
At10%inliers with DTD distractors, mean AUROC is above0.9999 while outside-class final weight share is74–76%. Ranking quality and calibrated usable mass are distinct. Mean inlier score also has a cross-source order reversal; OBSERVABILITY.md is an explicitly post-result descriptive audit, not another confirmatory test.
R2(r64,mix0.5,ridge0.1) and CS_l2 are fixed archived comparators. The idea description called them source-selected too broadly: no fresh source-selection of a scalar for this factorial or this OSLO update was performed. A source-tuned scalar envelope is therefore a remaining comparison gap for any future method claim. This does not invalidate the frozen within-state mass intervention; it limits algorithmic interpretation. No new method is promoted.

## Historical correction2026-09-16
# Omitted history correction
The20260915gallery factorial already crossed N128/256, member fraction0/.5 and same/other-domain outsiders on1000six-view DTD/EuroSATtasks. Its prior complete NumPy validation is backed by all4current design/result hashes and the audit-code hash. This pass did not recompute old scores.
The20260916Pets study adds two exposed Pets pools, single-view representation, N100/300 and fraction.1, plus a new confirmatory within-study test. It is a cross-protocol extension, not first discovery of this factorial gap. The source-scalar completion directly tunes the same OSLO final-update scalar, whereas the older report transfers historical retrieval/ridge settings; this comparator provenance adds information but no new predictor or algorithm.
The earlier seven-ref archive audit inspected only20260913refs and omitted a live20260915analysis folder. Its claim that the factorial was untested was too broad. The omission was found through a cross-reference in an inherited information-budget report. Coverage now indexes all17analysis/*/REPORT.md files in this known legacy workspace;4were absent from current,13matched hashes.16small reference files were mirrored; no large result arrays copied. This is not an exhaustive all-quest report audit.
Downgrade: new-mechanism/first-factorial interpretation rejected; numerical validity retained. No further N/fraction/centering expansion is justified. New source-domain acquisition alone does not resolve a missing predictive mechanism. Next qualify a structurally different observable family only if history and prior art leave a concrete falsifiable gap; do not restart failed utility/calibration networks.

