# Source-selected scalar comparator

Both source settings were selected before target scoring; all24target cells and source-selection counts passed independent audit. This is an auxiliary comparator, not a new method claim.

| Source | alpha0 | alpha0.25 | alpha0.5 | alpha0.75 | alpha1 | Selected |
|---|---:|---:|---:|---:|---:|---:|
| dtd_test | 76.797 | 77.627 | 77.978 | 76.475 | 66.095 | 0.5 |
| eurosat | 67.907 | 67.892 | 67.667 | 66.188 | 59.820 | 0 |

## Paired target averages
| Pool | Group | Selector | alpha | Accuracy% | Delta OSLO pp [95%CI] | Delta balanced pp [95%CI] |
|---|---|---|---:|---:|---|---|
| canonical | all | dtd_test | 0.5 | 95.550 | +2.005 [+1.828, +2.189] | +0.000 [+0.000, +0.000] |
| canonical | all | eurosat | 0 | 94.684 | +1.139 [+0.905, +1.374] | -0.866 [-0.978, -0.755] |
| canonical | low_fraction | dtd_test | 0.5 | 94.846 | +3.398 [+3.145, +3.658] | +0.000 [+0.000, +0.000] |
| canonical | low_fraction | eurosat | 0 | 94.561 | +3.113 [+2.815, +3.419] | -0.285 [-0.379, -0.189] |
| canonical | high_fraction | dtd_test | 0.5 | 96.959 | -0.779 [-0.949, -0.613] | +0.000 [+0.000, +0.000] |
| canonical | high_fraction | eurosat | 0 | 94.931 | -2.808 [-3.131, -2.478] | -2.029 [-2.254, -1.810] |
| fresh | all | dtd_test | 0.5 | 95.164 | +1.635 [+1.468, +1.809] | +0.000 [+0.000, +0.000] |
| fresh | all | eurosat | 0 | 94.114 | +0.584 [+0.355, +0.808] | -1.050 [-1.177, -0.928] |
| fresh | low_fraction | dtd_test | 0.5 | 94.425 | +2.949 [+2.713, +3.195] | +0.000 [+0.000, +0.000] |
| fresh | low_fraction | eurosat | 0 | 93.965 | +2.489 [+2.219, +2.769] | -0.461 [-0.558, -0.361] |
| fresh | high_fraction | dtd_test | 0.5 | 96.643 | -0.995 [-1.166, -0.834] | +0.000 [+0.000, +0.000] |
| fresh | high_fraction | eurosat | 0 | 94.413 | -3.224 [-3.573, -2.889] | -2.229 [-2.474, -1.997] |

## Every target cell
| Cell | OSLO | Balanced | DTD-selected | EuroSAT-selected | Zero | Support C10 |
|---|---:|---:|---:|---:|---:|---:|
| canonical_dtd_N100_pi0.1 | 94.275 | 94.971 | 94.971 | 94.515 | 94.515 | 94.333 |
| canonical_dtd_N100_pi0.5 | 97.797 | 97.131 | 97.131 | 94.877 | 94.877 | 94.333 |
| canonical_dtd_N100_pi0 | 90.963 | 94.304 | 94.304 | 94.421 | 94.421 | 94.333 |
| canonical_dtd_N300_pi0.1 | 93.331 | 94.821 | 94.821 | 94.491 | 94.491 | 94.333 |
| canonical_dtd_N300_pi0.5 | 98.067 | 97.203 | 97.203 | 94.875 | 94.875 | 94.333 |
| canonical_dtd_N300_pi0 | 84.613 | 94.235 | 94.235 | 94.405 | 94.405 | 94.333 |
| canonical_pets_nontask_N100_pi0.1 | 94.141 | 95.379 | 95.379 | 94.707 | 94.707 | 94.333 |
| canonical_pets_nontask_N100_pi0.5 | 97.437 | 96.717 | 96.717 | 94.987 | 94.987 | 94.333 |
| canonical_pets_nontask_N100_pi0 | 90.907 | 94.752 | 94.752 | 94.629 | 94.629 | 94.333 |
| canonical_pets_nontask_N300_pi0.1 | 94.120 | 95.424 | 95.424 | 94.712 | 94.712 | 94.333 |
| canonical_pets_nontask_N300_pi0.5 | 97.653 | 96.787 | 96.787 | 94.984 | 94.984 | 94.333 |
| canonical_pets_nontask_N300_pi0 | 89.237 | 94.883 | 94.883 | 94.611 | 94.611 | 94.333 |
| fresh_dtd_N100_pi0.1 | 94.219 | 94.547 | 94.547 | 93.835 | 93.835 | 93.744 |
| fresh_dtd_N100_pi0.5 | 97.648 | 96.821 | 96.821 | 94.323 | 94.323 | 93.744 |
| fresh_dtd_N100_pi0 | 90.475 | 93.661 | 93.661 | 93.744 | 93.744 | 93.744 |
| fresh_dtd_N300_pi0.1 | 93.763 | 94.485 | 94.485 | 93.864 | 93.864 | 93.744 |
| fresh_dtd_N300_pi0.5 | 98.005 | 96.923 | 96.923 | 94.336 | 94.336 | 93.744 |
| fresh_dtd_N300_pi0 | 83.811 | 93.637 | 93.637 | 93.739 | 93.739 | 93.744 |
| fresh_pets_nontask_N100_pi0.1 | 94.240 | 95.064 | 95.064 | 94.152 | 94.152 | 93.744 |
| fresh_pets_nontask_N100_pi0.5 | 97.293 | 96.363 | 96.363 | 94.469 | 94.469 | 93.744 |
| fresh_pets_nontask_N100_pi0 | 90.979 | 94.424 | 94.424 | 94.085 | 94.085 | 93.744 |
| fresh_pets_nontask_N300_pi0.1 | 94.256 | 95.104 | 95.104 | 94.200 | 94.200 | 93.744 |
| fresh_pets_nontask_N300_pi0.5 | 97.603 | 96.464 | 96.464 | 94.525 | 94.525 | 93.744 |
| fresh_pets_nontask_N300_pi0 | 90.067 | 94.480 | 94.480 | 94.099 | 94.099 | 93.744 |

## Evidence and limits
Nominal descriptive paired95% CI,10000bootstrap draws stratified by historical task seed, rng260916814; fixed exposed image pools, not independent-domain evidence; equal-weight cells and paired task-level aggregation. No new confirmatory hypothesis.
Auxiliary fixed scalar comparison, both source selections reported; no target tuning and no change to five-shot91.56% incumbent.
Source labels and DTD RGB exclusions inherited, raw pixels not rehashed
Both Petsc pools previously exposed; conditional bootstrap is not new-image generalization
Source gallery is split from archived query bank, differs from original removed gallery
Equal cell weighting is a declared design distribution, not an estimated deployment distribution
Source protocols use200tasks/domain and25queries/task; target protocols retain500tasks/pool and75queries/task. Source inference shares the fixed2-step fit for every scalar; no query feature or label enters fit. The scalar alters only the final support/gallery convex combination; centering and latent weights stay fixed.
Source construction labels are used to build offline diagnostic galleries. Target sampling labels and inlier fractions are never inputs to scalar selection or prediction. No target-alpha sweep occurred: only the two selected controls and mandatory0/.5 bridges were evaluated.
All score argmax, integer-correct counts, source argmax tie rule, paired task identities and input/output hashes were verified. Read protocol.json, execution_lock.json, selection.json, source_validation.json, target_validation.json, output_audit.json, summary.json and results.csv.

## Interpretation and selection uncertainty
DTD selects alpha0.5, exactly the existing mass-balanced comparator; EuroSAT selects alpha0, exactly the gallery-centered zero-update comparator. Thus this completion adds source-selection provenance, not a new target predictor. Both settings are reported; no target-based choice between them is permitted.
Across the12equally weighted target cells, DTD selection improves over the original OSLO transfer by2.0053pp and1.6347pp; EuroSAT selection improves by1.1393pp and0.5844pp. At50%inliers the respective losses versus original OSLO average0.7793/0.9947pp and2.808/3.224pp. Neither fixed scalar is uniformly preferable. These averages depend on the artificial equal weighting of gallery conditions.
EuroSAT alpha0 wins only9of60,000condition-query outcomes over alpha0.25 (0.015pp); repeated conditions are paired, not independent samples. The post-selection paired diagnostic in source_selection_uncertainty.json is not selection-adjusted and supports no confident domain-specific optimum. Alpha0 retains gallery-dependent centering and must never be labeled a no-gallery classifier.
Algorithmic interpretation is now more constrained: mass balancing coincides with a conventional source-selected scalar, so it cannot alone support novelty. A proposed adaptive mechanism must beat both frozen source controls on matched tasks, preserve high-inlier benefit, and use only observable inputs. Existing exposed Pets outcomes cannot establish independent generalization.

## Historical correction2026-09-16
# Omitted history correction
The20260915gallery factorial already crossed N128/256, member fraction0/.5 and same/other-domain outsiders on1000six-view DTD/EuroSATtasks. Its prior complete NumPy validation is backed by all4current design/result hashes and the audit-code hash. This pass did not recompute old scores.
The20260916Pets study adds two exposed Pets pools, single-view representation, N100/300 and fraction.1, plus a new confirmatory within-study test. It is a cross-protocol extension, not first discovery of this factorial gap. The source-scalar completion directly tunes the same OSLO final-update scalar, whereas the older report transfers historical retrieval/ridge settings; this comparator provenance adds information but no new predictor or algorithm.
The earlier seven-ref archive audit inspected only20260913refs and omitted a live20260915analysis folder. Its claim that the factorial was untested was too broad. The omission was found through a cross-reference in an inherited information-budget report. Coverage now indexes all17analysis/*/REPORT.md files in this known legacy workspace;4were absent from current,13matched hashes.16small reference files were mirrored; no large result arrays copied. This is not an exhaustive all-quest report audit.
Downgrade: new-mechanism/first-factorial interpretation rejected; numerical validity retained. No further N/fraction/centering expansion is justified. New source-domain acquisition alone does not resolve a missing predictive mechanism. Next qualify a structurally different observable family only if history and prior art leave a concrete falsifiable gap; do not restart failed utility/calibration networks.

