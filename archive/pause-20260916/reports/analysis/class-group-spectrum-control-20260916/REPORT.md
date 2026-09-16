# Fixed added-spectrum class-group orientation control

## Conclusion
The true class-group direction is more harmful than the fixed random-group reference even after matching the entire added covariance spectrum. True-direction accuracy is85.00%, versus89.3204% averaged across99reference directions per block. The sole contrast is-4.3204percentage points, with a descriptive nominal95% whole-block bootstrap interval[-5.1863,-3.4404]. The interval is entirely negative under the frozen rule.

The added component eigenvalue/strength difference alone cannot explain this directional contrast in the tested estimator. This is an intervention on empirical class-group geometry; it does not identify pure semantic information or a deployable suppression rule.

## Design and scope
Reuse all50E13blocks and5000original image identities, all six cached views and both frozen encoders. There are1250unique queries; two five-shot support sets predict the same query set within a block, giving2500prediction events per arm. Random partition references are averaged within blocks and are not independent replications. No new images, encodings, fit selection, query-dependent adaptation or hyperparameter search.

The calibration pool contains25images in eight empirical classes with counts5,3,6,4,1,4,1,1. It is distinct from the five task classes. Weighted class-centre rows sqrt(n_c/25)(mu_c-mu) reproduce parent S and have rank7. Retain the seven descending true-S eigenvalues and substitute each existing random-group basis, paired by eigenvalue order. Keep W, coefficient k=.1*896/trace(W), normalization and centred ridge1 fixed. Offline calibration labels remain an extra information budget.

The added S eigenvalues, trace, rank and coefficient are fixed. The total W+S spectrum can change because orientation relative to W changes. Therefore this is not a total-transform-spectrum control. Sign flips cancel; all5000partition eigensystems meet the frozen separation/rank tolerances, so no tied eigenspace or arbitrary nullspace is used. Ordering remains a specified construction convention, not the only possible orientation intervention.

## Results

| Arm | Accuracy (%) | Role |
|---|---:|---|
| Within-image correction W |90.12|Parent reference; descriptive here|
| W plus true class-group S |85.00|Audited parent bridge and true arm|
| W plus fixed-spectrum random-group directions, mean |89.3204|New frozen reference|

| Sole endpoint | Mean (pp) | Descriptive nominal95% interval (pp) |
|---|---:|---:|
| True direction minus matched random-direction mean |-4.3204|[-5.1863,-3.4404]|

Relative to W, true S corrects58and damages186prediction events, exactly reproducing E13. The random-direction mean is0.7996pp below W (descriptive only). It is not legitimate to divide4.3204by the original5.12loss and call that a causal percentage: the reference intervention and the interaction with W do not define a unique additive causal decomposition.

## Validation
All100true-group heads were recomputed before any new reference-direction scores. Maximum parent score difference1.1657e-15, zero prediction mismatches. Producer maximum added-spectrum error2.4980e-16 and trace error3.6082e-16. The independent audit recomputed all5000spectral constructions and all saved accuracies, the endpoint and bootstrap interval.

For blocks0and49, partitions0,1,99and both support roles, NumPy Gram eigendecomposition, dense896dim transformation and primal ridge independently reconstruct12heads. Maximum score error6.3838e-16, zero prediction mismatches. This is not all-head independent reconstruction or feature re-encoding. Input/code and parent output hashes passed before and after. Raw complete.json and summary.json retain computed_pending_audit; validation.json supplies subsequent passed status.

Managed bash-662d6eb5 exited0. Classification including parent bridge took139.20s and audit3.16s, excluding setup and the score-free qualification. Initial rank-four draft failed because it incorrectly equated task-way count with calibration-group count. The failed draft/code and decision-3f680334 are preserved; corrected weighted rank7qualification passed all5000systems before any new score. No numerical tolerance, partitions, images, endpoint or classifier settings were changed to rescue results.

## Evidence movement and limits
- E13 remains a prospective known-domain new-image result. E14 reuses those now-exposed data and is developmental follow-up, not another independent confirmation.
- The added-component spectrum-only explanation is insufficient under the specified replacement. Relative orientation contributes to this fitted-classifier difference.
- Total-transform interactions, finite-sample class centres, geographical or historical overlap, pretrained data exposure, and cross-domain validity remain unresolved. Calibration categories need not match task categories, and their sample counts are unequal.
- Neither a universal geometry mechanism, novelty of covariance regularization, nor an inference-time gain predictor follows. The nearest directional and spectral work already overlaps the broad explanation.

## Handoff
Candidate idea-875711b9; decision-a13d3333; computation science-cd372dbc; validation science-fcae536d; parent report-a4b47623. Preserve as E14 alongside E13 in the unselected outline and current synthesis. Do not launch another spectrum/strength grid. Next reassess the restricted paper/report claim with this confound resolved while keeping the applicability and novelty gaps explicit.
