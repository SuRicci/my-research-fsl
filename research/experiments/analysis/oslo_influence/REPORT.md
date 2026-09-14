# Retained gallery influence in a failed OSLO transfer

The source-default gallery-only transfer remains rejected. This diagnostic accounts for the final update coefficients; it does not test a remedy.

All4,000 traces reproduced the frozen candidate predictions exactly. Gallery labels were inspected only after fitting. The protocol, code, arrays and complete statistics are retained in this folder.

| Condition | Gallery coefficient (%) | Out-of-episode share of gallery weight (%) | Mean inlier AUC |
|---|---:|---:|---:|
| canonical_pets_k1 | 99.597 | 80.408 | 0.9280 |
| canonical_dtd_k1 | 99.485 | 100.000 | undefined (no in-class gallery) |
| canonical_pets_k5 | 98.053 | 80.731 | 0.9477 |
| canonical_dtd_k5 | 97.424 | 100.000 | undefined (no in-class gallery) |
| fresh_pets_k1 | 99.605 | 80.826 | 0.9263 |
| fresh_dtd_k1 | 99.487 | 100.000 | undefined (no in-class gallery) |
| fresh_pets_k5 | 98.071 | 80.862 | 0.9510 |
| fresh_dtd_k5 | 97.432 | 100.000 | undefined (no in-class gallery) |

Matched one-shot galleries have strong ranking separation (mean task AUROC about0.93) while retaining about80% of their total weight on out-of-episode classes. Ranking quality does not establish calibrated membership probabilities or low total contaminant weight.
In mismatched one-shot galleries, about194 effective gallery samples contribute per class against one labeled support sample. The mean gallery coefficient in the final prototype average is about99.5%. This is an algebraic averaging coefficient; it is not a causal importance score or a measure of vector-direction contribution.
Existing ablations already show that latent inlier weighting improves substantially on closed-set fitting, while zero-update and strong local heads avoid much of the mismatch degradation. This analysis therefore narrows the failure boundary: soft inlier weighting alone did not control accumulated out-of-class influence under this transferred protocol.
Limitations: same domain/classes in the prospective image-disjoint confirmation, exposed canonical outcomes, source task changed from query-transductive to gallery-inductive, no near-duplicate or pretraining-overlap audit. The recorded default-family failure is not a general rejection of OSLO or all robust semi-supervised methods.
Next decision must distinguish a new independently justified mechanism from an explanatory intervention. Neither a larger parameter grid nor a claimed improvement follows from these diagnostics.
