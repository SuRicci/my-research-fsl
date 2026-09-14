# Selected source maps: whole-feature geometry does not isolate transfer harm

The deferred whole-geometry preservation hypothesis is not ready for training from this diagnostic. Across the fixed models, larger updates often accompany worse saved transfer accuracy, but pairwise relational distortion does not supply a consistent adverse association beyond update magnitude. This is observational evidence with small, correlated model groups; it does not falsify all geometry regularizers or prove the cause of transfer failure.

## Evidence and comparison
All36source-composition selected models were reused, including6zero-step models. No parameters, checkpoints, classifiers, source roles or target predictions were changed. All13424eligible cached image vectors were transformed (DTD1877,EuroSAT5400,Flowers6147). Each dataset used4096prespecified class-uniform image pairs:2048same-class plus2048different-class, sampled with replacement across pairs. The distinct unordered-pair counts were4039/4095/4063. These are sampled relations, not exhaustive all-image pair counts.
The diagnostic first normalizes cached fused vectors in float64, then applies the immutable residual map and renormalizes. It measures displacement, raw update energy, cosine distortion and sampled class-gap change. Original float32 saved predictions provide source/transfer accuracy differences; no equality of these diagnostic transformed vectors with every old float32 intermediate is claimed.

## Source-observable relation to saved transfer accuracy
Correlations below exclude the6zero-step models. Positive partial correlation means more source pair-distortion accompanies less transfer harm after linear displacement adjustment, contrary to a simple monotonic penalty motivation. The sign is not a validated predictive rule.

| Source / shot | Nonzero models | Source displacement correlation | Source distortion correlation | Distortion correlation adjusted for displacement |
|---|---:|---:|---:|---:|
| dtd_k1 | 6 | -0.9572 | -0.8991 | 0.1691 |
| dtd_k5 | 9 | -0.9717 | -0.9645 | 0.0403 |
| eurosat_k1 | 7 | -0.9736 | -0.9523 | 0.5857 |
| eurosat_k5 | 8 | -0.9383 | -0.9316 | -0.0346 |

One-shot partial correlations measured on the already exposed transfer-domain features are-0.9285(DTD-source toEuroSAT) and+0.2542(EuroSAT-source toDTD), inconsistent in sign. Those features are diagnostic only and cannot become a source-selection signal. On Flowers, the corresponding source-observable adjusted correlations are positive in all4source/shotgroups. No pooled significance tests or independence over domains are claimed.

The mixed source models have a larger sampled target same-class-minus-different-class cosine gap than their original-only counterparts in both1shotdirections, yet both remain below the frozen zero-head classifier. An improved bulk class gap therefore does not establish better episodic classification. This secondary observation is descriptive, not a selected objective or causal result.

## Null and numerical validation
A fixed signed coordinate permutation changes mean squared vector displacement by2.0 while preserving pairwise cosine values to3.34e-16. This illustrates why vector movement is not itself geometric damage. The6identity models have all diagnostic changes below1e-12. All108model/domain sampled NumPy64 reconstructions pass: maximum feature error4.72e-16 and cosine-delta error1.78e-15. An independent covariance-based check validates all48correlations, including analytic partial correlations, to1.13e-14. Every recorded model/input hash matches after execution. Primary elapsed time24.17s; no new download or paid compute.

## Decision boundary and next action
Do not promote a generic whole-Gram preservation penalty or a magnitude threshold from these results. The existing pre-idea reopen condition is unmet: source distortion beyond magnitude does not consistently track harm, and identity remains a stronger classifier in the parent result. A selective regularizer would need a different, source-observable constraint and a comparison against identity plus magnitude-matched updates; no such benefit exists here.
Next compare a structurally different source-training objective (adversarial task distribution or held-out-domain regret) against the already failed source-mixture and confidence-selector families. First audit exact prior method, permissions and computation; do not automatically launch another adapter grid. Retain the cumulative query-consistency/scatter stack as the conditional scientific comparator, and the older OSLO manuscript as a separate narrow technical note.

Scope: auxiliary posthoc diagnosis of a historical source-map family. It is not a causal explanation for Caltech reversal in the untrained scatter stack, not canonical Pets evidence, and not a method improvement. Source/shot correlations involve only6-9selectednonzero models and are confounded by arm and initialization; linear adjustment does not remove all confounding. No submission-readiness claim.
