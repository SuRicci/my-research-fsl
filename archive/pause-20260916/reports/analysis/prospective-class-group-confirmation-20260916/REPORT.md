# Prospective EuroSAT class-group confirmation

## Conclusion
The two-part pattern specified before extraction is supported in this fixed known-domain experiment. Relative to 99 group-size-preserving random partitions, the true-label contrast between adding class-centre covariance and adding within-class original covariance is -7.5604 percentage points (nominal 97.5% whole-block bootstrap interval [-9.5938, -5.7681]). Adding the true class-centre component to within-image covariance reduces accuracy by 5.12 points [-6.72, -3.60]. Both upper bounds are below zero, as required by the frozen criterion.

This supports a repeatable empirical class-grouping effect for the whole tested estimator. It does not identify a pure semantic cause, establish a safe suppression rule, or validate a new few-shot method.

## Fixed design and information permissions
50 blocks use 5,000 distinct RGB-qualified EuroSAT originals. Each block assigns 25 calibration images and three disjoint 25-image sets: support R, support A and evaluation B. Each classification problem is five-way, five-shot with five queries per class; R and A separately predict the same B and are averaged inside the block. Thus there are 1,250 unique query images and 2,500 prediction events per arm, not 2,500 independent queries. Calibration uses 25 extra offline labels per block. No query labels select transformations, no gallery is used, and no hyperparameters or partitions were selected after results.

W denotes within-image view-residual covariance, U the covariance of image means around their empirical class centres, S the covariance of those centres, and B=U+S. All arms retain the same W-derived coefficient, six frozen views, 896-dimensional fused representation and centred ridge classifier with regularization 1. The protocol, image assignment and random partitions were frozen before fresh feature extraction. See PROTOCOL.md and the input/code locks.

## All true-partition arms

| Covariance used | Accuracy (%) | Difference from W (pp; descriptive) |
|---|---:|---:|
| Identity transformation | 87.60 | -2.52 |
| W | 90.12 | 0.00 |
| W + U | 90.48 | +0.36 |
| W + S | 85.00 | -5.12 |
| W + U + S | 85.32 | -4.80 |

Only the two prespecified endpoints below carry confirmation intervals. The small U increment and W-versus-identity difference are descriptive here; no new inferential endpoints were added.

| Prespecified endpoint | Mean (pp) | Nominal 97.5% interval (pp) |
|---|---:|---:|
| True S-minus-U contrast minus mean random-partition contrast | -7.5604 | [-9.5938, -5.7681] |
| Add true S to W | -5.1200 | [-6.7200, -3.6000] |

The true S-minus-U contrast is -5.48 pp; the mean random-partition contrast is +2.0804 pp. These are whole-estimator comparisons, not semantic-null p-values. Against W, W+S corrects 58 and damages 186 prediction events (net -128/2500); W+U corrects 44 and damages 35 (net +9/2500).

## Relationship to prior evidence
The parent diagnostic reported a EuroSAT true-versus-random contrast of -9.319 pp and an unestablished DTD contrast. This confirmation uses new assigned image identities and fresh calibration pools in the same known domain and class universe. The direction persists under the joint change in calibration, support and query images. Old and new intervals are not pooled, and magnitudes are not treated as directly interchangeable estimates because the block designs differ. DTD has no new confirmation in this experiment.

## Validation and provenance
The managed pipeline bash-0efd8ec2 exited 0 after extracting both encoders, completing all 50 blocks and running the frozen audit. All saved endpoints and intervals were recomputed; all 5,000 partition decompositions and image-role/block uniqueness checks passed. Independent dense eigen-decomposition and primal ridge reconstruction of 24 prespecified heads gave maximum score error 7.7716e-16, with zero prediction mismatches. This is a sampled numerical reconstruction, not an independent re-encoding or refitting of every head.

The old-image bridge passed all six views for four preexisting images with both encoders. An earlier concurrent ZIP-read failure was repaired only in I/O, with the original failure and prior hashes retained. No coefficient, task assignment, encoder, or scientific threshold changed. summary.json and complete.json retain their original computed_pending_audit snapshots; validation.json supplies the subsequent passed status.

Encoding took 1677.96 s, classification 375.13 s and audit 8.93 s in the successful resumed pipeline. These exclude earlier setup, bridge checks and failed work and are not end-to-end deployment latency. Cached weights were reused locally.

## Limits
- Same already examined EuroSAT domain, classes, views and coefficient formula. This is prospective new-image source-domain evidence, not unseen-domain generalization.
- Exact RGB exclusions and audited history do not prove geographic independence, absence of near duplicates, full historical non-use or exclusion from pretraining.
- Class grouping changes directions, spectrum and strength together. Empirical class centres are not pure population semantics, and within-class residuals are not pure nuisance.
- Bootstrap intervals describe whole-block variation under this finite frozen design. They do not remove domain selection or previous researcher-choice uncertainty.
- Both calibration and classification originals change together; their separate contribution is unidentified.
- Extra calibration labels support a diagnostic intervention, not an unlabelled deployment-time selector.

## Handoff
Computational node science-79ac86a0 and validation science-d8db463c preserve execution and numerical trust. Map this as E13 in the unselected candidate and retain E12 separately. The next question is whether the strengthened empirical distinction has a defensible contribution relative to existing augmentation and covariance work. No new samples, parameter sweeps, replacement pools or deployment claims are justified by this result alone.
