# Caltech101 prospective transfer: improvement not established

Recorded 2026-09-13T12:36:46.024520+00:00. Both fixed source configurations fail the preregistered improvement gate. The full original numerical audit failed; bounded diagnosis localizes one shared score-ensemble neighbor-selection discrepancy. Negative candidate-versus-support-ridge comparisons do not depend on that compromised control.

## Fixed comparison

500 paired five-way one-shot tasks, 15 queries per class, five seeds, three fixed 1024-image galleries, two pre-existing source parameter sets and all ten methods. Same six-view frozen CLIP/DINO encoding budget for primary controls; original R2 uses fewer views. No class names, gallery labels, query-batch adaptation or Caltech selection.

| Method | DTD settings (%) | EuroSAT settings (%) |
|---|---:|---:|
| original_r2 | 98.032000 | 98.032000 |
| mean_r2 | 98.063111 | 98.063111 |
| mean_CS_l2 | 98.014222 | 98.014222 |
| score_ensemble_r2 | 98.079111 | 98.079111 |
| augmented_support_ridge | 98.330667 | 98.330667 |
| mean_logistic_C1 | 98.341333 | 98.341333 |
| mean_logistic_C10 | 98.392000 | 98.392000 |
| scatter_r2 | 98.029333 | 98.042667 |
| scatter_blend | 98.032000 | 98.045333 |
| query_consistency | 98.032000 | 97.978667 |

## Paired primary contrasts

| Source setting | Comparator | Difference (pp) | Conditional 95% interval (pp) |
|---|---|---:|---|
| dtd | original_r2 | 0.000000 | [-0.129778, 0.128000] |
| dtd | mean_r2 | -0.031111 | [-0.112911, 0.043556] |
| dtd | mean_CS_l2 | 0.017778 | [-0.055133, 0.090667] |
| dtd | score_ensemble_r2 | -0.047111 | [-0.127111, 0.026667] |
| dtd | augmented_support_ridge | -0.298667 | [-0.426667, -0.179556] |
| dtd | mean_logistic_C1 | -0.309333 | [-0.432000, -0.189311] |
| dtd | mean_logistic_C10 | -0.360000 | [-0.485333, -0.237333] |
| dtd | scatter_r2 | 0.002667 | [-0.034667, 0.038222] |
| dtd | scatter_blend | 0.000000 | [0.000000, 0.000000] |
| eurosat | original_r2 | -0.053333 | [-0.184889, 0.075556] |
| eurosat | mean_r2 | -0.084444 | [-0.165356, -0.012444] |
| eurosat | mean_CS_l2 | -0.035556 | [-0.111111, 0.037333] |
| eurosat | score_ensemble_r2 | -0.100444 | [-0.177778, -0.027556] |
| eurosat | augmented_support_ridge | -0.352000 | [-0.478244, -0.231111] |
| eurosat | mean_logistic_C1 | -0.362667 | [-0.486222, -0.242667] |
| eurosat | mean_logistic_C10 | -0.413333 | [-0.534244, -0.290667] |
| eurosat | scatter_r2 | -0.064000 | [-0.104889, -0.025756] |
| eurosat | scatter_blend | -0.066667 | [-0.093333, -0.040889] |

## Gallery dependence

| Source setting / gallery | Candidate (%) | Versus C10 logistic (pp; 95% interval) |
|---|---:|---|
| dtd_caltech101 | 98.360000 | -0.032000 [-0.152000, 0.080000] |
| dtd_dtd | 97.709333 | -0.682667 [-0.848000, -0.525333] |
| dtd_eurosat | 98.026667 | -0.365333 [-0.552000, -0.189333] |
| eurosat_caltech101 | 98.352000 | -0.040000 [-0.160000, 0.069333] |
| eurosat_dtd | 97.656000 | -0.736000 [-0.898667, -0.581333] |
| eurosat_eurosat | 97.928000 | -0.464000 [-0.648000, -0.285333] |

## Integrity and downgrade

Both encoders completed 8677 images and six distinct views in 3840.47 seconds. File hashes, finite values, normalization, complete image order and target/gallery disjoint identities passed. Classification completed in 167.23 seconds; all six prediction banks are retained.

Original audit science-58218531 / bash-19112838 failed at score_ensemble_r2, task 0, DTD gallery: absolute error 1.9640636e-4 exceeds the fixed 2e-5 tolerance. The failed artifact and frozen code/thresholds remain unchanged. Output analysis.json retains its original pending audit field; RESULT.json is the final disposition.

Bounded diagnosis science-3460dae1 checked all 240 prespecified method-sample combinations: 238 passed, 2 failed. Those two are the same source-independent control copied into the two source configurations, not two independent events. Only view 5 / class 2 changes the top64 boundary, between gallery ids 86 and 281, separated by 1.0905868e-7 in float64 similarity. Conditioning a dense solve on the original float32 neighbor identities reduces ensemble score error to 6.40e-8; this is attribution evidence, not a replacement independent pass.

All eight ridge-based methods were checked on the original 30 cell-task samples; only the score ensemble failed. Candidate maximum score error is 7.70e-15. None of these sampled predicted classes changed. Every saved score tensor agrees with its predictions and accuracies; all 72 independently recomputed intervals match exactly. Numerical checks cover fixed samples, not every query. Logistic controls reuse frozen sklearn, source parity and convergence checks; no second logistic optimizer is claimed.

The candidate loses to augmented-support ridge by 0.298667 and 0.352000 pp (both paired intervals below zero) and to C10 logistic by 0.360000 and 0.413333 pp. Both candidate configurations fail the practical improvement gate and the DTD-gallery guardrail. This negative comparison is retained with the stated partial-audit boundary. No source setting is selected by Caltech.

The first diagnostic attempt completed its calculations but failed serializing NumPy neighbor ids. A builtin-int conversion repaired serialization only; both source snapshots and logs are preserved. No numerical formula, sample, tolerance or method parameter changed.

## Handoff

This one-domain result cannot support broad generalization, algorithm novelty, pretraining nonexposure or submission readiness. It does not replace the canonical Pets comparator or rehabilitate previous failed source/incremental gates. Existing OSLO paper receives reference-only metadata, without claim links.

Next: use existing source-domain evidence to reconsider a small, structurally differentiated candidate frontier. Keep Caltech as exposed diagnostic evidence and exclude it from parameter/source choice. Do not repeat unchanged encoding, evaluation, preflight or full failed audit; reopen only with a separately justified numerical protocol or genuinely new evidence.
