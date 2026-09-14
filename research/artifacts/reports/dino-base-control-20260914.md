# Fixed DINOv2 base matched-control result

The stronger frozen representation does not pass the predeclared promotion gate. The retained recipe remains effective against matched simple heads on the exposed source datasets, but its gain over the previous small-backbone incumbent is only +0.224 percentage points (95% conditional paired CI -0.070 to +0.524), below the +0.5 threshold and compatible with zero.

## Accuracy

| Method | DTD (%) | EuroSAT (%) | Equal-domain mean (%) |
|---|---:|---:|---:|
| base_consistency | 78.2507 | 76.9453 | 77.5980 |
| base_scatter | 78.1920 | 76.9453 | 77.5687 |
| base_mean_r2 | 78.0653 | 72.4547 | 75.2600 |
| base_mean_CS | 77.8680 | 73.1987 | 75.5333 |
| base_support_ridge | 76.8507 | 71.1840 | 74.0173 |
| base_logistic_C1 | 76.8773 | 69.8000 | 73.3387 |
| base_logistic_C10 | 76.7600 | 70.7147 | 73.7373 |
| base_augmented_ridge | 76.8347 | 71.9920 | 74.4133 |
| small_consistency | 78.1573 | 76.5907 | 77.3740 |

## Paired comparisons

| Control | Candidate minus control (pp) | 95% conditional CI (pp) |
|---|---:|---|
| base_mean_r2 | +2.3380 | [+2.1493, +2.5194] |
| base_mean_CS | +2.0647 | [+1.8840, +2.2493] |
| base_support_ridge | +3.5807 | [+3.3373, +3.8287] |
| base_logistic_C1 | +4.2593 | [+4.0093, +4.5207] |
| base_logistic_C10 | +3.8607 | [+3.5960, +4.1340] |
| base_augmented_ridge | +3.1847 | [+2.9547, +3.4247] |
| small_consistency | +0.2240 | [-0.0700, +0.5240] |

## Gate and interpretation

The sole pooled control failing the superiority requirements is the previous small-backbone incumbent. Every per-domain comparison has a nonnegative point estimate, and all four gallery-specific lower bounds versus that incumbent exceed -0.5 pp. Failure therefore rejects this fixed promotion package, not the existence of all representation benefit. The candidate is +2.0647 pp over matched mean-CS and +2.3380 pp over matched mean-R2, but most benefit predates this backbone change.

The incremental consistency term over base scatter is +0.0293 pp [0.0033, 0.0560]. This small conditional effect is not a new robust improvement claim. Recipe-by-backbone interactions are +0.1727 pp [-0.0493, 0.3940] versus mean-R2 and +0.2307 pp [0.0087, 0.4500] versus mean-CS; these exploratory intervals do not establish a causal scaling law.

## Validation and provenance

- All 14,153 image features validated for hashes, RGB identities, dimensions, finite values and unit norm.
- 1,000 unique 5-way 1-shot tasks, reused over 2 galleries: 2,000 task-gallery conditions and 9 methods. Cross-gallery support/query/class/seed identity verified.
- All 1,350,000 stored predictions checked against saved score argmax and query labels; 34 intervals recomputed separately.
- Original chained audit reconstructed primary and scatter at task 99 in every cell. Coverage audit additionally checks every declared index [0, 99, 199, 299, 499]: 40 method-task-cell cases, max absolute difference 1.4432899320127035e-15. Locked input hashes unchanged.
- Logistic optimizer is not independently implemented; its stored arithmetic and padding controls were verified.
- Compute session bash-8890632d and coverage-audit session bash-0bfb6cd3 completed successfully. Classification/analysis wall time 747.68 seconds.
- Protocol, run_manifest, four score banks, analysis.json, audit.json, declared_indices_audit.json and result_summary.json are in experiments/main/dino-base-control-20260914.

## Boundaries and next action

- 1000 unique tasks, reused over two galleries; not 2000 independent tasks
- Conditional intervals do not account for repeated development-set exploration
- Same images and views but unequal encoder compute and different pretrained checkpoint; not a pure model-size intervention
- Original complete.json audit_pending describes compute-time status; audit.json and declared_indices_audit.json now establish validation completion
- Logistic optimizer not independently reimplemented; NumPy independent algebra covers primary and scatter at all 5 declared indices in each of 4 cells
- Five canonical Pets metrics remain unmeasured and unchanged; no target reevaluation. This is an auxiliary result, not a replacement baseline or a canonical main-run submission.
- Reject promotion of this fixed base recipe. Preserve features for justified reuse; do not sweep checkpoints, dimensions, or hyperparameters.
- Next: one bounded feasibility comparison of the previously identified verification-head mechanism against evidence synthesis. Do not start an experiment until a concrete selected protocol is durable.
