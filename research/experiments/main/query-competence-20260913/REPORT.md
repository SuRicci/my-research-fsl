# Query-level source competence: fixed recipe rejected

The learned selector reaches77.275333% versus77.334000% for the retained equal blend,77.316667% for the nonlearned maximum-margin selector,77.188% for the centered expert and76.968% for the raw expert. Its paired difference from the incumbent is-0.058667pp,95%CI[-0.103333,-0.013333]. The prespecified0.5pp gain and positive interval criteria fail. Both domain means and all four cell mean differences versus the incumbent are negative. This rejects the fixed score-profile/logistic recipe, not dynamic selection in general.

| Target_gallery | Learned % | Blend % | Margin % | Learned minus blend pp [95%CI] |
|---|---:|---:|---:|---:|
| dtd_dtd | 79.7760 | 79.8880 | 79.8987 | -0.1120 [-0.2267, 0.0000] |
| dtd_eurosat | 76.2427 | 76.2667 | 76.2533 | -0.0240 [-0.1600, 0.1120] |
| eurosat_dtd | 73.4000 | 73.4853 | 73.4480 | -0.0853 [-0.1253, -0.0480] |
| eurosat_eurosat | 79.6827 | 79.6960 | 79.6667 | -0.0133 [-0.0347, 0.0080] |

## Why this comparison is faithful
Candidate idea-3c1c2eb3, early contract report-efa574a8, run/query-competence-20260913. Primary hypothesis: source correctness supervision provides useful query-specific expert selection beyond previous failed support-only objectives. Three fixed logistic heads use18class-invariant features: five sorted scores from each of three experts and three pairwise agreement bits. Source TRAIN standardization only; fixedL2/C1/lbfgs/tol1e-8/max_iter1000, no feature/loss/threshold/regularization grid. Selection uses maximum competence logit with blend/stack/raw tie order. A sigmoid is mathematically monotone; no claim of calibrated target probabilities.

Each source domain has100training and100source-selection tasks with RGB-disjoint identity halves. Each task has ordinary and all-task-excluded1024-source-image galleries. Labels enter only source training targets and source gallery construction. Same-source gamma is10(DTD) or1(EuroSAT); reverse-direction scored outputs were not used for training. Both source models were frozen before opposite-domain evaluation, and their hashes remained unchanged. Source-selection outcomes are descriptive because that half previously selected gamma; no new checkpoint or threshold was chosen from them.

Target evaluation uses the same1000unique one-shot tasks, five seeds per domain and two paired galleries (2000conditions,150000query occurrences). No new image features, targetlabels in inference, query-batch adaptation, target normalization, Pets/Caltech evaluation, or five-shot changes. Canonical five Pets metricids remain unchanged and unmeasured: this is a deliberately scoped auxiliary report, not a canonical main benchmark substitution. Data identities recur across tasks; paired bootstrap is conditional on the reused image pools, not independent domain/population confirmation.

## Failure counts and scope
The learned selector changes1456predicted labels, fixes473incorrect blend predictions, and spoils561correct ones, for a net88additional errors. The unchanged-label selections are not additional errors or successes. The previous finite-label oracle offered2825possible repairs, but that upper bound did not guarantee learnability from these18score-profile features.

Source-selection increments versus blend are-0.006667pp onDTD and+0.073333pp onEuroSAT; source training increments are+0.006667pp and+0.213333pp respectively. Thus substantial gains were already absent at the source-selection stage. The current evidence does not isolate whether feature sufficiency, model restriction, source sample coverage or gallery/domain shift explains the failure. It does not justify a target-driven feature/C/loss rescue.

Pooled comparison with the margin control is-0.041333pp[-0.088000,0.006667]. Beating the raw expert by+0.307333pp does not count as success when the retained blend is stronger. Scores, predictions, model coefficients, source and target identities, feature-source hashes, and all source outcomes are retained.

## Verification and resources
Formal bash-7271fa72 completed49.782s. All6000expert/task endpoint checks exactly matched parent predictions, and all20sampled parent score matrices matched with maximumerror0. New source first-task independent dense experts matched to6.69e-15 or better. All12banks passed class/query-permutation checks; both models converged without warnings.

Independent bash-16c97d04 reconstructed210000source/target query occurrences from scalar sorted features, explicit sigmoid/tie decisions and source-only scalers. All saved selected/margin/fixed predictions and accuracy values match; independent scalar bootstrap matches the primary interval to1e-10. Science run science-868ea88a and validation science-aeaa6b28 separate execution correctness from the failed scientific hypothesis.

No downloads, encoder runs, cloud use or additional spend. Small outputs and preserved source assets; actual storage count is in closure_receipt.json. The paper branch checkpoint was preserved before using an artifact-managed run branch in the same physical worktree; no data copy was made.

## Decision and next question
Retain the fixed blend77.334% and close this exact competence recipe. Do not tune the descriptor set, model regularization or a switching threshold on these exposed outcomes. Neither the oracle nor another source fit alone justifies new-target promotion. The next step should challenge the information bottleneck outside this output-selection recipe: use existing predictions/features and failed-recipe history to identify whether an unused representation/measurement source could address errors shared by all three experts before proposing further training. This is an unresolved candidate question, not a new accepted method or impossibility claim.
