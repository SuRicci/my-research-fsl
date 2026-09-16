# Ordered crop correspondence: completed fixed diagnostic

The fixed ordered six-view readout did not improve the incumbent on the paired exposed development tasks. All three promotion conditions failed. The candidate is closed without crop weighting, classifier-regularization search, alternate shuffle selection, or encoder expansion. The incumbent remains unchanged.

## Scientific question and scope

Does preserving known crop-position correspondence recover useful information discarded by view averaging and unordered covariance/set descriptors? This is a conventional spatial coding comparator, not a new equivariant method. The intervention uses the same images, views, support labels, support-derived geometry and C10 logistic estimator as the accepted comparator. No class names, gallery, extra supervision, encoder training, or joint query adaptation enter this run.

## Full results

| Fixed readout | DTD (%) | EuroSAT (%) | Macro (%) |
|---|---:|---:|---:|
| incumbent | 91.010667 | 92.109333 | 91.560000 |
| collapsed | 90.992000 | 92.058667 | 91.525333 |
| ordered | 90.997333 | 92.048000 | 91.522667 |
| shuffle0 | 90.994667 | 92.053333 | 91.524000 |
| shuffle1 | 91.008000 | 92.056000 | 91.532000 |
| shuffle2 | 90.997333 | 92.053333 | 91.525333 |
| shuffle3 | 90.992000 | 92.045333 | 91.518667 |
| shuffle4 | 90.992000 | 92.032000 | 91.512000 |

Five randomized-control accuracies are averaged for the primary comparison; their predictions are not ensembled, and these controls are not five independent experimental replications.

| Frozen contrast | Difference (pp) | Conditional paired 95% interval | One-sided 98.333333% lower |
|---|---:|---|---:|
| ordered-incumbent | -0.037333 | [-0.069333, -0.005333] | -0.072000 |
| ordered-collapsed | -0.002667 | [-0.032000, +0.026667] | -0.034667 |
| ordered-mean_shuffle_accuracy | +0.000267 | [-0.019200, +0.019467] | -0.020533 |

The prespecified macro-gain floor was +0.3 percentage points, all three lower bounds had to exceed zero, and neither domain could lose accuracy relative to the incumbent. All conditions failed. There is no evidence here that matching crop positions is useful at this fixed representation and regularization. The close agreement with collapsed and shuffled controls is insufficient to prove equivalence or establish an upper bound for other spatial models.
- dtd: ordered descriptor corrected 15 incumbent mistakes and destroyed 20 correct predictions, across 37,500 queries.
- eurosat: ordered descriptor corrected 42 incumbent mistakes and destroyed 65 correct predictions, across 37,500 queries.

## Protocol and reproducibility

Each domain uses the same500five-way,five-shot tasks with75queries per task as the accepted comparison. CLIP512andDINO384feature blocks are normalized and equally weighted using the trusted original loader. Per-task joint covariance shrinkage is estimated only from the25support originals and six views. DTD gamma1andEuroSAT gamma0.1remain fixed. The classifier minimizes summed multinomial cross-entropy plus L2penalty with C10and an unpenalized intercept.

For transformed view y_iv, a_i² is the mean squared view norm and m_i is the mean. Ordered features concatenate six views and divide by sqrt(6)*a_i. Collapsed features are m_i/a_i, exactly preserving the ordered representation mean component. The incumbent remains m_i/||m_i||. No transformed view is separately normalized. Five control seeds independently permute the four corners within each original while preserving original and center views, both encoder blocks, per-image means/covariances/norms and the complete view multiset. The same original has the same permutation across tasks. Exact extraction code hash and crop identities are in the selected-idea provenance file.

Paired bootstrap uses10,000draws stratified by domain and five historical task seeds, with seed260916811. Intervals describe conditional task-sampling precision on already exposed pools; overlapping image reuse and prior search prevent interpreting them as independent-domain or search-wide confirmation. Even though the nominal interval against the incumbent is below zero, the practical loss is small and no broad inferiority theorem is claimed.

## Validation and resources

All1,000incumbent predictions matched exactly; maximum score error was 1.6e-10. Direct full-feature versus support-Gram scores, shared view permutation, shuffled mean/covariance action/norm, query singleton and query-order invariants passed on four prespecified tasks, in addition to synthetic and two preflight tasks. Independent dense896-dimensional geometry and direct5376-dimensional logistic checks at four tasks had maximum score discrepancy 3.94e-12. Every saved prediction and per-image permutation was recomputed; all10,000bootstrap draws were independently reconstructed with maximum discrepancy 5.05e-14pp. This is independent implementation validation within the same environment, not an external replication.
Measured execution took 119.06seconds on local CPU4threads. Numerical preflight, literature review, and post-run audit are excluded from that measured duration. No model weights or feature cache copies were created. Input SHA256locks, environment versions, task arrays, raw scores, permutations and bootstrap outputs are retained.

## Decision and limitations

Select closure of this fixed spatial readout and return to evidence-grounded idea qualification. Reject coefficient/crop-weight/C tuning because it would turn a failed prespecified test into exposed-set selection. Reject encoder expansion because the diagnostic did not justify its cost or permissions. Reject immediate negative-paper packaging: this additional bounded failure does not by itself establish a useful general thesis, and prior skeptical review already found insufficient differentiation.

Unresolved: whether a different architecture or independent domain benefits from position correspondence; this run does not answer those questions. Reopen only with a distinct mechanism and independently justified information/permission contract, not the current task scores. The next bounded action is to reassess the residual bottleneck from preserved error evidence and prior reports before selecting another family. No new training or new benchmark is implicitly authorized by this result.

## Evidence

- protocol.json; execution_lock.json; input_manifest.json; preflight.json
- dtd.npz and eurosat.npz: all scores, predictions, exact task identities and permutations
- validation.json; audit.json; bootstrap.npz; summary.json
- run.py; analyze.py; audit.py
- artifacts/idea/ordered-view-information-20260916/REPORT.md,DRAFT.md,view_provenance.json
- literature/ordered-view-information-20260916/reading_scope.json records partial-paper reading scopes.
