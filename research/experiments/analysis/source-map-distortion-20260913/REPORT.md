# Saved source-map distortion audit

## Verdict
Saved learned maps change pairwise feature geometry, but the magnitude of that change is not a reliable standalone explanation of transfer loss. In particular, mixed-source maps can distort target pairwise similarities more than original-source maps while transferring better. This audit does not justify a generic Gram-preservation penalty or another penalty sweep. Preserve the untrained/current stronger reference and proceed to evidence integration before additional writing.

## Scope
All36previously selected residual maps were inspected. Deterministic256-image subsets come from original-source selection images, Flowers selection classes and existing target-evaluation pools; exact indices and pool hashes are in result.json. No training, model selection, new accuracy evaluation, new dataset or encoder pass occurred. Accuracy changes are parsed from the old full prediction banks. This one-view source-composition study is not a direct experiment on the newer six-view77.334%stack.

## Key counterexample
For EuroSAT-source one-shot fits, mixing source domains increases target Gram RMSE from0.09769to0.13080, yet reduces transfer loss from6.06667to5.17778pp versus the zero map. For five-shot fits the analogous RMSE rises0.06872to0.07151 while loss shrinks0.98222to0.77600pp. More preserved global target Gram entries therefore do not monotonically identify the better learned arm in these observations. Source Gram changes and target Gram changes also differ strongly by domain. This is a descriptive counterexample to a simple distortion proxy, not a causal theorem or evidence that all regularizers fail.

## Complete group summary
|Source/shot/arm|Source-validation gain pp|Target minus zero pp|Source Gram RMSE|Target Gram RMSE|
|---|---:|---:|---:|---:|
|dtd_k1_original|1.98222|-1.52800|0.23410|0.05745|
|dtd_k1_flowers|0.00000|0.00000|0.00000|0.00000|
|dtd_k1_mixed|1.76889|-0.89511|0.21577|0.05715|
|dtd_k5_original|0.90769|-1.43111|0.21568|0.07345|
|dtd_k5_flowers|0.05641|-0.01689|0.04374|0.01754|
|dtd_k5_mixed|0.78462|-0.99556|0.17477|0.04788|
|eurosat_k1_original|12.24444|-6.06667|0.57276|0.09769|
|eurosat_k1_flowers|0.04444|-0.08444|0.00503|0.01700|
|eurosat_k1_mixed|11.36000|-5.17778|0.56522|0.13080|
|eurosat_k5_original|3.71282|-0.98222|0.41186|0.06872|
|eurosat_k5_flowers|0.04103|-0.08178|0.02969|0.07320|
|eurosat_k5_mixed|3.42564|-0.77600|0.37980|0.07151|

## Validation and limits
All selected weight hashes andinputfeature hashes were checked. IndependentNumPy residual-map evaluation agrees withTorch within2.932e-7. The orthogonal synthetic control has substantial point displacement(mean1-cosine1.0555) but GramRMSE1.585e-16, confirming why point displacement alone is insufficient. Identity mappings recover zeroGramdifference. Script/code/protocol locks rechecked afterrun; no failed scientific execution.
Runtime bash-31594c22 completed1.44seconds. These models share data,initializationseeds and selection exposure; the36fits are not36independent generalization trials. Selectedstep0models should not be counted as trained improvements. No claim about unseen domains or causal effect of regularization is supported.

## Next edge
Do not open generic source-geometry-preservation training from this audit. A future training idea would need a source-defined selective geometry/objective that changes something beyond identity recovery and survives an explicit stronger alternative; current descriptor evidence does not identify one. The immediate next task is to reconcile the preserved technical-note outline and evidence ledger with completed auxiliary studies, separating relevant manuscript evidence from reference-only optimization history. No automaticpaperclosure or submissionclaim.
