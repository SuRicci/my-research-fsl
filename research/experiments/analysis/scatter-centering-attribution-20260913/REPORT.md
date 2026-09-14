# One-shot centering attribution

Parent: report-f208e67c / science-57067ff7. This is a post hoc mechanistic analysis on the same development tasks, not an independently selected new method or a confirmation dataset.

## Result
Neither fixed hybrid resolves the remaining tradeoff. The original scatter-plus-centering package remains the highest pooled mean among these four counterfactuals; no replacement is selected from individual domains.

For DTD with the EuroSAT gallery, scatter without centering achieves76.2427%. Centering only the retrieval geometry gives75.9253%; centering only the classifier features gives75.7627%; centering both gives75.9440%. Relative to no centering, the isolated retrieval change loses0.3173pp (paired95%CI[-0.5760,-0.0693]) and the isolated feature change loses0.4800pp ([-0.6773,-0.2827]). However, adding centered retrieval after feature centering recovers0.1813pp ([-0.0480,0.4187]). Attribution depends on which other component is held fixed; the evidence does not identify one universally harmful component.

In the opposite mismatched-gallery condition (EuroSAT queries, DTD gallery), centering the classifier with raw retrieval helps1.6027pp ([1.2987,1.9093]); subsequently changing retrieval loses0.2373pp ([-0.4640,-0.0160]). Matched-gallery cells also show different effects. Neighbor overlap ranges from36.665% to80.775% across the four conditions, indicating that centering materially changes the selected gallery set, but overlap alone is not a validated predictor of benefit.

Pooled over both domains and gallery conditions, the complete package improves0.2200pp ([0.0900,0.34735]) over scatter alone. The two simple hybrids do not exceed the complete package in pooled mean. These findings preserve the parent increment over meanCS while rejecting a universal repair by swapping only the fixed retrieval or classification geometry. Selecting a different hybrid from each target domain would be an oracle choice and is not proposed.

## Validation and comparability
All2000 task conditions share the parent supports, queries, galleries, source-selected gamma, scatter transform, double precision, mixing weight0.5 and ridge penalty0.1. Only one-shot cases are included; the five-shot penalty finding is already resolved in the parent study. All4000 endpoint task comparisons exactly match the parent predictions. Eighty fixed counterfactual score cases match independent dense NumPy eigendecomposition and primal ridge, with exact neighbors/predictions and maximum score error1.0381e-14. Paired intervals use the parent's already independently audited5000-replicate bootstrap helper; this analysis does not claim a second independent implementation of that helper.

Run bash-77ef4328 completed47.487s, with approximately0.75MB outputs. No new data, downloads, encoders, training, gallery labels or query adaptation. Parent files remain unchanged. The outputs contain full task predictions, sampled scores, neighbor overlap, protocol and hashes.

## Handoff
Close this one-question analysis. Keep the verified77.188% cumulative stack as a conditional improvement candidate and the75.54% meanCS stack as the accepted reference; the stronger promotion gate remains failed. Do not repeat the hybrids, the five-shot penalty grid, or a target-specific component choice. The next idea pass should inspect whether fixed prediction fusion can retain both geometries without learning a failed support-only selector; first check previous fusion evidence and exclude overlap. This is a hypothesis to challenge, not an approved or measured improvement. Generic source-trained geometry regularization remains the outside-family alternative. Before future manuscript work, reconcile the current paper scope and map only relevant auxiliary evidence.
