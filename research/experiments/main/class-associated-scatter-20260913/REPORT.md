# Class-associated scatter: fixed recipe does not improve the retained stack

The class-associated candidate reaches77.31867% versus77.33400% for the retained shared-scatter raw/centered blend. The paired difference is-0.01533pp,95%CI[-0.03333,0.00268]. The0.5ppincrement,positiveCI,class-association and nonnegative-domain gates fail. Retain the incumbent; do not tune shrinkage,gamma,score weights or class calibration using these outcomes. This is a fixed-recipe rejection, not proof that every class-covariance estimator is ineffective.

## Comparison and scope
Selected idea-c62de373; decision-41620c4e; run/class-associated-scatter-20260913; parent report-64b6342f. Same fixed six-view CLIP/DINO caches,1000one-shot tasks,5seeds/domain,2galleries/task,75queries/task. No five-shot or newPets/Caltech outcomes. CanonicalPetsfive-metric contract remains unchanged. These are repeatedly exposed development tasks; intervals are conditional on fixed image pools and repeated gallery conditions are averaged within task.
Half pooled plus half class-specific crop covariance,common pooled-trace scale,inherited sourcegamma10/1; classmetric transforms every input role before unchanged top64/mix0.5/ridge0.1 raw/centered score average. The associated output uses the matching classcolumn from each complete metric head. Equal-compute control averages all fiveheads; foreign control averages the otherfour for each class. No scores are claimed to be calibrated probabilities.

## Primary comparisons
|Reference|Candidate minus reference pp|Paired95%CI pp|
|---|---:|---|
|incumbent|-0.01533|[-0.03333,0.00268]|
|metric_ensemble|-0.00667|[-0.02133,0.00800]|
|foreign_metrics|-0.01067|[-0.02667,0.00533]|
|scatter_cs|0.13067|[0.04533,0.21667]|
|scatter_r2|0.35067|[0.26933,0.43468]|
|mean_cs|1.77867|[1.58200,1.97268]|

Improvement over older components does not demonstrate improvement over the best retained combination. No prospective promotion or method novelty is claimed.

## Per-condition result
|Target/gallery|Candidate%|Incumbent%|Difference pp|Corrected / newly wrong queries|
|---|---:|---:|---:|---|
|eurosat_eurosat|79.66933|79.69600|-0.02667|6 / 16|
|eurosat_dtd|73.47733|73.48533|-0.00800|16 / 19|
|dtd_eurosat|76.25333|76.26667|-0.01333|35 / 40|
|dtd_dtd|79.87467|79.88800|-0.01333|24 / 29|

The candidate changes239of150000query-condition predictions. It corrects81previous errors and makes104previously correct predictions wrong. This small observed intervention is descriptive; query-condition records are not150000independent samples.
EuroSAT andDTD domain means decline0.01733pp and0.01333pp. Allcell differences satisfy the loose-0.5ppnoninferiority floor, but that does not establish efficacy. The EuroSAT matched cell interval is slightly negative; no broad harm claim is inferred from one small cell effect.

## Estimator limitation and post-result interpretation
For fixed0<=rho<1, C_c(rho)=(1-rho)Cbar+rho Cc has the same range as pooledCbar: Cc uses a subset of its residual rows and the pooled coefficient is positive. As gamma tends to infinity, the inverse-square-root transform tends to the common null-space projector. Whenever transformed feature norms stay nonzero and rankings have no ties, all these classmetrics converge to the same normalized predictions. This algebraic limit is a deduction, not a measured finite-gamma equivalence or proof of why every error changed. The complete score-bank diagnostic documents that this particular finite-strength intervention is small. It weakens any broad interpretation that the run tested wholly separate class nuisance subspaces. Do not rescue the run with a rho/gamma grid; a different estimator would require a new justified hypothesis and information source.

## Verification and reproducibility
Numerical validation bash-fb7d0395 completed7.31seconds. Four synthetic dense cases plus40real metric/gallery comparisons passed; maxreal scoreerror1.8902e-14. Neighborindices and allthree aggregation predictions match independent NumPy eigensolves/primal ridge. Shared/gamma0/zero-scatter/identical-class nulls,classpermutation and querypartition invariants passed.
Formal bash-31953351 completed276.79seconds. All8000parent taskprediction comparisons match. Fullbanks,taskids,scores and code/inputlocks are saved. Independent audit bash-54bdd714 recomputed allmeans/transitions and primarypooled intervals with scalar bootstrap accumulation, agreeing within1e-10. Science nodes: science-420d3af4 (run),science-564e0772 (numericvalidation),science-e1b7a5de (completeaudit).
Outputs at main completion:68,997,639bytes; no new data/model downloads or paid resources. Currentfree disk>11GiB. Operational tool-schema corrections are recorded in RUN.md; no method, gate or formal command changed after outcome exposure.

## Next route
Close class-associated fixedshrinkage qualification. Preserve77.334% incumbent and the earlier failedpromotion history. The remaining higher-value outside-family question is source adaptation: inspect existing selected maps for non-isometric displacement and whether preservation could do more than return toward the zero map. This should start as a bounded diagnostic of saved models and existing predictions, not a new training grid. If that audit fails to yield a source-defined falsifiable mechanism, repair accumulated auxiliary evidence into the paper outline/ledger before further prose.
