# Fixed gallery-density correction: verified negative development result

The primary local correction reaches77.268667% versus77.374% for the cumulative parent, a difference of-0.105333pp,95%CI[-0.186667,-0.023333]. It fails all five predeclared gate components. Retain the existing parent; this rejects the single fixed recipe, not every reference-normalization method.

| Target/gallery | Parent% | Local% | Global% | Shuffled% | Local-parent pp[95%CI] |
|---|---:|---:|---:|---:|---|
|dtd_dtd|80.010667|79.720000|79.866667|79.877333|-0.290667[-0.437333,-0.146667]|
|dtd_eurosat|76.304000|76.560000|76.186667|76.096000|0.256000[0.074667,0.440000]|
|eurosat_dtd|73.485333|73.464000|73.394667|73.682667|-0.021333[-0.170667,0.130733]|
|eurosat_eurosat|79.696000|79.330667|79.565333|79.696000|-0.365333[-0.522667,-0.210667]|

The local recipe repairs1308query occurrences and spoils1466, for158net extra errors. DTD macro changes-0.017333pp andEuroSAT-0.193333pp. The DTD-query/EuroSAT-gallery condition improves0.256pp, but both matched-gallery cells decline. That isolated positive condition cannot replace the failed predeclared overall target. These are repeated occurrences within1000unique episodes, not independent images.

Local-minus-global is+0.015333pp[-0.062667,0.090683]; local-minus-shuffled is-0.069333pp[-0.159333,0.016000]. The current evidence does not attribute a reliable advantage to local density alignment. Selected mean density decreases, but the unique-neighbor statistic changes in different directions across heads/cells; neither statistic is itself classification success.

## Fixed method and comparison boundary
Candidateidea-dd6d44bc,decision-38882713,branchrun/gallery-density-20260913. In each inherited raw/centered scatter head, a label-blind128image gallery reference gives a self-excluded top10mean cosine penalty. Retrieval ranks cosine minus0.5penalty, then retains exactly64neighbors with uniform mean/half influence. Original ridge0.1, sourcegammaDTD10/EuroSAT1, etaDTD0/EuroSAT10, query-local consistency and equal head averaging are unchanged. All strengths/reference rules were fixed before outcomes; no fitting or search. Global-reference mean and identity-shuffled penalty controls preserve the retrieval and influence budgets.

Same500one-shot tasks perDTD/EuroSAT domain,fiveseeds,two paired full galleries:2000taskconditions/1000uniqueepisodes. The five canonicalPetsmetrics remain unchanged and unmeasured. No Caltech result or tuning. Both domains and imagepools are historically exposed development. Paired5000bootstrap replicates stratifydomain/seed and average galleries within task. Intervals are descriptive and conditional on fixedpools; no independent-domain or population-level claim.

## Validation and resources
Formalbash-71fbe8c3 finished197.808s. All2000parent score matrices match with maximumerror0; all saved score/prediction/accuracy mappings agree. Independentbash-6393e17c verified600000predictionentries,12first/middle/last taskconditions through NumPy SVD, independent neighbor identities and direct augmented-design ridge; maxscoreerror9.99e-16. All21confidenceintervals reproduced by count-weighted bootstrap, maxerror1.11e-16pp. Initial6numericalcases and query/class/gallerypermutation checks passed. Original historical full-audit failures remain unchanged; this study does not retroactively repair them.
All inputs use the existingtorch/NumPy environment and verifiedfeature hashes. New recipe code/protocol plus inherited code hashes are retained. No downloads,newencoding,clouduse or extraspend. The first validation-artifact submission lacked its parent link and the first update omitted its node_type; both were rejected before recording, then corrected. One attempted Git checkout did not create a branch; artifact.branch created the new branch in the same physicalworktree. These control-plane errors did not rerun or alter the scientific experiment.

## Decision and next question
Close this fixed density recipe with no k/reference/coefficient/selector rescue. Lower background similarity and modest one-cell gains do not warrant promotion. Numerical-contract repair remains separate from algorithmic improvement. Next inspect feasibility of an information family absent from global-vector edits, such as spatial descriptors from the already-local encoders, against evidence synthesis. Do not assume spatial features are available, informative, affordable, or a selected new method until source capability, disk cost and prior art are checked. Any next evaluation follows a separately selected durable idea under the existing autonomous authorization.

Prior-work comparison:artifacts/idea/gallery_density_survey.md and pre_idea_drafts/gallery-local-density.md. CSLS/NNN already establish reference normalization; this is a restricted transfer experiment, not an original normalization algorithm or SOTA result.
