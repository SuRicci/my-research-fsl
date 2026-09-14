# Matched-budget source composition: verified development result

## Verdict

The fixed mixed-source protocol does not qualify for canonical target evaluation. Mixing improves over the same head trained on the original source alone in both development directions, but remains worse than the untrained fixed feature head and strong one-view references. Do not interpret the local mixed-versus-original gain as an overall algorithm improvement or a general source-diversity law.

## Fixed comparison

36fits:two directions,two shots,three paired initialization seeds,three training-source arms. Every fit has100training episodes,200updates,batch5 and checkpoint opportunities0/50/100/200. Encoders are frozen CLIP-B16 and DINOv2-S14,224singleview, equal block normalization/fusion. The unchanged rank32 residual ReLU head feeds centered ridge0.1. Each fit receives1000episode presentations.

Original-source train/selection images are disjoint within each class. Flowers has a deterministic82training/20selection class split after6147exact-RGB-eligible rows. Mixed training uses50original and50Flowers tasks; original-only and replacement-only each use100tasks. All arms select on the same50original+50Flowers validation tasks. This is training composition conditional on shared Flowers validation-label exposure, not equal total distinct data exposure. Source5shot has13queries/class; evaluation always15. Actual unique images/classes and exact task manifests are retained in outputs/exposure.json and outputs/tasks/.

All36choices lock before any transfer evaluation. The2000unique evaluation tasks reuse5seeds and500tasks/domain/shot;8gallery comparison conditions do not create independent replications for gallery-free heads. DTD/EuroSAT are exposed development; Flowers official test is explicitly source development. No new Pets/Caltech result, canonical baseline five metric ids unchanged.

## Primary paired differences

| Evaluation direction | Mixed - original pp [95% CI] | Mixed - zero head pp [95% CI] | Mixed - R2 macro pp | Mixed - CS_l2 macro pp |
|---|---:|---:|---:|---:|
|dtd_to_eurosat|0.6329 [0.3627, 0.8862]|-0.8951 [-1.1360, -0.6578]|-2.2684|-3.0564|
|eurosat_to_dtd|0.8889 [0.5911, 1.2018]|-5.1778 [-5.7316, -4.6347]|-6.3444|-6.2444|

All three paired initialization effects are positive for mixed versus original in both1shotdirections. However, mixed-minus-zero95%intervals lie below zero in both directions. Flowers-only1shot selection chooses zero training in all3DTD-source cases and2/3EuroSAT-source cases; near-zero adaptation is a strong local counterexample to a benefit claim. The frozen gate requires>=0.5pp/positive lower CI versus original AND all matched oneview controls in both directions, plus all-cell lower CI>=-0.5pp; it fails. Mixture-specific complementarity also fails.

## Full condition accuracy (%)

| Direction / gallery / shot | Original | Flowers | Mixed | Zero ridge | R2 | CS_l2 | Selected linear |
|---|---:|---:|---:|---:|---:|---:|---:|
|dtd_to_eurosat_eurosat_k1|67.4347|68.9627|68.0676|68.9627|74.1227|73.9227|68.0933|
|dtd_to_eurosat_dtd_k1|67.4347|68.9627|68.0676|68.9627|66.5493|68.3253|68.0933|
|dtd_to_eurosat_eurosat_k5|88.1902|89.6044|88.6258|89.6213|87.1360|89.6293|89.6213|
|dtd_to_eurosat_dtd_k5|88.1902|89.6044|88.6258|89.6213|87.1360|89.6293|89.6213|
|eurosat_to_dtd_dtd_k1|69.3520|75.3342|70.2409|75.4187|78.4160|78.5173|75.3360|
|eurosat_to_dtd_eurosat_k1|69.3520|75.3342|70.2409|75.4187|74.7547|74.4533|75.3360|
|eurosat_to_dtd_dtd_k5|89.2578|90.1582|89.4640|90.2400|90.2720|90.1013|90.2400|
|eurosat_to_dtd_eurosat_k5|89.2578|90.1582|89.4640|90.2400|90.2720|90.1013|90.2400|

## Selection-versus-transfer diagnostic

This posthoc breakdown reads existing predictions only; it does not reselect a model or establish why harm occurs. In EuroSAT-source1shot, the mixed head improves the common selection score by11.36pp: +23.18pp on held-out EuroSAT images and -0.46pp on held-out Flowers classes. Yet it loses5.18pp to the zero head on DTD evaluation. Thus even common two-domain source validation can favor a transformation that transfers poorly to another domain. This sharpens the next question toward task/domain conditioning and preservation of pretrained geometry; it does not justify retuning this mixture from the evaluation outcomes.

## Independent verification and uncertainty

Formal bash-184440b0 completed in30.03s. Audit bash-98cb17e0 independently reconstructed324score cases (maximum absolute error1.31e-06<3e-5), reevaluated1602000query predictions and checked60paired statistics. Code/input hashes, selected models, shared initialization and source-selection lock passed. Numerical precheck bash-0528cc2b passed1600taskrows/12cases after the first import collision was repaired; failure bash-c4f47d48 retained.

Intervals use5000paired task draws stratified by5taskseeds and average over3fixedtrainingseeds. They are conditional on fixed image pools and these trainingseeds; not uncertainty over new source domains, pretrained data or a random trainingseed population. No prospective held-out improvement is established.

## Resource and cleanup

Only compact predictions, selected models, fixed audit scores and task manifests were written: study currently10.22MiB.13temporary index JSON files were losslessly compressed and byte-roundtrip verified, saving41.54MiB logicalbytes. Restore original index.json from adjacent gzip before rerunning historical extraction; receipt artifacts/reports/source-composition-cleanup-20260913.json records every hash/path. Features,weights,source code,predictions andlogs preserved. Free disk10.106GiB. No new data/model download or added spend.

## Handoff

Reject fixed mixture promotion and preserve the local source-composition effect. Do not widen mixture/learning-rate/checkpoint grids using exposed evaluation results. Before another method, inspect existing per-encoder error and prior source-selection evidence to determine whether task-conditioned representation selection has room beyond static fusion; compare with preservation-constrained adaptation and a clean diagnostic paper route. Reuse the SUR/URT/URL/TSA literature map and audit exact closest-work overlap. The earlier technical note remains a draft checkpoint; newer auxiliary studies are still unmapped into its authoritative outline/evidence ledger, so no writing/finalization claim follows from this result.

## Protocol metadata clarification after audit
The executable protocol is retained byte-for-byte under lock.json. It was derived from the previous learned-order contract and retains an inactive cost_boundary description about six-view head applications. That inherited prose does not describe this run: all arms use exactly one view and the same head, while unique training image/label exposure differs by arm as recorded. The new source_task_contract and actual task manifests are authoritative for source budgets. The one-view linear_mean reference reuses a penalty selected on the earlier six-view source study; it is a fixed historical recipe evaluated on one-view inputs, not freshly selected on the common100source-validation tasks. R2 and CS_l2 are fixed references; zero_ridge01 directly isolates the trained residual map. These limitations weaken blanket equal-tuning claims but do not turn the failed gate into a success: the candidate loses to the direct zero-head control in both directions. No executable config, predictions, model choices, tolerance, or metric gate was changed after outcomes.
