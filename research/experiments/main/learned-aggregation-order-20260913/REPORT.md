# Learned nonlinear aggregation order: source qualification report

Verdict: reject this fixed recipe for target promotion. The before-mean versus after-mean contrast is small and domain-dependent; no new Pets/Caltech outcome or novel-method claim.

## Matched-order result

| Evaluation domain | 1-shot before-minus-after (pp) | Paired95%CI | Three training-seed effects(pp) |
|---|---:|---|---|
| dtd_to_eurosat | -0.157333 | [-0.228444, -0.086222] | -0.133333, -0.149333, -0.189333 |
| eurosat_to_dtd | 0.170667 | [0.072889, 0.273778] | 0.125333, 0.584000, -0.197333 |

The prospective criterion required >=0.5pp over every named control in both directions, positive paired lower bounds, all-cell noninferiority and at least2/3positive training seeds. The first direction has all3seed effects negative; the second mean is below0.5pp. Both fixed-direction gates fail.

## Absolute performance and comparator boundary

| Domain | Shot | Before-mean accuracy(%) | After-mean accuracy(%) | Before-minus frozen tuned linear mean(pp) |
|---|---:|---:|---:|---:|
| dtd_to_eurosat_k1_eurosat | 1 | 65.532444 | 65.689778 | -4.819556 |
| dtd_to_eurosat_k5_eurosat | 5 | 90.273778 | 90.279111 | -0.454222 |
| eurosat_to_dtd_k1_dtd | 1 | 71.903111 | 71.732444 | -4.720889 |
| eurosat_to_dtd_k5_dtd | 5 | 90.028444 | 90.056000 | -0.806222 |

The learned-order comparison matches architecture, parameter count, source labels/tasks/updates, initialization/order seeds, checkpoint selection and fixed ridge penalty. Comparisons against historical linear_mean include its frozen source-selected ridge penalty; they establish competitiveness but are not an isolated causal estimate of training effects. Source selection gains alongside poor cross-domain results are consistent with domain-specific adaptation, not proof of a unique failure mechanism.

## Protocol and uncertainty

Two source domains,1/5shot,3training seeds per method,24trained heads; each has57,344parameters,200updates from100fixed source episodes. Source selection uses100identity-disjoint episodes; checkpoints0/50/100/200 are ranked by integercorrect, CE, earliest step. All choices were locked before target scoring. Four stored evaluation banks contain2000unique episodes; two gallery contexts yield8comparison cells. The same support-only predictions are reused across gallery contexts, not treated as independent samples. Training seeds are averaged as accuracy, not ensembled as predictions. Bootstrap uses5000paired task draws stratified by task seed and remains conditional on the fixed image pools and these3trainingseeds. DTD/EuroSAT remain development-exposed.

## Verification and provenance

Formal run bash-828c6bf6 / science-1b8c2b4b completed in 153.08s. Precheck bash-3be22a56 passed after two documented import/config entry repairs; no formal run was restarted. Independent audit bash-b52fe59e: 192selected-model/task reconstructions, 96checkpoint rank checks and 150paired statistic checks passed. Maximum saved-score error 1.845626541e-06, fixed tolerance3e-5; statistical error 9.71e-16. Parent scatter strict audit failure remains failed and is not superseded.

Inputs/code hashes: lock.json, outputs/manifest.json. Raw predictions, full comparisons, selection histories, checkpoints, and copied-bank hashes: outputs/cells, outputs/analysis.json, outputs/selection_lock.json, outputs/selection, outputs/models, outputs/references. Baseline five-metric Pets contract unchanged; these auxiliary metrics must not populate canonical target ids.

## Resources and cleanup

All views and14frozen controls were reused, without copying old prediction banks. Current outputs approximately49.80MiB, below0.10GiB allowance; free disk last measured10.342GiB. More map applications are an explicit cost of before-mean processing, not compute-matched training. Timing per model is in selection_lock and inference_timing. No image/model download or remote compute.

672transient runner paths were removed from future Git tracking;671existing files totaling296,910,573bytes were verified preserved. Filesystem bytes deleted=0; old history preserved. This prevents repeated runner snapshots, not a claim of reclaimed disk. Receipt outputs/runtime_tracking_cleanup.json; .gitignore excludes/.ds/codex-home/.

## Handoff

Next primary skill: idea, focused on reduced-view accuracy/encoding-cost qualification as the retained outside-family alternative. Do not widen the failed adapter horizon/rank/learning-rate grid from these outcomes. Reopen this fixed line only after a specific implementation error or independently justified training-permission change. Reduced-view work must count encoder applications and measure accuracy under frozen source selection; cache timing alone cannot prove deployment speedups. No need to repeat feature extraction, failed kernel/scatter grids, or numerical precheck. Before future writing, map this and preceding source studies into the authoritative outline/evidence ledger; current paper remains a draft checkpoint.
