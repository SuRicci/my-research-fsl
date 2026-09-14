# Finite-view source objective: verified fixed-recipe rejection

All 18 fits completed and the independent audit passed. The primary worst-view objective failed both transfer gates. The retained scatter/consistency classifiers remain the comparison anchors.

## Matched transfer results

Accuracies average training-seed accuracies and paired gallery contexts. Intervals resample tasks within task seeds, conditional on the exposed fixed pools and three training seeds. They are not independent-domain uncertainty.

| Source to target | Mean feature CE | Mean view CE | Worst view CE | Worst minus scatter (pp, 95% interval) | Worst minus consistency (pp, 95% interval) |
|---|---:|---:|---:|---|---|
| EuroSAT to DTD | 72.6951% | 72.5102% | 72.7467% | -5.3307 [-5.7151, -4.9280] | -5.4107 [-5.7912, -5.0160] |
| DTD to EUROSAT | 74.0693% | 74.0409% | 74.2418% | -2.3489 [-2.5942, -2.0987] | -2.3489 [-2.5942, -2.0987] |

Worst-view versus mean-feature gains were +0.0516 pp on DTD (interval crosses zero) and +0.1724 pp on EuroSAT. Worst-view versus mean-view gains were +0.2364/+0.2009 pp. All are below the fixed 0.5 pp minimum; better loss-control results do not imply improvement over the retained classifier.

## Validity and limits

- Same residual map, six frozen views, source tasks, steps and inference; losses differ. Exact identity recovers the retained scatter classifier. Equal steps/view exposure is not equal FLOPs.
- Source selection chose step 100 for all DTD fits and step 200 for all EuroSAT fits. All 18 choices were locked before transfer. No target-driven rank, learning-rate, view or selector retuning.
- 1000 distinct tasks, 2000 task-gallery conditions. 72 checkpoint ranks, 270000 selected source predictions and arithmetic of 1350000 evaluation predictions checked. Independent equations checked for all 18 selected models and 36 target-model samples; this is not an independent full reimplementation of every prediction. Eight primary paired intervals independently reproduced.
- Locked inputs/code and all selected model hashes match. Raw compute files retain their original audit-pending flag; RESULT.json and audit.json carry the final verified state.
- No canonical Pets metric or untouched target qualification was measured. No new OSLO-manuscript claim, novel robust method, or robustness theorem is established.
- Earlier pre-training resource failures and the deterministic streaming amendment remain preserved.

## Route implication

Close this exact recipe without a parameter sweep. Large negative transfer survives retaining the strong head, so merely restoring head alignment did not rescue these source-trained maps. The result does not prove why transfer fails, nor rule out source learning generally. The two previously prioritized objective families are now measured; next compare remaining research value against evidence synthesis before spending on another variant.

## Reproduction and evidence

Run: /opt/anaconda3/envs/torch/bin/python experiments/main/finite-view-head-objective-20260914/study.py --phase run (refuses to overwrite existing manifest).
Audit: /opt/anaconda3/envs/torch/bin/python experiments/main/finite-view-head-objective-20260914/audit.py.
Read protocol.json, outputs/code_lock.json, outputs/selection_lock.json, outputs/analysis.json, outputs/audit.json and RESULT.json. Compute log bash-f86bf840; audit log bash-b4a2ea89.
