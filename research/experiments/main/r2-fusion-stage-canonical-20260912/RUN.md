# Shared-retrieval late ridge fusion: small R2 gain, strong gate failed

- Run id: `r2-fusion-stage-canonical-20260912`
- Branch: `run/r2-fusion-stage-canonical-20260912`
- Parent branch: `idea/012-idea-a921652c`
- Worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-fusion-stage-canonical-20260912`
- Idea: `idea-a921652c`
- Baseline: `r2-canonical-local`
- Baseline variant: `mps32-v1`
- Dataset scope: `dtd-eurosat-canonical-locked-20260912`
- Verdict: `supported_with_limits`
- Status: `completed`

## Hypothesis

Separate per-view fits on identical R2 augmented prototypes produce >=0.5pp one-shot macro gain with paired CI>0 and no cell loss>0.5pp.

## Setup

Original canonical frozenCLIP/DINO features, DTD-only development, query independence and shared top64 retrieval.20 settings per trainable family, deterministic selection on400dev tasks before4000one-shot evaluation tasks.2000five-shot task rows inheritR2.

## Execution

Detached bash-d32701e1: /opt/anaconda3/envs/torch/bin/python evaluate.py && /opt/anaconda3/envs/torch/bin/python package_results.py; exit0, main29.4s. Full manifest/code/input/seed checks.

## Results

Macro72.9803 versusR2 72.7733: +0.2070pp CI[0.1257,0.2893]. Versusdev-selected early_shared+0.1740pp CI[0.1107,0.2397]. VersusCS_l2 -0.1530pp CI[-0.2703,-0.0330]. Seven baseline rows exactly reconciled.

## Conclusion

Strong gate failed: small benefit below predeclared0.5pp threshold, and known CS_l2 remains stronger. Retain conditional result; no robust incumbent, new fusion algorithm or untouched-pool claim. Continue through decision to bounded R4 trust audit.

## Metrics Summary

- `dtd_matched_1shot_accuracy` = 77.1573
- `dtd_5shot_accuracy` = 89.9813
- `dtd_mismatched_1shot_accuracy` = 74.048
- `eurosat_mismatched_1shot_accuracy` = 65.9013
- `eurosat_matched_1shot_accuracy` = 74.8147
- `eurosat_5shot_accuracy` = 86.808
- `macro_1shot_accuracy` = 72.9803

## Baseline Comparison

- `dtd_matched_1shot_accuracy`: run=77.1573 baseline=76.8627 delta=0.2947 (better)
- `dtd_mismatched_1shot_accuracy`: run=74.048 baseline=74.0787 delta=-0.0307 (worse)
- `dtd_5shot_accuracy`: run=89.9813 baseline=89.9813 delta=0 (worse)
- `eurosat_mismatched_1shot_accuracy`: run=65.9013 baseline=66.1173 delta=-0.216 (worse)
- `eurosat_matched_1shot_accuracy`: run=74.8147 baseline=74.0347 delta=0.78 (better)
- `eurosat_5shot_accuracy`: run=86.808 baseline=86.808 delta=0 (worse)
- `macro_1shot_accuracy`: run=72.9803 baseline=72.7733 delta=0.207 (better)

## Changed Files

- `experiments/main/r2-fusion-stage-canonical-20260912/fusion.py`
- `experiments/main/r2-fusion-stage-canonical-20260912/evaluate.py`
- `experiments/main/r2-fusion-stage-canonical-20260912/validate.py`
- `experiments/main/r2-fusion-stage-canonical-20260912/package_results.py`

## Evidence Paths

- `experiments/main/r2-fusion-stage-canonical-20260912/RESULTS.md`
- `experiments/main/r2-fusion-stage-canonical-20260912/outputs/validation_report.json`
- `experiments/main/r2-fusion-stage-canonical-20260912/outputs/all_metrics.json`
- `experiments/main/r2-fusion-stage-canonical-20260912/outputs/run_manifest.json`

## Config Paths

- `experiments/main/r2-fusion-stage-canonical-20260912/protocol.json`
- `experiments/main/r2-fusion-stage-canonical-20260912/selection.json`

## Notes

- Canonical metric rows unique; comparator matrix in evidence file.
- Paired task bootstrap stratified by seed and galleries coupled within domain; conditional on reused pools.
- All5shot rows inherited, no candidate5shot mechanism claim.
- Previous submission rejected only because dataset_scope used descriptive prose instead of exact scope id; no metric, code or protocol change.

## Evaluation Summary

- Claim Update: {'small_local_fusion_effect': 'supported_with_limits', 'robust_half_point_gain': 'refuted', 'strongest_control_superiority': 'refuted', 'novel_algorithm': 'unsupported'}
- Baseline Relation: {'primary_delta_pp': 0.207, 'ci95_pp': [0.12566666666666657, 0.2893333333333331], 'cs_l2_delta_pp': -0.153, 'seven_baseline_values_reconciled': True, 'five_shot_inherited': True}
- Failure Mode: Insufficient magnitude and weaker than establishedCS_l2; same historically seen pools, not independent-image/domain confirmation.
- Next Action: decision -> R4 trusted-reuse audit

## Delivery Policy

- Research paper required: `True`
- Recommended next route: `analysis_or_write`
- Reason: Research paper mode is enabled. The run looks promising, so the next route should usually strengthen the evidence and move toward analysis or writing rather than stopping at the algorithm result alone.
