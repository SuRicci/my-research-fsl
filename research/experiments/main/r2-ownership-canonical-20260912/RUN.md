# Exclusive gallery ownership fails the strong one-shot improvement gate

- Run id: `r2-ownership-canonical-20260912`
- Branch: `run/r2-ownership-canonical-20260912`
- Parent branch: `idea/012-idea-5670808c`
- Worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-ownership-canonical-20260912`
- Idea: `idea-5670808c`
- Baseline: `r2-canonical-local`
- Baseline variant: `mps32-v1`
- Dataset scope: `dtd-eurosat-canonical-locked-20260912`
- Verdict: `negative`
- Status: `completed`

## Hypothesis

Ownership improves one-shot macro beyond R2 and CS_l2 and count/mass controls.

## Setup

Same frozen canonical features, splits, original tasks, four-lambda DTD development budget and all seven required metrics. Five-shot primary inherits R2.

## Execution

Existing completed bash-b96deafe,dev7.817643s,eval53.689414s; this is an artifact mapping correction, not a rerun.

## Results

Macro72.785%;vsR2+0.011667pp CI[-0.115333,0.138342],vsCS_l2-0.348333pp CI[-0.477667,-0.221992]. Positive comparisons to count/mass controls do not rescue the failed system gate.

## Conclusion

Reject ownership promotion. Correct duplicate metric-row identifiers from run-b513efd1: canonical rows now contain only primary ownership values; all comparator results remain in comparison.json. Measured evidence and decision-61eadffc unchanged.

## Metrics Summary

- `dtd_matched_1shot_accuracy` = 76.9853
- `dtd_5shot_accuracy` = 89.9813
- `dtd_mismatched_1shot_accuracy` = 73.92
- `eurosat_mismatched_1shot_accuracy` = 65.7693
- `eurosat_matched_1shot_accuracy` = 74.4653
- `eurosat_5shot_accuracy` = 86.808
- `macro_1shot_accuracy` = 72.785

## Baseline Comparison

- `dtd_matched_1shot_accuracy`: run=76.9853 baseline=76.8627 delta=0.1227 (better)
- `dtd_mismatched_1shot_accuracy`: run=73.92 baseline=74.0787 delta=-0.1587 (worse)
- `dtd_5shot_accuracy`: run=89.9813 baseline=89.9813 delta=0 (worse)
- `eurosat_mismatched_1shot_accuracy`: run=65.7693 baseline=66.1173 delta=-0.348 (worse)
- `eurosat_matched_1shot_accuracy`: run=74.4653 baseline=74.0347 delta=0.4307 (better)
- `eurosat_5shot_accuracy`: run=86.808 baseline=86.808 delta=0 (better)
- `macro_1shot_accuracy`: run=72.785 baseline=72.7733 delta=0.0117 (better)

## Changed Files

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-ownership-canonical-20260912/experiments/main/r2-ownership-canonical-20260912/evaluate.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-ownership-canonical-20260912/experiments/main/r2-ownership-canonical-20260912/validate_implementation.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-ownership-canonical-20260912/experiments/main/r2-ownership-canonical-20260912/package_results.py`

## Evidence Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-ownership-canonical-20260912/experiments/main/r2-ownership-canonical-20260912/outputs/eval/comparison.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-ownership-canonical-20260912/experiments/main/r2-ownership-canonical-20260912/outputs/eval/metrics_summary.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-ownership-canonical-20260912/experiments/main/r2-ownership-canonical-20260912/outputs/eval/validation.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-ownership-canonical-20260912/experiments/main/r2-ownership-canonical-20260912/outputs/run_completion.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-ownership-canonical-20260912/experiments/main/r2-ownership-canonical-20260912/implementation_validation.json`

## Config Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-ownership-canonical-20260912/experiments/main/r2-ownership-canonical-20260912/protocol.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-ownership-canonical-20260912/experiments/main/r2-ownership-canonical-20260912/selection.json`

## Notes

- Correction supersedes erroneous per-metric comparison projection in run-b513efd1; raw data unchanged.
- Canonical metric ids uniquely identify primary ownership; comparator values remain traceable in full comparison.json.
- No evaluation rerun; all pools viewed.

## Evaluation Summary

- Claim Update: Robust improvement refuted for the fixed intervention; incremental count-control signal remains partial.
- Baseline Relation: No established macro advantage over R2; worse than CS_l2.
- Failure Mode: Single-pass ownership fails strong-system gate.
- Next Action: Continue decision-61eadffc: official iLPC same-permission source/cost audit.

## Delivery Policy

- Research paper required: `True`
- Recommended next route: `analysis_or_write`
- Reason: Research paper mode is enabled. The run looks promising, so the next route should usually strengthen the evidence and move toward analysis or writing rather than stopping at the algorithm result alone.
