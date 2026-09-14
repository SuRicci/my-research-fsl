# Loss-sensitive neighborhood adaptation: second family comparison

- Run id: `r2-logloss-canonical-20260912`
- Branch: `run/r2-logloss-canonical-20260912`
- Parent branch: `idea/012-idea-e1645299`
- Worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-logloss-canonical-20260912`
- Idea: `idea-e1645299`
- Baseline: `r2-canonical-local`
- Baseline variant: `mps32-v1`
- Dataset scope: `dtd-eurosat-canonical-locked-20260912`
- Verdict: `negative`
- Status: `completed`

## Hypothesis

Cross-entropy on full retrieved neighborhoods provides robust gains beyond R2 and mass-matched support/mean classifiers.

## Setup

Same accepted canonical frozen CLIP/DINO features, seven metrics, four1shot cells and two5shot safeguards. Same locked task draws;5seeds×200episodes/cell. No gallery labels, query-batch adaptation or encoder training. Equal four-value regularization grid on DTD development for seven tunable heads; selected neighborhood/mean/support logistic lambda1. FixedR2 and tunedR2 reported.

## Execution

Validated weighted solver replication/query independence. Development4800fits73.70s; main completed in91.65s, raw scores and convergence metadata saved. ReferenceR2 score parity, support/query disjointness, predictions, seven metrics and paired bootstrap validated. Isolated threadpoolctl3.5 dependency fixed an environment failure before fitting; no method change.

## Results

Neighborhood logistic macro72.376% vsR272.7733%; delta-0.3973pp CI[-0.4877,-0.3083]. Versus tunedR2 -0.4303pp. Versus mean logistic +0.0167pp, versus distributionridge -0.365pp. Operational loss-by-neighborhood contrast -0.039pp CI[-0.0677,-0.0097]. All promotion gates fail.

## Conclusion

Reject log-loss neighborhood route as a robust improvement. Full-neighborhood benefit beyond means is tiny for logistic and no larger than ridge under this development tuning. No novel classifier claim; preserve R2 and move to a distinct representation assumption.

## Metrics Summary

- `dtd_matched_1shot_accuracy` = 76.748
- `dtd_5shot_accuracy` = 89.9813
- `dtd_mismatched_1shot_accuracy` = 73.836
- `eurosat_mismatched_1shot_accuracy` = 66.6093
- `eurosat_matched_1shot_accuracy` = 72.3107
- `eurosat_5shot_accuracy` = 86.808
- `macro_1shot_accuracy` = 72.376

## Baseline Comparison

- `dtd_matched_1shot_accuracy`: run=76.748 baseline=76.8627 delta=-0.1147 (worse)
- `dtd_mismatched_1shot_accuracy`: run=73.836 baseline=74.0787 delta=-0.2427 (worse)
- `dtd_5shot_accuracy`: run=89.9813 baseline=89.9813 delta=0 (worse)
- `eurosat_mismatched_1shot_accuracy`: run=66.6093 baseline=66.1173 delta=0.492 (better)
- `eurosat_matched_1shot_accuracy`: run=72.3107 baseline=74.0347 delta=-1.724 (worse)
- `eurosat_5shot_accuracy`: run=86.808 baseline=86.808 delta=0 (better)
- `macro_1shot_accuracy`: run=72.376 baseline=72.7733 delta=-0.3973 (worse)

## Changed Files

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-logloss-canonical-20260912/experiments/main/r2-logloss-canonical-20260912/evaluate.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-logloss-canonical-20260912/experiments/main/r2-logloss-canonical-20260912/package_results.py`

## Evidence Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-logloss-canonical-20260912/experiments/main/r2-logloss-canonical-20260912/outputs/eval/comparison.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-logloss-canonical-20260912/experiments/main/r2-logloss-canonical-20260912/outputs/eval/RESULT.md`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-logloss-canonical-20260912/experiments/main/r2-logloss-canonical-20260912/outputs/eval/solver_summary.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-logloss-canonical-20260912/experiments/main/r2-logloss-canonical-20260912/outputs/eval/run_manifest.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-logloss-canonical-20260912/experiments/main/r2-logloss-canonical-20260912/implementation_validation.json`

## Config Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-logloss-canonical-20260912/experiments/main/r2-logloss-canonical-20260912/protocol.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-logloss-canonical-20260912/experiments/main/r2-logloss-canonical-20260912/selection.json`

## Notes

- Repeated fixed-pool comparison is developmental, not independent-image confirmation.
- Five-shot behavior is unchangedR2 by design.
- Loss-by-neighborhood contrast compares separately development-tuned heads and is not fixed-parameter causal attribution.

## Evaluation Summary

- Takeaway: Log-loss neighborhood adaptation is0.397pp worse than canonicalR2 and fails promotion.
- Claim Update: weakens
- Baseline Relation: worse
- Comparability: high
- Failure Mode: direction
- Next Action: revise_idea

## Delivery Policy

- Research paper required: `True`
- Recommended next route: `analysis_or_write`
- Reason: Research paper mode is enabled. The run looks promising, so the next route should usually strengthen the evidence and move toward analysis or writing rather than stopping at the algorithm result alone.
