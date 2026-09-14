# Old-space interpolation geometry under fixed bridge permissions

- Run id: `r4-transport-geometry-20260913`
- Branch: `run/r4-transport-geometry-20260913`
- Parent branch: `idea/012-idea-134c9aa0`
- Worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r4-transport-geometry-20260913`
- Idea: `idea-134c9aa0`
- Baseline: `r4-canonical-local`
- Baseline variant: `semantic-mps32-v1`
- Dataset scope: `r4-dtd-eurosat-semantic-20260912`
- Verdict: `negative`
- Status: `completed`

## Hypothesis

Centered interpolation neighborhoods add>=0.5pp macro mAP over old_centered and beat equally selected raw transport.

## Setup

Canonical20DTDdevelopment+40evaluation jobs,32/64bridges,five seeds,fullgallery. Original V0delta/PRESS. Both geometries selected globalalpha2 from identical DTDdevelopment grid[0,.5,1,2]. All9metrics13variants retained.

## Execution

Numerical validation bash-3cca6ae6 passed; main bash-d3e38239 completed60jobs in36.896seconds; independent full-rank/identity audit bash-57db6d6a passed114000new method-query rankings, maxAP error4.44e-16. No encoding/downloads.

## Results

Primary center_a2 macro43.054342 vsoldcentered44.325898: -1.271557pp,95%[-3.039835,+0.685570]. Equalraw/centered_residual_2 contrast+0.413821pp,95%[-0.230425,+1.148866]. DTD+2.109602pp;EuroSAT-4.652715pp. Posthoc smaller-alpha rows remain diagnostics.

## Conclusion

Strong gate failed. Close this kernel/preprocessing family on current exposed grid. Primary is not replaceable by a better posthoc alpha. No robust upgrade, novelty or independent-generalization claim. Next decision re-evaluates calibration/transfer evidence.

## Metrics Summary

- `dtd_cross_bridge32_map` = 30.7997
- `dtd_cross_bridge64_map` = 30.8721
- `dtd_matched_bridge32_map` = 33.8704
- `dtd_matched_bridge64_map` = 35.6128
- `eurosat_cross_bridge32_map` = 58.1887
- `eurosat_cross_bridge64_map` = 57.4748
- `eurosat_matched_bridge32_map` = 50.8671
- `eurosat_matched_bridge64_map` = 46.7492
- `macro_map` = 43.0543

## Baseline Comparison

- `dtd_cross_bridge32_map`: run=30.7997 baseline=30.6791 delta=0.1205 (better)
- `dtd_cross_bridge64_map`: run=30.8721 baseline=30.6791 delta=0.193 (better)
- `dtd_matched_bridge32_map`: run=33.8704 baseline=30.6791 delta=3.1912 (better)
- `dtd_matched_bridge64_map`: run=35.6128 baseline=30.6791 delta=4.9336 (better)
- `eurosat_cross_bridge32_map`: run=58.1887 baseline=57.9727 delta=0.216 (better)
- `eurosat_cross_bridge64_map`: run=57.4748 baseline=57.9727 delta=-0.4978 (worse)
- `eurosat_matched_bridge32_map`: run=50.8671 baseline=57.9727 delta=-7.1055 (worse)
- `eurosat_matched_bridge64_map`: run=46.7492 baseline=57.9727 delta=-11.2235 (worse)
- `macro_map`: run=43.0543 baseline=44.3259 delta=-1.2716 (worse)

## Changed Files

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r4-transport-geometry-20260913/experiments/main/r4-transport-geometry-20260913/run.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r4-transport-geometry-20260913/experiments/main/r4-transport-geometry-20260913/analyze.py`

## Evidence Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r4-transport-geometry-20260913/experiments/main/r4-transport-geometry-20260913/RESULTS.md`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r4-transport-geometry-20260913/experiments/main/r4-transport-geometry-20260913/outputs/analysis.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r4-transport-geometry-20260913/experiments/main/r4-transport-geometry-20260913/outputs/all_metrics.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r4-transport-geometry-20260913/experiments/main/r4-transport-geometry-20260913/outputs/validation_report.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r4-transport-geometry-20260913/experiments/main/r4-transport-geometry-20260913/outputs/selection.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r4-transport-geometry-20260913/experiments/main/r4-transport-geometry-20260913/outputs/completion.json`

## Config Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r4-transport-geometry-20260913/experiments/main/r4-transport-geometry-20260913/protocol.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r4-transport-geometry-20260913/experiments/main/r4-transport-geometry-20260913/PLAN.md`

## Notes

- Seen image pools; fixed-gallery class/seed bootstrap is not independent-domain evidence.
- All13baseline variants reconcile; new-new remains evaluator-only reference, not an upper bound.
- Initial record attempt rejected generic scope full; corrected metadata to exact unchanged canonical scope, without rerun or metric change.

## Evaluation Summary

- Claim Update: {'robust_geometry_upgrade': 'refuted', 'geometry_only_benefit': 'inconclusive', 'independent_generalization': 'unsupported', 'novel_algorithm': 'unsupported'}
- Baseline Relation: {'primary_delta_pp': -1.2715565497906027, 'primary_ci95_pp': [-3.039835210316906, 0.6855698352983759], 'equal_control_delta_pp': 0.413821125128365, 'all9metrics13variants_reconciled': True}
- Failure Mode: DTD-selectedalpha2 transfers poorly toEuroSATmatched; interpolation geometry contrastCIincludes0.
- Next Action: decision -> close local kernel route; reconsider calibration/information boundary

## Delivery Policy

- Research paper required: `True`
- Recommended next route: `revise_idea`
- Reason: Research paper mode is enabled, but the current run does not beat the baseline clearly enough. Revise the direction or strengthen the method before writing.
