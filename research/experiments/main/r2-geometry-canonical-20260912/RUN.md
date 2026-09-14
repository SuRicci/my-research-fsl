# Known support-centering improves the local frozen R2 comparison

- Run id: `r2-geometry-canonical-20260912`
- Branch: `run/r2-geometry-canonical-20260912`
- Parent branch: `idea/012-idea-54125bda`
- Worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-geometry-canonical-20260912`
- Idea: `idea-54125bda`
- Baseline: `r2-canonical-local`
- Baseline variant: `mps32-v1`
- Dataset scope: `dtd-eurosat-canonical-locked-20260912`
- Verdict: `positive`
- Status: `completed`

## Hypothesis

Known support-mean centering and L2 normalization improve frozenR2 geometry beyond raw/tunedR2 and simple heads under unchanged inductive permissions.

## Setup

Same canonical features,5-way75queries,5seeds200episodes/cell, four1shot plus two5shot conditions. Supportmean defines transform for support/query/gallery. Raw/tunedR2,gallery-centering and centered/rawsupportcontrols included. Per-shot ridge strengths selected using same four-lambda grid on DTDdevelopment classes. CenteredR2 selects0.1 for1shot and1 for5shot, equal to fixedR2 strengths.

## Execution

Implementation validated exactzero-originR2recovery,batchagreement,queryindependence. Dev14.61s; main56.62s. Recomputed raw predictions/accuracies, verified featurehashes, episodeindices and originalR2scores. Stored1shotneighborindices. Five-thousand paired seed-stratified bootstrap preserving gallery pairing.

## Results

CenteredR2 macro73.1333% vsR272.7733%, +0.3600pp CI[0.2603,0.4613]; versus tunedR2+0.3270pp CI[0.2183,0.4367]. All predeclared promotion gates pass. EuroSATmismatched1shot+2.0067pp; matched1shot-0.3693pp; DTDmismatched-0.208pp and matched+0.0107pp. EuroSAT5shot+1.4027pp; DTD5shot-0.0293pp.

## Conclusion

Promising local standard-method transfer; promote only provisionally pending new-image confirmation. Gains are concentrated in EuroSAT mismatch and5shot; support-centering is prior art and not a novel algorithm. No independent-pool or broadSOTA claim.

## Metrics Summary

- `dtd_matched_1shot_accuracy` = 76.8733
- `dtd_5shot_accuracy` = 89.952
- `dtd_mismatched_1shot_accuracy` = 73.8707
- `eurosat_mismatched_1shot_accuracy` = 68.124
- `eurosat_matched_1shot_accuracy` = 73.6653
- `eurosat_5shot_accuracy` = 88.2107
- `macro_1shot_accuracy` = 73.1333

## Baseline Comparison

- `dtd_matched_1shot_accuracy`: run=76.8733 baseline=76.8627 delta=0.0107 (better)
- `dtd_mismatched_1shot_accuracy`: run=73.8707 baseline=74.0787 delta=-0.208 (worse)
- `dtd_5shot_accuracy`: run=89.952 baseline=89.9813 delta=-0.0293 (worse)
- `eurosat_mismatched_1shot_accuracy`: run=68.124 baseline=66.1173 delta=2.0067 (better)
- `eurosat_matched_1shot_accuracy`: run=73.6653 baseline=74.0347 delta=-0.3693 (worse)
- `eurosat_5shot_accuracy`: run=88.2107 baseline=86.808 delta=1.4027 (better)
- `macro_1shot_accuracy`: run=73.1333 baseline=72.7733 delta=0.36 (better)

## Changed Files

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-geometry-canonical-20260912/experiments/main/r2-geometry-canonical-20260912/evaluate.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-geometry-canonical-20260912/experiments/main/r2-geometry-canonical-20260912/package_results.py`

## Evidence Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-geometry-canonical-20260912/experiments/main/r2-geometry-canonical-20260912/outputs/eval/comparison.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-geometry-canonical-20260912/experiments/main/r2-geometry-canonical-20260912/outputs/eval/RESULT.md`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-geometry-canonical-20260912/experiments/main/r2-geometry-canonical-20260912/outputs/eval/run_manifest.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-geometry-canonical-20260912/experiments/main/r2-geometry-canonical-20260912/implementation_validation.json`

## Config Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-geometry-canonical-20260912/experiments/main/r2-geometry-canonical-20260912/protocol.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-geometry-canonical-20260912/experiments/main/r2-geometry-canonical-20260912/selection.json`

## Notes

- Support-mean centering thenL2 is an explicit known comparator inFeietalICCV2021; no method noveltyclaim.
- Third-family evaluation reuses viewed fixed image pools and taskdraws; CIs conditional, not independent-image confirmation.
- No cellmean regression exceeds0.5pp, but most1shot macro gain comes from EuroSATmismatchedgallery.

## Evaluation Summary

- Takeaway: Known support-centering gives a positive local0.36pp macro gain and passes preset gates; new-image confirmation needed.
- Claim Update: strengthens
- Baseline Relation: mixed
- Comparability: high
- Failure Mode: none
- Next Action: analysis_campaign

## Delivery Policy

- Research paper required: `True`
- Recommended next route: `analysis_or_write`
- Reason: Research paper mode is enabled. The run looks promising, so the next route should usually strengthen the evidence and move toward analysis or writing rather than stopping at the algorithm result alone.
