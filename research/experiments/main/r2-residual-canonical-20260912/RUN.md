# Support-anchored residual transfer: locked canonical FSL comparison

- Run id: `r2-residual-canonical-20260912`
- Branch: `run/r2-residual-canonical-20260912`
- Parent branch: `idea/012-idea-ca651025`
- Worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912`
- Idea: `idea-ca651025`
- Baseline: `r2-canonical-local`
- Baseline variant: `mps32-v1`
- Dataset scope: `dtd-eurosat-canonical-locked-20260912`
- Verdict: `negative`
- Status: `completed`

## Hypothesis

Removing gallery mean transfer while retaining retrieved residual scatter can improve robustness to mismatched galleries without sacrificing matched-gallery accuracy.

## Setup

Frozen canonical CLIP ViT-B/16 QuickGELU and native DINOv2 ViT-S/14; fixed half-fusion. DTD and EuroSAT four paired 1-shot query/gallery conditions and two 5-shot safeguards. Five seeds × 200 episodes per cell. No gallery labels, encoder training, or query-batch adaptation. DTD development classes disjoint from evaluation classes; historically viewed image pools.

## Execution

Completed separate baseline replay, DTD-only development selection, and locked evaluation. Residual lambda=1; tuned support lambda=0.1. Validated mean/scatter identity, support/query disjointness, raw predictions, per-query independence, paired gallery tasks and exact R2 reference agreement. Five-thousand stratified paired bootstrap replicates.

## Results

Residual macro 1-shot accuracy 71.133% versus R2 72.7733%; delta -1.6403 pp, paired 95% CI [-1.793,-1.491]. EuroSAT mismatched +2.1307 pp, matched -5.6267 pp; DTD matched -2.9973 pp, mismatched -0.068 pp. Residual also trails support ridge by 0.2383 pp and mass-matched/isotropic controls by about 0.05 pp. All three promotion gates fail. Five-shot behavior intentionally unchanged.

## Conclusion

Reject residual-only transfer as a robust improvement under this fixed protocol; preserve canonical R2 as incumbent. Useful matched-gallery mean signal dominates the proposed robustness benefit. This does not rule out all residual methods or other classifier families.

## Metrics Summary

- `dtd_matched_1shot_accuracy` = 73.8653
- `dtd_5shot_accuracy` = 89.9813
- `dtd_mismatched_1shot_accuracy` = 74.0107
- `eurosat_mismatched_1shot_accuracy` = 68.248
- `eurosat_matched_1shot_accuracy` = 68.408
- `eurosat_5shot_accuracy` = 86.808
- `macro_1shot_accuracy` = 71.133

## Baseline Comparison

- `dtd_matched_1shot_accuracy`: run=73.8653 baseline=76.8627 delta=-2.9973 (worse)
- `dtd_mismatched_1shot_accuracy`: run=74.0107 baseline=74.0787 delta=-0.068 (worse)
- `dtd_5shot_accuracy`: run=89.9813 baseline=89.9813 delta=0 (worse)
- `eurosat_mismatched_1shot_accuracy`: run=68.248 baseline=66.1173 delta=2.1307 (better)
- `eurosat_matched_1shot_accuracy`: run=68.408 baseline=74.0347 delta=-5.6267 (worse)
- `eurosat_5shot_accuracy`: run=86.808 baseline=86.808 delta=0 (better)
- `macro_1shot_accuracy`: run=71.133 baseline=72.7733 delta=-1.6403 (worse)

## Changed Files

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/evaluate.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/package_results.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/validate_implementation.py`

## Evidence Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/outputs/eval/RESULT.md`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/outputs/eval/comparison.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/outputs/eval/run_manifest.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/outputs/baseline/metrics_summary.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/implementation_validation.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/provenance.json`

## Config Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/protocol.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-residual-canonical-20260912/experiments/main/r2-residual-canonical-20260912/selection.json`

## Notes

- Confidence intervals are conditional on fixed, historically viewed image pools; new task seeds do not establish independent-image/domain generalization.
- Historical cached features and canonical features are separate comparator variants.
- Mean-only and distribution attribution controls use candidate lambda=1, not separately optimized strongest alternatives.
- Five-shot equality follows the preserved support-only baseline by design.
- Initial record attempt failed only dataset_scope formatting; no data or computation changed for this corrected submission.

## Evaluation Summary

- Takeaway: Residual-only gallery transfer loses 1.64 pp to canonical R2 and fails all three promotion gates.
- Claim Update: weakens
- Baseline Relation: worse
- Comparability: high
- Failure Mode: direction
- Next Action: revise_idea

## Delivery Policy

- Research paper required: `True`
- Recommended next route: `revise_idea`
- Reason: Research paper mode is enabled, but the current run does not beat the baseline clearly enough. Revise the direction or strengthen the method before writing.
