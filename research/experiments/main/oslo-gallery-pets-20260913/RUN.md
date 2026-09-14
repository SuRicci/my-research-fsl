# Gallery-only OSLO fails strong-baseline qualification on canonical and fresh Pets images

- Run id: `oslo-gallery-pets-20260913`
- Branch: `run/oslo-gallery-pets-20260913`
- Parent branch: `idea/012-idea-aca445b7`
- Worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913`
- Idea: `idea-aca445b7`
- Baseline: `r2-pets-transfer`
- Baseline variant: `frozen-v1`
- Dataset scope: `pets-frozen-gallery-transfer-v1`
- Verdict: `rejected_strong_baseline_gate`
- Status: `completed`

## Hypothesis

Source-default latent inlier weighting, transferred to gallery-only fitting, improves matched-gallery one-shot performance over R2/CS_l2 without material degradation elsewhere.

## Setup

Frozen CLIP/DINO fusion.5-way1/5-shot,15queries/class,500tasks/cell; canonical test pool and prospective1804 image-disjoint same-domain trainval images;1876-image Pets or DTD galleries. Seven methods,eight cells,4000tasks. No query adaptation or gallery-label access.

## Execution

Source parity12cases and query invariants passed. Frozen protocol and source hashes preceded fresh feature extraction. Local managed bash-4896958c completed encode+evaluate, exit0. bash-db3c14ef passed complete task/prediction/frozen-baseline audit and representative score replays.

## Results

Matched1shot canonical94.9173 vs R2 96.2400 (-1.3227pp,95%[-1.7227,-0.9333]); fresh94.8747 vs96.0160 (-1.1413pp,[-1.5173,-0.7627]). Mismatch1shot64.1920/63.9920 vs93.9307/93.3893; all37breeds lose in both pools. Both5shot conditions also degrade. All measurements and paired intervals retained.

## Conclusion

Reject source-default OSLO_G as an improvement: both pools fail frozen strong-control superiority and non-degradation gates. Inlier weighting improves on the closed-set update but remains below strong controls. Retain R2/CS_l2; close this default family without a Pets tuning grid. Prospective same-domain confirmation supports this bounded failure, not independent-domain robustness, novelty or SOTA.

## Metrics Summary

- `pets_matched_1shot_accuracy` = 94.9173
- `pets_mismatched_1shot_accuracy` = 64.192
- `pets_matched_5shot_accuracy` = 97.1947
- `pets_mismatched_5shot_accuracy` = 93.304
- `pets_macro_1shot_accuracy` = 79.5547
- `pets_fresh_matched_1shot_accuracy` = 94.8747
- `pets_fresh_mismatched_1shot_accuracy` = 63.992
- `pets_fresh_matched_5shot_accuracy` = 97.1893
- `pets_fresh_mismatched_5shot_accuracy` = 92.9787
- `pets_fresh_macro_1shot_accuracy` = 79.4333

## Baseline Comparison

- `pets_matched_1shot_accuracy`: run=94.9173 baseline=96.24 delta=-1.3227 (worse)
- `pets_mismatched_1shot_accuracy`: run=64.192 baseline=93.9307 delta=-29.7387 (worse)
- `pets_matched_5shot_accuracy`: run=97.1947 baseline=98.4427 delta=-1.248 (worse)
- `pets_mismatched_5shot_accuracy`: run=93.304 baseline=98.4427 delta=-5.1387 (worse)
- `pets_macro_1shot_accuracy`: run=79.5547 baseline=95.0853 delta=-15.5307 (worse)
- `pets_fresh_matched_1shot_accuracy`: run=94.8747 baseline=None delta=n/a (not comparable)
- `pets_fresh_mismatched_1shot_accuracy`: run=63.992 baseline=None delta=n/a (not comparable)
- `pets_fresh_matched_5shot_accuracy`: run=97.1893 baseline=None delta=n/a (not comparable)
- `pets_fresh_mismatched_5shot_accuracy`: run=92.9787 baseline=None delta=n/a (not comparable)
- `pets_fresh_macro_1shot_accuracy`: run=79.4333 baseline=None delta=n/a (not comparable)

## Changed Files

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/oslo.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/run_eval.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/validate_source.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/analyze_results.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/PLAN.md`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/CHECKLIST.md`

## Evidence Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/REPORT.md`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/outputs/analysis.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/outputs/validation.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/outputs/source_validation.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/assets/feature_manifest.json`

## Config Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/protocol.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/oslo-gallery-pets-20260913/experiments/main/oslo-gallery-pets-20260913/locked_sources.json`

## Notes

- All five canonical required metric ids retained; additional fresh-pool metrics do not replace baseline metrics.
- No independent-domain, pretraining-overlap or near-duplicate audit claim.
- record_main_experiment uses active workspace; evidence belongs to run/oslo-gallery-pets-20260913.
- First submission rejected only because dataset_scope='full' mismatched canonical scope id; corrected to exact accepted id without changing data, code, metrics or comparison protocol.

## Evaluation Summary

- Not recorded.

## Delivery Policy

- Research paper required: `True`
- Recommended next route: `revise_idea`
- Reason: Research paper mode is enabled, but the current run does not beat the baseline clearly enough. Revise the direction or strengthen the method before writing.
