# Support-only bounded prevalidated ridge selection

- Run id: `r2-prevalidation-canonical-20260912`
- Branch: `run/r2-prevalidation-canonical-20260912`
- Parent branch: `idea/012-idea-8c090fcf`
- Worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-prevalidation-canonical-20260912`
- Idea: `idea-8c090fcf`
- Baseline: `r2-canonical-local`
- Baseline variant: `mps32-v1`
- Dataset scope: `dtd-eurosat-canonical-locked-20260912`
- Verdict: `supported_with_limits`
- Status: `completed`

## Hypothesis

Support-only exact ridge LOO classification-loss selection improves five-shot accuracy over dev-tuned raw/radius controls without query information.

## Setup

Canonical frozen fused CLIP/DINO, existing six cells/1000paired tasks each. New five-shot bounded PreVal-inspired selector uses four fixedlambda values and positive outputscale[0,100]. PRESS-MSE same-grid selector control. Four one-shot cells inherit parent CS_l2 policy verbatim; their improvement is not attributed to this pass.

## Execution

Validation bash-6d527c71:600 deleted refits maximumerror1.409e-9; scalar loss optimizer2.498e-16; query batch8.006e-12. Main bash-6c9919d1 exit0,16.399seconds; allrawpredictions recomputed/validated against scores and labels, all7metricids present. Double precision exactLOO, parentfloat32 endpointdifference<1e-5.

## Results

Five-shot candidate DTD89.852% (-.129333pp vsR2), EuroSAT89.505333%(+2.697333pp). Two-cell mean gain+1.284pp CI[1.160667,1.405333] vsdev-tunedraw;+.478pp[.384,.573333] vsradius. VsPRESS-MSE only+.021333pp[-.012667,.054667], not established. Scaleupperbound chosen94/1000DTD and58/1000Euroepisodes; nozero-scalechoices. Inheritedone-shotmacro73.133333% (+.360pp vsR2) unchanged fromparent.

## Conclusion

Retain support-only selection as a known practical improvement with EuroSAT concentration; additional calibrated-loss advantage over simpler PRESS is unsupported. Prefer confirming simpler PRESS over retaining unnecessary calibration complexity.

## Metrics Summary

- `dtd_matched_1shot_accuracy` = 76.8733
- `dtd_5shot_accuracy` = 89.852
- `dtd_mismatched_1shot_accuracy` = 73.8707
- `eurosat_mismatched_1shot_accuracy` = 68.124
- `eurosat_matched_1shot_accuracy` = 73.6653
- `eurosat_5shot_accuracy` = 89.5053
- `macro_1shot_accuracy` = 73.1333

## Baseline Comparison

- `dtd_matched_1shot_accuracy`: run=76.8733 baseline=76.8627 delta=0.0107 (better)
- `dtd_mismatched_1shot_accuracy`: run=73.8707 baseline=74.0787 delta=-0.208 (worse)
- `dtd_5shot_accuracy`: run=89.852 baseline=89.9813 delta=-0.1293 (worse)
- `eurosat_mismatched_1shot_accuracy`: run=68.124 baseline=66.1173 delta=2.0067 (better)
- `eurosat_matched_1shot_accuracy`: run=73.6653 baseline=74.0347 delta=-0.3693 (worse)
- `eurosat_5shot_accuracy`: run=89.5053 baseline=86.808 delta=2.6973 (better)
- `macro_1shot_accuracy`: run=73.1333 baseline=72.7733 delta=0.36 (better)

## Changed Files

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-prevalidation-canonical-20260912/experiments/main/r2-prevalidation-canonical-20260912/prevalidated.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-prevalidation-canonical-20260912/experiments/main/r2-prevalidation-canonical-20260912/validate.py`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-prevalidation-canonical-20260912/experiments/main/r2-prevalidation-canonical-20260912/evaluate.py`

## Evidence Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-prevalidation-canonical-20260912/experiments/main/r2-prevalidation-canonical-20260912/outputs/eval/comparison.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-prevalidation-canonical-20260912/experiments/main/r2-prevalidation-canonical-20260912/outputs/eval/RESULT.md`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-prevalidation-canonical-20260912/experiments/main/r2-prevalidation-canonical-20260912/outputs/eval/run_manifest.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-prevalidation-canonical-20260912/experiments/main/r2-prevalidation-canonical-20260912/validation.json`

## Config Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/r2-prevalidation-canonical-20260912/experiments/main/r2-prevalidation-canonical-20260912/protocol.json`

## Notes

- Known PreVal-inspired adaptation, not source-exact PreVal reproduction or novel algorithm.
- One-shot results are inherited and carry no new causal selector claim.
- Bootstrap intervals condition on already-viewed image pools; no broad-domain or SOTA inference.

## Evaluation Summary

- Takeaway: Support-only model selection improves five-shot mean; calibration adds no established benefit over PRESS.
- Claim Update: strengthens
- Baseline Relation: mixed
- Comparability: high
- Failure Mode: none
- Next Action: analysis_campaign

## Delivery Policy

- Research paper required: `True`
- Recommended next route: `analysis_or_write`
- Reason: Research paper mode is enabled. The run looks promising, so the next route should usually strengthen the evidence and move toward analysis or writing rather than stopping at the algorithm result alone.
