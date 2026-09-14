# Frozen iLPC Pets qualification fails strong-control and mismatch gates

- Run id: `fsl-pets-transfer-20260913`
- Branch: `run/fsl-pets-transfer-20260913`
- Parent branch: `idea/012-idea-01adcb83`
- Worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913`
- Idea: `idea-01adcb83`
- Baseline: `r2-pets-transfer`
- Baseline variant: `frozen-v1`
- Dataset scope: `pets-frozen-gallery-transfer-v1; all37breeds,4cells,500pairedtasks/cell`
- Verdict: `failed_primary_gate`
- Status: `completed`

## Hypothesis

Frozen gallery-only iLPC-z improves Pets matched one-shot accuracy by at least0.5pp over R2,CS_l2 and supportC10 with positive paired95% bounds; mismatch and5shot are mandatory stress conditions.

## Setup

Official Pets trainval unlabeled1876image gallery;3669exact-RGB-unique test images supply disjoint supports/queries. Equal1876image DTD mismatch gallery. Frozen canonical CLIP-B16/DINO-S14,MPSfloat32 withfloat16storage,fusion0.5.5way,15queries/class,1/5shot,5seedsx100tasks/cell. No target tuning,query adaptation,gallery labels or encoder training.

## Execution

Acquisition bash-9512022b; assets audit bash-31c1ec3f. Baseline bash-48b70ed6 and audit bash-1af2e4af; confirmed baseline-0503efcd before candidate. Candidate bash-c587b30c,536.36s,4CPUworkers/BLAS1. Full audit bash-5e6a4443; aggregate verification bash-c6f63cc3. First auditor bash-362ca54c failed before outcomes on Path-type mismatch; fixed only analyzer, no experiment rerun.

## Results

Matched1shot96.0533% versusR296.2400%,delta-0.1867pp95%[-0.5920,+0.2000]; versusCS_l2delta-0.2107[-0.6053,+0.1813]. Gain over supportC10+1.7200[1.2799,2.1520] does not beat gallery controls. Mismatch1shot58.5573% vsR293.9307%,delta-35.3733[-36.3573,-34.4160],loss in500/500tasks. Matched5shot97.8187% vsR298.4427%,delta-0.6240[-0.8480,-0.4133]; mismatch5shot92.6693% vsR298.4427%,delta-5.7733[-6.1600,-5.4000]. Newscope macro1shot77.3053% vsR295.0853%,delta-17.78[-18.3254,-17.2560].

## Conclusion

Prespecified independent-domain superiority gate failed. Matched gain exists over support-only logistic but not strong gallery controls; severe mismatch regression rejects robust-replacement claim. Retain historical EuroSAT-local benefit as conditional. No new algorithm or paper-readiness claim.

## Metrics Summary

- `pets_matched_1shot_accuracy` = 96.0533
- `pets_mismatched_1shot_accuracy` = 58.5573
- `pets_matched_5shot_accuracy` = 97.8187
- `pets_mismatched_5shot_accuracy` = 92.6693
- `pets_macro_1shot_accuracy` = 77.3053

## Baseline Comparison

- `pets_matched_1shot_accuracy`: run=96.0533 baseline=96.24 delta=-0.1867 (worse)
- `pets_mismatched_1shot_accuracy`: run=58.5573 baseline=93.9307 delta=-35.3733 (worse)
- `pets_matched_5shot_accuracy`: run=97.8187 baseline=98.4427 delta=-0.624 (worse)
- `pets_mismatched_5shot_accuracy`: run=92.6693 baseline=98.4427 delta=-5.7733 (worse)
- `pets_macro_1shot_accuracy`: run=77.3053 baseline=95.0853 delta=-17.78 (worse)

## Changed Files

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/analyze_results.py`

## Evidence Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/REPORT.md`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/outputs/analysis.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/outputs/validation.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/outputs/paired_accuracy.npz`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/outputs/ilpc_complete.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/outputs/baseline/validation.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/outputs/baseline/confirmed_comparator.json`

## Config Paths

- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/protocol.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/outputs/run_manifest_ilpc.json`
- `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/fsl-pets-transfer-20260913/experiments/main/fsl-pets-transfer-20260913/outputs/asset_validation.json`

## Notes

- Confidence intervals condition on fixed image pools and seeds; reused images across tasks are not new independent datasets.
- Exact RGB disjointness only; near duplicates and pretraining independence unproven.
- Local formula-based iLPC-z, not full official-entrypoint reproduction; local frozen R2 is not SWAT.
- No cherry-picked canonical rows:5unique primary metric ids; comparators remain in evidence tables.
- Baseline code and evaluator/protocol unchanged through candidate execution.

## Evaluation Summary

- Claim Update: {'matched_superiority': 'not_supported; pairedCIcrosszero vsR2/CS_l2 and upperboundsbelowprecommitted0.5pp', 'robust_replacement': 'refuted', 'benefit_over_support_logistic': 'supported_only_for_matched1shot'}
- Baseline Relation: R2 comparison anchor and CS_l2/support controls retained; no incumbent promotion. Old7R2/9R4metrics unchanged in their original scopes.
- Failure Mode: direction_underperforming; mismatch pseudolabel propagation harmful, mechanistic causality not established by this comparison alone.
- Next Action: Decision: close this frozen qualification without Pets tuning; choose a structurally distinct route or bounded failure synthesis from retained evidence.

## Delivery Policy

- Research paper required: `True`
- Recommended next route: `revise_idea`
- Reason: Research paper mode is enabled, but the current run does not beat the baseline clearly enough. Revise the direction or strengthen the method before writing.
