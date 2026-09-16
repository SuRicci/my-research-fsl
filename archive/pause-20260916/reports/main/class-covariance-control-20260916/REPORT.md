# Conventional class-covariance control — measured result

中文摘要：本轮完成两域共 1,000 个固定任务。类别各自的协方差有小幅正作用，但完整候选平均准确率 89.96%，低于现有参照 91.56%；源先验学习相对未训练先验下降约 1.09 个百分点。保留机制证据，关闭这套固定训练配方，不晋升为最优方法。额外源监督与已暴露开发集的限制均保留。

The predeclared covariance-utility tests pass, but the trained classifier is inferior to both its untrained-prior control and the existing support-only incumbent. Retain the narrow within-family effect; do not promote a new method or claim equal-budget superiority.

## Question and fixed comparison

Research type: auxiliary/development controlled classifier comparison. Null: class-specific posterior covariance provides no positive macro gain over covariance tying under the same prior. Alternative: the fixed contrast is positive in macro and nonnegative in each domain. A second predeclared check compares independently trained QDA and tied models. Both use the identical frozen six-view joint geometry.

The normalized NIW/Student-t model follows the published MetaQDA probabilistic construction, with a locally specified512-episode optimization recipe. This is neither an original method nor an exact upstream benchmark reproduction. The inspected upstream precision blend and missing MAP determinant are not silently copied. Source: https://arxiv.org/html/2101.02833 ; pinned author code and distinctions are in artifacts/idea/class-prior-qualification-20260916/SOURCE_AUDIT.md.

Source priors train independently in opposite directions: EuroSAT5training/5validation classes to DTD, and DTD23training/24validation classes to EuroSAT. Training and validation classes/images are disjoint; source and target have no shared RGB hashes. Each target reuses500fixed5-way5-shot15-query tasks. All pools are previously exposed development. Target queries are never used for fitting/configuration/checkpoint selection. The source-supervised models have additional training labels relative to logistic; logistic is contextual rather than an equal-training-resource comparison.

## Full accuracy surface

| Arm | DTD (%) | EuroSAT (%) | Macro (%) |
|---|---:|---:|---:|
| Initial prior / class-specific | 90.642667 | 91.453333 | 91.048000 |
| Initial prior / tied | 90.432000 | 91.229333 | 90.830667 |
| QDA-trained prior / class-specific | 89.792000 | 90.122667 | 89.957333 |
| Same QDA-trained prior / tied | 89.432000 | 89.906667 | 89.669333 |
| Tied-trained prior / class-specific | 89.805333 | 89.888000 | 89.846667 |
| Tied-trained prior / tied | 89.450667 | 89.733333 | 89.592000 |
| Support-only logistic C10 | 91.010667 | 92.109333 | 91.560000 |

## What the controlled contrasts establish

- same_prior_covariance_tying: macro +0.288000pp,95%CI[0.21333333333333357, 0.3613333333333334],one-sided95%lower0.225333pp; DTD+0.360000pp,EuroSAT+0.216000pp. Frozen gate PASSED.
- separately_trained_tied: macro +0.365333pp,95%CI[0.2053333333333334, 0.524],one-sided95%lower0.230667pp; DTD+0.341333pp,EuroSAT+0.389333pp. Frozen gate PASSED.

The same-prior comparison changes only inference covariance tying. The independently trained tied control also loses. This supports a small conditional advantage of class-specific covariance within these models; it does not establish that this family improves the strongest classifier.

Source learning with QDA reduces target macro by-1.090667pp versus initialization; descriptive95%CI[-1.242666666666667, -0.9440000000000001]. Initial class specificity already contributes0.217333pp,CI[0.15066666666666662, 0.2839999999999998]. The trained candidate trails the contextual incumbent by-1.602667pp. Secondary contrasts are descriptive and did not select an arm.

Source-held-out classes show the same warning: QDA learning changes EuroSAT accuracy93.20->92.306667 and DTD92.386667->90.226667; query cross-entropy also worsens. This is consistent with poor generalization of the fixed source recipe. It does not isolate overfitting, optimization dynamics and domain mismatch, or prove all meta-learned priors ineffective. No early-checkpoint rescue or hyperparameter sweep was run.

Bootstrap intervals use10000task resamples stratified by dataset and five historical seeds, conditional on fixed exposed image pools. They do not estimate new-image/new-domain uncertainty. Primary tests use seed260916721, secondary descriptive intervals260916722.

## Validation and cost

- All1000task prediction arrays recomputed from scores; full task identities/class labels preserved. Logistic bridges have0prediction changes; maximum score discrepancy1.119e-10.
- All1224source training/validation episodes audited for partition membership, class alignment and within-task image uniqueness.
- Independent raw-second-moment dense density checks cover24full896-dimensional arm/task combinations(two tasks/domain,six arms), maximum score error6.821e-13; all audited predictions agree. This is sampled dense verification, not all-score independent recomputation.
- Synthetic density conventions independently checked against SciPy; gradients, class permutation, support order and query singleton checks passed.
- Source training142.19s; target evaluation163.46s; local CPU, no download or encoder training. Low-rank posterior computation preserves the full prior covariance; no diagonal truncation.

## Judgment and limits

Supported: small class-covariance advantage in the declared conventional family. Not supported: improvement over the incumbent, benefit from this source-learning recipe, a novel generic Bayesian FSL method, or independent confirmation. The two primary gates pass, so calling this an all-negative run would be wrong. Nonetheless additional confirmation data is not the next priority: it would not remedy known-method overlap or the stronger-incumbent deficit. That prioritization is an explicit decision beyond the within-family utility gate.

Retain all7arms and source deterioration. Do not tune covariance scales, training length, checkpoints or domain-specific rescue on these exposed outcomes. Keep91.56%incumbent. Next: update the idea board with this newly measured boundary, then qualify a genuinely different question from existing evidence; do not return to a previously rejected narrowed-thesis packaging route.

## Evidence

MATCHED_CONTRACT.md; partition_audit.json; model.py;run.py;audit.py; numerical_validation.json; execution_lock.json; source_complete.json; target_complete.json; summary.json; secondary_effects.json; validation.json; full source/target NPZ scores and fixed prior state dictionaries. Source bash-7c96d2b9,target bash-df5dc01e,audit bash-b77d0de9.
