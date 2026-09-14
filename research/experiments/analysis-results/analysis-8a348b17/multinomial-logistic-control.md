# Five-shot multinomial logistic control

- Source branch: `analysis/idea-8c090fcf/analysis-8a348b17-multinomial-logistic-control`
- Source worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-8a348b17-multinomial-logistic-control`
- Source result: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-8a348b17-multinomial-logistic-control/experiments/analysis/analysis-8a348b17/multinomial-logistic-control/RESULT.md`
- Status: `completed`

## Goal

Measure ordinary multinomial logistic regression on21savedpairedcells and compare existing ridge,geometry andprototype controls.

## Paper Contract Binding

- Selected outline: `none`
- Section id: `none`
- Item id: `multinomial-logistic-control`
- Exp id: `none`
- Paper role: `none`
- Analysis role: `none`
- Reviewer question: TBD
- Target display: `none`
- Claim links: none

## Core Requirement

Full protocol only.

## Setup

Support-only multinomial-lbfgs; normalized canonical CLIP/DINO/fusion; C=[1,10,100,1000] on200originalDTDdevelopmenttasks perrepresentation, ties prefer smallerC. SelectedC100CLIP,C10DINO/fusion frozen before all21saved1000-taskcells. C1 also retained; no query adaptation, gallery information or encoder update.

## Execution

Managed bash-467ca7f7 completed exit0 in430.21s,44,400fits, zero convergence warnings. Source hashes, class labels, support/query separation, query-subset invariance and savedscore/prediction/summary agreement passed. CPUfourworkers/oneBLASthread; no new downloads or encoding.

## Results

No broad logistic superiority: versus development-tuned ridge,5positive,8negative,8zero-containing pointwise95% task intervals. Original/newEuroSATfusionlogistic88.073/87.751%, belowfixedridge.1 by1.313/1.299pp. FixedC1isfrequentlyunder-calibrated; gainsagainstoriginalridge1canreflectregularizationchoice. Countsdescriptive,notaglobalmultiplicity-correctedtest.

## Claim Impact

Strong-classifier comparison gap closed. Retain conditional known-method effects; no evidence that logistic or PRESS creates a generally improved method. New-method and SOTAclaimsremainunsupported.

## Manuscript Update Hint

Use full21celltable, tuned/default classifier distinction, exact permissions, solvercost, and pooled-analysis limits.

## Evaluation Summary

- Not recorded.

## Comparison Baselines

- None recorded.
