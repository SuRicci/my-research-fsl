# Canonical CUB Flowers and CIFAR-FS panel

- Source branch: `analysis/idea-8c090fcf/analysis-1b946833-canonical-additional-domains`
- Source worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-1b946833-canonical-additional-domains`
- Source result: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-1b946833-canonical-additional-domains/experiments/analysis/analysis-1b946833/canonical-additional-domains/RESULT.md`
- Status: `completed`

## Goal

Download officialverifiedimages/labels, encode exactcanonical CLIP/DINO, execute predeclared9five-shot cells withoutnewtuning.

## Paper Contract Binding

- Selected outline: `none`
- Section id: `none`
- Item id: `canonical-additional-domains`
- Exp id: `none`
- Paper role: `none`
- Analysis role: `none`
- Reviewer question: TBD
- Target display: `none`
- Claim links: none

## Core Requirement

Full protocol only.

## Setup

Three source-verified official image-test pools; fixed CLIP/DINO/fused features; nine five-way five-shot cells with five seeds x200 paired tasks. Identical frozen four-coefficient grid and development-only choices for all controls. Full sources and exclusions documented in SOURCE_AUDIT.md.

## Execution

Source verification bash-3fd260a6 and encoding/evaluation bash-4e5703a5 completed with exit0. Encoding1528.96s; evaluation31.70s. Nine cells x1000 tasks passed identity/class-label alignment, support/query disjointness and array-summary agreement checks.

## Results

No supported PRESS advantage over development-tuned fixed ridge in any of nine cells; three CLIP losses and CIFAR-FS DINO loss have negative pointwise95% task intervals. Fused PRESS-minus-fixed.001 is +.0013pp Flowers,0 CIFARFS,+.0013pp CUB. High Flowers/CUB accuracy ceilings and domain-dependent geometry effects limit generalization. This update repairs initially nested metric rows into81 explicit numeric entries; measured data are unchanged.

## Claim Impact

Broad PRESS superiority and adaptive-benefit claims downgraded. Known geometric/penalty effects remain conditional. No new-method or SOTA claim.

## Manuscript Update Hint

Include all81 accuracies acrossninecells plus conditional task intervals, source/split limitations and ceiling effects.

## Evaluation Summary

- Not recorded.

## Comparison Baselines

- None recorded.
