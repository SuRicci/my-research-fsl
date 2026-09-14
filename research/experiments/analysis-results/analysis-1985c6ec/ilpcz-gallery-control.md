# iLPC-z local frozen-gallery comparator

- Source branch: `analysis/idea-5670808c/analysis-1985c6ec-ilpcz-gallery-control`
- Source worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-1985c6ec-ilpcz-gallery-control`
- Source result: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-1985c6ec-ilpcz-gallery-control/experiments/analysis/analysis-1985c6ec/ilpcz-gallery-control/RESULT.md`
- Status: `completed`

## Goal

TBD

## Paper Contract Binding

- Selected outline: `none`
- Section id: `none`
- Item id: `ilpcz-gallery-control`
- Exp id: `none`
- Paper role: `none`
- Analysis role: `none`
- Reviewer question: TBD
- Target display: `none`
- Claim links: none

## Core Requirement

Full protocol only.

## Setup

Published iLPC-z formulas reimplemented locally; frozen canonical CLIP/DINO fusion; support labels and independent unlabeled full galleries; queries excluded from adaptation. Four graph settings selected only on original DTD development tasks; k20 alpha0.5 selected for both shots; final logistic C10.

## Execution

bash-0c1f8465 completed exit0 in 16023.08s; 2400 development fits and 6000 evaluation tasks. bash-212ae974 full validator/packager completed exit0. Four local CPU workers, BLAS1; graph and factorization reuse validated; no iteration or gallery truncation.

## Results

Macro one-shot 60.3897% vs R2 72.7733%, delta -12.3837pp [95% paired CI -12.7637,-12.0023]. Matched EuroSAT +13.648pp one-shot and +4.968pp five-shot; DTD matched -7.6573pp one-shot/-11.4013pp five-shot; mismatched galleries -31.3133pp DTD and -24.212pp EuroSAT.

## Claim Impact

Reject universal replacement/robust advantage. Retain large conditional EuroSAT matched-gallery benefit as known-comparator evidence, motivating an enablement/reliability question rather than a novelty or original-table reproduction claim.

## Manuscript Update Hint

Pre-outline comparator and failure-boundary evidence. Preserve as local adaptation result; no active outline yet. Future Results/Limitations must include all six cells and severe cross-domain losses.

## Evaluation Summary

- Comparability: Same image/features/support-label/query permissions and paired task identities; method and full-gallery adaptation differ intentionally.

## Comparison Baselines

- `r2-canonical-local` · variant `mps32-v1`
