# New-image confirmation within existing datasets

- Source branch: `analysis/idea-54125bda/analysis-bb5b0939-new-image-confirmation`
- Source worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-bb5b0939-new-image-confirmation`
- Source result: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-bb5b0939-new-image-confirmation/experiments/analysis/analysis-bb5b0939/new-image-confirmation/RESULT.md`
- Status: `completed`

## Goal

Freeze parentparameters and compare on previouslyunusedDTDval1andEuroSATtrainimages.

## Paper Contract Binding

- Selected outline: `none`
- Section id: `none`
- Item id: `new-image-confirmation`
- Exp id: `none`
- Paper role: `none`
- Analysis role: `none`
- Reviewer question: TBD
- Target display: `none`
- Claim links: none

## Core Requirement

Full protocol only.

## Setup

Frozen parent parameters, canonical frozen CLIP/DINO. 1,870 unique DTD val1 and 2,000 EuroSAT remaining train images, zero decoded-RGB overlap with original query/gallery pools. Same original galleries and class split; five fresh seeds, 1,000 paired tasks per six cells. Independently remeasure R2 on this pool.

## Execution

Managed bash-e7a2836a completed exit 0 at 2026-09-12T10:56:49Z. Reference-feature cosine gate >0.99999; code validates raw scores, disjoint support/query indices, task pairing and R2 parity. No parameter tuning on confirmation.

## Results

Macro candidate 72.590% vs same-pool R2 72.265%, delta +0.325 pp, paired 95% CI [0.228,0.423675]; vs frozen dev-tuned R2 +0.261333 pp [0.153317,0.372667]. Per-cell deltas: DTD matched -0.125333, DTD mismatch -0.277333, DTD 5-shot +0.020; EuroSAT mismatch +1.962667, matched -0.260, 5-shot +1.309333 pp. All preset point-estimate guardrails pass; gains heterogeneous.

## Claim Impact

Supports reproducible practical improvement within two existing datasets; no new-domain, independent image-sampling confidence, or novel-algorithm claim. Support centering is prior art.

## Manuscript Update Hint

Not recorded.

## Evaluation Summary

- Takeaway: New images preserve modest average benefit with EuroSAT concentration.
- Claim Update: strengthens
- Baseline Relation: mixed
- Comparability: high
- Failure Mode: none
- Next Action: analysis_campaign

## Comparison Baselines

- `r2-canonical-local`
