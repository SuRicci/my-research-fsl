# Ridge scale controls across encoders and pools

- Source branch: `analysis/idea-54125bda/analysis-2cc2d420-ridge-scale-robustness`
- Source worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-2cc2d420-ridge-scale-robustness`
- Source result: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-2cc2d420-ridge-scale-robustness/experiments/analysis/analysis-2cc2d420/ridge-scale-robustness/RESULT.md`
- Status: `completed`

## Goal

Compare raw fixed lambda1, equal-grid tuned raw ridge, radius-scaled ridge and known support-centered L2 ridge using original/new DTD and EuroSAT pools.

## Paper Contract Binding

- Selected outline: `none`
- Section id: `none`
- Item id: `ridge-scale-robustness`
- Exp id: `none`
- Paper role: `none`
- Analysis role: `none`
- Reviewer question: TBD
- Target display: `none`
- Claim links: none

## Core Requirement

Full protocol only.

## Setup

12 five-shot cells: original/new pools x DTD/EuroSAT x CLIP/DINO/fusion. 1,000 paired tasks each. Raw/radius/CS_l2 each tune identical four coefficient values on original DTD development only. Fixedlambda1 and prototype included.

## Execution

bash-1afee3ab completed exit0. Development2.436s, evaluation30.628s. Fused parent endpoint score error0; uniform-scale identity max1.759e-6. All feature hashes/task indices validated.

## Results

Radius scaling loses to equally tuned global ridge in every CLIP cell: originalDTD-.290667pp, originalEuroSAT-.494667, newDTD-.400,newEuroSAT-.521333, all pairedCIs below0. DINO DTD approximatelyneutral (-.004/+.005333) but EuroSAT positive (+1.001333/+.937333). Fusion positive on DTD (+.060/+.105333) and EuroSAT (+1.552/+1.512), and exceeds CS_l2 in all four fusedcells. Detailed twelve-cell outcomes retained without selecting only positive encoders.

## Claim Impact

Reject universal support-radius superiority. Retain encoder/domain-dependent practical benefit for frozen fused features, especially EuroSAT, and known regularization-scale explanation. Equal global tuning absorbs CLIP improvements. No novel method or broad-domain claim.

## Manuscript Update Hint

Not recorded.

## Evaluation Summary

- Takeaway: Scale adaptation is representation- and domain-dependent after equal global tuning.
- Claim Update: narrows
- Baseline Relation: mixed
- Comparability: high
- Failure Mode: none
- Next Action: revise_idea

## Comparison Baselines

- None recorded.
