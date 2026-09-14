# PRESS versus every fixed penalty on twelve cells

- Source branch: `analysis/idea-8c090fcf/analysis-5cd86318-press-fixed-penalty-challenge`
- Source worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-5cd86318-press-fixed-penalty-challenge`
- Source result: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-5cd86318-press-fixed-penalty-challenge/experiments/analysis/analysis-5cd86318/press-fixed-penalty-challenge/RESULT.md`
- Status: `completed`

## Goal

Evaluate PRESS and allfourfixedpenalties on original/new DTD/EuroSAT withCLIP,DINO,fusion; reuse equal-development tuned andradiuscontrols.

## Paper Contract Binding

- Selected outline: `none`
- Section id: `none`
- Item id: `press-fixed-penalty-challenge`
- Exp id: `none`
- Paper role: `none`
- Analysis role: `none`
- Reviewer question: TBD
- Target display: `none`
- Claim links: none

## Core Requirement

Full protocol only.

## Setup

All12priorfive-shot cells andsamefourpenalties; exactPRESS withoutcalibration versusallfixedpenalties, priorDTD-devtuned andradius controls. No selection onquerylabels.

## Execution

bash-0f67ac18 completed exit0,157.063seconds. Parentmaximumscoreerror4.303e-7; full pairedscore/taskrecords saved.

## Results

Allfourfusioncells PRESS vsfixedlambda.001 CIincludeszero, with deltas+.018667,+.026667,+.022667,+.029333pp. PRESS vsdev-tunedCLIP isnegativeall4cells (3CIs below0); DTD-DINO-.577333pp old/-.349333new. EuroSAT fusion gains vsdev-tunedbaseline remain+2.673333/+2.474667pp but fixedlambda.01/.001 largelymatches. Fixedlambda.1 iscompetitiveacrosscells asretrospectivediagnostic.

## Claim Impact

Downgrade adaptive-selection explanation for fused-feature gains; small fixedpenalties account for most benefit. No universal PRESS superiority; no newalgorithmnovelty. Do notpromote retrospectivelybestfixedvalue asprospectivelyvalidated.

## Manuscript Update Hint

Not recorded.

## Evaluation Summary

- Takeaway: Most fused gains are explainable by fixed small penalties; adaptive superiority is unsupported.
- Claim Update: narrows
- Baseline Relation: mixed
- Comparability: high
- Failure Mode: none
- Next Action: analysis_campaign

## Comparison Baselines

- None recorded.
