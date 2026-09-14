# Retrieval/head factorial and failure summary

- Source branch: `analysis/idea-54125bda/analysis-bb5b0939-retrieval-head-attribution`
- Source worktree: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-bb5b0939-retrieval-head-attribution`
- Source result: `/Users/decoqwq/DeepScientist/quests/012/.ds/worktrees/analysis-analysis-bb5b0939-retrieval-head-attribution/experiments/analysis/analysis-bb5b0939/retrieval-head-attribution/RESULT.md`
- Status: `completed`

## Goal

Separate centeredretrievalandcenteredclassification usingcachedparenttasks.

## Paper Contract Binding

- Selected outline: `none`
- Section id: `none`
- Item id: `retrieval-head-attribution`
- Exp id: `none`
- Paper role: `none`
- Analysis role: `none`
- Reviewer question: TBD
- Target display: `none`
- Claim links: none

## Core Requirement

Full protocol only.

## Setup

Full original six cells, five seeds/1000 paired tasks each; fixed parent image features and indices. 1-shot 2x2 raw/centered retrieval by raw/centered classifier, lambda .1/top64/blend .5. 5-shot adds raw ridge lambda_eff=mean centered-support squared radius to raw lambda1 and per-sample CS_l2.

## Execution

bash-d73da95a exit0,76.939 seconds. Parent raw/centered score maximum error exactly0; uniform-radius scaling vs regularization identity maximum error8.047e-7.

## Results

1-shot macro effects: combined+.360pp CI[.260325,.461333], retrieval alone+.133[.039,.229342], head alone+.317667[.223,.407342], interaction-.090667[-.177008,-.001]. EuroSAT mismatch is largest benefit; matched cells and DTD mismatch degrade. Five-shot radius control gains+.060pp DTD and+1.552pp EuroSAT over raw, exceeding full CS_l2 by+.089333pp and+.149333pp respectively (paired CIs exclude0). Neighbor overlap 40.2%-80.5%; overlap/delta correlations small and descriptive.

## Claim Impact

Downgrade any special five-shot per-sample geometry explanation: a simpler scalar regularization control explains and exceeds its gain. Retain practical, heterogeneous 1-shot improvement and known-method attribution; no novelty claim.

## Manuscript Update Hint

Not recorded.

## Evaluation Summary

- Takeaway: Simpler regularization control exceeds the five-shot centered method; one-shot mechanisms are heterogeneous.
- Claim Update: narrows
- Baseline Relation: mixed
- Comparability: high
- Failure Mode: none
- Next Action: analysis_campaign

## Comparison Baselines

- None recorded.
