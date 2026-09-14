# Query-local consistency: finite auxiliary qualification

## Verdict
The source-selected query consistency recipe reaches **77.374%** versus the retained blend **77.334%**, a **+0.040pp** paired difference (95% interval **[+0.013983,+0.067333]pp**). All prespecified checks except the minimum +0.5pp gain pass. The qualification gate therefore fails; this is not independent target promotion or a useful-scale improvement. Preserve the exact recipe as a conditional development optimization point and the simpler77.334%parent as reference. Close this finite recipe without eta extension, new loss or query-feature rescue. Do not discard its measured small gain, and do not inflate it into a generalization or novelty claim.

## Method and permissions
The original six-view mean, support-scatter map, retrieval sets, blended prototypes, centered ridge penalty0.1, and equal raw/centered head combination remain fixed. For each query independently, add eta times its mean squared across-view score residual to the original centered ridge objective. A six-dimensional Woodbury correction evaluates the solution. Other queries and query/gallery labels are never inference inputs; no state carries between queries. This is a known tangent/consistency regularization transfer, not a new invariance principle or MEMO replication. Equation and seven-paper overlap audit: artifacts/idea/pre_idea_drafts/query-local-consistency.md and artifacts/idea/query_consistency_survey.md.

## Frozen design
100source episodes per domain, each with ordinary and class-excluded source galleries, reuse exact prior source-bank indices. Consistency and trace-matched isotropic controls independently choose eta from[0,0.1,1,10] by pooled integer correctness; ties prefer smaller eta. Both choices freeze before target evaluation. Source DTD chooses eta0 for both; source EuroSAT chooses consistency10/isotropic0. No target-based choice. A third comparator averages query-view scores using the unchanged parent head. Full evaluation retains500episodes per domain, five seeds and two galleries:1000sampled episodes/2000task conditions/150000query occurrences per target method. Source400conditions/30000query occurrences per source-grid method. Gallery conditions share tasks and are paired, not independent replications.

| Domain and gallery | Parent accuracy (%) | Consistency (%) | Difference (pp) | Paired95% interval (pp) | Repairs / spoiled |
|---|---:|---:|---:|---|---:|
|dtd_dtd|79.888000|80.010667|+0.122667|[0.050666666666666534, 0.1973333333333333]|119 / 73|
|dtd_eurosat|76.266667|76.304000|+0.037333|[-0.04000000000000014, 0.11733333333333348]|125 / 111|
|eurosat_dtd|73.485333|73.485333|+0.000000|[0.0, 0.0]|0 / 0|
|eurosat_eurosat|79.696000|79.696000|+0.000000|[0.0, 0.0]|0 / 0|

DTD domain mean improves0.080pp[0.027967,0.134667]; EuroSAT is unchanged because its source selected zero correction. DTD mismatched-gallery CI includes zero: neither every-cell positive improvement nor universal safety is established. Total244repairs and184spoiled predictions yield60net corrections. Isotropic equals parent because both source choices are zero; query-view mean reaches77.258%. Consistency exceeds it by0.116pp[0.079983,0.152667]. These comparisons do not show that all possible isotropic penalties fail.

## Verification and execution
Full run bash-d210498f completed in102.880s. Prevalidation bash-212448aa passed16dense comparisons and six invariances, max6.67e-16. Initial same-name import failure bash-82dd4beb was repaired before computations. Independent audit bash-f0263e97 verified all900000saved predictions across8banks, all source integer choices,21intervals (max error0pp), and240query-method score vectors on20sampled actual tasks using NumPy direct-primal solves, max7.22e-16. Frozen preprocessing is reused from the independently audited parent; the dense head solver is independent. The first audit command hit another reference import collision; exact-path loading fixed it without rerunning the scientific experiment. Both failure receipts retained.
All2400source/target parent task predictions matched saved parent results; maximum parent score error1.67e-15. Code/config locks, source-bank hashes, exact predictions/scores/task indices and source choices are saved.

## Evidence boundary and next step
All datasets and image pools are repeatedly exposed development assets. The paired bootstrap conditions on those pools and does not account for cross-experiment method search. No canonical Pets or Caltech metric was measured, overwritten or inferred. This auxiliary report plus Science Evidence Graph deliberately does not masquerade as a five-metric Pets main experiment. The existing gallery-only OSLO paper receives only a reference record; its four evidence groups and claims remain unchanged because this query-adapted protocol differs.

Next: decision on independent validation feasibility for the accumulated stack, with source parameter lock and exposure audit. Do not launch another covariance/selector/eta rescue. Retain77.374%as a provisional development point and77.334%as its cheaper parent; neither new target readiness nor quest completion is implied. Local-only, zero additional spend, no downloads or encoding; disk reserve remains above10GiB.
