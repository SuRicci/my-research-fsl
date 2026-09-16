# Optimistic isotropic-cap screen: rejected for insufficient usefulness

The frozen candidate idea-f2d1d671 failed its pre-outcome engineering gate. Across 1,000 existing five-shot tasks (75,000 query appearances), even the deliberately optimistic cap permits only 16,644 omitted query-view applications out of 450,000 (3.6987%). DTD is 5.2827%; EuroSAT is 2.1147%. The gate required at least 10% overall and 5% in each domain. Every eligible exit occurs after view five, so at most one view is omitted for those queries. There are no view-four exits despite all queries satisfying the preliminary norm condition at that prefix. Maximum view-four pairwise lower margins are -0.7703 (DTD) and -1.0344 (EuroSAT), far from zero.

These are optimistic eligibility counts, not measured encoder work saved or latency improvements. The radius was frozen at 0.99 per remaining view, below the required unit-feature universal radius, with no floating-point padding. Zero observed disagreements do not convert this rule into a safe deployment implementation.

## Evidence and audit
Both full-reference accuracy metrics reproduce the accepted contract: DTD 91.010667%, EuroSAT 92.109333%, macro 91.56%. All 75,000 full-reference predictions match the audited parent; maximum score error is 4.261e-11. The audit independently recomputed every first-positive stopping index, exact counts, task identities, labels, seeds, and the fixed gate. Source/input hashes match, including the nested reference dependency manifest. The separate synthetic cap check covers 200 cases against a one-dimensional angular optimizer, with maximum discrepancy 8.882e-16. Measured screen runtime was about 96 seconds; this is analysis runtime, not deployment latency.

## Scope and decision
Close this isotropic, canonical-order cap package. Do not tune view order, radius, threshold, classifier C, or geometry gamma to rescue it. The negative result does not reject all adaptive inference. The broad full-model-preserving stopping principle is already established in prior work; no novelty claim is made.

Next qualification is allowed only if it retains uncertainty geometry discarded by the isotropic relaxation, with an independent mathematical example and a separate frozen contract. An alternative is independent-image confirmation of the incumbent; it remains necessary for strong generalization claims but is not the next way to answer this compute question. Calibrated confidence exits require a different calibration/guarantee contract and are not silently substituted.

Durable evidence: protocol.json, execution_lock.json, validation_synthetic.json, dtd.npz, eurosat.npz, summary.json, AUDIT.json. Historical full-view scores and the baseline remain unchanged. This is a pre-outline auxiliary analysis, not a new main method or a submission-ready paper result.
