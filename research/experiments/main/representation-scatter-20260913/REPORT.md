# Real-view scatter: completed source qualification and numerical audit

Updated 2026-09-13T04:41:25.703476+00:00. Auxiliary development evidence; no new Pets or Caltech result.

## Outcome

The fixed scatter recipe does not meet the prespecified cross-source improvement gate. Retain the EuroSAT-specific gain; do not promote this recipe to Pets or claim general robust improvement. Computation succeeded; the original frozen end-to-end numerical audit failed one of its 88 comparisons and remains failed.

| Opposite-source selection → evaluation | 1-shot macro gain vs six-view mean R2 (pp) | Paired 95% interval (pp) |
|---|---:|---|
| dtd_to_eurosat | 3.4547 | [3.1213, 3.8200] |
| eurosat_to_dtd | 0.0640 | [-0.0987, 0.2293] |

Both directions were required to improve by at least 0.5pp with positive paired lower bounds against every prespecified control. EuroSAT meets the directional macro gate; DTD fails it against mean R2, score ensembling and mean CS_l2. DTD with a mismatched gallery also fails the per-cell noninferiority requirement: candidate minus augmented-support ridge is -0.5013pp, interval [-0.8907, -0.1147] (exact endpoints in analysis.json).

## Information and geometry

| Evaluation cell | View mean minus original (pp) | Scatter minus view mean (pp) |
|---|---:|---:|
| dtd_to_eurosat_eurosat_k1 | 2.2027 | 3.3733 |
| dtd_to_eurosat_dtd_k1 | 2.0560 | 3.5360 |
| dtd_to_eurosat_eurosat_k5 | 1.2507 | 2.8453 |
| dtd_to_eurosat_dtd_k5 | 1.2507 | 2.8453 |
| eurosat_to_dtd_dtd_k1 | 1.3973 | -0.0240 |
| eurosat_to_dtd_eurosat_k1 | 1.3360 | 0.1520 |
| eurosat_to_dtd_dtd_k5 | 0.6640 | 0.0240 |
| eurosat_to_dtd_eurosat_k5 | 0.6640 | 0.0240 |

View averaging improves over the original representation in all eight cells, with positive paired interval lower bounds. It carries about1.34–1.40pp of DTD1shot improvement and0.664pp of DTD5shot improvement. The scatter increment on DTD is close to zero. On EuroSAT, scatter additionally contributes3.37–3.54pp in1shot and2.85pp in5shot. EuroSAT gains have the same sign for all five task seeds; DTD signs are mixed. The identical5shot gallery-condition outputs reflect the frozen support-only R2 five-shot head and are not independent replications.

This is post-hoc descriptive decomposition. It does not establish the causal reason for the domain difference, generality to unseen domains, or impossibility of the entire covariance family. Augmented information can help while a particular geometric correction fails to improve broadly.

## Verification and limits

- All eight cells,10methods,4000task conditions completed (bash-825ca7da / science-53e01eec). These are2000unique task episodes with two gallery conditions each, not4000independent tasks.
- All aggregate accuracies, task identities, exact integer gamma choices, and original-control scores against historical outputs were checked. All four source prediction banks match the pre-tie-repair banks exactly.
- Frozen dense auditor bash-7dee487f / science-55e35668 failed at DTD source1shot task99, excluded gallery, gamma0.1. Score error0.003034226 exceeds frozen2e-5.
- Diagnostic bash-f9e21814 / science-e7fc0ac0:87/88 end-to-end comparisons pass, including all40sampled evaluation tasks. The single failed source case has one top64neighbor swap at float32 similarity tie; dense similarity difference4.3203e-7. Transform error1.2388e-7, identical predicted classes. Conditioning the independent NumPy regression on the actual float32 neighbor identities yields3.3859e-7score error and identical predictions. This conditional check does not turn the frozen audit into a pass.
-90predefined mean/interval comparisons independently reproduced with bootstrap multiplicities; maximum difference1.7764e-15pp. Bootstrap reflects task variation conditional on fixed image pools. It is not a random-domain confidence interval.
- Strict promotion is declined for two separate reasons: the measured robust-gain gate fails, and the original full-score validation remains failed. No tolerance relaxation, outcome-based gamma expansion, or rerun to rescue this result.
- The first diagnostic report write failed on NumPy int64 JSON serialization; the rerun changed serialization only. Both logs and diagnostic script hashes are retained.

## Retained assets and next action

Eight validated feature caches retained; redundant chunk file count=0; current free space=10.940GiB. No new deletion was needed in this pass. Old source outcomes, locks, code and numerical failure evidence remain available.

Choose the next research family through a fresh problem-first idea pass. Start from the measured information-versus-geometry contrast and verify closest literature before selecting a method. Compare representation aggregation/encoding-information routes, class-signal-preserving adaptation, and a simpler cost-aware view-mean comparator. These are search families, not approved methods. Do not reopen utility-weighting grids, source-coverage grids or scatter-gamma grids unchanged. Keep Pets metric contract and reserved Caltech evaluation untouched. Before future paper writing, map utility/coverage/scatter evidence into the authoritative outline/evidence ledger on the paper line.

Evidence: protocol.json; evaluation_contract.json; evaluation_lock.json; output_audit_contract.json; outputs/analysis.json; outputs/output_audit_diagnostics.json; outputs/statistical_audit.json; outputs/complete.json.
