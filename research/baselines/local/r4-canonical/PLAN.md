# Frozen V0 local comparator

Objective: verify unchanged V0 and all nine original lab methods plus old centering, partial whitening, alpha2 centered residual and evaluator-only oracle on existing paired features. decision-eeb1407d; audit artifacts/intake/r4-v0-reuse-20260912/state_audit.md. This is a distinct semantic retrieval baseline, not historical result reproduction.

Success: original-source numerical equivalence, paired feature hashes/identity and RGB role disjointness, full finite ranking permutations, AP independently recalculated, eight eval cells and separate four development cells, flat canonical metric contract with traceable values. Only then confirm r4-canonical-local/semantic-mps32-v1.

No empirical tuning: all method settings inherited. DTD original class split reserved for later dev/eval; EuroSAT only eval. Future result interpretation must compare geometry and alignment controls, not only raw old scores. Macro and per-cell values retained; no new-method claim.

Efficient route: cached normalized feature vectors, CPU BLAS6, no download/reencoding. Estimate output<=1GiB; enforce actual10GiB disk floor. One source/evaluator/identity validation; then one managed real run. Abort on identity/hash mismatch, nonfinite score, metric inconsistency or resource floor.

Next: implement wrapper and freeze manifest; validate numerical/scoring contract; run and independently audit outputs; confirm or record blocked state; decision/idea.

## Exit
Accepted baseline-2ec8a7ee after decision-e5661c2b. Fixed old_centered is strongest uniform comparator; V0 remains a trusted separately measured variant. No historical metric-parity claim. Next anchor idea; original measurement and revised independent audit both durably resolved.
