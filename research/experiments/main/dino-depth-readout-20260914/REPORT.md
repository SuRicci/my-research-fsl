# Fixed DINO multi-depth readout: verified rejection

The prespecified multi-depth representation fails the source qualification gate. It improves EuroSAT while degrading DTD; pooled accuracy versus retained final-layer consistency changes by -1.1793 pp (95% paired interval [-1.4073, -0.9546]). These intervals describe fixed, repeatedly exposed development pools, not population-level or held-out target generalization.

| Evaluation | Multi-depth accuracy | Final-layer accuracy | Difference, pp (95% paired CI) |
|---|---:|---:|---:|
| dtd | 74.8173% | 78.1573% | -3.3400 [-3.6680, -3.0226] |
| eurosat | 77.5720% | 76.5907% | +0.9813 [+0.6800, +1.2947] |

## Protocol and verification
Eight fixed methods, 1000 unique one-shot tasks and 2000 gallery conditions; the two galleries are averaged within each task before pooled inference. Reuse the same six image views and frozen encoders. Fixed blocks 9-12, equal DINO block energy and half CLIP/half DINO weights; gamma is dimension-adjusted by 896/d. Equal data and query-local permissions; head compute cost is not matched.

The primary beats the four simpler multi-depth controls when pooled, but loses to both retained final-layer controls. Passing weaker controls does not qualify it. DTD also violates the nonnegative directional point-gain and gallery noninferiority requirements. The full gate and all comparisons remain in protocol.json and outputs/analysis.json.

Managed computation and audit both exited zero. Audit independently checked 1,200,000 prediction entries and 22 paired intervals, verified task identities and exact retained-control predictions, and recomputed the primary equations on four prespecified cases (maximum score error below 9e-16). The logistic solver was not independently reimplemented. Input hashes, method order, array shapes and finite values were rechecked after completion.

## Decision boundary and next action
Reject this fixed readout; do not resweep layers, weights or strength using these outcomes. Do not promote the EuroSAT-only gain to a robust improvement. Retain the existing conditional final-layer scatter/consistency incumbent. Return to evidence synthesis and remaining-value assessment; no target qualification or Caltech reevaluation is justified by this gate failure.

All five canonical Pets metrics remain unmeasured and unchanged. This is auxiliary development evidence, recorded as a report and Science Evidence Graph outcome; it is not submitted as a canonical main result with copied baseline numbers. Keep manuscript-facing claims unchanged until paper-contract mapping is reviewed.

## Durable evidence
RESULT.json; protocol.json; outputs/analysis.json; outputs/audit.json; outputs/run_manifest.json; all four cell NPZs. Managed sessions: bash-5721cf20 and bash-5c66513c. Science nodes: science-620dec73 and science-01c5426c.
