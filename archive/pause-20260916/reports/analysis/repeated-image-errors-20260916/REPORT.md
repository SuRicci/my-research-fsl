# Repeated-image residual errors: descriptive follow-up

Question: after the ordered-crop readout failed, does existing evidence support treating the residual error as a fixed set of inherently unrecognizable images? This one-question analysis reads accepted incumbent predictions on the same1000tasks; no new model is trained or selected.

| Domain | Query originals | Mixed-correctness originals | Errors on mixed originals | Always-wrong repeated originals | Repeated query/classset groups | Mixed query/classset groups |
|---|---:|---:|---:|---:|---:|---:|
| dtd | 1877 | 904 | 99.347% | 1 | 0 | 0 |
| eurosat | 5392 | 1346 | 98.040% | 13 | 1019 | 92 |

Most errors occur on originals that are classified correctly in another observed episode. Thus these saved observations do not support treating all remaining error as fixed image-level inability. They do not prove that support resampling or a new estimator will correct those errors prospectively.

Repeated query/classset contexts hold the exact query and five actual labels fixed; changing support originals can still change the deterministic prediction. Such observed switching, where present, establishes support-context dependence within these tasks. It does not isolate support quality from support-estimated geometry or classifier parameters. DTD has no matched repeat groups in this sample, so the same conditional statement cannot be made there.

## Limits and comparison boundary

No intervention, new scores, test-domain confirmation, causal attribution outside matched contexts, or performance gain is claimed. The class set and supports vary together in general; finite repeated visits do not define inherent image difficulty. Class-pair rankings are descriptive and exposed; they cannot train or select a method under the current information contract. The top10confusion pairs account for different amounts of opportunity and should not be read as a fixed set of recoverable mistakes.

## Integrity and execution

Both domains retain37,500queries; source hashes are saved. Per-original counts and errors were independently checked by vectorized aggregation. Class identity per original is consistent, and identical query/classset/support sets have deterministic correctness. The first bounded execution timed out because lazy NPZ field access repeatedly decompressed arrays; a once-per-domain materialization fixed the I/O issue. Only the successful rerun produces these results. No weights or feature copies were created.

## Route implication

The fixed-image impossibility premise is weakened; a task-context question remains more defensible than another crop-information or static per-image confidence tweak. The next idea pass should compare a support/competitor-conditioned decision objective against the already-closed support-view, confidence, covariance and generic head families. It must first identify a mathematical or information distinction and nearest-prior overlap; otherwise reject it without another run. The existing result does not select a new classifier, justify a parameter search, or establish paper novelty.

Evidence: protocol.json, summary.json, per-image and directed-classpair CSV files, analyze.py and execution_note.md. Parent mainrun-1ec98c8a; routingdecision-7e97595d. This report is pre-outline reference evidence, not manuscript-ready empirical support for a new method.
