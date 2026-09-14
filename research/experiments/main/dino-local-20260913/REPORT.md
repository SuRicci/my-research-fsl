# Fixed native DINO local matching: rejected

Source development pooled fusion 75.786000% versus parent 77.374000%; delta -1.588000pp, 95% paired interval [-1.8066833333333334, -1.366]. All fixed advancement gates fail.

| Domain | Parent | Patch only | Fusion | Mean fusion | Fusion minus parent |
|---|---:|---:|---:|---:|---:|
| dtd | 78.1573 | 74.1200 | 77.5480 | 77.4187 | -0.6093 |
| eurosat | 76.5907 | 64.8373 | 74.0240 | 74.0853 | -2.5667 |

Fusion-minus-mean control 0.034000pp, CI [-0.03000000000000003, 0.0973333333333332]. Correct descriptor ownership exceeds shuffled ownership, but this positive control does not establish superiority over the retained parent.

Net added errors: 2382 over 150000 paired query-condition evaluations. These contain 1000 unique source episodes repeated across two galleries, not 2000 independent episodes.

Extraction: all 7277 image identities and cached CLS checked, 264.962s. Native CLS first batch each domain matched exactly; block pooling formulations max error 5.7220459e-06. Numerical audit: 13 cases, max local score error 1.6653345e-15, all750000 predictions, all21 independent intervals (max error 3.5527137e-15pp). Classification time 90.788s before standalone interval verification.

Fixed protocol: native DINO16x16 tokens -> nonoverlapping4x4 pooling -> normalized16descriptors; nearest support-class matching; equal per-query standardized-score blend. Sources/protocol/code/output hashes are durable in protocol.json, assets/manifest.json and outputs/code_lock.json/audit.json. Two earlier capability failures retained; actual extraction restored the existing CPU fallback setting.

Boundary: same one-shot source episodes and label/query permission as the saved cumulative parent; new original-view native descriptors are an explicit representation change. Source data were repeatedly exposed. No Caltech/Pets evaluation, no canonical required-metric replacement, no independent-domain generalization or new-method contribution claim. The fixed recipe is closed; do not tune its blend on this result.

Next: one lightweight saved-output failure analysis to quantify error overlap and finite source-only headroom. An oracle upper bound, if calculated, is diagnostic and never a deployable method. Preserve native feature cache for justified future use; do not re-encode.
