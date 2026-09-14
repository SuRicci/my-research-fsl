# Reduced-view efficiency: fixed source rule not qualified

The predeclared source-only policy does not establish a <=3-view replacement for the six-view reference. Both source domains select the six-view fallback. This is failure to certify this fixed noninferiority rule, not proof that all reduced-view methods have unacceptable true accuracy loss. No new Pets or Caltech result and no novel-algorithm claim.

## Source-only selection

| Source selecting policy | Selected views | Closest reduced set by worst lower bound (descriptive) | Worst source mean delta (pp) | Worst paired95%lower bound (pp) |
|---|---|---|---:|---:|
| dtd | [0, 1, 2, 3, 4, 5] | [0, 1, 4] | -0.2667 | -0.5467 |
| eurosat | [0, 1, 2, 3, 4, 5] | [0, 2, 3] | -0.2400 | -0.6800 |

Selection required every R2, support-centered CS_l2 and fixed source-tuned linear classifier cell, across1/5shot and ordinary/excluded source galleries, to have paired95%lower bound at least-0.5pp. Source selection used100episodes per shot. Among feasible subsets the rule preferred fewer views, then worst-cell mean, macro mean and lexicographic order. Both source decisions were locked before evaluating any new cross-domain results. Each selected policy retains six views, so its measured gain relative to its own full-six reference is exactly zero and its structural cost ratio is1.

## Descriptive cross-domain frontier

The complete prospective frontier contains17 original-anchored subsets: original only, five2-view sets, ten3-view sets, and six-view reference. None of the16reduced fixed sets has all24 classifier/shot/gallery lower bounds at least-0.5pp in the saved evaluation curves. These unselected curves cannot be used to choose and promote a new policy. Individual intervals are conditional on fixed image pools; they do not establish a family-wide impossibility claim.

| Fixed view set | Worst evaluation cell | Classifier | Delta versus six views (pp) | Paired95%interval (pp) |
|---|---|---|---:|---|
| [0] | dtd_to_eurosat_k1_eurosat | linear_mean | -2.2587 | [-2.6960, -1.8133] |
| [0, 1] | dtd_to_eurosat_k1_eurosat | linear_mean | -1.4187 | [-1.7733, -1.0666] |
| [0, 2] | dtd_to_eurosat_k1_eurosat | linear_mean | -1.1360 | [-1.4640, -0.8080] |
| [0, 3] | dtd_to_eurosat_k1_eurosat | linear_mean | -0.9813 | [-1.3120, -0.6427] |
| [0, 4] | dtd_to_eurosat_k1_dtd | CS_l2 | -0.7920 | [-1.1467, -0.4533] |
| [0, 5] | dtd_to_eurosat_k1_dtd | r2 | -1.1440 | [-1.4560, -0.8293] |
| [0, 1, 2] | dtd_to_eurosat_k1_dtd | CS_l2 | -1.0160 | [-1.3120, -0.7200] |
| [0, 1, 3] | dtd_to_eurosat_k1_eurosat | linear_mean | -0.8560 | [-1.1253, -0.5813] |
| [0, 1, 4] | dtd_to_eurosat_k1_dtd | r2 | -0.4293 | [-0.6693, -0.1920] |
| [0, 1, 5] | dtd_to_eurosat_k1_dtd | r2 | -0.8347 | [-1.0960, -0.5706] |
| [0, 2, 3] | dtd_to_eurosat_k1_eurosat | linear_mean | -0.3573 | [-0.5787, -0.1413] |
| [0, 2, 4] | dtd_to_eurosat_k1_eurosat | linear_mean | -0.4853 | [-0.7600, -0.2053] |
| [0, 2, 5] | dtd_to_eurosat_k1_eurosat | linear_mean | -0.7040 | [-0.9520, -0.4507] |
| [0, 3, 4] | dtd_to_eurosat_k1_dtd | CS_l2 | -0.5520 | [-0.8400, -0.2587] |
| [0, 3, 5] | dtd_to_eurosat_k1_dtd | r2 | -0.7600 | [-1.0240, -0.5013] |
| [0, 4, 5] | dtd_to_eurosat_k1_eurosat | linear_mean | -0.6773 | [-0.9387, -0.4107] |

View indices:0 is original resize224/center224;1–4 are resize256 top-left/top-right/bottom-left/bottom-right224crops;5 is the resize256 center crop. The original anchor is a deliberately restricted family, not a search over every possible augmentation or policy. Some three-view mean losses are below0.5pp while uncertainty crosses the margin; do not label those losses as conclusively excessive.

## Measured local encoder cost

| Views | CLIP plus DINO median seconds per32images | Speed relative to six views | Encoder applications relative to six |
|---|---:|---:|---:|
| [0] | 1.2340 | 6.003x | 0.1667 |
| [0, 1] | 2.4538 | 3.019x | 0.3333 |
| [0, 1, 2] | 3.7130 | 1.995x | 0.5000 |
| [0, 1, 2, 3, 4, 5] | 7.4084 | 1.000x | 1.0000 |

Timing uses sequential frozen CLIP-B16 and DINOv2-S14 on local Apple MPS,32fixed source images (16perdomain), one warmup per subset/backbone and five randomized interleaved repetitions. Image decoding, crop generation and model loading are excluded; tensor preprocessing, device transfers, forward pass and CPU feature normalization are included. DINO bicubic interpolation used the supported CPU fallback for all view counts. This is measured encoding-stage latency under the fixed view-batched implementation, not full application latency or a universal hardware speedup. Gallery encoding can be amortized offline; query/support encoding savings depend on the deployment schedule. The benchmarked3view set is [0,1,2], not a qualified source-selected3view policy.

## Verification and comparison integrity

- Formal computation bash-7456b532 / science-62cef04f completed499.94s:2000uniqueevaluationepisodes,5taskseeds,8galleryconditions,3classifiers,17subsets.272prediction files cover selection and evaluation. Five-shot gallery duplicates and support-only classifiers do not form independent replications.
- Precheck bash-064eebd5 passed10synthetic/real cases, including exactduplicate retrieval ties and query partition invariance; maxscoreerror3.11e-15.
- Full independent audit bash-5d6db1b7 / science-d0429619 passed3264classifier-task score reconstructions (max4.47e-15),408evaluation frontier statistics (max8.88e-16pp) and independently recovered both source choices. Source ranking statistics were checked in addition to the408evaluationstatistics.
- This study explicitly recomputes all classifiers in float64 with descending cosine and stable index tie order. Historicalbridge40original/fullsix prediction comparisons has0flips and0accuracydelta. The older strict scatter-audit failure remains failed; no tolerance or historical evidence was rewritten.
- Actual encoder benchmark bash-f5893a8b / science-2a7f09b0 verifies7retained encoding inputs including both model weights.40timed encoder/subset measurements are finite and median/speed ratios independently checked.
- Locks, manifests and sampled float64scores retain exact provenance; all full task predictions are stored compactly asuint8. No test outcome selected a subset or altered the frozen margin.

## Resource and cleanup record

Study size at closure: 15.35MiB; free disk 10.215GiB. No image/model download. This pass removed 26untracked regenerable Python bytecode files (195506logicalbytes); source files, weights, cachedfeatures and raw predictions/logs were preserved. Bytecode may be regenerated by imports; do not treat its logical size as durable disk relief.

## Decision and handoff

Do not promote the fixed reduced-view rule or widen its subset grid from these evaluation outcomes. Retain six-view controls and the measured encoding-cost boundary. Next plausible inquiry is source-training diversity, after local label/identity/provenance checks: the prior training experiments use two source domains, whereas12intact restored feature containers cover six datasets. Container integrity alone does not establish label alignment or encoder parity. This is a feasibility direction, not a selected learned method; see artifacts/idea/broad_source_feasibility.md and broad_source_container_inventory.json. Task-conditioned aggregation and evidence synthesis remain alternatives; unchanged adapter/subset rescue is rejected.

The existing technical-note paper remains a draft. Map relevant newer auxiliary evidence into its authoritative outline/evidence ledger before any writing; do not infer submission readiness from this source report. No completion approval.
