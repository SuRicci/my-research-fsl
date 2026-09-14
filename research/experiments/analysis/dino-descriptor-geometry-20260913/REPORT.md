# Native pooled descriptor geometry: shared image component dominates

All7277source query-pool images were checked without labels, predictions or fitting. Feature hashes match the verified extraction manifest. Two algebraic identities cross-check pairwise cosine and mean/residual energy.

| Domain | Images | Mean off-diagonal token cosine | Mean image-component energy | Mean residual energy | Mean Gram participation rank |
|---|---:|---:|---:|---:|---:|
| dtd | 1877 | 0.832545 | 0.843011 | 0.156989 | 1.411237 |
| eurosat | 5400 | 0.824134 | 0.835125 | 0.164875 | 1.426412 |

Inference: the normalized16-descriptor sets are geometrically dominated by a shared per-image component; effective participation rank is about1.4, far below the maximum16. This motivates examining image-mean residual descriptors as a different representation before adding a more complex support reconstruction solver. It does not establish semantic usefulness, foreground quality, register artifacts, or classification improvement. Image-mean subtraction can amplify low-energy noise and remains an unmeasured hypothesis.

Next bounded idea work: inspect primary literature for descriptor centering/residual pooling and compare fixed per-image residual matching with FRN-style support reconstruction. Prefer a single zero-fit residual candidate only if the prior-art and validity check justify it; otherwise reject both. Reuse cache, no new downloads/encoding. Freeze any classification recipe before execution and keep same source-only developmental boundary.
