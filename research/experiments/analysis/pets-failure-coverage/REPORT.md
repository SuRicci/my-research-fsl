# Pets failure coverage

Repeated task occurrences, not independent images. Labels are retrospective diagnostics only. DTD class ids are never compared numerically to Pets ids. No causal mediation or newly calibrated selector is established.

| Condition | Breeds losing to R2 | Mean selected gallery images | Outside episode classes | Correct among all selected | Max linear residual |
|---|---:|---:|---:|---:|---:|
| pets_pets_k1 | 21/37 | 1254.9 | 80.01% | 19.23% | 5.77e-15 |
| pets_dtd_k1 | 37/37 | 1769.4 | 100.00% | 0.00% | 1.29e-14 |
| pets_pets_k5 | 25/37 | 1270.5 | 80.18% | 19.43% | 6.66e-15 |
| pets_dtd_k5 | 37/37 | 1664.6 | 100.00% | 0.00% | 1.29e-14 |

The external-domain gallery is retained extensively despite broad query losses. Small linear-system residuals and exact classification-head replays make a numerical-solver explanation less plausible; they do not prove which modeling assumption causes the failure. Matched-gallery contamination is also substantial, so domain matching alone does not establish episode relevance or superiority over the existing retrieval control.

No new gate was fitted. Next: challenge the whole-gallery, all-pseudolabel assumption and compare structurally distinct information/objective routes using prior literature and an untouched qualification plan.
