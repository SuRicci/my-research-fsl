# Finite query-selector headroom: necessary condition passes
The existing fixed blend remains the measured incumbent at77.334%. An unattainable best-label choice among the raw-scatter, centered-scatter and fixed-blend predictions reaches79.217333%, a conditional increment of1.883333pp with paired95% interval[1.799333,1.968000]. Both development domains have positive ceiling: DTD1.412pp[1.312,1.516], EuroSAT2.354667pp[2.225333,2.488]. All five taskseed means in each domain are positive. A one-expert-per-task label oracle reaches78.322%; it is also unattainable.

The frozen necessary ceiling gate of0.5pp passes. This does not qualify a trained method, imply a continuous-mixture upper bound, or promote evaluation on new target data. It only justifies checking source-only feasibility and closest prior art before any new fit.

| Target_gallery | Blend % | Query oracle % | Ceiling pp | Correctable errors | Vulnerable correct |
|---|---:|---:|---:|---:|---:|
| dtd_dtd | 79.8880 | 81.1147 | 1.2267 | 460 | 506 |
| dtd_eurosat | 76.2667 | 77.8640 | 1.5973 | 599 | 723 |
| eurosat_dtd | 73.4853 | 76.7200 | 3.2347 | 1213 | 1668 |
| eurosat_eurosat | 79.6960 | 81.1707 | 1.4747 | 553 | 637 |

Across150000query occurrences (1000unique tasks, each under two galleries), 2825blend errors can be corrected by another expert, but3534correct blend predictions can be spoiled by at least one alternative. The latter count is an opportunity for damage, not an observed harm from a selector. Queries/images recur across episodes; counts are not independent samples. Galley conditions are averaged within each task before domain/pooled bootstrap, stratified by fixed taskseed and domain,5000replicates. All intervals are conditional on reused image pools and do not cover population or prospective-domain uncertainty.

Input audit covers all four immutable prediction banks, their hashes, exact method order, full prediction/accuracy parity, parent published means and task identities across galleries. An independent scalar reconstruction checked450000three-expert predictions and reproduced the pooled interval to1e-10. Full parent score banks are NOT available: only five sampled tasks per cell retain scores. Existing oracle for CLIP/DINO recipes is a different family and was not reused as this result.

Source-data risk: inherited one-shot gamma is10 when trained/selected from DTD and1 when trained/selected from EuroSAT. Thus simply training a gate on the existing opposite-direction evaluation scores would import parameters selected from the eventual target domain. Those banks support this diagnostic but are not automatically valid source-only gate training inputs. Next inspect source identity partitions and reuse same-source-selected gamma for any prospective training bank; no target-label tuning.

Parent decision-f7f139bf/report-64b6342f; science-855526f5. New computation bash-84538266, independent verification bash-df855952. No classifier training, downloads, new scores, target datasets, baseline metric changes or manuscript claims.
