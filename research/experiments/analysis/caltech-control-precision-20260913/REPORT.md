# Caltech fixed comparator repair: verified, transfer verdict unchanged

The deterministic float64 comparator resolves the recorded end-to-end numerical mismatch on the fixed audit cases. All 112500 saved predictions are unchanged from the earlier control. This repairs comparison trust without providing an accuracy improvement.

| Gallery | Repaired control accuracy (%) | Changed predictions |
|---|---:|---:|
| caltech101 | 98.394667 | 0 |
| dtd | 97.784000 | 0 |
| eurosat | 98.058667 | 0 |

The 15 fixed real-task checks and one synthetic check passed with maximum score error 3.331e-16; original tolerance remained 2e-5. Independent saved-prediction and count-weighted bootstrap verification passed all 12 intervals, maximum interval discrepancy 2.776e-17 percentage points. The 500 unique tasks are shared across three galleries; 1500 conditions are not independent tasks.

Compared with this control, the DTD-trained primary configuration changes mean accuracy by -0.047111 percentage points (95% paired interval [-0.120889, 0.026667]); the EuroSAT-trained configuration changes it by -0.100444 points ([-0.173356, -0.026667]). Neither supports superiority. Gallery names denote gallery composition; the query target is Caltech101 throughout.

Boundary: this is a numerical variant on previously evaluated target assets. No model fitting, target tuning, fresh confirmation, or canonical Pets metric update was performed. Original failed audits, all original ten-method banks, and the separate source representation-scatter failure remain preserved. No manuscript C1-C4 claim changes follow from this auxiliary repair.

Execution and provenance: bash-12eb6fda finished successfully at 2026-09-13 17:26:38 UTC. Computation took 13.47 seconds. Recovery verified all recorded input, output, code, and historical failure hashes and reused the completed run. The previous science node remained marked running after output completion; it must now be updated, not duplicated.

Next: map this repair under the existing Caltech reference-only evidence item; return to idea selection using the retained source-development result and its failed external superiority test. Do not rerun or tune this fixed control.
