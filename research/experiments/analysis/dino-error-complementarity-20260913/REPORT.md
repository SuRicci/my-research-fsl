# Descriptive local/parent error complementarity
Parent report-6eb4cbf9, fixed native DINO local fusion rejected. This analysis changes no predictions and fits no parameter.

Across150000 query-condition evaluations, native patch-only predictions uniquely correct 6699 cases, whereas parent predictions uniquely correct 18542; both are wrong 27240 times. Descriptor information is not wholly redundant, but replacing parent decisions is usually harmful. All sixteen within-cell parent-margin quartiles contain more parent-only correct cases than patch-only correct cases.

The unfitted normalized margin difference has descriptive AUC 0.6212 to 0.6910 among cases with exactly one method correct. This does not establish a deployable selector: the subset itself uses ground truth, and all source outputs were already exposed. No threshold was fitted or evaluated.

The label-informed union upper bound is diagnostic only; it is not method accuracy. Two gallery conditions share each episode, and no independent-domain generalization is measured. These findings do not rescue equal fusion or justify a confidence switch. Original verified native feature cache is retained; no re-encoding needed.

Route: close this one-question analysis. Return to idea with a bounded representation-quality question: inspect within-image descriptor redundancy before choosing between image-mean residual local descriptors and support-class reconstruction, both on the existing cache. More blend weights, confidence thresholds or Caltech tuning are not justified by this result. A new candidate must be fixed and recorded before any further classification run. Existing narrow OSLO paper receives reference-only mapping, no new claim.
