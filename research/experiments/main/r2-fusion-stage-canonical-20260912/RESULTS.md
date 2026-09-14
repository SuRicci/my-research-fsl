# Fusion-stage paired result

Strong gate: FAIL. Development-selected comparator: early_shared.

| Method | DTD matched | DTD mismatch | EuroSAT mismatch | EuroSAT matched | 1-shot macro | DTD 5-shot* | EuroSAT 5-shot* |
|---|---:|---:|---:|---:|---:|---:|---:|
| late_shared | 77.1573 | 74.0480 | 65.9013 | 74.8147 | 72.9803 | 89.9813 | 86.8080 |
| early_shared | 77.0013 | 74.0720 | 65.6747 | 74.4773 | 72.8063 | 89.9813 | 86.8080 |
| early_native | 77.0013 | 74.0720 | 65.6747 | 74.4773 | 72.8063 | 89.9813 | 86.8080 |
| centered_native | 76.8733 | 73.8707 | 68.1240 | 73.6653 | 73.1333 | 89.9813 | 86.8080 |
| late_support | 73.6253 | 73.6253 | 68.5040 | 68.5040 | 71.0647 | 89.9813 | 86.8080 |
| clip_native | 66.4720 | 63.7000 | 63.5307 | 69.7800 | 65.8707 | 89.9813 | 86.8080 |
| dino_native | 76.4240 | 73.2507 | 61.9440 | 69.7173 | 70.3340 | 89.9813 | 86.8080 |
| r2 | 76.8627 | 74.0787 | 66.1173 | 74.0347 | 72.7733 | 89.9813 | 86.8080 |
| cs_l2_fixed | 76.8733 | 73.8707 | 68.1240 | 73.6653 | 73.1333 | 89.9813 | 86.8080 |

*Five-shot rows inherit R2 exactly and are not evidence for the fusion mechanism.

| Late_shared minus comparator | Macro delta, pp | 95% paired CI, pp |
|---|---:|---|
| early_shared | +0.1740 | [+0.1107, +0.2397] |
| early_native | +0.1740 | [+0.1107, +0.2397] |
| centered_native | -0.1530 | [-0.2703, -0.0330] |
| late_support | +1.9157 | [+1.7457, +2.0847] |
| clip_native | +7.1097 | [+6.7413, +7.4693] |
| dino_native | +2.6463 | [+2.4833, +2.8087] |
| r2 | +0.2070 | [+0.1257, +0.2893] |
| cs_l2_fixed | -0.1530 | [-0.2703, -0.0330] |

## Gate and limitations

{
  "macro_at_least_half_pp": false,
  "macro_ci_positive": true,
  "no_cell_loss_over_half_pp": true,
  "dev_selected_control": "early_shared",
  "control_ci_positive": true,
  "interior_weight": true,
  "pass": false
}

same historically seen image pools; paired task bootstrap stratified by seed, galleries paired within query domain; conditional intervals
The primary method was selected before evaluation. Reported comparator differences do not authorize post-hoc primary replacement. Fusion is an established technique; no new algorithm or independent-domain validation is implied.
