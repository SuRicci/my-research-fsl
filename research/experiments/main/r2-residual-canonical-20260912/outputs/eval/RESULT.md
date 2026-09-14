# Paired experiment result

Paired locked task draws within historically viewed image pools; seed-stratified bootstrap preserves shared-gallery pairing. No independent-image or SOTA claim.

| Condition | R2 % | Residual % | Delta pp | 95% paired CI pp |
|---|---:|---:|---:|---|
| dtd_matched_1shot_accuracy | 76.863 | 73.865 | -2.997 | [-3.247, -2.737] |
| dtd_5shot_accuracy | 89.981 | 89.981 | +0.000 | [0.000, 0.000] |
| dtd_mismatched_1shot_accuracy | 74.079 | 74.011 | -0.068 | [-0.309, 0.171] |
| eurosat_mismatched_1shot_accuracy | 66.117 | 68.248 | +2.131 | [1.816, 2.441] |
| eurosat_matched_1shot_accuracy | 74.035 | 68.408 | -5.627 | [-5.904, -5.341] |
| eurosat_5shot_accuracy | 86.808 | 86.808 | +0.000 | [0.000, 0.000] |

Decision gate: {'positive_r2_macro_ci': False, 'no_cell_loss_above_0_5pp': False, 'beats_simple_controls': False, 'simple_controls': {'support_tuned': False, 'support_proto': True, 'isotropic': False}, 'passed': False}

Macro control comparison:
{
  "r2": {
    "delta_pp": -1.640333333333333,
    "paired_ci95_pp": [
      -1.7929999999999997,
      -1.491
    ]
  },
  "residual": {
    "delta_pp": 0.0,
    "paired_ci95_pp": [
      0.0,
      0.0
    ]
  },
  "support_proto": {
    "delta_pp": 0.7210000000000001,
    "paired_ci95_pp": [
      0.5386583333333336,
      0.8943333333333334
    ]
  },
  "support_ridge": {
    "delta_pp": -0.23833333333333326,
    "paired_ci95_pp": [
      -0.31366666666666654,
      -0.1636666666666666
    ]
  },
  "support_tuned": {
    "delta_pp": -0.23833333333333326,
    "paired_ci95_pp": [
      -0.31366666666666654,
      -0.1636666666666666
    ]
  },
  "support_mass_matched": {
    "delta_pp": -0.051666666666666666,
    "paired_ci95_pp": [
      -0.07866666666666655,
      -0.02500000000000001
    ]
  },
  "isotropic": {
    "delta_pp": -0.05133333333333333,
    "paired_ci95_pp": [
      -0.07799999999999987,
      -0.024999999999999942
    ]
  },
  "mean_only": {
    "delta_pp": -1.552333333333333,
    "paired_ci95_pp": [
      -1.7060083333333331,
      -1.4019833333333334
    ]
  },
  "distribution": {
    "delta_pp": -1.6079999999999997,
    "paired_ci95_pp": [
      -1.7606833333333334,
      -1.4589999999999999
    ]
  },
  "random_residual": {
    "delta_pp": -0.04933333333333323,
    "paired_ci95_pp": [
      -0.08366666666666653,
      -0.014999999999999881
    ]
  }
}