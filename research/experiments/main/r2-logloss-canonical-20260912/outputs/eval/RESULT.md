# Paired experiment result

Paired locked task draws within historically viewed image pools; seed-stratified bootstrap preserves shared-gallery pairing. No independent-image or SOTA claim.

| Condition | R2 % | Neighborhood logistic % | Delta pp | 95% paired CI pp |
|---|---:|---:|---:|---|
| dtd_matched_1shot_accuracy | 76.863 | 76.748 | -0.115 | [-0.280, 0.049] |
| dtd_5shot_accuracy | 89.981 | 89.981 | +0.000 | [0.000, 0.000] |
| dtd_mismatched_1shot_accuracy | 74.079 | 73.836 | -0.243 | [-0.392, -0.085] |
| eurosat_mismatched_1shot_accuracy | 66.117 | 66.609 | +0.492 | [0.304, 0.675] |
| eurosat_matched_1shot_accuracy | 74.035 | 72.311 | -1.724 | [-1.903, -1.545] |
| eurosat_5shot_accuracy | 86.808 | 86.808 | +0.000 | [0.000, 0.000] |

Decision gate: {'positive_r2_macro_ci': False, 'no_cell_loss_above_0_5pp': False, 'beats_simple_controls': False, 'simple_controls': {'r2_tuned': False, 'support_logistic': True, 'mean_logistic': True, 'support_ridge': True, 'mean_ridge': False, 'support_proto': True, 'distribution_ridge': False}, 'passed': False}

Macro control comparison:
{
  "r2": {
    "delta_pp": -0.39733333333333337,
    "paired_ci95_pp": [
      -0.48766666666666664,
      -0.3083166666666666
    ]
  },
  "support_proto": {
    "delta_pp": 1.964,
    "paired_ci95_pp": [
      1.7633333333333334,
      2.1676666666666664
    ]
  },
  "distribution_logistic": {
    "delta_pp": 0.0,
    "paired_ci95_pp": [
      0.0,
      0.0
    ]
  },
  "mean_logistic": {
    "delta_pp": 0.016666666666666684,
    "paired_ci95_pp": [
      0.0019999999999999736,
      0.03133333333333319
    ]
  },
  "support_logistic": {
    "delta_pp": 1.3106666666666666,
    "paired_ci95_pp": [
      1.157658333333333,
      1.4643416666666664
    ]
  },
  "distribution_ridge": {
    "delta_pp": -0.36500000000000005,
    "paired_ci95_pp": [
      -0.4133333333333334,
      -0.31566666666666693
    ]
  },
  "mean_ridge": {
    "delta_pp": -0.30933333333333346,
    "paired_ci95_pp": [
      -0.3589999999999999,
      -0.2596666666666668
    ]
  },
  "support_ridge": {
    "delta_pp": 1.0046666666666666,
    "paired_ci95_pp": [
      0.8426666666666667,
      1.1693499999999997
    ]
  },
  "r2_tuned": {
    "delta_pp": -0.4303333333333333,
    "paired_ci95_pp": [
      -0.5363333333333332,
      -0.3223333333333333
    ]
  }
}