# Paired experiment result

Paired locked task draws within historically viewed image pools; seed-stratified bootstrap preserves shared-gallery pairing. No independent-image or SOTA claim.

| Condition | R2 % | Support-centered R2 % | Delta pp | 95% paired CI pp |
|---|---:|---:|---:|---|
| dtd_matched_1shot_accuracy | 76.863 | 76.873 | +0.011 | [-0.133, 0.157] |
| dtd_5shot_accuracy | 89.981 | 89.952 | -0.029 | [-0.093, 0.032] |
| dtd_mismatched_1shot_accuracy | 74.079 | 73.871 | -0.208 | [-0.387, -0.036] |
| eurosat_mismatched_1shot_accuracy | 66.117 | 68.124 | +2.007 | [1.743, 2.271] |
| eurosat_matched_1shot_accuracy | 74.035 | 73.665 | -0.369 | [-0.541, -0.196] |
| eurosat_5shot_accuracy | 86.808 | 88.211 | +1.403 | [1.281, 1.521] |

Decision gate: {'positive_r2_macro_ci': True, 'no_cell_loss_above_0_5pp': True, 'beats_simple_controls': True, 'simple_controls': {'r2_tuned': True, 'centered_support': True, 'centered_proto': True, 'support_ridge': True, 'support_proto': True, 'gallery_r2': True}, 'passed': True}

Macro control comparison:
{
  "r2": {
    "delta_pp": 0.3600000000000001,
    "paired_ci95_pp": [
      0.2603250000000003,
      0.46133333333333304
    ]
  },
  "support_proto": {
    "delta_pp": 2.721333333333333,
    "paired_ci95_pp": [
      2.5176583333333338,
      2.9230166666666664
    ]
  },
  "centered_proto": {
    "delta_pp": 2.0613333333333337,
    "paired_ci95_pp": [
      1.9193333333333333,
      2.2050083333333332
    ]
  },
  "r2_tuned": {
    "delta_pp": 0.3270000000000001,
    "paired_ci95_pp": [
      0.21833333333333332,
      0.43666666666666676
    ]
  },
  "centered_r2": {
    "delta_pp": 0.0,
    "paired_ci95_pp": [
      0.0,
      0.0
    ]
  },
  "centered_support": {
    "delta_pp": 2.2713333333333328,
    "paired_ci95_pp": [
      2.1303249999999996,
      2.415333333333333
    ]
  },
  "support_ridge": {
    "delta_pp": 1.762,
    "paired_ci95_pp": [
      1.617325,
      1.9083333333333332
    ]
  },
  "gallery_r2": {
    "delta_pp": 1.108333333333333,
    "paired_ci95_pp": [
      0.9913166666666663,
      1.228333333333333
    ]
  }
}