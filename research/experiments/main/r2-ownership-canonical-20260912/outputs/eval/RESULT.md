# Ownership paired main result

Exploratory paired task inference on viewed fixed image pools; shared-gallery pairing preserved. Five-shot primary is inherited R2; no new-method or SOTA inference.

| Condition | Ownership % | R2 % | CS_l2 % |
|---|---:|---:|---:|
| dtd_matched_1shot_accuracy | 76.985 | 76.863 | 76.873 |
| dtd_5shot_accuracy | 89.981 | 89.981 | 89.952 |
| dtd_mismatched_1shot_accuracy | 73.920 | 74.079 | 73.871 |
| eurosat_mismatched_1shot_accuracy | 65.769 | 66.117 | 68.124 |
| eurosat_matched_1shot_accuracy | 74.465 | 74.035 | 73.665 |
| eurosat_5shot_accuracy | 86.808 | 86.808 | 88.211 |

Gate: {"strong_macro": {"r2": false, "centered_r2": false}, "cell_loss": {"r2": true, "centered_r2": false}, "ownership_attribution": {"count_nearest": true, "count_random": true, "mass_only": true}, "passed": false}

Macro comparisons:
{
  "r2": {
    "delta_pp": 0.011666666666666763,
    "paired_ci95_pp": [
      -0.11533333333333316,
      0.13834166666666678
    ]
  },
  "support_proto": {
    "delta_pp": 2.373,
    "paired_ci95_pp": [
      2.1826583333333334,
      2.560683333333333
    ]
  },
  "ownership": {
    "delta_pp": 0.0,
    "paired_ci95_pp": [
      0.0,
      0.0
    ]
  },
  "count_nearest": {
    "delta_pp": 0.39433333333333326,
    "paired_ci95_pp": [
      0.28500000000000003,
      0.5036749999999999
    ]
  },
  "count_random": {
    "delta_pp": 0.27200000000000013,
    "paired_ci95_pp": [
      0.1629916666666667,
      0.38001666666666667
    ]
  },
  "mass_only": {
    "delta_pp": 0.22399999999999998,
    "paired_ci95_pp": [
      0.09765833333333337,
      0.3499999999999999
    ]
  },
  "exclusive_mass": {
    "delta_pp": 0.21100000000000016,
    "paired_ci95_pp": [
      0.08733333333333373,
      0.33233333333333326
    ]
  },
  "centered_ownership": {
    "delta_pp": -0.053333333333333136,
    "paired_ci95_pp": [
      -0.1820083333333334,
      0.07167500000000003
    ]
  },
  "r2_tuned": {
    "delta_pp": -0.021333333333333194,
    "paired_ci95_pp": [
      -0.14268333333333327,
      0.09967500000000011
    ]
  },
  "centered_r2": {
    "delta_pp": -0.34833333333333333,
    "paired_ci95_pp": [
      -0.4776666666666667,
      -0.2219916666666672
    ]
  },
  "support_tuned": {
    "delta_pp": 1.4136666666666668,
    "paired_ci95_pp": [
      1.259658333333333,
      1.5643416666666665
    ]
  }
}