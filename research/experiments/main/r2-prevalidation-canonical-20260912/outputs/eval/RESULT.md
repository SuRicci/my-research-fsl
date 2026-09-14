# Support-only prevalidation result

Known bounded PreVal-inspired transfer. Four1-shot cells inherited unchanged; only5-shot selection is new. Fixed-pool conditional paired intervals.

| Metric | R2 | Candidate | Delta pp | Paired95%CI |
|---|---:|---:|---:|---|
| dtd_matched_1shot_accuracy | 76.863 | 76.873 | +0.011 | [-0.133,0.157] |
| dtd_5shot_accuracy | 89.981 | 89.852 | -0.129 | [-0.249,-0.011] |
| dtd_mismatched_1shot_accuracy | 74.079 | 73.871 | -0.208 | [-0.387,-0.036] |
| eurosat_mismatched_1shot_accuracy | 66.117 | 68.124 | +2.007 | [1.743,2.271] |
| eurosat_matched_1shot_accuracy | 74.035 | 73.665 | -0.369 | [-0.541,-0.196] |
| eurosat_5shot_accuracy | 86.808 | 89.505 | +2.697 | [2.492,2.909] |

Five-shot strong-control contrasts:
{
  "r2": {
    "delta_pp": 1.284,
    "paired_ci95_pp": [
      1.1606666666666667,
      1.405333333333333
    ]
  },
  "parent_centered": {
    "delta_pp": 0.5973333333333336,
    "paired_ci95_pp": [
      0.4960000000000004,
      0.6993500000000002
    ]
  },
  "raw_tuned": {
    "delta_pp": 1.284,
    "paired_ci95_pp": [
      1.1606666666666667,
      1.405333333333333
    ]
  },
  "radius": {
    "delta_pp": 0.4780000000000004,
    "paired_ci95_pp": [
      0.3840000000000004,
      0.5733333333333339
    ]
  },
  "press": {
    "delta_pp": 0.021333333333333412,
    "paired_ci95_pp": [
      -0.012666666666666604,
      0.05466666666666686
    ]
  }
}

Gate:{'beats_tuned_raw': True, 'beats_radius': True, 'no_five_shot_loss_over_0_5pp': True, 'passed': True}

Interpretation: support-only selection improves five-shot average with EuroSAT concentration. Calibrated log-loss offers no established advantage over simpler PRESS-MSE. Positive one-shot delta is inherited from parent and must not be attributed to this selector.