# Source calibration qualification — rejected

This auxiliary development study does not qualify a new method. Eight complete paired cells contain 4,000 task-condition evaluations from 2,000 sampled tasks. Two separate source calibrators used 200 support/gallery training episodes, and source-only control selection used 200 additional labeled episodes. All image pools are historically exposed. No canonical Pets metric or Caltech outcome was produced.

## Paired one-shot transfer

| Training source → evaluation domain | Δ versus source-selected control (pp) | Paired 95% CI | Δ versus same-mass control (pp) |
|---|---:|---|---:|
| dtd → eurosat | 0.4787 | [0.3027, 0.6693] | 0.0000 |
| eurosat → dtd | -1.4493 | [-1.7160, -1.1679] | 0.0000 |

The positive DTD→EuroSAT mean is explained by the same-mass control: candidate and control make identical predictions across all evaluated conditions. EuroSAT→DTD loses against the source-selected mix/lambda control. Five-shot exactly equals R2 by design and trails CS_l2 on EuroSAT; this alone also fails the all-cell gate. No composition benefit is established.

## Complete accuracy surface

| Cell | Calibrated | Same mass | Fixed prior | R2 | CS_l2 | Logistic C1 | Logistic C10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dtd_to_eurosat_dtd_k1 | 68.4613 | 68.4613 | 67.6533 | 65.8747 | 67.9173 | 67.6213 | 68.2427 |
| dtd_to_eurosat_dtd_k5 | 86.8240 | 86.8240 | 86.8240 | 86.8240 | 89.3947 | 85.1200 | 88.1653 |
| dtd_to_eurosat_eurosat_k1 | 73.6693 | 73.6693 | 73.3680 | 73.6693 | 73.2560 | 67.6213 | 68.2427 |
| dtd_to_eurosat_eurosat_k5 | 86.8240 | 86.8240 | 86.8240 | 86.8240 | 89.3947 | 85.1200 | 88.1653 |
| eurosat_to_dtd_dtd_k1 | 75.5547 | 75.5547 | 75.7573 | 78.8133 | 78.8933 | 75.5760 | 75.4080 |
| eurosat_to_dtd_dtd_k5 | 90.5733 | 90.5733 | 90.5733 | 90.5733 | 90.3600 | 90.4747 | 90.6827 |
| eurosat_to_dtd_eurosat_k1 | 75.5547 | 75.5547 | 75.5653 | 75.1173 | 74.8560 | 75.5760 | 75.4080 |
| eurosat_to_dtd_eurosat_k5 | 90.5733 | 90.5733 | 90.5733 | 90.5733 | 90.3600 | 90.4747 | 90.6827 |

## Mechanism boundary

On matched EuroSAT, the DTD-trained estimator assigns a mean prior ≈1 while the actual task-membership fraction is0.5. On matched DTD, the EuroSAT-trained estimator assigns ≈0 while the actual mean is0.1064. Fully mismatched galleries correctly tend to0. The estimator thus behaves mostly as a domain-dependent on/off switch; its composition contribution vanishes at prediction level. The extremes agree with independent numerical optimization on audited real examples. This is compatible with score miscalibration or class-conditional shift; it does not identify the sole cause or falsify all learned relevance methods.

The in-distribution calibration premise did not transfer in this fixed setup. Do not retry a prior/temperature grid on these outcomes, treat the directional gain as a general improvement, or spend the reserved Caltech target on this candidate. Direct query-utility learning is a distinct hypothesis that remains untested; LST/DS3L are direct prior art, so it needs a separate scoped contract rather than being presented as a fix that this experiment validates.

## Verification and provenance

Source locks match; feature hashes/ids verified by the existing loader; calibration-support versus source-selection images are disjoint; support and query indices are disjoint within every task; all19 method outputs per cell are finite and predictions/accuracies recompute. Four first-task real-data EM estimates match an independent bounded MLE within1e-5. Five-shot scores exactly equalR2. Array hashes and full audit are outputs/validation.json; all intervals and method scores are outputs/analysis.json and outputs/cells/. Total run artifacts132858078bytes at audit; free11.14GiB. Managed run bash-29f564d1 completed in103seconds, independent audit bash-790ec7df.

## Handoff

Record as auxiliary source-qualification evidence, not a canonical main/test result: the prespecified source gate failed before Pets/Caltech progression. Candidate idea-452b63b3 is rejected for promotion. Keep previous baseline, main results and paper checkpoint unchanged. This run may be cited only as a separate development/transfer diagnostic. Next decision: bounded value check for direct task-risk learning versus closure; no automatic Caltech acquisition and no same-family calibration retry.
