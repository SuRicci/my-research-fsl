# Augmentation covariance as predictive information — fixed development result

The fixed package failed its predeclared promotion gate. These exposed development tasks cannot establish new-image generalization.

| Kernel | DTD logistic | EuroSAT logistic | Macro logistic | Macro ridge |
|---|---:|---:|---:|---:|
| raw_mean | 90.8693 | 89.4267 | 90.1480 | 89.6613 |
| corrected_mean | 91.0107 | 92.1093 | 91.5600 | 91.0960 |
| corrected_half_mean | 91.0027 | 91.8907 | 91.4467 | 90.7520 |
| corrected_cov | 90.8427 | 89.5520 | 90.1973 | 89.6400 |
| corrected_poly | 90.8240 | 92.0933 | 91.4587 | 91.1960 |
| corrected_perm0 | 90.7760 | 88.3280 | 89.5520 | 88.9880 |
| corrected_perm1 | 90.8987 | 88.6960 | 89.7973 | 89.1373 |
| corrected_perm2 | 90.8613 | 88.5893 | 89.7253 | 89.0853 |
| covariance_only | 59.3227 | 73.3840 | 66.3533 | 66.3280 |
| raw_cov | 90.6373 | 87.0693 | 88.8533 | 88.3413 |

## Primary fixed contrasts
All differences are percentage points. Intervals resample tasks within historical seeds and domains, conditional on exposed image pools.
| Comparator | Macro gain | 95% interval | one-sided 99% lower |
|---|---:|---|---:|
| corrected_mean | -1.3627 | [-1.5093333333333168, -1.2159999999999798] | -1.5347 |
| corrected_half_mean | -1.2493 | [-1.3893333333333544, -1.1106666666666878] | -1.4160 |
| corrected_poly | -1.2613 | [-1.4093333333333504, -1.1106666666666882] | -1.4400 |
| permutation_mean | 0.5058 | [0.3675555555555686, 0.6422222222222164] | 0.3409 |
| raw_mean | 0.0493 | [-0.11603333333336001, 0.21999999999999886] | -0.1467 |

## Evidence and limits
Same 500 tasks per domain, 6 cached views, 25 labeled support images and 75 individual queries. No query-batch fitting, extra labels, or encoder training. Kernel identity is established second-order representation theory, not a novelty claim. Corrected-mean gamma was selected in earlier exposed development.
All task identities and mean-head predictions bridge the audited parent; four predeclared tasks have dense covariance and alternate-head checks. Numerical and statistical audit files record exact bounds. All scores and predictions retained. Initial native-library failure is retained separately; no result selection follows that failure.
Promotion requires >=0.3pp gain over corrected mean, all five Bonferroni-adjusted one-sided bounds positive, and no domain point loss beyond0.3pp. Secondary ridge/raw-covariance performance cannot rescue this decision.
dtd: 270 corrected and 333 introduced errors out of 37500 query occurrences; images recur between tasks.
eurosat: 473 corrected and 1432 introduced errors out of 37500 query occurrences; images recur between tasks.
