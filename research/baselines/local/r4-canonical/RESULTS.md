# Frozen V0 local comparison

New DTD/EuroSAT semantic retrieval scope; canonical local feature variant, not historical CUB/Oxford reproduction. All methods frozen before this run. Values mAP%.

| Dataset | Bridge | Budget | Old | Center | White | V0 | Strongest control | Delta pp | Oracle |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| dtd | cross | 32 | 27.046 | 30.679 | 29.589 | 27.715 | centered_residual_2: 30.890 | -3.175 | 37.976 |
| dtd | cross | 64 | 27.046 | 30.679 | 29.589 | 28.195 | centered_residual_2: 30.918 | -2.723 | 37.976 |
| dtd | matched | 32 | 27.046 | 30.679 | 29.589 | 31.961 | centered_residual_2: 33.766 | -1.805 | 37.976 |
| dtd | matched | 64 | 27.046 | 30.679 | 29.589 | 34.754 | centered_residual_2: 35.381 | -0.626 | 37.976 |
| eurosat | cross | 32 | 54.435 | 57.973 | 52.601 | 55.027 | old_centered: 57.973 | -2.946 | 52.998 |
| eurosat | cross | 64 | 54.435 | 57.973 | 52.601 | 54.884 | old_centered: 57.973 | -3.089 | 52.998 |
| eurosat | matched | 32 | 54.435 | 57.973 | 52.601 | 57.116 | old_centered: 57.973 | -0.856 | 52.998 |
| eurosat | matched | 64 | 54.435 | 57.973 | 52.601 | 56.801 | old_centered: 57.973 | -1.171 | 52.998 |

Primary V0 equal-cell macro: 43.306736%.

Complete13method rows, development values, five seed values and paired intervals: outputs/all_metrics.json. Protocol and source hashes frozen; outputs/validation_report.json audits all60jobs. Queries: DTD240eval+230dev, EuroSAT100eval; five seeds reuse these same query images. AP compares semantic class matches, not instance labels. No gallery-new features or labels enter methods; evaluator-only full-new-gallery reference is not a mathematical upper bound or an allowed baseline. WIP is restricted m-bridge version, not full original training protocol.
