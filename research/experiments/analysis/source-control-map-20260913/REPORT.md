# Existing source-control evidence map

Eight saved prediction banks join on exact support/query/class/seed/label identities, covering 2000 gallery-conditioned tasks from 1000 distinct episodes. Saved scores, predictions and accuracy arithmetic agree. No new fitting, evaluation or Caltech parameter choice occurred. Original source numerical gates remain failed; this is a reconstruction of exposed-development evidence.

| Method | DTD (%) | EuroSAT (%) | Macro (%) |
|---|---:|---:|---:|
| original_r2 | 76.585333 | 70.336000 | 73.460667 |
| mean_r2 | 77.952000 | 72.465333 | 75.208667 |
| mean_CS_l2 | 77.813333 | 73.266667 | 75.540000 |
| score_ensemble_r2 | 77.834667 | 72.386667 | 75.110667 |
| augmented_support_ridge | 76.744000 | 72.157333 | 74.450667 |
| mean_logistic_C1 | 76.736000 | 70.232000 | 73.484000 |
| mean_logistic_C10 | 76.621333 | 71.205333 | 73.913333 |
| scatter_blend | 78.077333 | 76.590667 | 77.334000 |
| query_consistency | 78.157333 | 76.590667 | 77.374000 |

## Paired error overlap with support-only ridge

| Target / gallery | Candidate minus support ridge (pp) | Candidate-only correct | Ridge-only correct | Both wrong |
|---|---:|---:|---:|---:|
| dtd_dtd | 3.266667 | 1963 | 738 | 6758 |
| dtd_eurosat | -0.440000 | 1227 | 1392 | 7494 |
| eurosat_dtd | 1.328000 | 1831 | 1333 | 8610 |
| eurosat_eurosat | 7.538667 | 3458 | 631 | 6983 |

Counts are repeated query occurrences within fixed task/gallery conditions, not unique images or independent samples. Full per-control counts and bank hashes are in RESULT.json.

Interpretation: support-only replacement is not uniformly dominant on current source evidence. The candidate wins on both source domains, most strongly on EuroSAT, but loses on Caltech. This frames a transfer-dependence question; it does not establish a deployable selection rule, a causal mechanism, or novelty. Earlier failed competence/support-risk selectors cannot be silently reopened.

Next: focus the prior-work/feasibility audit on whether any previously unused observable or learning constraint can distinguish beneficial representation changes from harmful transfer. Compare numerical-faithfulness repair and evidence synthesis as alternatives. No new selected method exists yet.
