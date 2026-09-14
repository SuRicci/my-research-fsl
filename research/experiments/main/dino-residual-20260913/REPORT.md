# Fixed residual descriptor qualification

Verdict: reject the fixed recipe. Per-image shared energy was an observed geometric property, not evidence that the shared component should be discarded. Removing it failed every prespecified qualification gate.

| Method | Macro accuracy (%) |
|---|---:|
| parent | 77.374000 |
| residual | 42.225333 |
| residual_fusion | 69.071333 |
| raw_fusion | 75.786000 |
| support_center_fusion | 75.551333 |
| shuffled_fusion | 54.553333 |
| mean_fusion | 75.752000 |

Residual fusion minus parent: -8.302667 pp, descriptive paired95% interval [-8.652016666666666, -7.939333333333335]. DTD -11.845333pp; EuroSAT -4.760000pp. The same-pool raw local, shared-support centering and mean-fusion controls all outperform the residual fusion. Shuffled ownership is worse, which does not rescue qualification.

Across2000taskconditions/1000uniqueepisodes: 6397 repaired query occurrences, 18851 spoiled occurrences. These are repeated query occurrences, not unique images.

## Validation and provenance
Sevenmethods/1050000predictions audited;12real and4synthetic numerical cases; all35bootstrap intervals independently recomputed, maximum discrepancy 7.11e-15pp. Parent, original local and mean-fusion scores and predictions preserved exactly. Fixed tasks0,249,499 checked in eachcell. Class/query/patch permutations and query-batch independence checked. Norm<=1e-12maps to zero; no post-outcome rule changes.
Study161.655s: bash-c587dcc2. Independent statistics: bash-127aa926. Science nodes science-e969b98a/science-6d4d5d9a. Code/protocol/input/outputSHA256 in outputs/code_lock.json and audit.json. Environment torch2.3.0,numpy1.24.4,CPU4threads.

## Evidence boundary and next decision
Source-only auxiliary development; all five canonical Pets metrics unmeasured and unchanged. Caltech remains exposed and closed. This rejects the fixed per-image residual protocol, not every possible local descriptor method, centering method or learned reconstruction. Strong degradation is consistent with removing useful common signal and/or amplifying residual noise; this experiment cannot separate those explanations.
No coefficient sweep, threshold selector or extra encoding follows this failure. The next research question must preserve the observed useful content and challenge a different bottleneck; support-class reconstruction is one deferred alternative, requiring a frozen numerical/regularization contract and comparison against a broader source-trained classification alternative before promotion.

Primary prior-art overlap and limitations: artifacts/idea/residual_descriptor_survey.md. Existing narrow OSLO manuscript receives reference-only provenance, with no new main-text claim.
