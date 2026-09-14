# Source coverage changes transfer, but class-adaptive influence fails qualification

This auxiliary experiment compares ordinary versus mixed source-gallery training crossed with scalar versus class-adaptive influence. Uniform top-64 composition, frozen fused features, source labels, source selection and target tasks are controlled. The mixed-adaptive candidate fails its prespecified promotion gate. No canonical Pets or Caltech metric is generated.

## One-shot paired effects

| Source → target | Coverage effect, scalar (pp,95% CI) | Coverage effect, adaptive | Candidate − mixed scalar | Interaction |
|---|---|---|---|---|
| dtd → EuroSAT | +0.1080 [+0.0276, +0.1907] | -0.0289 [-0.1520, +0.0938] | -0.0920 [-0.2578, +0.0720] | -0.1369 [-0.2782, +0.0040] |
| eurosat → DTD | +0.3684 [+0.2018, +0.5391] | +1.7773 [+1.5747, +1.9818] | -0.3836 [-0.5498, -0.2195] | +1.4089 [+1.1858, +1.6262] |

Ordinary→mixed source training improves scalar influence by0.1080pp for DTD→EuroSAT and0.3684pp in reverse. For the adaptive model, coverage changes the two directional macros by−0.0289pp and+1.7773pp. However, the mixed-adaptive model remains−0.0920pp and−0.3836pp below the corresponding mixed scalar. Its reverse-direction deltas are negative in all three training allocations. Improved recovery from an underperforming adaptive model is not superiority over a strong simple control.

## Full accuracy surface

| Cell | Ordinary scalar | Ordinary adaptive | Mixed scalar | Mixed adaptive | R2 | CS_l2 |
|---|---|---|---|---|---|---|
| dtd_to_eurosat_dtd_k1 | 66.1724 | 67.6507 | 66.9600 | 68.0018 | 66.5493 | 68.3253 |
| dtd_to_eurosat_dtd_k5 | 84.8311 | 83.9973 | 84.8516 | 83.8924 | 87.1360 | 89.6293 |
| dtd_to_eurosat_eurosat_k1 | 74.3671 | 72.9787 | 73.7956 | 72.5698 | 74.1227 | 73.9227 |
| dtd_to_eurosat_eurosat_k5 | 86.5076 | 86.9111 | 86.4880 | 86.6924 | 87.1360 | 89.6293 |
| eurosat_to_dtd_dtd_k1 | 78.7182 | 78.6524 | 77.9342 | 78.1111 | 78.4160 | 78.5173 |
| eurosat_to_dtd_dtd_k5 | 89.9413 | 89.9404 | 89.8516 | 89.7920 | 90.2720 | 90.1013 |
| eurosat_to_dtd_eurosat_k1 | 73.7458 | 70.2267 | 75.2667 | 74.3227 | 74.7547 | 74.4533 |
| eurosat_to_dtd_eurosat_k5 | 89.3440 | 89.3084 | 89.5787 | 89.5609 | 90.2720 | 90.1013 |

## Scope and uncertainty

Eight target conditions contain4,000task-condition outcomes from2,000sampled tasks paired across gallery conditions. Forty-eight trained models arise from four arms × two source domains × two shots × three source-training allocations. The headline metric is the mean accuracy across these allocations, not a prediction ensemble. Three additional globalized-alpha controls per cell preserve each task's average influence while removing its class-specific allocation. Candidate minus globalized-alpha is positive in DTD→EuroSAT but does not have a positive lower interval endpoint in the reverse direction. This intervention does not establish a general advantage over scalar training.

All arms receive identical100source training tasks and100identity-disjoint selection tasks per source/shot/allocation. Selection is a common50:50mixture for every arm, so the training factor is conditional on stress-aware selection. Every source gallery contains1024distinct images; half of the mixed condition excludes all task classes. The ordinary condition samples the original gallery. The evaluation retains original full galleries, a shared size shift. Source5shot uses13queries/class and evaluation15; source1shot uses15. No inference gallery labels, target query-batch fitting, class text or encoder training.

Nominal95% intervals use5,000paired task bootstrap replicates stratified by the five evaluation seeds. Accuracy is averaged across the three fixed training allocations before bootstrapping. These are conditional fixed-pool intervals, not random-domain or training-seed-population intervals; multiple contrast intervals are descriptive and unadjusted. Source domains have already been used for development. Source class exclusion is a controlled stressor, not a faithful simulation of every natural domain shift.

## Reproducibility and validation

Command: /opt/anaconda3/envs/torch/bin/python experiments/main/source-coverage-factorial-20260913/run.py --phase run. Managed run: bash-4aa58cfb; real execution completed successfully. Config, source hashes, asset manifest, per-allocation models, checkpoint histories, all source gallery/task indices and target scores are stored in outputs/. Target simple controls reuse the preceding immutable score files only after hashes, task identity and first-task numerical reconstruction are checked.

Independent audit bash-5eb6c48e passed. It verified all task labels, all source gallery exclusions, source-only checkpoint selection statistics, every score/prediction/accuracy mapping and immutable control reuse. Independent NumPy reconstruction covered24first/central/last target tasks and360model-task predictions; maximum score error9.28e-7. Query partition invariance passed. The preliminary implementation check verified scalar nesting in R2, finite-difference gradients, class/gallery permutation and query partition invariance.

## Interpretation and handoff

The controlled data change helps a scalar model modestly in both directions and reduces some adaptive-model failure, but does not establish the selected robust-improvement hypothesis. Reject further tuning of this fixed adaptive-influence protocol. Retain the coverage result as auxiliary development evidence. Next compare genuinely additional image information with its encoding-budget-matched simple controls; do not infer representation insufficiency merely from these negative results. The older paper checkpoint remains separate and these results need mapping before a later writing pass.
