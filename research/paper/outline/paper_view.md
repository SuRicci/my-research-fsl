# When Inlier Ranking Fails to Control Gallery Influence in Frozen Few-Shot Classification

- Paper type: `technical_note`
- Outline maturity: `early`

## One-Sentence Paper Idea

- Central thesis: In a controlled gallery-only transfer of open-set prototype fitting, good inlier ranking can coexist with harmful accumulated weight from task-external samples.
- What readers learn: The aggregate mass and composition retained by a prototype update must be examined separately from ranking quality.

## Story Spine

- Problem: Independent unlabeled galleries can contain many classes outside a few-shot episode.
- Gap: Open-set weighting can rank relevant samples well without ensuring that the total retained contribution of irrelevant samples is small.
- Method: Evaluate a gallery-only transfer of established open-set prototype fitting, decompose its final averaging coefficients, and intervene separately on gallery mass and class composition.
- Main result: The source-default transfer underperforms strong local controls in two image-disjoint Pets pools; high inlier ranking quality coexists with dominant task-external weight.
- Scope limit: One target dataset, one frozen feature fusion, two gallery sources and fixed source defaults. The interventions are retrospective and oracle composition uses unavailable labels.

## Positioning

- closest_neighbor: OSLO open-set transductive few-shot learning; local retrieval-weighted and support-centered classifiers.
- novelty_boundary: A scoped empirical failure analysis of a restricted gallery-only transfer. The original objective, inlier weighting, prototype fitting and centering are established components.
- why_not_prior_work: Not recorded
- not_claiming: A new learning algorithm or state-of-the-art performance, A failure of original transductive OSLO, Independent-domain generalization, A deployable oracle filter or a theoretical accuracy ceiling

## Core Claims

- `C1` With fixed source defaults, gallery-only open-set fitting does not improve over R2 in the evaluated Pets conditions.
- `C2` Strong inlier ranking coexists with substantial total task-external gallery weight in the evaluated matched-gallery transfer.
- `C3` Final-step mass balancing alleviates mismatch degradation but does not close the matched one-shot gap; privileged composition removal gives a separate diagnostic gain.
- `C4` A fixed source-trained membership calibrator with prior correction fails the two-direction development-transfer gate and yields no prediction benefit over a same-mass control.

## From Facts To Interpretation

- `Observed performance -> scope` Source-default transfer fails a strong-control qualification even though it improves on closed-set fitting.
- `Observed weights -> interpretation` Ranking separation and total contaminant weight answer different questions.
- `Intervention -> bounded explanation` Changing final mass reduces mismatch harm while class composition retains explanatory value.

## Evidence Boundaries

- Observed facts: OSLO_G is below R2 in all eight cells, including paired negative matched one-shot effects in both pools., Mean matched one-shot inlier AUROC is about 0.93 but about 80% of gallery weight remains on task-external classes., Mass balancing largely rescues mismatch but still loses matched one-shot; privileged composition filtering improves matched one-shot., Separate source calibration failed its bidirectional gate and matched the same-mass control at prediction level.
- Allowed interpretations: Total retained mass can remain harmful despite favorable inlier ranking., Mass and composition have distinguishable effects in this constrained final-step diagnostic.
- Do not claim: New algorithm, Universal superiority or robustness, Original OSLO benchmark reproduction, Independent-domain confirmation, Deployable oracle membership, Causal directional importance from scalar averaging coefficients
- Evidence gaps: No prospective admissible remedy, No second target dataset for OSLO transfer, No near-duplicate or pretraining-overlap audit, No independent sampling uncertainty over domains, No broad hyperparameter or gallery-size sensitivity

## Method

- Paper name: Gallery-only OSLO transfer
- Intuition: Adapt class prototypes using labeled support and an independent unlabeled gallery, while keeping test queries outside fitting.
- Step: Initialize prototypes from labeled support in frozen fused feature space.
- Step: Alternate established soft class assignments, latent inlier weights and prototype updates using support and gallery only.
- Step: Classify each held-out query independently with fixed fitted prototypes.
- Step: Inspect final weight mass and apply retrospective mass or oracle-composition interventions for explanation.

## Evaluation

- Setting: Five-way inductive classification with independent unlabeled gallery adaptation and frozen encoders.
- datasets_or_benchmarks: Oxford-IIIT Pets original fixed image pool, Oxford-IIIT Pets prospective image-disjoint confirmation pool, DTD as a mismatched gallery
- baselines: R2 local retrieval-weighted classifier, CS_l2 support-centered control, Support-only multinomial logistic regression with fixed C1 and C10, Closed-set prototype fitting, Zero prototype update
- metrics: Classification accuracy, Paired task-bootstrap accuracy difference and 95% interval, Final gallery averaging coefficient, Task-external fraction of retained gallery weight, Inlier ranking AUROC
- controlled_factors: Frozen CLIP and DINO features with fixed fusion, Identical support/query task identities and seeds, No query adaptation or gallery-label use in eligible methods, Source defaults frozen before confirmation feature extraction, Intervals conditional on fixed pools, not domain sampling

## Analysis Plan

- `A1` Strong controls and source ablations (stronger-baseline comparison)
- `A2` Image-disjoint confirmation and condition breakdown (robustness or sensitivity)
- `A3` Ranking versus accumulated weight (mechanism or attribution check)
- `A4` Mass versus composition intervention (limitation or residual headroom analysis)
- `A5` Source calibration under domain shift (limitation or residual headroom analysis)

## Reviewer Objections

- The task differs from original OSLO. -> limitation
- The same target dataset and exposed diagnostic pool restrict generalization. -> claim_downgrade
- High ranking quality says nothing about calibrated probabilities or causal influence. -> writing
- Oracle labels and post hoc interventions cannot establish an admissible improved method. -> limitation
