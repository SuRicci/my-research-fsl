# Conditional Laplace comparison — practical gain rejected
One fixed conventional posterior-predictive recipe was tested on all1,000 exposed development tasks (75,000queries), preserving six-view joint geometry, C10, support labels and every comparison metric. No hyperparameter sweep.

| Domain | Original / refined MAP (%) | Posterior (%) | Delta(pp) |
|---|---:|---:|---:|
| DTD |91.010667|91.000000|-0.010667|
| EuroSAT |92.109333|92.085333|-0.024000|
| Equal-domain macro |91.560000|91.542667|-0.017333|

Macro paired conditional95%CI for the point estimate difference is[-0.042667,+0.008000]pp: no demonstrated accuracy improvement or statistically resolved harm. Numerical refinement changed0/75,000MAP predictions. Posterior averaging changed112decisions, yielding13fewer correct overall. This closes only the fixed recipe, not all Bayesian methods.

## Numerical uncertainty and gate
The preregistered criterion was>=+0.3pp macro, positive paired conditional95%lower bound and each-domain>=-0.2pp against both original and refined MAP. It failed.203queries retain the conservative near-tie/replicate-discrepancy flag at the maximum integration budget. Assigning every flagged prediction favorably yields macro91.70%(+0.14pp), still below+0.3pp. Unfavorably gives91.429333%. These are bounds over detected flagged queries, not rigorous certification of all quadrature errors. No need to spend more integration budget to rescue a failed practical threshold.

## Audits and provenance
Synthetic dense-autograd covariance comparison passed at7.11e-15 after one fixed same-objective Newton repair; original failed solver-tolerance attempt retained. All1,000original fits match saved logits to1.29e-10 and every reference prediction. Refined gradients<=1.51e-15; maximum logit correction0.000368. Four prespecified real-task autograd Hessians agree to1.34e-15; an extra quadrature scramble changes0/300audit predictions. All task identity arrays,40shards versus final files,integer-correct counts and locked hashes verified.
Scientific run: bash-3f5124b0,223.439s. First aggregator failed on shared75-label storage; broadcast-only repair recorded, no rerun/prediction change. Read protocol.json, execution_lock.json, validation.json, statistics_audit.json, summary.json and final dtd/eurosat.npz.

## Interpretation and next edge
The incumbent remains91.56%. Laplace prediction is established prior art, and this fixed version gives no practically useful gain here. Support-null uncertainty was retained, intercept and softmax gauge handled explicitly; this result cannot be attributed to dropping those terms. Do not tune prior strength, posterior temperature, or uncertainty truncation to chase these exposed tasks.
Next stage: idea/decision frontier reassessment from the new negative evidence. Do not silently promote the deferred kernel recipe without a new discriminating rationale. Full-paper goal remains open and independent-image evidence unavailable under the current cached-data permissions. No manuscript-readiness claim.

## Workflow recovery
MCP mutations retained the original turn workspace while disk/runtime refs activated this dedicated run. Source/results are committed on run/conditional-laplace-20260916. Official ArtifactService invoked via bash_exec resolves this workspace correctly; registration uses that same canonical implementation with strict metric validation, without hand editing daemon state.
