# R4 interpolation geometry experiment

Idea: isolate neighborhood geometry from final-score centering. H0: centered transport does not beat equalalpha raw transport; H1: it improves semantic mAP beyond fixed oldgeometry. Research type: paired method ablation and candidate evaluation, exploratory on seen pools.

Deliverable: protocol, minimal method/runner, validation, ranks+AP+identities, flat9metrics and all13baseline comparisons, aggregate/pairedCI report, durable main run then decision.

Code map: import frozen baseline functions read-only; add raw/centered kernel transport module; add runner using saved identity jobs; numerical and independent rank audit. No adjacent edits.

Minimum: comparable full results; solid: strong gate and attributable samealpha effect; maximum independent data/paper work deferred until solid.

Statistical contract: 5000 paired class-cluster/bridge-seed resamples, fixedgallery; macro and all8cells. Source query/bridge ids exactmatch. Alpha0,.5,1,2,globalDTDdev selection, samebudget forrawcontrol. Primarycannotchangeafterevaluation. Fixedminimumlambda ablation is diagnostic.

Success and abandonment: selected_idea.md gate, including>=0.5pp versus oldcentered and positive intervals versus equalcontrol and centered_residual_2; stop this family if failed, or ifonlyweights explain gain. No targetdomain-specific tuning.

Resource: noencodings/downloads; CPUfloat64 reused arrays and caches, projectedmemory<4GiB, save only newscore ranks rather than duplicate controls, expect<0.5GiB. Floor10GiB; recover duplicates first.

Exit: completed and rejected by strong gate. DurableRESULT.json/RUN.md,outputs/analysis.json andvalidation_report.json. Nextanchoridea on thisresultline; no repeatunlessnewindependentqualification.
