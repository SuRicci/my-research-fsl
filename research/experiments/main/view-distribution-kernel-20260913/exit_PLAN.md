# Active node — view distribution kernel source qualification

Selected idea-900e7c72, decision-0c5b874b; semantic idea/view-distribution-kernel-20260913 -> run/view-distribution-kernel-20260913 uses current physical worktree to preserve disk floor. Parent run/representation-scatter-20260913 is retained. Candidate-first submission is an explicit resource exception, not an unselected implementation.

Objective/question: do six-view distribution embeddings add transferable class information beyond an equally tuned nonlinear classifier on the mean?
H0: distribution kernel does not meet both-source improvement gates. H1: at least0.5pp1shotmacro gain with positive paired lower bounds against every fixed control and all cells noninferior at -0.5pp.
Research type/tier: controlled auxiliary/development study; reused exposed sources. No new canonical Pets metric. Minimum: valid frozen comparisons. Solid: all gates and numerical checks pass. Maximum: reserved confirmation and canonical target variant only after a separate decision.

Code map: experiments/main/view-distribution-kernel-20260913/view_kernel.py (finite-view kernel + centered head), run_kernel.py (reuse fixed loaders/tasks and frozen ten-control banks), verify_kernel.py (independent NumPy formula, intercept-constrained solve, invariants). Previous code/results read-only. No encoding/download; <=0.2GiB results, >=10GiB free, local CPU, zero spend, deadline2026-09-15T09:00Z.
Efficiencies: reuse exact cached data/old comparator outputs with hashes and task identity verification; compute support-only heads once per task, duplicate only gallery-conditioned comparisons; compute each query independently, no query-query matrix.

Design: source dtd/eurosat; shots1/5;100selection episodes and500evaluation episodes per domain/shot; exact old splits/seeds. Candidate normalized all-pairs RBF kernel, same-budget mean-RBF, original-RBF, linear mean ridge; three beta1/4/16 × lambda1/0.1/0.01; exact integer accuracy selection, tie order smaller beta then larger lambda. Linear selects same lambdas. Every source choice locked before reported eval outcomes. Historical ten controls are fixed and not retuned.

Validation: new independent NumPy float64 kernel plus augmented intercept linear-system solver, max score tolerance1e-8; query chunk/reorder invariance, repeated-view limit, linear-kernel collapse and PSD. Full bank identity/prediction/accuracy checks. Audit five fixed eval tasks per domain/shot and three selection tasks per domain/shot for all new methods. Previous scatter strict audit remains failed1/88, never overwritten. A passing source metric gate is only a candidate signal until inherited comparison risk and every new audit pass are resolved.

Uncertainty:5000paired bootstrap draws stratified by seed, same resampling per method, fixed-pool conditional; the identical support-only results across gallery conditions are not independent repetitions.
Abandonment: failed metric gate, unresolved numerical error, disk/resource/deadline violation; no grid widening based on evaluation.
Deliverables: frozen protocol/hash manifest; selection locks; per-task scores; independent audit; aggregate report with comparison boundaries; decision. Next: validation -> real measured run -> durable auxiliary report and decision. Existing paper is parked; remap new evidence before any writing.

Exit: computation and independent audits completed; fixed source metric gate refuted. Resume next idea pass, not the old kernel run.
