# Final-step counterfactual contract

{
  "parent_run_id": "oslo-gallery-pets-20260913",
  "decision_id": "decision-3accd6b9",
  "route": "analysis-lite",
  "question": "Does changing final gallery mass alone restore performance, or does task-external composition remain limiting?",
  "design": "Fixed final-step posterior weights and embeddings; cross raw/support-matched gallery mass with original/oracle in-episode composition. Original A is existing method. B total gallery mass=shot per class. C remove out-of-episode mass. D remove then normalize total gallery mass to shot. Zero mass falls back to support prototype.",
  "unchanged": "All two-step fitted weights, xi, S+G centering, features, support/query tasks and query independence. Oracle labels used after fit only.",
  "interpretation": "Retrospective finite intervention on final prototype only; no candidate promotion, no honest-inference comparison claim for oracle variants, no claim of global iterative mechanism or proven remedy.",
  "coverage": "All4000tasks/eightcells; same paired bootstrap5000stratifiedbyseed as parent. Compare to original, zero-update,R2,CS_l2. Report allcells.",
  "verification": "Original replay agrees<3e-6; oracle mismatch equals zero-update; balanced nonemptygallery coefficient=.5; mass partition valid.",
  "stop": "Exactly this four-condition diagnostic; record and decide, no sweeps or additional substeps.",
  "resources": "LocalCPU4threads, existing features, <30MiB addedfiles, free>=10GiB; no new worktree/download."
}

Exit:completed and validated. Balanced final-step mass rescues most mismatch harm but fails matched1shot strong controls. Oracle membership improves matched1shot but requires unavailable inference labels. No candidate promotion.
