# Existing-fit influence audit

{
  "parent_run_id": "oslo-gallery-pets-20260913",
  "parent_artifact_id": "run-2dac3539",
  "decision_id": "decision-5094098f",
  "route": "analysis-lite",
  "question": "How much of the final frozen OSLO_G prototype update comes from out-of-episode gallery samples versus correct-class evidence and labeled support?",
  "intervention": "none; deterministic instrumentation of existing two-step fits",
  "scopes": [
    "canonical",
    "fresh"
  ],
  "shots": [
    1,
    5
  ],
  "gallery_conditions": [
    "pets",
    "dtd"
  ],
  "fixed": "All original tasks, features, source parameters, centering and update rules; no refit variant or target selection. Diagnostic labels are used only after fit.",
  "observables": [
    "effective gallery mass per class",
    "gallery fraction of prototype averaging coefficient",
    "out-of-episode fraction of retained gallery weight",
    "correct-class and wrong-in-episode weight partition",
    "xi in/out separation"
  ],
  "validation": "All traced predictions must agree within3e-6 with frozen saved scores; mass components sum to total, all4000tasks covered.",
  "boundary": "Descriptive update attribution; no causal remedy or calibrated probability claim. DTD category universe treated as outside Pets target labels.",
  "resources": "LocalCPU4threads, existing features, <=30MiB extra outputs, no new worktree/download, free>=10GiB.",
  "next": "Record trace audit; decision on whether a separate intervention is warranted. Do not reopen OSLO source-default parameter grid.",
  "created_at": "2026-09-12T22:27:11.781633+00:00"
}

Exit:success. Complete descriptive attribution with exact prediction parity; this does not validate a remedy. No additional slice committed.
