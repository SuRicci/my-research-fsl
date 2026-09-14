# Active experiment: source-group objective qualification
Idea idea-3a020710; decision-955f5a64; branch run/source-group-objective-20260914.
Question: does relative worst-group source loss improve held-out classification beyond identity, matched mean/raw-max training and retained strong stack? Null: no useful >=0.5pp gain; alternative requires paired positive CIs and both source directions.
Read artifacts/idea/source_group_objective_20260914.md for complete claim, literature and selection contract.
Tier auxiliary/dev. Baseline Pets metrics remain unchanged and unmeasured; use report/science rather than canonical main metric submission.
Code-change map: add one study.py reusing immutable source-composition task/model helpers; add one independent audit.py; no existing baseline or model changes.
Inputs: exact existing mixed/selection source tasks, canonical singleview fusion, 3paired seeds. Three full-bank objectives x2selectors. Every fit gets100tasks/update,200updates, same optimizer and0/50/100/200selection opportunities. Choices freeze before evaluation.
Minimum: numeric objective/gradient, independent ridge and split invariants pass. Solid:18fits/all6arms/1000unique evaluation tasks, input/choice locks and independent validation, paired statistical verdict. Maximum only if gate passes: later5shot/multiview/canonical qualification, not authorized by a surrogate gain.
Efficiency: cached features, batched exact group losses,6CPUthreads; no encoding; save only selected states, compact predictions and selected audit scores. Added output <=20MiB, free>=10GiB and deadline2026-09-15T09:00Z.
Success: primary refmax/refmaxce >=0.5pp and positive task-CI vs identity/mean/rawmax/strong-stack in each direction; >=2/3 positive training-seed effects vs matched objectives. Identity recovery or lower loss alone is failure.
Stop/degrade: numerical/input failure->repair before outcomes; fixed gate failure->reject promotion without retuning. Existing tasks are exposed development, no generalization guarantee.
Next: implement->one check->detached real run->audit->report/science->decision. No public release or quest completion.

## Verified exit
Completed and audited; fixed gate rejected bydecision-6bde6800. Resultreport-54e9c664. No retraining/selectorgrid ontheseoutcomes. Futurecanonicalqualificationnotopened. SeeRESULT.json/REPORT.md.
