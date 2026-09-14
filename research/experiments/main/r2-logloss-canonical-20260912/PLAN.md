# Active run contract
Idea idea-e1645299; hypothesis and challenge: artifacts/idea/pre_idea_drafts/loss_sensitive_neighborhood.md.
Null: changing loss does not make full retrieved neighborhoods better than R2 and support/mean-only classifiers. Alternative: positive paired improvement plus loss-by-neighborhood interaction.
Baseline accepted r2-canonical-local/mps32-v1; seven required metrics unchanged, scope dtd-eurosat-canonical-locked-20260912. Same images/tasks/feature SHA, query75per5waytask,5seeds200each; DTDdev classes disjoint, same four1shot cells/two5shot safeguards. Repeated fixed-pool comparison labeled developmental. Five-shot preserves R2 by design.
Minimal changes: snapshot validated reference_eval.py and baseline_methods.py; new evaluate.py adds established weighted multinomial sklearn lbfgs; package comparison and validation. No source repo changes.
Candidate: real supports plus original16 rank-selected top64neighbors, pseudo totalweight1/class; CE sum plus lambda||W||²/2, unpenalized intercept. No residual recentering.
Controls: fixed/tuned R2, prototype, support ridge/logistic, mean ridge/logistic, distributionridge. All tunable heads use same DTDdev grid[.001,.01,.1,1], ties0.1,1,.01,.001. No eval retuning.
Minimum: solver weights, raw predictions/tasks, baseline parity and finite metrics verified. Solid: positive paired macro CI vs fixed/tunedR2, no cellloss>0.5pp, beyond support/mean-only. Maximum: independent expansion only if solid.
Validation budget1 synthetic+realdevbatch pass resolves solver invariance and runtime. ConvergenceWarnings invalidate run and require solver-only fix; parameter changes cannot use evaluation accuracy. Four solver workers, BLAS1, cachedfeatures; no MPS competition or downloads.
Protocol/code snapshots and command argv persisted. Stop at72hcap or nonfinite/leakage/resource issue. Result is not complete until main artifact and decision. Negative result routes to a differentfamily/claim boundary, not more blind tuning.
