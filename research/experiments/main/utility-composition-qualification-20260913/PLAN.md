# Utility composition qualification

Hypothesis: query-risk optimized relative weights transfer better than membership BCE and independently optimized scalar mixing. Null: no composition benefit or domain harm. Auxiliary exposed-development experiment, not canonical Pets main/test.

Minimal change map: utility.py holds neighbor features and differentiable fit; run.py reuses prior qualify.controls, ref.assets and bootstrap. Existing baseline sources remain read-only. Scope: two transfer directions × two gallery types × shots1/5,500pairedtasks each. Source100training/100selection episodes per shot, identity-disjoint folds. Fullbatch deterministic training80updates; checkpoints20/40/80selected only by source accuracy then CE. Same-label access controls.

Success: both1shot macro gains>=.5pp and95%lower>0 over every named strong/control objective; all-cell95%lower>=−.5pp. Required composition gain beyond same-alpha and independently trained scalar. Negative outcomes and all method cells preserved.

Validation: finite difference versus autograd, direct primal-versus-dual ridge, uniform-weight R2 k1 parity, query splitting and class/gallery permutation equivariance, source image disjointness, feature manifests, full counts, score→prediction→accuracy consistency.

Budget: reused local torch interpreter,CPU6threads; <.3GiB disk and>=10GiB free. First real run after one implementation check; no target grid. Failure routes to decision; success freezes prospective target protocol before new target use.

Source5shot feasibility amendment:13queries/class in train/selection, common to both domains and controls. Both existing1shot cells/model preserved by SHA256. Original source/protocol snapshot in pre_repair/. Target remains15queries/class.
