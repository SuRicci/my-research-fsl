# Exclusive ownership in frozen-feature retrieval augmentation
## Problem and hypothesis
In image-only inductive one-shot classification, independent top-k retrieval can assign one unlabeled image to several task classes. Canonical frozen R2 then averages that image into incompatible class prototypes. We hypothesize that a single task-relative ownership rule can improve this augmentation without query-set adaptation. The hypothesis concerns information allocation, not encoder misalignment or a new confidence-filtering principle.

## Method
Let normalized class support prototypes be p_c and normalized gallery features g_i. Form each original cosine top64 list N_c. Assign owner(i)=argmax_c p_c^T g_i. Keep K_c={i in N_c: owner(i)=c}; do not refill from more distant images. Define u_c=normalize(mean_{i in K_c}g_i) and a_c=normalize(0.5p_c+0.5u_c). If K_c is empty, a_c=p_c. Fit the existing centered dual ridge with unpenalized intercept on a_c and one-hot class targets. Ties use the smallest class index; audit ties and permutation equivariance away from ties. Queries only pass through the final classifier independently.

Primary variant uses raw canonical fused features and fixed half mixing, changing only gallery ownership. A centered-geometry ownership combination is an explicitly secondary ablation; it cannot replace a failed primary post hoc. Five-shot results inherit raw canonical R2 unchanged and carry no ownership attribution.

## Closest work and contribution scope
Prototype masking for distractive semi-supervised FSL (Ren et al.,2018), task-conditioned clustering (Seo et al.,2020), and iLPC cleaning (Lazarou et al.,2025) establish the family. Our procedure is a deliberately simpler, single-pass intervention on a locked frozen retrieval pipeline. No first-method, new-algorithm or SOTA claim is made. The potential contribution is a measured allocation failure, controlled practical remedy and explicit boundary under mismatched galleries, conditional on results.
Known CS_l2 (Fei et al.,2021) is mandatory: it already reduces shared assignments and improves the local macro. CLIP task-ambiguity discussion (Herzog and Wang,2026) is motivation only, not proof of this mechanism. CoMo/CaFo establish pretrained-model fusion; extra semantic/generative permissions are excluded.

## Fixed evaluation and anti-win conditions
Use accepted dtd-eurosat-canonical-locked-20260912 identities, frozen CLIP/DINO assets, four original one-shot cells and both five-shot cells. Five seeds ×200 paired tasks/cell, 5-way,15 queries/class. Preserve all7 required metrics, headline original macro1shot.
Select ridge lambda solely on original DTD development classes/tasks, fixed grid[0.1,1,0.01,0.001], ties in that order. All relevant readouts receive the same lambda budget. No evaluation labels in adaptation, tuning or filtering; no gallery labels; no semantic names, new models or query-batch statistics.
Primary gate: macro improvement at least0.2pp and paired95% lower bound>0 against both R2 and CS_l2; no one-shot cell point loss worse than0.5pp versus either. Require positive paired macro lower bounds against count-nearest, deterministic random-count, and mass-only controls to support ownership attribution. Report intervals conditional on viewed fixed pools, preserve shared-gallery task pairing, retain every cell.
Strong controls: R2 fixed and dev-tuned; CS_l2; support-only ridge and prototype. Allocation controls: keep closest same count; keep deterministic random same count; original mean with mixing reduced to0.5|K_c|/64. Secondary filter-plus-mass and centered-filter variants are ablations, not new selected candidates. Save ownership counts, collision statistics, scores, predictions, per-task accuracies and identities.
If any gate fails, downgrade to a negative/incremental diagnostic and do not tune thresholds or refill rules on these evaluation pools. If successful, freeze method and test the previously acquired alternate image pools plus stronger gallery-only cleaning comparison and existing additional domains where assets permit. All pools are already viewed; no fresh-holdout claim.

## Feasibility and handoff
Reuse geometry evaluator assets/tasks/ridge implementation. Ownership uses the same5×gallery similarity matrix and top64 budget; extra work is argmax/masking. CPU batching, existing normalized features, no downloads/encoding. Small deterministic validation must check reference endpoint, empty retention fallback, no-query coupling and mask counts; at most one smoke unless patched. Actual durations are to be measured.
Implementation belongs in a new artifact-managed run child after this idea line is accepted. Current evidence foundation is run/r2-prevalidation-canonical-20260912 because it contains the completed comparison campaign lineage; it does not make PreVal the scientific incumbent.

## References
Ren,M. et al. Meta-Learning for Semi-Supervised Few-Shot Classification. ICLR(2018). arXiv:1803.00676.
Seo,J.;Yoon,S.W.;Moon,J. Task-Adaptive Clustering for Semi-Supervised Few-Shot Classification(2020). arXiv:2003.08221.
Lazarou,M.;Stathaki,T.;Avrithis,Y. Exploiting unlabeled data in few-shot learning with manifold similarity and label cleaning. Pattern Recognition161,111304(2025).
Fei,N.;Gao,Y.;Lu,Z.;Xiang,T. Z-Score Normalization, Hubness, and Few-Shot Learning. ICCV142–151(2021).
Herzog,J.;Wang,Y. Reevaluating the Intra-Modal Misalignment Hypothesis in CLIP(2026). arXiv:2603.16100.
Zhang,R. et al. Collaboration of Pre-trained Models Makes Better Few-shot Learner(2022). arXiv:2209.12255.
Zhang,R. et al. Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners. CVPR(2023). arXiv:2303.02151.
