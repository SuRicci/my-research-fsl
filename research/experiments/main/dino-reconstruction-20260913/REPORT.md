# Fixed normalized reconstruction: audited negative development result

The fixed reconstruction fusion reaches75.323333% compared with77.374% for the retained cumulative parent, a difference of-2.050667percentage points(95%paired interval[-2.286000,-1.807983]). It fails all five predeclared gate components. Retain the cumulative parent; reject this fixed normalized-descriptor transfer.

| Query/gallery | Parent% | Reconstruction fusion% | Rank1 fusion% | Raw local fusion% | Delta-parent pp [95%CI] |
|---|---:|---:|---:|---:|---|
|dtd_dtd|80.010667|78.213333|78.034667|78.570667|-1.797333[-2.141333,-1.461333]|
|dtd_eurosat|76.304000|76.418667|76.224000|76.525333|0.114667[-0.229333,0.464000]|
|eurosat_dtd|73.485333|71.168000|71.400000|71.754667|-2.317333[-2.741333,-1.906600]|
|eurosat_eurosat|79.696000|75.493333|75.605333|76.293333|-4.202667[-4.648000,-3.778667]|

The fused candidate repairs4389query occurrences and spoils7465; these are repeated occurrences within1000unique episodes, not independent images.

The domain-average changes are-0.841333pp onDTD and-3.260000pp onEuroSAT. Full-span minusrank1 fusion is+0.007333pp[-0.110017,0.125333] pooled, while the gain changes sign across domains. This provides no reliable overall advantage from the extra support-map directions. It is not proof that every non-mean component is uninformative. Reconstruction fusion is also below raw-local fusion by0.462667pp and below mean fusion by0.428667pp. The shuffled association control is strongly worse but does not rescue the candidate gate.

## Fixed comparison and evidence boundary
Same source assets and500five-way1shot tasks per domain,fiveseeds,75queries/classification task,and two paired gallery conditions;2000taskconditions/1000unique episodes. Parent,raw local fusion andmean fusion scores and predictions are bit-for-bit inherited. No new query/gallery identities,query-batch adaptation,gallery labels,encoder training,target fitting or selection. Canonical fivePetsmetrics are unchanged and unmeasured; this is an auxiliary/dev result, not a canonical main-test submission. Both source domains and pools were repeatedly exposed. All intervals are descriptive and conditional on these fixed pools; galleries are averaged within episode before domain/seed-stratified5000replicate bootstrap(seed26091399). No external-domain,novelty,SOTA or paper-readiness claim.

## Mechanism and prior work
Fixed known-method transfer from Wertheimer,Tang,Hariharan,Feature Map Reconstruction Networks,CVPR2021(arXiv2012.01506):https://arxiv.org/html/2012.01506v2 and official source https://github.com/Tsingularity/FRN/blob/main/models/FRN.py . Reuse the existing normalized4x4DINO descriptors,renormalize in float64,and reconstruct each query map from each class support map withlambda=16/384+1e-6,rho=1. Negative mean squared residual is the local score,equally fused after per-query standardization with the parent. Rank1 support control repeats the normalized support-map mean16times, preserving count,norm andlambda while keeping the query map. These cached vectors do not retain raw pre-normalization magnitude. Unlike FRN,the encoder/alpha/beta were not trained for reconstruction. Failure applies to this fixed transplant,not to trained FRN or learned source calibration.

## Verification and resources
Formal compute161.201s. Torch dual solve and independentNumPySVD agree on12first/middle/last real conditions and4synthetic cases; max local score error4.44e-16, max fusion error3.24e-14, tolerance1e-10unchanged. Zero,constant,rank1 maps and query/class/patch permutations plus one-query equivalence passed. Independent complete-output audit verified1050000prediction entries and35intervals(max discrepancy7.11e-15pp). Input anddesign hashes unchanged.
Existingtorch2.3.0/NumPy1.24.4,CPU4threads; no new downloads/encoding/extra spend. New output size38.73MiB; free disk10.103GiB. The uniqueCaltecharchive was inspected and retained. Actual design freeze is2026-09-13T14:17:26.555385+00:00; inheritedprotocol.frozen_at is explicitly superseded byPROVENANCE_NOTE.json,withoutchanging the locked protocol. Invalid science status submissions(succeeded,pending) were rejected before recording and corrected to allowed states; they did not alter computations.

## Decision implication
Close this fixed local reconstruction recipe withoutlambda,rho,fusion-weight ordescriptor-pooling rescue sweeps. Nearest matching,demeaning andfixed reconstruction all failed to add reliable value under current cached-map permissions. Do not generalize this to all spatial representation learning. Next prioritize the known numerical parity blocker in saved comparator evidence: define one deterministic high-precision repair and budget contract before rerunning any affected evaluation. This improves evidence trust; it is not a promised accuracy gain or an accepted new algorithm. Original failed audits and untouched historical results remain immutable.
