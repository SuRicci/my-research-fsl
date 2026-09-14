# Source-group objective screen: verified negative qualification

The fixed reference-relative worst-source objective does not qualify for further target evaluation. It improves over matched mean-risk training, but fails to reliably beat the untrained reference and is substantially below the retained strong multi-view stack. This rejects this fixed protocol, not every distributionally robust objective.

## Primary result
| Direction | Candidate accuracy % | Minus identity pp [95%CI] | Minus matched mean pp [95%CI] | Minus raw max pp [95%CI] | Minus strong stack pp [95%CI] |
|---|---:|---|---|---|---|
|dtd_to_eurosat|69.0453|+0.0827 [-0.0693, +0.2382]|+0.7449 [+0.5689, +0.9280]|+0.0827 [-0.0693, +0.2382]|-7.5453 [-8.1049, -7.0200]|
|eurosat_to_dtd|74.3973|-1.0213 [-1.2951, -0.7555]|+0.3111 [+0.0471, +0.5849]|-1.0213 [-1.2951, -0.7555]|-3.7600 [-4.1462, -3.3858]|

## All prespecified arms
| Objective / selector | DTD to EuroSAT % | EuroSAT to DTD % |
|---|---:|---:|
|mean/pooled_ce|66.5244|64.9333|
|mean/reference_max_ce|68.3004|74.0862|
|raw_max/pooled_ce|66.5378|65.5876|
|raw_max/reference_max_ce|68.9627|75.4187|
|reference_max/pooled_ce|68.6400|72.3244|
|reference_max/reference_max_ce|69.0453|74.3973|

Every raw-max fit selects step0 under reference-relative validation, exactly recovering identity. The reference-max candidate selects step50 in all six primary configurations but does not establish transfer safety. Changing the selector can remove much of the damage from training without creating an improvement over identity. These are matched empirical interventions; they do not identify a universal causal mechanism for domain shift.

## Protocol and interpretation
18fits: two primary source domains, three paired initializations, three objectives. Each fit receives the exact same100source tasks per full-bank update,200updates and common0/50/100/200checkpoint opportunities; two selectors choose independently from the same history. Original-source and Flowers data/splits remain exactly those in the mixed-source study. The encoder,896dimfusion,rank32map,Adamsettings and ridge head are unchanged. Full-bank exposure is20000task presentations perfit versus1000in the historical minibatch study. Therefore the new-versus-old study comparison is descriptive; objective and selector effects are identified only among the current matched arms.

One-shot qualification uses1000unique historical development tasks,500perdirection,5taskseeds. Gallery comparison conditions do not create extra independent tasks. Mean accuracy averages the three fixed trainingseeds, not a logit ensemble. Intervals use5000pairedtask resamples stratified bytaskseed and are conditional on these pools and trainingseeds. They are not uncertainty over newdomains or a random population of trainingseeds. Strong stack uses six views and gallery access, while the current learned candidate has one view and no gallery. It is a required practical comparison, not a same-input causal control; identity and mean/rawmax isolate same-input gains. Existing oneviewR2/CS_l2/selected-linear references are also reported in outputs/analysis.json. Canonical five Pets metrics remain unchanged and unmeasured. No Pets or Caltech outcome was generated.

The primary gate requires each direction to improve>=0.5pp with positive pairedCI over identity, matchedmean/rawmax under the same selector, and strongstack; at least2/3trainingseeds must improve against the matched objectives. Both directions fail. Neither posthoc selector changes nor targetparameter tuning are allowed to rescue the gate.

## Verification and execution
Numerical precheck bash-2ce21d0f: independent double-precision ridge error8.88e-16; maximum finite-difference objective-gradient error2.78e-12; identity null and pooled-constant gradient equality exact; source bank identities,labels and train/selection separation verified.
Formal run bash-1da1787d completed all18fits and both prediction banks, then failed because historical oneviewcontrol cells store tasks separately. Summary-only recovery bash-43548293 corrected the task-bank link, preserved hashes of all76existing outputs, and did not retrain or regenerate predictions. Original code,lock andmanifest retained alongside summary_repair.json. The protocol/selection/gate did not change.
Independent audit science-2978ee2a checked36model-selector configurations,1350000predictions,180NumPy ridge reconstructions(maxerror6.9061e-7),8primary paired intervals,checkpoint ranks andinputlocks. Other reported intervals use the same checked estimator but were not each independently resampled. The result cannot be reported as novel MMR or a transfer guarantee: MMR subtracts group-optimal risk, while this experiment subtracts an identity reference. Literature derivation and alternatives are in artifacts/idea/source_group_objective_20260914.md.
A precheck import collision incidentally reran a legacy validation script and refreshed its validation.json. That file was untracked and no pre-import hash was captured, so historical byte equality is not established. Its script reads historical data/models/predictions and only rewrites validation.json; it reported passed. The import was corrected to an explicit path. This limitation is recorded in legacy_import_audit.json; do not interpret the refreshed timestamp as the original validation time.

## Handoff
Reject promotion of the fixed source-group objective; preserve the local improvement over failed mean-risk training and the independent identity counterexample. Do not reopen a group-weight,learningrate,checkpoint or selector grid using these outcomes. Retain the conditional query-consistency/scatter incumbent. Next compare the remaining real-view training objective against existing failures and the already-testedR4 line; require a concrete differentiating observable before another run. Existing narrowOSLOpaper remains separate; map this auxiliary history asreference-only, not support for itsC1-C4 claims. No quest-completion or submission-readiness claim.

Resource record: 194.85s summed fit walltime; 2.49s summary recovery. Current outputs 8.370MiB; current free 10.017GiB. Local CPU6threads,zero spend,no newencoding/download. Deadline2026-09-15T09:00Z. Verified chunk cleanup reclaimed25955508bytes; retained feature hashes checked before and after deletion.
