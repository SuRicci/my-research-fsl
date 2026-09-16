# Support-view CE: matched conventional comparator
The fixed corrected-view support objective does not improve the accepted corrected-mean classifier. Macro accuracy is91.545333% versus91.560000%, difference-0.014667pp with conditional paired95%interval[-0.032000,+0.002667]pp. The predeclared>=0.3pp gain gate fails; this closes the fixed recipe without tuning, not all augmentation objectives.

| Training | DTD mean-query % | EuroSAT mean-query % | Macro mean-query % | Macro view-score average % |
|---|---:|---:|---:|---:|
|identity_mean|90.869333|89.426667|90.148000|90.150667|
|identity_views|90.872000|89.776000|90.324000|90.328000|
|joint_mean|91.010667|92.109333|91.560000|91.540000|
|joint_views|91.000000|92.090667|91.545333|91.538667|

## Meaning and limits
Without geometry, the view-trained objective increases macro accuracy by0.176pp, mostly EuroSAT(+0.349333pp). Those factorial contrasts are descriptive secondary outcomes, not a newly selected primary test. Corrected means remain stronger than raw-view training; direct empirical augmentation CE does not explain away their advantage in this finite fixed recipe. This does not prove a universal causal mechanism or theoretical equivalence. Query score averaging is separately reported and was not selected after outcomes.
Each original support image has total weight1; the classifier optimizes sum-image average-view CE plus||W||²/(2C), C10. No extra independent images or labels were added. Same geometry uses fixed DTDgamma1 and EuroSATgamma0.1, unit normalized feature rows, support-only fitting and independent queries. Four fits by two inference modes span all1000frozen five-shot tasks. These pools are exposed development data, not untouched new-image/domain evaluation. Bootstrap stratifies tasks by five fixed task seeds; intervals condition on these pools and do not cover route-selection uncertainty. This is a standard comparator, not an invention of augmentation loss.

## Numerical and provenance qualification
All2000mean-head task outputs bridge saved comparator predictions; every input/code lock remained equal. All600000query predictions and3000000scores were checked against shapes, labels and task identities.16independent explicit-softmax fits and8identical-view null fits at4prespecified tasks passed. The independent equation implementation uses the production optimizer's stopping defaults; it does not independently certify the exact convex optimum. The original1e-3gradient bound passed. A synthetic comparison to tighter optimizer stopping initially exceeded the1e-4score threshold; the mismatch and matched-stopping amendment are preserved, and neither production solver nor scientific gate was relaxed. No target scores existed then.
Complete paired bootstrap statistics used10000replicates and independently reconstructed task multiplicities. This checks arithmetic of the chosen estimand, not adequacy of a new resampling design. Independent equations were checked at4tasks, not every fit. No full independent feature/data-loader reimplementation is claimed.
Archive qualification restored13code/text files, with hashes, distinguishing older source-trained finite-view objectives and augmented-support ridge. It did not restore weights or prove no similar method exists elsewhere.

## Reproduction
Frozen protocol.json; run.py; summarize.py; execution_lock.json; input_manifest.json; synthetic_validation.json; precheck_amendment.json; validation.json; statistical_audit.json; summary.json; complete.json; dtd.npz; eurosat.npz. Runtime83.78s, local CPU, zero spend, no encoding. Main command bash-f91a2697; summary/audit bash-34ca5c90. Environment and command recorded in execution_lock.json. Failed synthetic call bash-6a4458bf and its diagnostic bash-744974fe are retained.

## Handoff
Retain corrected-mean incumbent and all prior negative packages. Fixed support-view C10 is closed, reopen only with independent evidence changing information or objective rather than a parameter grid. Full paper objective remains active. Next qualified question may shift from accuracy to guaranteed preservation of full-view decisions with fewer query views; require archive/literature and a non-vacuous certificate before any new run. Existing reduced-view accuracy studies do not themselves establish a certificate or encoding latency saving.
