# Task-local frozen representation masks: verified development result

## Verdict
The fixed SUR-style mask protocol fails qualification and will not be expanded to Pets or Caltech. Support likelihood improves while every evaluated task places more than 99% of effective weight on DINO. Fixed mixed representations remain the incumbent; no general impossibility or new-algorithm claim follows.

## Protocol and evidence
Selected idea-7e538f3a; run/task-local-mask-20260913. Two frozen encoders, single224view,40Adadelta supportNCC steps at100 with explicitlocalrho0.9/eps1e-6. Primary uses selectedsquaredblockweights then CS_l2. RawR2 and nativeNCC readouts are diagnostics. Compare equalcomposition and both singleencoderCS_l2 endpoints, unchanged support/gallerypermissions, masks independent of queryvectors/labels and gallery. No source model or hyperparameterselection. This transfersSURselection to a differentencoderbank/classifier and is not exactpublishedSURreproduction.
2000unique5waytasks:500/domain/shot over5fixedseeds.8conditions include twogalleries; duplicatedgalleryfree5shot/NCC outcomes are not independentreplications. DTD/EuroSAT are exposeddevelopment. CanonicalPetsbaseline and itsrequiredmetricids are unchanged. This auxiliarydevelopmentreport is not submitted as a newPetsmainexperiment.

## Primary one-shot effects
|Domain|Mask CS_l2 (%)|Equal CS_l2 (%)|Difference pp [95% CI]|
|---|---:|---:|---:|
|dtd|75.9013|76.4853|-0.5840 [-0.8253, -0.3373]|
|eurosat|67.5173|71.1240|-3.6067 [-3.9694, -3.2493]|

Allfivefixedseed effects versus equalcomposition are negative in bothdomains. Mask-minusDINOendpoint is only0.0093pp/0.0187pp, farbelow the0.5ppqualificationgate. These smallendpointdeviations do not rescue the negativeprimarycomparison. Paired5000taskbootstrap stratifies taskseed and averages galleryconditions within eachtask; intervals are conditional on reusedpools and do not capture prospective-domain uncertainty.

## Full condition accuracy (%)
|Condition|Mask CS|Equal CS|CLIP CS|DINO CS|R2|Mask R2|NCC|Mask NCC|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|dtd_dtd_k1|78.1440|78.5173|67.8347|78.1493|78.4160|77.7147|74.4320|74.1280|
|dtd_eurosat_k1|73.6587|74.4533|63.1627|73.6347|74.7547|73.5440|74.4320|74.1280|
|dtd_dtd_k5|89.1627|90.1013|85.4613|89.1573|90.2720|89.6827|89.7760|88.9173|
|dtd_eurosat_k5|89.1627|90.1013|85.4613|89.1573|90.2720|89.6827|89.7760|88.9173|
|eurosat_dtd_k1|65.3467|68.3253|64.5627|65.3280|66.5493|63.0587|68.3653|64.3013|
|eurosat_eurosat_k1|69.6880|73.9227|69.7440|69.6693|74.1227|70.0507|68.3653|64.3013|
|eurosat_dtd_k5|86.1200|89.6293|86.4613|86.1067|87.1360|84.0427|85.4080|81.2587|
|eurosat_eurosat_k5|86.1200|89.6293|86.4613|86.1067|87.1360|84.0427|85.4080|81.2587|

## Mechanism diagnosis
Posthoc analysis of the existing supports shows that100% ofDTD and97.2% ofEuroSATone-shot tasks have CLIPcosine larger than DINOcosine for every off-diagonal classpair. For these tasks the supportNCCobjective is strictlyincreasing in theCLIPeffectiveweight: thecorrect-classcosine is1, while allwrong-classlogits increase. This algebraic conditional explains theobjectivepreference without appealing toquerylabels. It does not show thatallCLIPgeometry or alltaskselection fails.
Meanone-shotCLIPweights are0.003196(DTD) and0.003137(EuroSAT),with tinytaskvariation. Unmodifiedmixedfeatures exceedDINOon55.2%/81.2% ofpairedtasks, so objectivepreference doesnot reliably indicateclassifierbenefit. NativeNCCalso deteriorates, which rules out attributing the entirefailure only to insertion into retrieval/ridge. Newdiagnostic is posthoc, no newmodel or tuning.

## Verification and resources
Independentexplicit-feature optimization plus manualAdadelta updates reproduced12masks (maxerror2.7e-09); NumPy64 reconstructed192scorecases (maxerror2.1e-06<3e-5) withidenticalpredictions. All600000fixedR2/CSreferencepredictions exactlymatchthepriorbank; all2.4millionpredictionentries checkedforrange/shape. Code/inputhashlock passed. Theactual4000taskconditionrows mapto2000uniquetasks.
Precheck bash-5b4357ea stoppedbeforedataload atdiskfloor. Afterverifiedcleanup,bash-dd34d00f passed2000taskidentitychecks and gradient/permutation/batchequivalence. Formal bash-c302b34b completed68.89s. Audit bash-58010fd3 firstfailed onmissingoldnamesfield; repairedfromrv_study fixedheadorder, then bash-17ad2342 passed. Formalcode,protocol and outcomesunchanged.
Removedtwoinactiveancestorworktreecopies only, retainedbothbranches/commits andthreeuntrackedfiles withverifiedhashesandrestorationinstructions. Observedfree8.2705->10.8458GiB; concurrentfilesystemactivity prevents exactattribution of thatdelta. Receipt artifacts/reports/taskmask-cleanup-receipt.json. Anoldrawlistinventorycrashedartifactprojectionafterrefresh; exactrawbytes preservedunderartifacts/idea/raw_inventory andpointerformatrepaired.

## Next anchor
Reject thisfixedobjective/optimizerprotocol; do not tuneiteration,temperature ormaskthreshold against theseoutcomes. Nextidea should test whether available withinclass information or a head-aligned permittedvalidation objective can distinguish representations; revisit existing multi-viewcandidate rejection notes before proposing it. Geometrypreservingglobalupdates remain deferred, because thecurrentdata do not establish that moreglobaltraining helps. Existingpaper remainscheckpoint; integrate this auxiliaryevidence only afteroutline/ledgerreview. No completionapproval requested.
