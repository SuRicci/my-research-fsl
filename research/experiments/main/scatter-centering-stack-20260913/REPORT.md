# Cumulative scatter-centering: verified increment with a retained boundary

## Conclusion
Adding locked support-view scatter to six-view mean plus support centering improves pooled one-shot accuracy from75.5400% to77.1880%, +1.6480pp (paired95%CI1.46195 to1.83600). It passes the prespecified increment/no-harm gate versus meanCS. Five-shot mean increases90.65333% to91.36533%, +0.7120pp (0.58533 to0.84267).

This does not pass the stronger full-stack promotion gate relative to scatter alone. Pooled one-shot +0.2200pp (0.0900 to0.34735) coexists with DTD macro -0.11333pp (-0.2640 to0.0280), and the DTD mismatched-gallery cell -0.29867pp (-0.53067 to-0.07733) breaches the predeclared -0.5pp lower bound. Retain positive incremental evidence; do not claim universal superiority or authorize prospective expansion under this failed gate.

## Matched per-domain effects versus the retained meanCS stack
| Domain | Shot | Delta pp | Paired95%CI pp |
|---|---:|---:|---|
| EuroSAT | 1 | +3.20667 | [2.88267,3.54133] |
| DTD | 1 | +0.08933 | [-0.08133,0.25333] |
| EuroSAT | 5 | +1.46933 | [1.28800,1.66133] |
| DTD | 5 | -0.04533 | [-0.22133,0.12533] |

## Component attribution already resolved
At one shot, the center x scatter interaction is -0.11133pp (-0.23468 to0.01467). No positive synergistic interaction is established; useful pooled stacking need not be superadditive. At five shots, the CS package changes ridge penalty from1 to0.1. Holding lambda0.1 fixed, adding centering to scatter loses0.22533pp (-0.30933 to-0.14530), with losses in bothdomains. The uncentered scatterlambda0.1 readout averages91.59067%. Thus a comparison only to legacy scatterR2 would misattribute some penalty benefit to centering. No posthoc newgamma/order/selector chosen.

## Protocol and validity
Selected idea-71814a9c and decision-af86142e. Actual semanticrun scatter-centering-stack-20260913 descends from completed support-riskselection; currentruntime projection can lag actualGit. Same frozen CLIP/DINO features, sixviews, equalencoderfusion, same500tasks per domain/shot with5seeds, matched and mismatched galleries;2000unique tasks produce4000taskconditions. All6packages sharefloat64; sourcegamma inherited from prior opposite-domain selections with no reselection. Four primarypackages plus two shared-penalty diagnostics. Queryvectors do not fit scatter, center, neighbors orparameters; no querylabels or gallerylabels inclassifier.

18 prechecks, including original float32 failurecase, pass newfloat64 implementation. Independent final audit checks alltask/aggregate contracts,240sampled dense NumPy eigendecomposition/primalridge scorecases with exactneighbors/predictions, and62paired statistics via independent bootstrapmultiplicity calculation. Maxscoreerror1.0381e-14; maxCIerror8.88e-16pp. Sampling is not full allscore reconstruction. Whole meanR2 and meanCS predictionbanks exactly match historicalfloat32. ScatterR2 has105prediction changes over alltaskconditions (five-shot galleryduplicates included); one-shot accuracy unchanged, EuroSATfive-shot -0.01067pp andDTDfive-shot -0.05867pp versusoldprecision. Originalfloat32 results/frozenaudit remain immutable, not retroactively markedpassed.

Run bash-a366bd26 completed85.764s. Final audit bash-b6717efb completed56.003s. Initial audit bash-de78a1ca failed before numericwork because legacy verify module shadowedlocalverify; exact-path importrepair preserved inoutputs/audit_import_repair.json. No classification rerun or measuredcode change. Initialsciencewrites missed requiredparent/node_type fields, rejected without creatingnodes; correctedrecords carry validlineage.

## Evidence boundary and continuation
Auxiliary exposed-development study, conditional fixed-image-pool intervals; not newPets/Caltech measurements, independentdatasetconfirmation, exactpaperreproduction oralgorithmnovelty. Five canonicalPetsmetricids andbaselinecontract unchanged. The currentnumericallyverified higher-scoring stack is retained as a conditional candidate; fullpromotion false and historicalmeanCS remains the acceptedreference. Keep scatteralone as a serious complementary challenger.
Next bounded question: atone-shot, does center-dependent neighborselection or centeredclassification cause the remaining DTDmismatch loss? Cross the neighbor set and featuregeometry with allothercomponents fixed; reuse sameassets and predictions forcausal decomposition, not source/targetretuning. This information separates a compatible nextcomponent from another blind stack. Five-shot penalty finding is alreadyresolved; do notrerun thatgrid. Beforefuturewriting, maptheseauxiliaryresults and earlierunmapped studies tooutline/evidenceledger; priorpapercheckpoint unchanged.

Files: protocol.json,code_lock.json,model.py,study.py,verify.py,summarize.py,audit.py; outputs/manifest.json,complete.json,numeric_validation.json,analysis.json,audit.json and eightNPZs. Resources localonly/zero-spend, newoutputs~1.9MB beforeaudit, free~10.7GiB,deadline2026-09-15T09:00Z. No rawdata/model/featuredeletion or newdownloads needed.
