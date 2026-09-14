# Support validation for finite representation selection: verified development result

## Outcome
The fixed protocol fails qualification. Both one-shot domains lose against the same-sixview equal mixture, and balanced image validation at five shots also fails to select a better representation. Fixed reference classifiers remain incumbent. This rejects the prespecified method, not every possible support estimator.

## Protocol
Idea idea-8cbaafc7; branch run/support-risk-selection-20260913; decision-d3308c1f. For each task, choose among equal CLIP/DINO mixture and either encoder endpoint by centered ridge prediction MSE on held-out support information. Ridge penalty0.1, fixed tie order equal/DINO/CLIP. At1shot, six folds each withhold a view of every support image; the remaining five views form support means. At5shot, five balanced folds each withhold one entire image per class and train on four images per class. A separate5shot view-fold selector is diagnostic only. Final features use all six views; query vectors and labels never enter selection. Unlabeled gallery may enter one-shot selection through the unchanged retrieval head.
Same2000unique development tasks,500/domain/shot, five taskseeds and two paired gallery conditions. All inputs, code and protocol are hashed. DTD/EuroSAT are historically exposed development data. No new Pets/Caltech evaluation and no change to the accepted Pets metric contract. This auxiliary result is recorded as report-77f9a6fd, not represented as a new canonical benchmark result.

## One-shot comparison to equal same-view features
|Domain|Selected accuracy %|Fixed equal accuracy %|Difference pp [95% CI]|
|---|---:|---:|---:|
|dtd|77.4653|77.8133|-0.3480 [-0.5107, -0.1880]|
|eurosat|72.5947|73.2667|-0.6720 [-0.8760, -0.4640]|

All five fixed taskseed differences are negative in each domain. Intervals resample tasks within fixed seeds, averaging both gallery conditions within each task before bootstrapping. They are conditional on reused image pools; they are not population-level or prospective-domain guarantees. Full comparisons against every prespecified fixed choice and nine previously saved controls are in outputs/analysis.json. Improvement over original single-view controls does not pass the matched sixview comparison.

## Conditions and diagnostic observations
|Condition|Selected %|Equal %|Selected-minus-equal pp|CV accuracy %|CV-minus-query pp|
|---|---:|---:|---:|---:|---:|
|dtd_dtd_k1|79.4720|79.8160|-0.3440|100.0000|20.5280|
|dtd_eurosat_k1|75.4587|75.8107|-0.3520|100.0000|24.5413|
|dtd_dtd_k5|90.0933|90.6507|-0.5573|90.2560|0.1627|
|dtd_eurosat_k5|90.0933|90.6507|-0.5573|90.2560|0.1627|
|eurosat_dtd_k1|70.2880|70.4987|-0.2107|99.9867|29.6987|
|eurosat_eurosat_k1|74.9013|76.0347|-1.1333|99.9800|25.0787|
|eurosat_dtd_k5|89.3093|90.6560|-1.3467|89.5280|0.2187|
|eurosat_eurosat_k5|89.3093|90.6560|-1.3467|89.5280|0.2187|

The selected1shot view predictors are almost perfectly accurate on their correlated held-out views, while unseen-image task accuracy remains70.3-79.5%. That discrepancy documents optimistic view validation under this fixed crop distribution. It does not prove that view correlation alone caused every selection error. The independent-image5shot estimator has a much smaller mean accuracy discrepancy (0.16/0.22pp), yet its selected classifier loses0.5573/1.3467pp versus equal features. Accuracy-level mean calibration therefore does not establish correct task-specific ranking by squared loss. No posthoc loss/temperature/weight tuning was performed.
The five-shot diagnostic view selector exceeds the primary image selector by0.2267/1.2480pp, but remains below the equal reference. It is not promoted after observing that contrast. Gallery-free five-shot duplicates are not independent replications. Classifier, fold-size mismatch and risk-estimator variance remain possible explanations; current evidence does not isolate their causal contributions.

## Verification
Formal bash-a5f133a9 completed140.63seconds. Precheck bash-bf952744 passed all2000task identities and four domain/shot equal-reference/batch checks. Independent NumPy64 bash-fc75ecd9 reproduced72fixed query cases and396fold score matrices; maxima1.959e-6/2.482e-6 below3e-5. All300000equal-sixview reference predictions match the previous bank exactly;1.5millionprediction entries checked and all selected indices reconstructed. Independent scalar bootstrap bash-6acd141c reproduced both primary intervals to1e-10. Science nodes science-57fa28d3 and science-c2d15085 preserve execution/validation separately.
No failed formalrun or changed outcome-driven configuration. A functions orchestration syntax error occurred before a later statistics tool call; no command ran until corrected. The earlier missing PyTorch catalog card was handled by checking catalog absence and using the already working local torch2.3.0/NumPy1.24.4 environment. It did not alter scientific execution.

## Resources and next question
Outputs use1.408MiB; free space at report time10.743GiB. No additional dataset/model downloads, new encoding or cleanup deletion was needed. Earlier cleanup receipts and all retained assets remain intact.
Close this finite validation protocol and do not rescue it with another loss, weight grid or confidence threshold on the same tasks. Before another model is proposed, assess whether constrained global updates have a substantive causal motivation beyond interpolating back toward the identity; compare that opportunity with consolidating accumulated evidence. Reuse current source-composition/learned-order checkpoints and curves for this assessment. Existing paper checkpoint is preserved, and all new auxiliary outcomes must be mapped to its outline/evidence ledger before further writing.
