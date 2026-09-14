# Direct utility weighting: source qualification did not pass

The fixed utility-trained composition model does not provide robust transfer gains. This auxiliary study uses already exposed DTD and EuroSAT features; it produces no new canonical Pets or Caltech result. All eight prespecified cells are complete and independently validated.

## Paired one-shot transfer

| Training source → evaluation domain | vs learned scalar (pp, 95% CI) | vs BCE (pp, 95% CI) | vs same-alpha uniform (pp, 95% CI) |
|---|---|---|---|
| dtd → eurosat | -0.1760 [-0.2694, -0.0840] | +0.1200 [+0.0400, +0.1987] | +0.0533 [-0.0160, +0.1254] |
| eurosat → dtd | -1.4827 [-1.7053, -1.2547] | -1.6333 [-1.8587, -1.3933] | -1.2267 [-1.4480, -0.9933] |

The DTD-trained utility model improves over BCE by a small amount but loses to the independently optimized uniform-weight scalar. Its composition-only comparison includes zero. In the reverse direction it loses to all three controlled alternatives. These observations reject this fixed promotion hypothesis; they do not prove that every utility-learning or meta-learning method fails.

## Full accuracy surface

| Cell | Utility | Scalar | BCE | Same alpha | R2 | CS_l2 |
|---|---|---|---|---|---|---|
| dtd_to_eurosat_dtd_k1 | 65.3040 | 66.1307 | 65.0027 | 65.1680 | 66.5493 | 68.3253 |
| dtd_to_eurosat_dtd_k5 | 85.0107 | 85.0293 | 85.0293 | 85.0267 | 87.1360 | 89.6293 |
| dtd_to_eurosat_eurosat_k1 | 74.8533 | 74.3787 | 74.9147 | 74.8827 | 74.1227 | 73.9227 |
| dtd_to_eurosat_eurosat_k5 | 86.4027 | 86.4160 | 86.3600 | 86.4133 | 87.1360 | 89.6293 |
| eurosat_to_dtd_dtd_k1 | 77.2933 | 78.7440 | 79.3680 | 78.7520 | 78.4160 | 78.5173 |
| eurosat_to_dtd_dtd_k5 | 88.8400 | 89.5067 | 89.5280 | 89.5093 | 90.2720 | 90.1013 |
| eurosat_to_dtd_eurosat_k1 | 68.5520 | 70.0667 | 69.7440 | 69.5467 | 74.7547 | 74.4533 |
| eurosat_to_dtd_eurosat_k5 | 87.7920 | 88.0507 | 88.1733 | 88.0480 | 90.2720 | 90.1013 |

All method labels, including both support-logistic and source-selected mix/ridge configurations, are provided in outputs/accuracy.csv and analysis.json. Distinct configuration labels need not represent distinct algorithms or predictions. The utility candidate misses both the ≥0.5pp directional-superiority gate and the all-cell noninferiority gate.

## Protocol and evidence limits

Both source directions use 100 training and 100 identity-disjoint source-selection episodes per shot. One-shot training uses 15 queries per class. Five-shot source training and selection use 13 because the smallest DTD image half contains 18 images; the target evaluation keeps 15 queries per class in all conditions. All controls share the same permitted source labels and episode allocation. The BCE training intercept is a nuisance parameter that cancels in the normalized relative weights. Checkpoints were chosen using source accuracy, then source cross-entropy, then earlier update. Encoder parameters remain fixed and query batches never fit state.

Four thousand task-condition evaluations correspond to 2,000 sampled tasks paired over two gallery conditions. Intervals use 5,000 paired bootstrap replicates stratified by seed and are conditional on the fixed image pools and single deterministic source-training allocation. The fixed80-update training budget does not establish globally optimized models. No new-method originality or independent-domain generalization is established.

Validation checked every task label and support/query exclusion, disjoint source-training/selection image sets, all score→prediction→accuracy mappings, source-only checkpoint selection, immutable pre-repair one-shot hashes and numerical finiteness. Trained predictions on24first/central/last tasks were independently reconstructed using NumPy; query splitting preserved predictions. Synthetic primal-versus-dual ridge and finite-difference gradients also passed.

## Interpretation and next question

Replacing membership supervision with query-risk supervision did not resolve domain dependence in this controlled implementation. Global mixing and relative composition must remain separate controls. Source-task distribution and feature information remain competing untested explanations, not established causes. Next study selection should challenge those assumptions; do not run another weight/temperature grid on these target outcomes. Existing baselines and the earlier technical report remain the retained result.
