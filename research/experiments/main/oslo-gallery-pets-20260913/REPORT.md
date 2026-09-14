# Gallery-only OSLO source-default qualification

Fixed source-default OSLO gallery transfer; no query adaptation. Canonical outcomes were exposed before direction selection; fresh images were prospectively encoded after freezing, but share domain/classes. Intervals condition on fixed pools and seeds. No independent-domain, universal-safety, algorithmic-novelty, or SOTA claim. Exact RGB overlap excluded; near-duplicate and pretraining overlap remain unmeasured.

Primary qualification gate passed: False

| Pool / gallery / shot | OSLO_G | Closed set | Zero update | R2 | CS_l2 | Support C1 | Support C10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical_pets_k1 | 94.9173 | 90.5040 | 94.7467 | 96.2400 | 96.2640 | 94.2773 | 94.3333 |
| canonical_dtd_k1 | 64.1920 | 36.9280 | 94.4027 | 93.9307 | 93.9627 | 94.2773 | 94.3333 |
| canonical_pets_k5 | 97.1947 | 94.3760 | 98.4427 | 98.4427 | 98.5520 | 98.4080 | 98.5627 |
| canonical_dtd_k5 | 93.3040 | 49.4053 | 98.4640 | 98.4427 | 98.5520 | 98.4080 | 98.5627 |
| fresh_pets_k1 | 94.8747 | 91.7040 | 94.2213 | 96.0160 | 96.0480 | 93.7307 | 93.7440 |
| fresh_dtd_k1 | 63.9920 | 38.0827 | 93.7387 | 93.3893 | 93.4400 | 93.7307 | 93.7440 |
| fresh_pets_k5 | 97.1893 | 94.6987 | 98.3760 | 98.3547 | 98.3147 | 98.3360 | 98.3840 |
| fresh_dtd_k5 | 92.9787 | 48.7600 | 98.3547 | 98.3547 | 98.3147 | 98.3360 | 98.3840 |

| Cell | Comparator | Difference (pp) | Paired 95% interval (pp) |
|---|---|---:|---|
| canonical_pets_k1 | closed_set | +4.4133 | [+3.8399, +5.0401] |
| canonical_pets_k1 | zero_update | +0.1707 | [-0.2481, +0.5813] |
| canonical_pets_k1 | r2 | -1.3227 | [-1.7227, -0.9333] |
| canonical_pets_k1 | CS_l2 | -1.3467 | [-1.7414, -0.9600] |
| canonical_pets_k1 | support_logistic_C1 | +0.6400 | [+0.1813, +1.0880] |
| canonical_pets_k1 | support_logistic_C10 | +0.5840 | [+0.1439, +1.0161] |
| canonical_dtd_k1 | closed_set | +27.2640 | [+25.9040, +28.6109] |
| canonical_dtd_k1 | zero_update | -30.2107 | [-31.2293, -29.2560] |
| canonical_dtd_k1 | r2 | -29.7387 | [-30.7441, -28.7920] |
| canonical_dtd_k1 | CS_l2 | -29.7707 | [-30.7920, -28.8187] |
| canonical_dtd_k1 | support_logistic_C1 | -30.0853 | [-31.1094, -29.1307] |
| canonical_dtd_k1 | support_logistic_C10 | -30.1413 | [-31.1547, -29.1840] |
| canonical_pets_k5 | closed_set | +2.8187 | [+2.3840, +3.3041] |
| canonical_pets_k5 | zero_update | -1.2480 | [-1.5573, -0.9599] |
| canonical_pets_k5 | r2 | -1.2480 | [-1.5520, -0.9600] |
| canonical_pets_k5 | CS_l2 | -1.3573 | [-1.6694, -1.0560] |
| canonical_pets_k5 | support_logistic_C1 | -1.2133 | [-1.5227, -0.9253] |
| canonical_pets_k5 | support_logistic_C10 | -1.3680 | [-1.6880, -1.0693] |
| canonical_dtd_k5 | closed_set | +43.8987 | [+42.4506, +45.3414] |
| canonical_dtd_k5 | zero_update | -5.1600 | [-5.5921, -4.7520] |
| canonical_dtd_k5 | r2 | -5.1387 | [-5.5733, -4.7280] |
| canonical_dtd_k5 | CS_l2 | -5.2480 | [-5.6933, -4.8293] |
| canonical_dtd_k5 | support_logistic_C1 | -5.1040 | [-5.5333, -4.6987] |
| canonical_dtd_k5 | support_logistic_C10 | -5.2587 | [-5.7040, -4.8427] |
| fresh_pets_k1 | closed_set | +3.1707 | [+2.6987, +3.6640] |
| fresh_pets_k1 | zero_update | +0.6533 | [+0.2507, +1.0533] |
| fresh_pets_k1 | r2 | -1.1413 | [-1.5173, -0.7627] |
| fresh_pets_k1 | CS_l2 | -1.1733 | [-1.5413, -0.8053] |
| fresh_pets_k1 | support_logistic_C1 | +1.1440 | [+0.7227, +1.5761] |
| fresh_pets_k1 | support_logistic_C10 | +1.1307 | [+0.7172, +1.5653] |
| fresh_dtd_k1 | closed_set | +25.9093 | [+24.6186, +27.2054] |
| fresh_dtd_k1 | zero_update | -29.7467 | [-30.6961, -28.7919] |
| fresh_dtd_k1 | r2 | -29.3973 | [-30.3413, -28.4560] |
| fresh_dtd_k1 | CS_l2 | -29.4480 | [-30.3814, -28.5013] |
| fresh_dtd_k1 | support_logistic_C1 | -29.7387 | [-30.6747, -28.7759] |
| fresh_dtd_k1 | support_logistic_C10 | -29.7520 | [-30.6933, -28.8000] |
| fresh_pets_k5 | closed_set | +2.4907 | [+2.0827, +2.9040] |
| fresh_pets_k5 | zero_update | -1.1867 | [-1.4774, -0.9120] |
| fresh_pets_k5 | r2 | -1.1653 | [-1.4720, -0.8773] |
| fresh_pets_k5 | CS_l2 | -1.1253 | [-1.4267, -0.8427] |
| fresh_pets_k5 | support_logistic_C1 | -1.1467 | [-1.4453, -0.8666] |
| fresh_pets_k5 | support_logistic_C10 | -1.1947 | [-1.4960, -0.9093] |
| fresh_dtd_k5 | closed_set | +44.2187 | [+42.7732, +45.7253] |
| fresh_dtd_k5 | zero_update | -5.3760 | [-5.7814, -4.9653] |
| fresh_dtd_k5 | r2 | -5.3760 | [-5.7813, -4.9680] |
| fresh_dtd_k5 | CS_l2 | -5.3360 | [-5.7360, -4.9199] |
| fresh_dtd_k5 | support_logistic_C1 | -5.3573 | [-5.7547, -4.9520] |
| fresh_dtd_k5 | support_logistic_C10 | -5.4053 | [-5.8133, -4.9947] |

Full gate decisions, per-seed/per-breed results and fitting diagnostics: outputs/analysis.json.
Complete eight-cell task, identity, prediction, frozen-baseline and score-replay audit: outputs/validation.json.
Source parity and query permutation/split/insertion checks: outputs/source_validation.json.
Protocol and inference source hashes were frozen before fresh feature extraction: protocol.json and locked_sources.json.
Paired bootstrap: 5,000 resamples within five task-seed strata; tasks across gallery conditions use shared resamples. No independent-domain uncertainty estimate.
Known OSLO source: Boudiaf et al., CVPR 2023, https://arxiv.org/abs/2301.08390; author code commit 9240a4e630874c50459e069db609db9163a81c03. Gallery-only fitting changes the task from source transductive query adaptation.
All seven methods use identical frozen CLIP/DINO features. Existing encoders and images were reused; zero new downloads and no remote compute.
