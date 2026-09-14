# Final-step mass and composition intervention

This retrospective diagnostic leaves the source-default OSLO_G rejection unchanged. Oracle membership uses true gallery labels and is not an eligible method under the inference contract. Earlier fitted weights, centering and feature geometry are held fixed.

| Condition | Original | Balanced mass | Oracle membership | Oracle + balanced | Zero update | R2 | CS_l2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical_pets_k1 | 94.9173 | 95.7573 | 97.9227 | 96.9387 | 94.7467 | 96.2400 | 96.2640 |
| canonical_dtd_k1 | 64.1920 | 94.3013 | 94.4027 | 94.4027 | 94.4027 | 93.9307 | 93.9627 |
| canonical_pets_k5 | 97.1947 | 98.5893 | 98.7947 | 98.8347 | 98.4427 | 98.4427 | 98.5520 |
| canonical_dtd_k5 | 93.3040 | 98.4240 | 98.4640 | 98.4640 | 98.4640 | 98.4427 | 98.5520 |
| fresh_pets_k1 | 94.8747 | 95.3920 | 97.8667 | 96.7440 | 94.2213 | 96.0160 | 96.0480 |
| fresh_dtd_k1 | 63.9920 | 93.6133 | 93.7387 | 93.7387 | 93.7387 | 93.3893 | 93.4400 |
| fresh_pets_k5 | 97.1893 | 98.3920 | 98.5787 | 98.6133 | 98.3760 | 98.3547 | 98.3147 |
| fresh_dtd_k5 | 92.9787 | 98.3227 | 98.3547 | 98.3547 | 98.3547 | 98.3547 | 98.3147 |

| Condition | Intervention | Comparator | Difference (pp) | Paired95% interval (pp) |
|---|---|---|---:|---|
| canonical_pets_k1 | mass_balanced | original | +0.8400 | [+0.5120,+1.1760] |
| canonical_pets_k1 | mass_balanced | r2 | -0.4827 | [-0.6827,-0.2933] |
| canonical_pets_k1 | oracle_membership | original | +3.0053 | [+2.5440,+3.4881] |
| canonical_pets_k1 | oracle_membership | r2 | +1.6827 | [+1.3973,+1.9733] |
| canonical_pets_k1 | oracle_balanced | original | +2.0213 | [+1.6293,+2.4187] |
| canonical_pets_k1 | oracle_balanced | r2 | +0.6987 | [+0.5440,+0.8533] |
| canonical_dtd_k1 | mass_balanced | original | +30.1093 | [+29.1626,+31.1255] |
| canonical_dtd_k1 | mass_balanced | r2 | +0.3707 | [+0.2267,+0.5200] |
| canonical_dtd_k1 | oracle_membership | original | +30.2107 | [+29.2560,+31.2293] |
| canonical_dtd_k1 | oracle_membership | r2 | +0.4720 | [+0.3279,+0.6240] |
| canonical_dtd_k1 | oracle_balanced | original | +30.2107 | [+29.2560,+31.2293] |
| canonical_dtd_k1 | oracle_balanced | r2 | +0.4720 | [+0.3279,+0.6240] |
| canonical_pets_k5 | mass_balanced | original | +1.3947 | [+1.1253,+1.6827] |
| canonical_pets_k5 | mass_balanced | r2 | +0.1467 | [+0.0427,+0.2560] |
| canonical_pets_k5 | oracle_membership | original | +1.6000 | [+1.2987,+1.9173] |
| canonical_pets_k5 | oracle_membership | r2 | +0.3520 | [+0.2107,+0.4933] |
| canonical_pets_k5 | oracle_balanced | original | +1.6400 | [+1.3360,+1.9600] |
| canonical_pets_k5 | oracle_balanced | r2 | +0.3920 | [+0.2960,+0.4907] |
| canonical_dtd_k5 | mass_balanced | original | +5.1200 | [+4.7120,+5.5493] |
| canonical_dtd_k5 | mass_balanced | r2 | -0.0187 | [-0.0987,+0.0560] |
| canonical_dtd_k5 | oracle_membership | original | +5.1600 | [+4.7520,+5.5921] |
| canonical_dtd_k5 | oracle_membership | r2 | +0.0213 | [-0.0533,+0.0907] |
| canonical_dtd_k5 | oracle_balanced | original | +5.1600 | [+4.7520,+5.5921] |
| canonical_dtd_k5 | oracle_balanced | r2 | +0.0213 | [-0.0533,+0.0907] |
| fresh_pets_k1 | mass_balanced | original | +0.5173 | [+0.2079,+0.8373] |
| fresh_pets_k1 | mass_balanced | r2 | -0.6240 | [-0.7840,-0.4560] |
| fresh_pets_k1 | oracle_membership | original | +2.9920 | [+2.5333,+3.4587] |
| fresh_pets_k1 | oracle_membership | r2 | +1.8507 | [+1.5707,+2.1440] |
| fresh_pets_k1 | oracle_balanced | original | +1.8693 | [+1.4986,+2.2533] |
| fresh_pets_k1 | oracle_balanced | r2 | +0.7280 | [+0.5600,+0.9120] |
| fresh_dtd_k1 | mass_balanced | original | +29.6213 | [+28.6639,+30.5520] |
| fresh_dtd_k1 | mass_balanced | r2 | +0.2240 | [+0.0800,+0.3733] |
| fresh_dtd_k1 | oracle_membership | original | +29.7467 | [+28.7919,+30.6961] |
| fresh_dtd_k1 | oracle_membership | r2 | +0.3493 | [+0.1947,+0.5040] |
| fresh_dtd_k1 | oracle_balanced | original | +29.7467 | [+28.7919,+30.6961] |
| fresh_dtd_k1 | oracle_balanced | r2 | +0.3493 | [+0.1947,+0.5040] |
| fresh_pets_k5 | mass_balanced | original | +1.2027 | [+0.9653,+1.4587] |
| fresh_pets_k5 | mass_balanced | r2 | +0.0373 | [-0.0720,+0.1493] |
| fresh_pets_k5 | oracle_membership | original | +1.3893 | [+1.1307,+1.6694] |
| fresh_pets_k5 | oracle_membership | r2 | +0.2240 | [+0.0827,+0.3547] |
| fresh_pets_k5 | oracle_balanced | original | +1.4240 | [+1.1493,+1.7120] |
| fresh_pets_k5 | oracle_balanced | r2 | +0.2587 | [+0.1547,+0.3627] |
| fresh_dtd_k5 | mass_balanced | original | +5.3440 | [+4.9387,+5.7493] |
| fresh_dtd_k5 | mass_balanced | r2 | -0.0320 | [-0.1093,+0.0453] |
| fresh_dtd_k5 | oracle_membership | original | +5.3760 | [+4.9653,+5.7814] |
| fresh_dtd_k5 | oracle_membership | r2 | -0.0000 | [-0.0693,+0.0720] |
| fresh_dtd_k5 | oracle_balanced | original | +5.3760 | [+4.9653,+5.7814] |
| fresh_dtd_k5 | oracle_balanced | r2 | -0.0000 | [-0.0693,+0.0720] |

Balancing final gallery mass against labeled support removes most mismatched-gallery degradation, but remains below R2 in matched1shot in both pools. Thus excessive final-step mass is an actionable part of the observed failure, while changing this scalar alone does not supply the required strong-baseline improvement.
Removing task-external gallery samples with oracle labels increases matched1shot performance beyond R2. This identifies a constrained intervention with useful information, not an implementable estimator or a guaranteed ceiling. Earlier iteration weights remain fixed and may already contain contamination.
Balanced mass and oracle composition are diagnostic counterfactuals defined after seeing the failed main result. They are neither prospectively confirmed candidates nor evidence for algorithmic novelty. The same-domain fresh pool is now exposed.
All4,000 original predictions replayed exactly; oracle mismatch agrees with zero-update. Full paired intervals condition on fixed pools and task-seed strata. The initial import failure and its diagnostic-only repair are recorded in recovery.json.
No further diagnostic variants are committed. Next route: consolidate supported evidence and limitations before selecting any new trainable membership mechanism or independent evaluation domain.
