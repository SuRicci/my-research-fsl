# iLPC-z enablement feasibility audit

Already seen pools/tasks; no fitted gate, threshold or new prediction method. Oracle switching uses unavailable query labels and is a non-deployable upper bound.

| Phase/cell | Tasks | Mean delta vs R2, pp | Wins/ties/losses | Seed delta range, pp |
|---|---:|---:|---|---|
| development/dtd_dtd_k1 | 200 | -8.033 | 41/10/149 | -8.813 to -7.253 |
| development/dtd_eurosat_k1 | 200 | -36.020 | 1/0/199 | -36.853 to -35.187 |
| development/dtd_dtd_k5 | 200 | -8.420 | 10/11/179 | -8.720 to -8.120 |
| evaluation/dtd_dtd_k1 | 1000 | -7.657 | 220/45/735 | -8.913 to -5.973 |
| evaluation/dtd_eurosat_k1 | 1000 | -31.313 | 1/1/998 | -32.473 to -29.733 |
| evaluation/dtd_dtd_k5 | 1000 | -11.401 | 32/26/942 | -12.400 to -10.267 |
| evaluation/eurosat_dtd_k1 | 1000 | -24.212 | 16/8/976 | -24.900 to -23.540 |
| evaluation/eurosat_eurosat_k1 | 1000 | +13.648 | 894/20/86 | +12.553 to +14.220 |
| evaluation/eurosat_eurosat_k5 | 1000 | +4.968 | 811/42/147 | +4.593 to +5.353 |

Development oracle switching upper bounds (unavailable query-label knowledge): dtd_dtd_k1: 1.160 pp, dtd_eurosat_k1: 0.007 pp, dtd_dtd_k5: 0.100 pp

Seed consistency can retain a conditional effect, but task-level benefit frequencies and an oracle upper bound do not prove an observable, transferable selector. Original development is DTD-only; no EuroSAT selector may be tuned from these evaluation effects.
