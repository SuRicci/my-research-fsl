# Directional stopping fixed usefulness verdict
The candidate fails its pre-outcome usefulness gate. Do not promote it or tune nearby stopping bounds.

| Domain | Omitted query-view applications | Fraction of full budget | Prediction disagreements |
|---|---:|---:|---:|
| dtd | 11216 | 4.984889% | 0 |
| eurosat | 5159 | 2.292889% | 0 |
| Macro / total | 16375 / 450000 | 3.638889% | 0 |

Candidate idea-0760972a, decision-851c76cf, parent run-c1ac97bc. Each domain has500five-shot tasks and37500query occurrences. Gate:>=10%macro and>=5%each domain,zero disagreements. Both domain gates and the macro gate failed. Existing corrected-mean accuracy remains91.56%; all75000reference predictions match archived incumbent outputs. Maximum score difference4.261e-11. No new accuracy claim.

Protocol/code/numerical document hashes were frozen before outcomes. The real-arithmetic bound accounts for the nonzero intercept and normalization floor; the guard is conditional on the documented standard CPU arithmetic model. Independent high-precision synthetic qualification and real-data dense/scalar audit passed. AUDIT.json reconstructs all earliest-stop counts, identities and output hashes; dense audit covers12queries/60prefixes from4fixed tasks. These checks are not formal binary verification or a guarantee for arbitrary hardware.

The screen uses already-computed feature caches and took95.159s. Potential query-view omissions are neither measured encoder speedup nor net end-to-end savings. Support feature costs and batching overhead are excluded. The earlier isotropic-cap result3.6987%used an intentionally optimistic radius0.99; the present B1.001guarded bound is not a controlled head-to-head dominance comparison. Both failed their practical screen.

Incident: the initial audit was terminated by its60scommand deadline because it repeatedly decompressed NPZ arrays in a scalar loop. One-time materialization plus a detached audit passed; measured execution was not repeated. See AUDIT_RECOVERY.md.

Decision implication: close this candidate and the immediate cap/directional bound family for this setup. Do not claim all safe stopping methods are ineffective. Reopen only with independent structural evidence and a new explicit contract, not threshold/radius/order tuning on these outcomes. Return to idea for a different bottleneck, starting from existing data/permissions and independent-evidence feasibility. Full-paper objective remains active; no completion approval and no submission-ready manuscript.
