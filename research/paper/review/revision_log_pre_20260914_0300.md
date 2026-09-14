# Revision log — 2026-09-13
| ID | Why it matters | Exact change | Kind | Status | Blocks submission |
|---|---|---|---|---|---|
| R1 | Closely related mass-based outlier selection omitted | Related work: add POT with training and diagnostic distinction; copy-ready paragraph below | verified citation | completed; citation resolved and PDF inspected | yes for mature positioning |
| R2 | Historical records inflated apparent evidence quantity | Sync matrix/ledger displays; report33 total,29 reference-only,4 paper groups,7 tables | metadata | completed;33 records/four paper groups | yes if counts are used |
| R3 | One-target diagnosis cannot support broad algorithmic conclusions | Retain current scope paragraph, source-default restriction and oracle caveats | claim boundary | retained; no rewrite needed | yes for broad claims |
| R4 | No public package or submission checklist | Keep draft_checkpoint and no public-availability claim | submission | deferred | yes |

Copy-ready Related work addition:
Partial optimal transport has also been used to derive an outlier mass score and train an OOD detector jointly with a semi-supervised classifier (Ren et al.,2024). Transported mass in that learned selection objective differs from the scalar averaging coefficients inspected here. Our retrospective interventions characterize a fixed gallery-only predictor and do not establish an alternative to learned open-set selection.

Evidence: official IJCAI2024 paper Sections2–4 and publisher BibTeX; primary URLs in review.md. Remaining limitation: no head-to-head POT comparison or new remedy.

PDF proof: nine pages, eleven references, seven unchanged numeric tables. Bibliography spacing adjusted after observed overflow; all pages checked. System Python fitz import failed after successful compilation; existing pdftoppm completed proof without package installation.
