# No-refit Pets failure coverage

Parent run-cb50d2d5, decision-ed211b59. Goal: describe breadth of loss and pseudo-label contamination from saved outputs before selecting another route. Evidence: parent outputs/analysis.json, validation.json, per-task NPZ and assets/identities.json. No new model fitting, predictions or parameter choices. All37breeds/fourconditions retained. Gallery labels permitted only retrospectively for diagnostics; never enter adaptation. Exact DTD label integers must never be matched to Pets label integers.

Success: complete per-breed sign counts, selected matched-gallery in-episode fraction/pseudo-label accuracy, DTD entirely external-domain gallery flag, and numerical/cost summaries. Evidence is descriptive, not causal. Abandon unsupported label mapping if identities or source hashes fail. Use compact report instead of campaign branch because this is one immutable no-refit question, sole parent clear, no supplementary experiment. Then decision/idea reframing; no Pets tuning.

Exit: complete report-b835005e and decision-7abc1872. Bothmismatchcells loseall37breeds;matched1shotselectedlabels80.01%outsideepisode. No numericalfailureevidence. No further slices justified before ideareframing.
