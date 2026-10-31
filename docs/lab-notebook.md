# Lab notebook

The raw development log, kept as written during the sprint. Polished results are in [03-results.md](03-results.md).


## 2026-10-26
Done
- Repo, pipeline (events, label A onsets, patient-week table, time and patient splits), 9 tests.
- Baselines: 2x rule, logistic regression, XGBoost.
- FIM-PP zero-shot (separate env in `~/projects/vendor/OpenFIM`, scored via `scripts/fimpp_score.py`).
- TabPFN code ready (`models/tabpfn_model.py`), waiting on license token.

Pain points
- XGBoost on macOS needs `brew install libomp`.
- FIM-PP pins transformers 4.56.2, so it lives in its own venv and talks to us through parquet files.
- FIM-PP binary-searches event times: pad with a large time, never zeros, or intensities are garbage.
- FIM-PP appends event times to the evaluation grid: our grid is the first 29 outputs.
- 364 anchors have no community events before them (one person is inpatient-only), so FIM-PP has no score there.

- Identifier audit: client first name + hospital name in staff free text (2 list columns). Stripped from
  working copy (`scripts/strip_free_text.py`); feature table unchanged. Data is Mac-only (no lab server).
- TabPFN licenses accepted (2.5 and 3.5). Key in git-ignored `.env`.
- TabPFN zero-shot is the weakest model so far (time split AUROC: 2.5 = 0.61 to 0.63, 3.5 = 0.65;
  patient split 2.5 = 0.63) vs logistic 0.71 to 0.72 and FIM-PP zero-shot 0.68.
- Drift check: replacing running totals with yearly rates (`--stationary`) helps logistic/XGBoost a little
  (0.715 / 0.68), not TabPFN. Patient split (no drift) shows the same TabPFN gap, so drift isn't the cause.
- Full TabPFN context (7,783 rows, 8 estimators) was slower (11 min) and no better than 4,000 rows.

More pain points
- tabpfn 9.2 defaults to TabPFN-3.5, not 2.5: pin the version explicitly.
- XGBoost and PyTorch each load OpenMP on macOS and crash in one process: XGBoost is imported lazily.
- Every extra variant scored on the same test set inflates the best number: stop tuning after the planned runs.

- Fine-tuning hurt both pretrained models (time split): TabPFN-2.5 0.605 -> 0.602 (its own random-row
  validation said 0.986: memorised people), FIM-PP 0.68 -> 0.63 (73 train timelines, 18 held-out people,
  500 epochs, picked epoch 400 by held-out NLL; the script's own best-model uses train loss, ignore it).

Next
- Decide the main model for the dashboard/poster with Nithin, then evaluation write-up, dashboard, report.
