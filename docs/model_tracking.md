## Model Tracking Log

Purpose

- Tie modeling decisions (features, thresholds, gates) to outcomes.
- Make runs comparable and reproducible over time.
- Provide a single place to justify keeping or discarding changes.

How to use

- Create one entry per tagged run (or backtest window).
- Link to artifacts in `data/models/` and `logs/` when possible.
- Keep comparisons scoped: one meaningful change per entry.

Naming guidelines

- Use explicit tags for gates and context, e.g. `2025W7_2025W7_2025W8`, `apple`, or `wk200`.
- The training script normalizes tags for filenames; record both the raw tag and the normalized one.

What to capture (minimum)

- Context: goal of the change (e.g., calibrate better log-loss; reduce noisy overrides).
- Data: date, sources, whether externals were refreshed; last completed week.
- Gates: `--traingate`, `--testgate`, `--predictgate`.
- Config: triage thresholds/top-N; edge gating thresholds (pos/neg/abs/flip offset).
- Metrics: accuracy, mean log-loss, AUC; market baseline metrics for the same window.
- Overrides: flips vs. favorite, hit rate on flips, average edge.
- Calibration: before/after if applicable; method (e.g., Platt scaling).
- Decision: keep/discard; rationale.

Artifact pointers

- `data/models/training_metadata_<tag>.json` — summary written by 05_train_model.py
- `data/models/feature_importance_<tag>.csv` — feature gains
- `data/models/shap_summary_<tag>.csv` — SHAP mean |SHAP| by feature
- `data/models/calibration_<tag>.json` — optional Platt coefficients
- `logs/cv_results_<tag>.md` — top CV results (if search ran)
- `logs/predict_<tag>.md` — prediction log for a week
- `data/predictions/predictions_<tag>.csv` — per-game probabilities
- `data/predictions/ground_truth_<tag>.csv` — per-game outcomes + correctness

Template

```
### Run: <human-readable name>
- Date: <YYYY-MM-DD>
- Tag (raw → normalized): <raw> → <normalized>
- Goal: <why are we running this?>

Data
- Last completed week: <YYYY Ww>
- Externals updated: <yes/no>  (team_pbp_weekly.csv / injuries_weekly.csv / snap_counts_weekly.csv / betting_lines_weekly.csv)
- Feature file: data/normalized/training_features_normalized.csv  (fallback: data/features/training_features.csv)

Gates
- Train: <YYYYWw>
- Test: <YYYYWw>
- Predict: <YYYY Ww>

Config
- Triage: gain >= <x>, top-N <n> (or disabled)
- Edge gating: abs=<thresh>, pos=<thresh_pos>, neg=<thresh_neg>, max_flip_offset=<off>
- Seed: 42; Booster params: lr=<>, leaves=<>, feature_fraction=<>, bagging_fraction=<>, lambda_l2=<>, min_data_in_leaf=<> (if tuned)

Metrics
- Model: accuracy=<>, log-loss=<>, AUC=<>
- Market baseline (same window): accuracy=<>, log-loss=<>
- Calibration (if used): accuracy=<>, log-loss=<>, method=<sigmoid/Platt>

Overrides
- Flips vs. favorite: <count> / <total> (<>%)
- Hit rate on flips: <correct>/<flips> (<>%)
- Avg model edge on flips: <0.xx>

Artifacts
- Metadata: data/models/training_metadata_<normalized>.json
- Feature importance: data/models/feature_importance_<normalized>.csv
- SHAP summary: data/models/shap_summary_<normalized>.csv
- Predictions: data/predictions/predictions_<normalized>.csv
- Ground truth: data/predictions/ground_truth_<normalized>.csv
- Logs: logs/predict_<normalized>.md, logs/cv_results_<normalized>.md (if present)

Decision
- Keep/Discard: <one>
- Rationale: <why?>
```

Example (abbreviated)

```
### Run: Apple (calibrated)
- Date: 2025-10-28
- Tag: apple → apple
- Goal: Improve probability calibration vs. market

Data
- Last completed week: 2025 W7
- Externals updated: team pbp + injuries + lines

Gates
- Train/Test: 2025W7
- Predict: 2025 W8

Config
- Triage: gain >= 5.0, top-N 120
- Edge gating: abs=0.05, pos=0.10, neg=0.12, max_flip_offset=0.05

Metrics
- Model: acc=73.0%, log-loss=0.5711, AUC=0.78
- Market: acc=73.3%, log-loss=0.5737
- Calibration: improved log-loss after Platt scaling (sigmoid)

Overrides
- Flips: 11/272 (4.0%); hit=6/11 (54.5%); avg edge=0.12

Artifacts
- Metadata: data/models/training_metadata_apple.json
- Predictions: data/predictions/predictions_apple.csv

Decision
- Keep: better log-loss with negligible accuracy change; aligns with objectives.
```

Release criteria (suggested)

- Matches or exceeds market accuracy OR demonstrates meaningfully better log‑loss
- Stable across weeks with expected variance (see METRICS_GUIDE.md)
- Overrides gated; thin flips avoided without justification
- Reproducible: artifacts + config captured; random seed pinned

