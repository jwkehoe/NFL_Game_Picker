Command-line flags for the main scripts in this repo.

These reflect the current 01–07 pipeline scripts and helpers.

—

scripts/01_get_nfl_game.py

- No CLI flags. Uses env var `AUTO_NFL_USE_NFLDATAPY=1` to prefer nfl_data_py for schedules.
- Outputs: `data/raw/nfl_games_all.csv`, `logs/get_nfl_games.md`

Examples:
- `AUTO_NFL_USE_NFLDATAPY=0 python scripts/01_get_nfl_game.py`

—

scripts/02_normalize_data.py

- No CLI flags. Normalizes raw/processed/features into `data/normalized/*`.
- Inputs: `data/raw/nfl_games_all.csv`, `data/processed/elo_history.csv`, `data/features/training_features.csv`
- Outputs: `data/normalized/nfl_games_all_normalized.csv`, `data/normalized/elo_history_normalized.csv`, `data/normalized/training_features_normalized.csv`

Examples:
- `python scripts/02_normalize_data.py`

—

scripts/03_calculate_elo.py

- No CLI flags. Computes Elo from normalized games.
- Params (in-code): BASE=1500, K=20, HOME_ADV=65
- Inputs: `data/normalized/nfl_games_all_normalized.csv`
- Outputs: `data/processed/elo_history.csv`, `data/normalized/elo_history_normalized.csv`

Examples:
- `python scripts/03_calculate_elo.py`

—

scripts/04_build_features.py

- No CLI flags. Builds features from normalized games + Elo and appends rows for scheduled future games.
- Inputs: `data/normalized/nfl_games_all_normalized.csv`, `data/normalized/elo_history_normalized.csv`
- Outputs: `data/features/training_features.csv`, `data/normalized/training_features_normalized.csv`

Examples:
- `python scripts/04_build_features.py`

—

scripts/05_train_model.py

Flags:
- `--mode train` (optional)
- `--traingate` `YYYYWw` e.g. `2024W16`
- `--testgate` `YYYYWw` e.g. `2024W17`
- `--predictgate` `'YYYY Ww'` e.g. `'2024 W18'` (if omitted, infers next week)
- `--learning_rate` float (default 0.05)
- `--num_leaves` int (default 31)
- `--max_depth` int (default -1)
- `--num_boost_round` int (default 1000)
- `--early_stopping` int (default 50)
- `--bagging_fraction` float (default 0.8)
- `--feature_fraction` float (default 0.9)
- `--seed` int (default 42)
- `--disable_triage` (flag) skip feature triage
- `--feature_triage_gain` float (default 5.0)
- `--feature_triage_top_n` int (default 120)
- `--tag` str (used in output filenames)
- `--debug` (flag)
- `--rolling_seasons` int (restrict training window)
- `--rolling_weeks` int (restrict training window)

Notes:
- Reads features from `data/normalized/training_features_normalized.csv` or falls back to `data/features/training_features.csv`.
- Saves artifacts to `data/models/`.

Examples:
- `python scripts/05_train_model.py --mode train --traingate 2025W7 --testgate 2025W7 --predictgate '2025 W8' --tag apple`
- `python scripts/05_train_model.py --mode train --rolling_weeks 200 --testgate 2025W7 --predictgate '2025 W8' --tag wk200`

—

scripts/06_predict_games.py

Flags:
- `--predictgate` `'YYYY Ww'` (required)
- `--model_tag` tag used in training outputs (required)

Notes:
- Loads `data/models/nfl_model_<tag>.txt` (+ optional ensemble `gb_model_<tag>.joblib` and `calibration_<tag>.json`).
- Reads features from `data/normalized/training_features_normalized.csv` or falls back to `data/features/training_features.csv`.
- Outputs CSV under `data/predictions/` and a log in `logs/`.

Examples:
- `python scripts/06_predict_games.py --predictgate '2025 W8' --model_tag apple`

—

scripts/07_post_game_ground_truth.py

Flags:
- `--model_tag` tag used in predictions file (required)
- `--gate` `'YYYY Ww'` (required)

Notes:
- Joins predictions with final scores to compute accuracy and outputs a per-game CSV + Markdown summary.

Examples:
- `python scripts/07_post_game_ground_truth.py --model_tag apple --gate '2025 W8'`

—

Other

- `scripts/run_weekly_backtest.py` has no CLI flags; it writes summary CSVs and logs for a season‑long walk‑forward backtest using defaults defined in the script and environment variables (see source).

—

Utilities

- scripts/run_master_pipeline.sh
  - Full 01→07 flow with gates and optional rolling windows.
  - Flags: `--traingate YYYYWk`, `--testgate YYYYWk`, `--predictgate YYYYWk`, `--rolling-seasons N`, `--rolling-weeks N`, `--skip-ground-truth`.
  - Example: `bash scripts/run_master_pipeline.sh --traingate 2025W7 --testgate 2025W7 --predictgate 2025W8`

- scripts/example_predict_2025w8.sh
  - Functional example to train + predict for 2025 Week 8 using the standard pipeline.
  - Prefer run_master_pipeline.sh for general use.

- scripts/run_rolling_ensemble.py
  - Research utility: trains separate home/away point regressors and ensembles with baseline to form win probs.
  - No flags; tweak CURRENT_SEASON/PREDICTION_WEEKS in file.
  - Outputs: `data/predictions/rolling_ensemble_week{6,7,8}.csv` and `logs/rolling_ensemble_summary.md`.

- scripts/update_weekly_external.py
  - Fetch external datasets for merging into features.
  - Flags: `--start-season INT`, `--end-season INT`, `--season-types REG,POST`.
  - Env: `AUTO_NFL_USE_NFLDATAPY=0` to disable (writes placeholders).
  - Outputs: CSVs in `data/external/` used by 04_build_features.py.

- scripts/nfldatapy_client.py
  - Internal library wrapper around nfl_data_py used by other scripts.
  - Optional via `AUTO_NFL_USE_NFLDATAPY`; raises `NflDataPyUnavailable` if disabled or missing.
  - See module docstring for details on requests and normalized outputs.
