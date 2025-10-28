# Weekly Workflow

End‑to‑end steps to refresh data, train, predict, and evaluate.

—

1) Fetch and normalize data

```bash
python scripts/01_get_nfl_game.py
python scripts/02_normalize_data.py
python scripts/03_calculate_elo.py
python scripts/04_build_features.py
```

Outputs:
- `data/raw/nfl_games_all.csv`
- `data/normalized/*`
- `data/features/training_features.csv`

—

Optional: Update weekly external datasets

```bash
python scripts/update_weekly_external.py            # all seasons found in raw
python scripts/update_weekly_external.py --start-season 2018
```

This writes `data/external/*.csv` which 04_build_features.py merges automatically.

—

2) Train (time‑aware)

Choose a training/evaluation window using season/week “gates”.

```bash
# Example: train up to 2025W7, evaluate on 2025W7, prepare predictions for 2025 W8
python scripts/05_train_model.py \
  --mode train \
  --traingate 2025W7 \
  --testgate 2025W7 \
  --predictgate "2025 W8" \
  --tag apple
```

Artifacts land under `data/models/` (model text, ensemble member, calibration, metadata, feature importances, SHAP summary).

—

3) Predict for a week

```bash
python scripts/06_predict_games.py --predictgate "2025 W8" --model_tag apple
```

Outputs:
- `data/predictions/predictions_<tag>.csv`
- `data/predictions/easy_<tag>.csv`
- `logs/predict_<tag>.md`

—

4) Post‑game ground truth (after games finish)

```bash
python scripts/07_post_game_ground_truth.py --model_tag apple --gate "2025 W8"
```

Outputs:
- `data/predictions/ground_truth_<tag>.csv`
- `logs/ground_truth_<tag>.md`

—

Tips

- Use gates consistently: training/test `YYYYWw` (e.g. 2025W7), predict `"YYYY Ww"` (e.g. "2025 W8").
- For season‑wide simulation, see `scripts/run_weekly_backtest.py` which automates the walk‑forward loop and writes weekly metrics.
