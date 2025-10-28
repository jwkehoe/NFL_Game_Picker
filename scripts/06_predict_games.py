#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_predict_games.py
───────────────────────────────────────────────
Use a trained LightGBM model to predict NFL game outcomes for a given week.

Inputs:
  • data/models/<model_tag>.txt
  • data/normalized/training_features_normalized.csv

Outputs:
  • data/predictions/predictions_<model_tag>.csv
  • logs/predict_<model_tag>.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import load

from adjustment_utils import apply_edge_threshold

# ------------------------------------------------------------
# Setup
DATA_DIR = Path("data")
FEATURES_PATH = DATA_DIR / "normalized" / "training_features_normalized.csv"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
LOG_DIR = Path("logs")

for d in (PREDICTIONS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_model(model_path: Path):
    print(f"📘 Loading model → {model_path}")
    booster = lgb.Booster(model_file=str(model_path))
    expected_cols = booster.feature_name()
    print(f"📐 Loaded feature schema ({len(expected_cols)} columns).")
    return booster, expected_cols

def prepare_features(df, expected_cols):
    """Ensure feature compatibility with model schema, including categorical columns."""
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
    extra = [c for c in df.columns if c not in expected_cols]
    if extra:
        df = df.drop(columns=extra)
    df = df[expected_cols]

    # Convert categorical columns to 'category' dtype if any
    # Assuming categorical columns are those with dtype 'object' or 'category'
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')

    return df


def encode_categoricals(df: pd.DataFrame):
    """Mirror training-time categorical encoding (object → category codes)."""
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        print(f"⚙️ Encoding categorical columns for inference: {list(cat_cols)}")
        for c in cat_cols:
            df[c] = df[c].astype("category").cat.codes
    return df

# ------------------------------------------------------------
def predict_games(args):
    model_path = MODELS_DIR / f"nfl_model_{args.model_tag}.txt"
    gb_model_path = MODELS_DIR / f"gb_model_{args.model_tag}.joblib"
    calibration_path = MODELS_DIR / f"calibration_{args.model_tag}.json"
    booster, expected_cols = load_model(model_path)

    gb_model = None
    if gb_model_path.exists():
        print(f"📘 Loading ensemble member → {gb_model_path}")
        gb_model = load(gb_model_path)
    else:
        print("⚠️ Gradient boosting ensemble member not found; falling back to LightGBM only.")

    calibration = None
    if calibration_path.exists():
        with open(calibration_path, "r", encoding="utf-8") as f:
            calibration = json.load(f)
        if {"coef", "intercept"} <= calibration.keys():
            print(f"🧮 Calibration loaded → {calibration_path}")
        else:
            calibration = None

    features_path = FEATURES_PATH
    legacy_features = DATA_DIR / "features" / "training_features.csv"
    if not features_path.exists():
        if legacy_features.exists():
            print(f"⚠️ Normalized features missing at {features_path}. Falling back to {legacy_features}.")
            features_path = legacy_features
        else:
            raise FileNotFoundError(
                "Missing normalized training features. Run 04_build_features.py before predicting."
            )

    print(f"📘 Loading features → {features_path}")
    df_raw = pd.read_csv(features_path)
    df = encode_categoricals(df_raw.copy())
    df = df.dropna(subset=["season", "week"])
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype(int)
    df_raw = df_raw.dropna(subset=["season", "week"])
    df_raw["season"] = pd.to_numeric(df_raw["season"], errors="coerce").astype(int)
    df_raw["week"] = pd.to_numeric(df_raw["week"], errors="coerce").astype(int)

    # Parse gate like "2025 W8"
    try:
        season_str, week_str = args.predictgate.strip().split()
        season = int(season_str)
        week = int(week_str.strip("Ww"))
    except Exception:
        raise ValueError(f"Invalid predictgate format: {args.predictgate}")

    df_pred = df[(df["season"] == season) & (df["week"] == week)]
    df_pred_raw = df_raw[(df_raw["season"] == season) & (df_raw["week"] == week)]
    if df_pred.empty:
        print(f"⚠️ No rows found for {args.predictgate}. Have you run build_features for that week?")
        print("❌ No rows to predict after filtering. Exiting gracefully.")
        return

    # Keep identification columns for output
    meta_cols = ["season", "week", "home_team", "away_team", "home_score", "away_score", "target"]
    available_meta = [c for c in meta_cols if c in df_pred_raw.columns]
    meta_df = df_pred_raw[available_meta].copy()

    if "baseline_prob" in df_pred:
        baseline = df_pred["baseline_prob"].astype(float)
    else:
        baseline = (
            df_pred.get("home_moneyline_prob", pd.Series([], dtype=float))
            .fillna(df_pred.get("win_prob"))
            .fillna(0.5)
        )
    baseline = baseline.reindex(df_pred.index).fillna(0.5).clip(0, 1)
    meta_df["baseline_prob"] = baseline.values

    feature_df = prepare_features(df_pred, expected_cols)
    residual_preds_lgb = booster.predict(feature_df)
    if gb_model is not None:
        residual_preds_gb = gb_model.predict(feature_df.fillna(0.0))
        residual_preds = (residual_preds_lgb + residual_preds_gb) / 2.0
    else:
        residual_preds = residual_preds_lgb

    raw_preds = baseline.values + residual_preds
    preds = apply_edge_threshold(raw_preds, baseline.values)
    if calibration and {"coef", "intercept"} <= calibration.keys():
        coef = calibration["coef"]
        intercept = calibration["intercept"]
        preds = 1.0 / (1.0 + np.exp(-(coef * preds + intercept)))
        preds = np.clip(preds, 1e-6, 1 - 1e-6)
    meta_df["win_prob"] = preds
    meta_df["model_edge"] = meta_df["win_prob"] - meta_df["baseline_prob"]
    meta_df["predicted_winner"] = meta_df.apply(
        lambda r: r["home_team"] if r["win_prob"] >= 0.5 else r["away_team"], axis=1
    )
    meta_df["confidence"] = (meta_df["win_prob"] - 0.5).abs() * 200

    # Output file paths
    out_csv = PREDICTIONS_DIR / f"predictions_{args.model_tag}.csv"
    easy_csv = PREDICTIONS_DIR / f"easy_{args.model_tag}.csv"
    out_md = LOG_DIR / f"predict_{args.model_tag}.md"

    meta_df.to_csv(out_csv, index=False)

    easy_df = meta_df[["home_team", "away_team", "confidence", "predicted_winner", "home_score", "away_score"]]
    easy_df.to_csv(easy_csv, index=False)

    log = f"""# Prediction Log
- Timestamp: {ts()}
- Model: {model_path}
- Features: {features_path}
- Calibration: {calibration_path if calibration_path.exists() else 'None'}
- Predict gate: {args.predictgate}
- Rows predicted: {len(meta_df):,}
- Output: {out_csv}
"""
    out_md.write_text(log, encoding="utf-8")

    print(f"🔮 Predicted {len(meta_df):,} games for {args.predictgate}")
    print(f"✅ Predictions saved → {out_csv}")
    print(f"🧾 Easy view saved → {easy_csv}")
    print(f"📝 Log → {out_md}")

# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict NFL games with trained model")
    parser.add_argument("--predictgate", required=True, help="e.g. '2025 W8'")
    parser.add_argument("--model_tag", required=True, help="Model tag, e.g. 2025_w7_2025_w8_na")

    # If invoked without parameters, print options and examples rather than running blindly
    import sys
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/06_predict_games.py --predictgate '2025 W8' --model_tag 2025_w7_2025_w7_2025_w8")
        print("  python scripts/06_predict_games.py --predictgate '2025 W7' --model_tag apple")
        print("  python scripts/06_predict_games.py --predictgate '2024 W17' --model_tag 2024w9_to_2024w16")
        sys.exit(0)

    args = parser.parse_args()
    predict_games(args)
