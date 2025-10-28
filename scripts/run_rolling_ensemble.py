#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rolling_ensemble.py
───────────────────────────────────────────────
Purpose
  Research utility that trains two LightGBM regressors to predict home/away
  points separately, converts the point differential to a win probability,
  and ensembles it with a market baseline (if available).

What it does
  - Loads normalized features (data/normalized/training_features_normalized.csv)
  - For weeks 6–8 of CURRENT_SEASON, repeats:
      1) Train on the recent rolling window (TRAIN_WINDOW_SEASONS, TRAIN_WINDOW_WEEKS)
      2) Predict home/away points for the target week
      3) Convert points→win prob via logistic transform
      4) Ensemble with baseline prob (mean)
      5) Save per-week CSV to data/predictions/ and a summary log to logs/

When to use
  - Exploratory comparison vs. residual LightGBM (scripts/05_train_model.py)
  - Educational example of decomposing to points and reassembling to wins
  - Not a production path; treat outputs as research artifacts

Outputs
  - data/predictions/rolling_ensemble_week{6,7,8}.csv
  - logs/rolling_ensemble_summary.md

Notes
  - No CLI flags; tweak CURRENT_SEASON, PREDICTION_WEEKS and windows in-code
  - Baseline comes from feature column `baseline_prob` if present; else 0.5
  - This script does not apply the project’s edge gating thresholds; it’s meant
    for model comparison, not direct pick generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data")
FEATURE_PATH = DATA_DIR / "normalized" / "training_features_normalized.csv"
PRED_DIR = DATA_DIR / "predictions"
LOG_DIR = Path("logs")

PRED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CURRENT_SEASON = 2025
PREDICTION_WEEKS = [6, 7, 8]
TRAIN_WINDOW_SEASONS = 3
TRAIN_WINDOW_WEEKS = 16


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_PATH)
    df.columns = [c.lower() for c in df.columns]
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="coerce").astype(int)
    return df


def filter_training_data(df: pd.DataFrame, train_end_week: int) -> pd.DataFrame:
    min_season = CURRENT_SEASON - (TRAIN_WINDOW_SEASONS - 1)
    df = df[df["season"] >= min_season]

    # Convert to absolute week index across seasons to enforce rolling window in weeks
    df = df.copy()
    df["_abs_week"] = df["season"] * 100 + df["week"]
    cutoff_abs = CURRENT_SEASON * 100 + train_end_week
    cutoff_abs -= (TRAIN_WINDOW_WEEKS - 1)
    df = df[df["_abs_week"] <= CURRENT_SEASON * 100 + train_end_week]
    df = df[df["_abs_week"] >= cutoff_abs]
    df = df.drop(columns=["_abs_week"])
    return df


def encode_categoricals(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train_df, pred_df], axis=0, ignore_index=True)
    encoders: Dict[str, LabelEncoder] = {}
    for col in combined.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
        encoders[col] = le
    encoded_train = combined.iloc[: len(train_df)].copy()
    encoded_pred = combined.iloc[len(train_df) :].copy()
    return encoded_train, encoded_pred


@dataclass
class WeekResult:
    week: int
    accuracy: float
    logloss: float
    rows: int


def train_models(train_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Prepare encoded data
    feature_cols = [c for c in train_df.columns if c not in {"home_score", "away_score", "target"}]
    encoded_train, encoded_pred = encode_categoricals(train_df[feature_cols], pred_df[feature_cols])

    def fit_lgbm(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMRegressor:
        model = lgb.LGBMRegressor(
            objective="regression",
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            n_estimators=400,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X, y)
        return model

    home_model = fit_lgbm(encoded_train, train_df["home_score"])
    away_model = fit_lgbm(encoded_train, train_df["away_score"])
    pred_home = home_model.predict(encoded_pred)
    pred_away = away_model.predict(encoded_pred)
    return pred_home, pred_away


def points_to_win_prob(home_points: np.ndarray, away_points: np.ndarray, k: float = 7.0) -> np.ndarray:
    diff = home_points - away_points
    return 1.0 / (1.0 + np.exp(-diff / k))


def evaluate_week(pred_df: pd.DataFrame, results_path: Path) -> WeekResult:
    prob_col = "ensemble_win_prob"
    actual_df = pred_df.dropna(subset=["home_score", "away_score"]).copy()
    if actual_df.empty:
        pred_df.to_csv(results_path, index=False)
        return WeekResult(week=int(pred_df["week"].iloc[0]), accuracy=float("nan"), logloss=float("nan"), rows=len(pred_df))

    actual_df["actual_winner"] = np.where(
        actual_df["home_score"] > actual_df["away_score"],
        actual_df["home_team"],
        np.where(actual_df["home_score"] < actual_df["away_score"], actual_df["away_team"], "TIE"),
    )
    actual_df["predicted_winner"] = np.where(actual_df[prob_col] >= 0.5, actual_df["home_team"], actual_df["away_team"])
    mask = actual_df["actual_winner"] != "TIE"
    if mask.any():
        y_true_bool = actual_df.loc[mask, "actual_winner"] == actual_df.loc[mask, "home_team"]
        y_pred_bool = actual_df.loc[mask, prob_col] >= 0.5
        accuracy = accuracy_score(y_true_bool.astype(int), y_pred_bool.astype(int))
    else:
        accuracy = float("nan")
    y_true = np.where(actual_df["actual_winner"] == actual_df["home_team"], 1, 0)
    logloss = log_loss(y_true, actual_df[prob_col].clip(1e-6, 1 - 1e-6), labels=[0, 1])
    pred_df.to_csv(results_path, index=False)
    return WeekResult(week=int(pred_df["week"].iloc[0]), accuracy=accuracy, logloss=logloss, rows=len(pred_df))


def rolling_ensemble():
    features = load_features()
    summary: List[WeekResult] = []
    train_end_week = 5

    for target_week in PREDICTION_WEEKS:
        train_df = features[
            (features["season"] < CURRENT_SEASON)
            | ((features["season"] == CURRENT_SEASON) & (features["week"] <= train_end_week))
        ].copy()
        train_df = filter_training_data(train_df, train_end_week)
        pred_df = features[
            (features["season"] == CURRENT_SEASON) & (features["week"] == target_week)
        ].copy()

        pred_home, pred_away = train_models(train_df, pred_df)
        baseline = pred_df.get("baseline_prob", pd.Series(np.full(len(pred_df), np.nan))).fillna(0.5).to_numpy()
        win_prob = points_to_win_prob(pred_home, pred_away)
        ensemble_prob = np.clip((win_prob + baseline) / 2.0, 1e-6, 1 - 1e-6)

        output = pred_df[["season", "week", "home_team", "away_team", "home_score", "away_score"]].copy()
        output["pred_home_points"] = pred_home
        output["pred_away_points"] = pred_away
        output["market_baseline"] = baseline
        output["model_win_prob"] = win_prob
        output["ensemble_win_prob"] = ensemble_prob
        output["predicted_winner"] = np.where(
            ensemble_prob >= 0.5, output["home_team"], output["away_team"]
        )
        output["confidence"] = np.abs(ensemble_prob - 0.5) * 200

        result_path = PRED_DIR / f"rolling_ensemble_week{target_week}.csv"
        week_result = evaluate_week(output, result_path)
        summary.append(week_result)

        print(
            f"Week {target_week}: rows={week_result.rows}, "
            f"accuracy={week_result.accuracy:.3f}, logloss={week_result.logloss:.3f}"
        )

        train_end_week = target_week

    summary_log = LOG_DIR / "rolling_ensemble_summary.md"
    lines = [
        "# Rolling Ensemble Summary",
        "| Week | Rows | Accuracy | LogLoss |",
        "| --- | --- | --- | --- |",
    ]
    for res in summary:
        lines.append(f"| {res.week} | {res.rows} | {res.accuracy:.3f} | {res.logloss:.3f} |")
    summary_log.write_text("\n".join(lines), encoding="utf-8")
    print(f"📝 Summary log → {summary_log}")


if __name__ == "__main__":
    rolling_ensemble()
