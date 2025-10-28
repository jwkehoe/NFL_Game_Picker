#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_weekly_backtest.py
───────────────────────────────────────────────
Sequentially trains a residual LightGBM model starting with seasons 2021-2024
through Week 8, then walks forward week-by-week (2024 Week 9 through 2025 Week 8,
including postseason) to:

  • Train on all prior results (skipping preseason)
  • Predict the upcoming week
  • Record per-game outcomes and weekly accuracy
  • Append the new week to the training window and repeat

Reports:
  • data/reports/weekly_backtest_games.csv
    Columns: Season, Week, Home, Away, Predicted, Winner, Score, Confidence,
             Public Favorite, Betting Odds, Money Line, Spread, Win, Model Edge
  • data/reports/weekly_backtest_summary.csv
    Columns: Season, Week, Games_Evaluated, Correct, Incorrect, Ties,
             Percent_Correct

Logs:
  • logs/weekly_backtest_training.log
  • logs/weekly_backtest_metrics.log
"""

from __future__ import annotations

import logging
from pathlib import Path
import os
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from adjustment_utils import apply_edge_threshold
TRIAGE_GAIN_THRESHOLD = float(os.getenv("AUTO_NFL_TRIAGE_GAIN", "5.0"))
TRIAGE_TOP_N = int(os.getenv("AUTO_NFL_TRIAGE_TOP_N", "120"))
DATA_DIR = Path("data")
REPORT_DIR = DATA_DIR / "reports"
LOG_DIR = Path("logs")
FEATURES_PATH = DATA_DIR / "normalized" / "training_features_normalized.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_LOG_PATH = LOG_DIR / "weekly_backtest_training.log"
METRICS_LOG_PATH = LOG_DIR / "weekly_backtest_metrics.log"

START_TRAIN_CUTOFF = 2024 * 100 + 8  # 2024 Week 8
END_WEEK_ABS = 2025 * 100 + 8        # stop at 2025 Week 8
MIN_SEASON = 2021


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH)
    df.columns = [c.lower() for c in df.columns]
    df = df[df["season"] >= MIN_SEASON]
    df = df[df["week"] >= 1]  # skip preseason
    df["abs_week"] = df["season"] * 100 + df["week"]
    df = df[df["abs_week"] <= END_WEEK_ABS]
    df = df.sort_values(["abs_week", "home_team"]).reset_index(drop=True)
    return df


def compute_baseline(series: pd.Series, fallback: pd.Series) -> np.ndarray:
    return (
        series.fillna(fallback)
        .fillna(0.5)
        .clip(0.0, 1.0)
        .to_numpy()
    )

def prepare_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    drop_cols = {
        "target",
        "home_score",
        "away_score",
        "abs_week",
    }
    X_train = train_df.drop(columns=list(drop_cols), errors="ignore")
    X_test = test_df.drop(columns=list(drop_cols), errors="ignore")
    combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    cat_cols = combined.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))
    X_train_enc = combined.iloc[: len(X_train)].copy()
    X_test_enc = combined.iloc[len(X_train) :].copy()
    return X_train_enc, X_test_enc


def auto_triage_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    gain_threshold: float,
    top_n: int,
    seed: int,
) -> tuple[pd.Index, list[str]]:
    if X_train.empty:
        return X_train.columns, []

    booster = lgb.train(
        {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l2": 5.0,
            "seed": seed,
        },
        lgb.Dataset(X_train.fillna(0.0), label=y_train),
        num_boost_round=200,
    )
    gain = booster.feature_importance(importance_type="gain")
    gain_series = pd.Series(gain, index=X_train.columns, dtype=float)

    keep_mask = gain_series >= gain_threshold if gain_threshold > 0 else gain_series > 0
    keep_cols = gain_series[keep_mask].index
    if top_n and len(keep_cols) < min(top_n, len(gain_series)):
        keep_cols = gain_series.sort_values(ascending=False).head(top_n).index

    dropped = sorted(set(X_train.columns) - set(keep_cols))
    return pd.Index(keep_cols), dropped


def public_favorite(row: pd.Series) -> str:
    spread = row.get("spread_line")
    if pd.notna(spread):
        if spread < 0:
            return row["home_team"]
        if spread > 0:
            return row["away_team"]
    home_ml = row.get("home_moneyline")
    away_ml = row.get("away_moneyline")
    if pd.notna(home_ml) and pd.notna(away_ml):
        if home_ml < away_ml:
            return row["home_team"]
        if home_ml > away_ml:
            return row["away_team"]
    baseline = row.get("baseline_prob", 0.5)
    return row["home_team"] if baseline >= 0.5 else row["away_team"]


def format_moneyline(value) -> str:
    if pd.isna(value):
        return ""
    try:
        return f"{int(value):+d}"
    except Exception:
        return str(value)


def main():
    features = load_features()
    if features.empty:
        raise SystemExit("No features available for requested window.")

    train_log = logging.getLogger("train")
    metrics_log = logging.getLogger("metrics")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO)
    train_handler = logging.FileHandler(TRAINING_LOG_PATH, mode="w", encoding="utf-8")
    metrics_handler = logging.FileHandler(METRICS_LOG_PATH, mode="w", encoding="utf-8")
    train_handler.setFormatter(logging.Formatter("%(message)s"))
    metrics_handler.setFormatter(logging.Formatter("%(message)s"))
    train_log.addHandler(train_handler)
    metrics_log.addHandler(metrics_handler)
    train_log.propagate = False
    metrics_log.propagate = False

    target_weeks = sorted(
        features.loc[
            (features["abs_week"] > START_TRAIN_CUTOFF) & (features["abs_week"] <= END_WEEK_ABS),
            ["season", "week", "abs_week"],
        ].drop_duplicates("abs_week").itertuples(index=False, name=None),
        key=lambda x: x[2],
    )

    if not target_weeks:
        raise SystemExit("No weeks available beyond the initial training cutoff.")

    train_cutoff = START_TRAIN_CUTOFF
    game_records: List[dict] = []
    summary_records: List[dict] = []

    with tqdm(target_weeks, desc="Backtest", unit="week") as iterator:
        for season, week, target_abs in iterator:
            train_df = features[
                (features["abs_week"] <= train_cutoff)
                & features["target"].notna()
            ].copy()

            if train_df.empty:
                train_log.info(f"Skipping Week {season} W{week}: no training data.")
                train_cutoff = target_abs
                continue

            test_df = features[features["abs_week"] == target_abs].copy()
            if test_df.empty:
                train_log.info(f"Skipping Week {season} W{week}: no rows found.")
                train_cutoff = target_abs
                continue

            baseline_train = compute_baseline(train_df.get("baseline_prob"), train_df.get("win_prob"))
            baseline_test = compute_baseline(test_df.get("baseline_prob"), test_df.get("win_prob"))

            residual_train = train_df["target"].to_numpy() - baseline_train
            residual_test = test_df["target"].fillna(0.0).to_numpy() - baseline_test

            X_train, X_test = prepare_dataset(train_df, test_df)
            keep_cols, dropped_cols = auto_triage_features(
                X_train,
                residual_train,
                TRIAGE_GAIN_THRESHOLD,
                TRIAGE_TOP_N,
                seed=42,
            )
            if len(keep_cols) == 0:
                raise RuntimeError("Feature triage removed all columns. Adjust AUTO_NFL_TRIAGE_* settings.")
            if dropped_cols:
                train_log.info(
                    f"Week {season} W{week}: triage removed {len(dropped_cols)} columns "
                    f"(gainτ={TRIAGE_GAIN_THRESHOLD}, top_n={TRIAGE_TOP_N})"
                )
            X_train = X_train.reindex(columns=keep_cols)
            X_test = X_test.reindex(columns=keep_cols, fill_value=0.0)

            lgb_model = lgb.LGBMRegressor(
                objective="regression",
                learning_rate=0.07,
                num_leaves=31,
                max_depth=-1,
                n_estimators=500,
                subsample=0.8,
                colsample_bytree=0.9,
                bagging_freq=5,
                reg_lambda=5.0,
                min_child_samples=40,
                random_state=42,
            )

            lgb_model.fit(
                X_train,
                residual_train,
                eval_set=[(X_test, residual_test)],
                eval_metric="l2",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )

            gb_model = GradientBoostingRegressor(
                random_state=42,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
            )
            gb_model.fit(X_train.fillna(0.0), residual_train)

            best_iter = lgb_model.best_iteration_ or lgb_model.n_estimators
            residual_pred_lgb = lgb_model.predict(X_test, num_iteration=best_iter)
            residual_pred_gb = gb_model.predict(X_test.fillna(0.0))
            residual_pred = (residual_pred_lgb + residual_pred_gb) / 2.0
            raw_prob = baseline_test + residual_pred
            win_prob = apply_edge_threshold(raw_prob, baseline_test)

            test_df = test_df.reset_index(drop=True)
            test_df["predicted_prob"] = win_prob
            test_df["baseline_prob_effective"] = baseline_test
            test_df["predicted_winner"] = np.where(win_prob >= 0.5, test_df["home_team"], test_df["away_team"])

            actual_winner = np.where(
                test_df["home_score"].notna() & test_df["away_score"].notna(),
                np.where(
                    test_df["home_score"] > test_df["away_score"],
                    test_df["home_team"],
                    np.where(
                        test_df["home_score"] < test_df["away_score"],
                        test_df["away_team"],
                        "TIE",
                    ),
                ),
                "",
            )
            test_df["actual_winner"] = actual_winner

            mask_eval = actual_winner != ""
            if mask_eval.any():
                y_true = np.where(actual_winner[mask_eval] == test_df.loc[mask_eval, "home_team"], 1, 0)
                ll = log_loss(y_true, win_prob[mask_eval], labels=[0, 1])
                acc = accuracy_score(y_true, (win_prob[mask_eval] >= 0.5).astype(int))
            else:
                ll = float("nan")
                acc = float("nan")

            baseline_pred = test_df.loc[mask_eval, "baseline_prob_effective"].to_numpy()
            if mask_eval.any():
                baseline_ll = log_loss(y_true, baseline_pred.clip(1e-6, 1 - 1e-6), labels=[0, 1])
                baseline_acc = accuracy_score(y_true, (baseline_pred >= 0.5).astype(int))
            else:
                baseline_ll = float("nan")
                baseline_acc = float("nan")

            metrics_log.info(
                f"{season} W{week}: model_logloss={ll:.4f}, model_acc={acc:.3f}, "
                f"baseline_logloss={baseline_ll:.4f}, baseline_acc={baseline_acc:.3f}, "
                f"train_rows={len(train_df)}, test_rows={len(test_df)}"
            )

            for _, row in test_df.iterrows():
                score = ""
                if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
                    score = f"{int(row['home_score'])}-{int(row['away_score'])}"
                confidence = f"{abs(row['predicted_prob'] - 0.5) * 200:.1f}%"
                public_fav = public_favorite(row)
                money_line = ""
                if pd.notna(row.get("home_moneyline")) or pd.notna(row.get("away_moneyline")):
                    money_line = f"{row['home_team']} {format_moneyline(row.get('home_moneyline'))} / {row['away_team']} {format_moneyline(row.get('away_moneyline'))}"
                spread = ""
                if pd.notna(row.get("spread_line")):
                    spread = f"{row['spread_line']:+.1f}"

                win_flag = ""
                if row["actual_winner"] == "":
                    win_flag = ""
                elif row["actual_winner"] == "TIE":
                    win_flag = "T"
                elif row["predicted_winner"] == row["actual_winner"]:
                    win_flag = "Y"
                else:
                    win_flag = "N"

                game_records.append({
                    "Season": int(row["season"]),
                    "Week": int(row["week"]),
                    "Home": row["home_team"],
                    "Away": row["away_team"],
                    "Predicted": row["predicted_winner"],
                    "Winner": row["actual_winner"],
                    "Score": score,
                    "Confidence": confidence,
                    "Public Favorite": public_fav,
                    "Betting Odds": spread,
                    "Money Line": money_line,
                    "Spread": spread,
                    "Win": win_flag,
                    "Model Edge": f"{(row['predicted_prob'] - row['baseline_prob_effective']) * 100:.1f}%",
                    "Predicted Prob": row["predicted_prob"],
                    "Baseline Prob": row["baseline_prob_effective"],
                })

            evaluated = test_df[mask_eval]
            correct = (evaluated["predicted_winner"] == evaluated["actual_winner"]).sum()
            incorrect = ((evaluated["predicted_winner"] != evaluated["actual_winner"]) & (evaluated["actual_winner"] != "TIE")).sum()
            ties = (evaluated["actual_winner"] == "TIE").sum()
            games_eval = len(evaluated) - ties
            percent = f"{(correct / games_eval * 100):.1f}%" if games_eval > 0 else "nan"

            summary_records.append({
                "Season": season,
                "Week": week,
                "Games_Evaluated": games_eval,
                "Correct": int(correct),
                "Incorrect": int(incorrect),
                "Ties": int(ties),
                "Percent_Correct": percent,
                "Model_LogLoss": ll,
                "Baseline_LogLoss": baseline_ll,
            })

            train_log.info(
                f"Week {season} W{week}: train_rows={len(train_df)}, eval_rows={len(evaluated)}, "
                f"correct={correct}, incorrect={incorrect}, ties={ties}"
            )

            train_cutoff = target_abs

    games_df = pd.DataFrame(game_records)
    summary_df = pd.DataFrame(summary_records)

    game_report_path = REPORT_DIR / "weekly_backtest_games.csv"
    summary_report_path = REPORT_DIR / "weekly_backtest_summary.csv"

    games_df.to_csv(game_report_path, index=False)
    summary_df.to_csv(summary_report_path, index=False)

    print(f"✅ Game-level report → {game_report_path}")
    print(f"✅ Weekly summary report → {summary_report_path}")
    print(f"📝 Logs written to {TRAINING_LOG_PATH} and {METRICS_LOG_PATH}")


if __name__ == "__main__":
    main()
