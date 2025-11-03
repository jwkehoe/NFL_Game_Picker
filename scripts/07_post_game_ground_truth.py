#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_post_game_ground_truth.py
───────────────────────────────────────────────
Merge model predictions with actual game outcomes once scores are final.

Inputs:
  • data/predictions/predictions_<model_tag>.csv
  • data/normalized/nfl_games_all_normalized.csv

Outputs:
  • data/predictions/ground_truth_<model_tag>.csv
  • logs/ground_truth_<model_tag>.md
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
PREDICTIONS_DIR = DATA_DIR / "predictions"
NORMALIZED_GAMES = DATA_DIR / "normalized" / "nfl_games_all_normalized.csv"
LOG_DIR = Path("logs")

for directory in (PREDICTIONS_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_gate(value: str) -> tuple[int, int]:
    cleaned = value.replace("W", " ").replace("w", " ").strip().split()
    if len(cleaned) != 2:
        raise ValueError(f"Invalid gate format '{value}'. Expected e.g. '2025 W8'.")
    season, week = map(int, cleaned)
    return season, week


def compute_actual_winner(row: pd.Series) -> str:
    home = row.get("home_score")
    away = row.get("away_score")
    if pd.isna(home) or pd.isna(away):
        return ""
    if home > away:
        return row["home_team"]
    if home < away:
        return row["away_team"]
    return "TIE"


def load_predictions(model_tag: str) -> pd.DataFrame:
    pred_path = PREDICTIONS_DIR / f"predictions_{model_tag}.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")
    preds = pd.read_csv(pred_path)
    preds.columns = [c.lower() for c in preds.columns]
    return preds, pred_path


def load_games() -> pd.DataFrame:
    if not NORMALIZED_GAMES.exists():
        raise FileNotFoundError(
            f"Missing normalized games file: {NORMALIZED_GAMES}. Run 01–04 before posting ground truth."
        )
    games = pd.read_csv(NORMALIZED_GAMES)
    games.columns = [c.lower() for c in games.columns]
    for col in ("season", "week"):
        games[col] = pd.to_numeric(games[col], errors="coerce").astype("Int64")
    return games


def post_ground_truth(model_tag: str, gate: str) -> None:
    season, week = parse_gate(gate)

    preds, pred_path = load_predictions(model_tag)
    games = load_games()

    games_gate = games[
        (games["season"] == season)
        & (games["week"] == week)
    ].copy()

    join_cols = ["season", "week", "home_team", "away_team"]
    merged = pd.merge(preds, games_gate, on=join_cols, suffixes=("_pred", "_actual"), how="outer")

    if "home_score_actual" in merged.columns:
        merged = merged.rename(
            columns={
                "home_score_actual": "home_score",
                "away_score_actual": "away_score",
                "result_actual": "result",
                "total_actual": "total",
            }
        )
    if "home_score_pred" in merged.columns:
        merged = merged.rename(
            columns={
                "home_score_pred": "home_score_pred",
                "away_score_pred": "away_score_pred",
            }
        )

    # Identify unmatched rows
    missing_pred = merged[merged["win_prob"].isna()]
    if not missing_pred.empty:
        print("⚠️ Some completed games were not found in the predictions file:")
        print(missing_pred[join_cols + ["home_score", "away_score"]])
    merged = merged[merged["win_prob"].notna()].copy()

    merged["actual_winner"] = merged.apply(compute_actual_winner, axis=1)
    merged = merged[merged["actual_winner"] != ""].copy()
    if merged.empty:
        print(f"⚠️ No completed games with final scores found for {gate}.")
        return

    merged["predicted_correct"] = merged["predicted_winner"] == merged["actual_winner"]
    merged["predicted_prob"] = np.where(
        merged["predicted_winner"] == merged["home_team"],
        merged["win_prob"],
        1 - merged["win_prob"],
    )
    merged["win_margin"] = merged["home_score"] - merged["away_score"]

    if merged.empty:
        print(f"⚠️ Predictions file did not contain rows for {gate}. Nothing to evaluate.")
        return

    accuracy = merged["predicted_correct"].mean()
    avg_conf_correct = merged.loc[merged["predicted_correct"], "predicted_prob"].mean()
    avg_conf_incorrect = merged.loc[~merged["predicted_correct"], "predicted_prob"].mean()

    out_csv = PREDICTIONS_DIR / f"ground_truth_{model_tag}.csv"
    merged.sort_values(["season", "week", "home_team"]).to_csv(out_csv, index=False)

    log_path = LOG_DIR / f"ground_truth_{model_tag}.md"
    table_rows = [
        "| Home | Away | Predicted Winner | Actual Winner | Correct | Predicted Prob | Win Margin |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in merged.sort_values(["season", "week", "home_team"]).iterrows():
        table_rows.append(
            f"| {row['home_team']} | {row['away_team']} | {row['predicted_winner']} | "
            f"{row['actual_winner']} | {'✅' if row['predicted_correct'] else '❌'} | "
            f"{row['predicted_prob']:.3f} | {row['win_margin']:.1f} |"
        )

    log_content = "\n".join([
        "# Ground Truth Report",
        f"- Timestamp: {ts()}",
        f"- Model tag: {model_tag}",
        f"- Gate: {gate}",
        f"- Predictions source: {pred_path}",
        f"- Games evaluated: {len(merged):,}",
        f"- Accuracy: {accuracy:.3%}",
        f"- Avg predicted prob (correct): {avg_conf_correct:.3f}",
        f"- Avg predicted prob (incorrect): {avg_conf_incorrect if not np.isnan(avg_conf_incorrect) else float('nan'):.3f}",
        f"- Output CSV: {out_csv}",
        "",
        "## Game Outcomes",
        *table_rows,
        "",
    ])
    log_path.write_text(log_content, encoding="utf-8")

    print(f"✅ Ground truth merged for {gate} — {len(merged)} games")
    print(f"🎯 Accuracy: {accuracy:.3%}")
    print(f"📈 Avg predicted prob (correct): {avg_conf_correct:.3f}")
    if not np.isnan(avg_conf_incorrect):
        print(f"📉 Avg predicted prob (incorrect): {avg_conf_incorrect:.3f}")
    print(f"🧾 Ground truth CSV → {out_csv}")
    print(f"📝 Report → {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge predictions with actual scores once games have finished.")
    parser.add_argument("--model_tag", required=True, help="Model tag used in predictions file (e.g. 2025w7_to_2025w8)")
    parser.add_argument("--gate", required=True, help="Season/week gate, e.g. '2025 W8'")
    args = parser.parse_args()

    post_ground_truth(model_tag=args.model_tag, gate=args.gate)


if __name__ == "__main__":
    main()
