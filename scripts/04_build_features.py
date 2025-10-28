#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_build_features.py
───────────────────────────────────────────────
Engineer model features using normalized game + Elo data.
Inputs:
  • data/normalized/nfl_games_all_normalized.csv
  • data/normalized/elo_history_normalized.csv
Outputs:
  • data/features/training_features.csv
  • data/normalized/training_features_normalized.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt

# --------------------------------------------------------------------- #
RAW_FILE = Path("data/normalized/nfl_games_all_normalized.csv")
ELO_FILE = Path("data/normalized/elo_history_normalized.csv")
OUT_FILE = Path("data/features/training_features.csv")
NORM_OUT_FILE = Path("data/normalized/training_features_normalized.csv")
Path("data/features").mkdir(parents=True, exist_ok=True)
NORM_OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[{timestamp}] ⚙️ Building features from normalized sources:")
print(f"  • Games → {RAW_FILE}")
print(f"  • Elo   → {ELO_FILE}")

# --------------------------------------------------------------------- #
try:
    games = pd.read_csv(RAW_FILE)
    elo = pd.read_csv(ELO_FILE)
    print(f"✅ Loaded {len(games):,} games, {len(elo):,} Elo records")
except Exception as e:
    print(f"❌ Failed to load source files: {e}")
    raise SystemExit(1)

# --- Normalize column names ---
games.columns = [c.lower() for c in games.columns]
elo.columns = [c.lower() for c in elo.columns]

# --- Normalize season, week, and numeric feature columns ---
for col in ["season", "week"]:
    games[col] = pd.to_numeric(games[col], errors="coerce").fillna(0).astype(int)
    elo[col] = pd.to_numeric(elo[col], errors="coerce").fillna(0).astype(int)

numeric_cols = [
    "away_rest", "home_rest", "away_moneyline", "home_moneyline", "spread_line",
    "away_spread_odds", "home_spread_odds", "total_line", "under_odds", "over_odds",
    "temp", "wind"
]
for col in numeric_cols:
    if col in games.columns:
        games[col] = pd.to_numeric(games[col], errors="coerce")

EXTERNAL_DIR = Path("data/external")
TEAM_PBP_PATH = EXTERNAL_DIR / "team_pbp_weekly.csv"
INJURIES_PATH = EXTERNAL_DIR / "injuries_weekly.csv"
SNAP_COUNTS_PATH = EXTERNAL_DIR / "snap_counts_weekly.csv"
BETTING_LINES_PATH = EXTERNAL_DIR / "betting_lines_weekly.csv"


def merge_team_features(base: pd.DataFrame, dataset: pd.DataFrame, prefix: str) -> pd.DataFrame:
    dataset.columns = [c.lower() for c in dataset.columns]
    if not {"season", "week", "team"}.issubset(dataset.columns):
        return base
    dataset = dataset.drop_duplicates(subset=["season", "week", "team"])
    value_cols = [c for c in dataset.columns if c not in {"season", "week", "team"}]
    dataset["team"] = dataset["team"].str.upper()

    def make_name(col: str, suffix: str) -> str:
        base_name = col
        if prefix and not col.startswith(prefix):
            base_name = f"{prefix}{col}"
        return f"{base_name}_{suffix}"

    home_df = dataset.rename(columns={"team": "home_team", **{c: make_name(c, "home") for c in value_cols}})
    away_df = dataset.rename(columns={"team": "away_team", **{c: make_name(c, "away") for c in value_cols}})

    base = base.merge(home_df, on=["season", "week", "home_team"], how="left")
    base = base.merge(away_df, on=["season", "week", "away_team"], how="left")
    return base


if TEAM_PBP_PATH.exists():
    pbp_df = pd.read_csv(TEAM_PBP_PATH)
    games = merge_team_features(games, pbp_df, prefix="pbp_")
else:
    print(f"ℹ️  External dataset not found: {TEAM_PBP_PATH}")

if INJURIES_PATH.exists():
    inj_df = pd.read_csv(INJURIES_PATH)
    games = merge_team_features(games, inj_df, prefix="inj_")
else:
    print(f"ℹ️  External dataset not found: {INJURIES_PATH}")

if SNAP_COUNTS_PATH.exists():
    snap_df = pd.read_csv(SNAP_COUNTS_PATH)
    games = merge_team_features(games, snap_df, prefix="snap_")
else:
    print(f"ℹ️  External dataset not found: {SNAP_COUNTS_PATH}")

if BETTING_LINES_PATH.exists():
    lines_df = pd.read_csv(BETTING_LINES_PATH)
    games = merge_team_features(games, lines_df, prefix="line_")
else:
    print(f"ℹ️  External dataset not found: {BETTING_LINES_PATH}")

def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive opponent-relative features (home minus away) for paired columns."""
    home_suffix = "_home"
    away_suffix = "_away"
    created = set()

    for col in df.columns:
        if not col.endswith(home_suffix):
            continue
        base = col[: -len(home_suffix)]
        away_col = f"{base}{away_suffix}"
        if away_col in df.columns and base not in created:
            df[f"{base}_diff"] = df[col] - df[away_col]
            created.add(base)
    return df


games = add_relative_features(games)

ADDITIONAL_FEATURE_COLUMNS = [
    col for col in games.columns
    if col.startswith(("pbp_", "inj_", "snap_", "line_", "baseline_prob", "market_prob", "market_edge"))
]

def moneyline_to_prob(value):
    if pd.isna(value):
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    if value > 0:
        return 100.0 / (value + 100.0)
    if value < 0:
        return -value / (-value + 100.0)
    return np.nan

# --- Helper to get prior Elo ---
def get_elo(team, season, week):
    mask = (elo["team"] == team) & (
        (elo["season"] < season) | ((elo["season"] == season) & (elo["week"] < week))
    )
    vals = elo.loc[mask, "elo_rating"]
    return vals.iloc[-1] if len(vals) > 0 else 1500.0

# --- Feature construction ---
rows = []
for _, g in games.iterrows():
    if pd.isna(g.get("home_score")) or pd.isna(g.get("away_score")):
        continue

    hs, as_ = g["home_score"], g["away_score"]
    h_team, a_team = g["home_team"], g["away_team"]
    s, w = g["season"], g["week"]

    h_elo = get_elo(h_team, s, w)
    a_elo = get_elo(a_team, s, w)
    elo_diff = h_elo - a_elo
    win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
    target = 1 if hs > as_ else 0

    home_rest = g.get("home_rest")
    away_rest = g.get("away_rest")
    rest_diff = np.nan
    if pd.notna(home_rest) or pd.notna(away_rest):
        rest_diff = (home_rest if pd.notna(home_rest) else 0) - (away_rest if pd.notna(away_rest) else 0)

    home_ml_prob = moneyline_to_prob(g.get("home_moneyline"))
    away_ml_prob = moneyline_to_prob(g.get("away_moneyline"))
    spread_line = g.get("spread_line") if "spread_line" in games.columns else np.nan
    total_line = g.get("total_line") if "total_line" in games.columns else np.nan
    temp = g.get("temp") if "temp" in games.columns else np.nan
    wind = g.get("wind") if "wind" in games.columns else np.nan
    div_game = g.get("div_game") if "div_game" in games.columns else np.nan

    market_prob_diff = np.nan
    if pd.notna(home_ml_prob) and pd.notna(away_ml_prob):
        market_prob_diff = home_ml_prob - away_ml_prob

    spread_abs = np.nan
    home_spread_favorite = np.nan
    if pd.notna(spread_line):
        spread_abs = abs(spread_line)
        home_spread_favorite = 1 if spread_line < 0 else 0

    baseline_prob = home_ml_prob if pd.notna(home_ml_prob) else win_prob
    market_edge = win_prob - baseline_prob if pd.notna(baseline_prob) else np.nan

    row_data = {
        "season": int(s),
        "week": int(w),
        "home_team": h_team,
        "away_team": a_team,
        "elo_home": h_elo,
        "elo_away": a_elo,
        "elo_diff": elo_diff,
        "win_prob": win_prob,
        "home_score": hs,
        "away_score": as_,
        "target": target,
        "home_rest": home_rest,
        "away_rest": away_rest,
        "rest_diff": rest_diff,
        "home_moneyline_prob": home_ml_prob,
        "away_moneyline_prob": away_ml_prob,
        "spread_line": spread_line,
        "spread_abs": spread_abs,
        "home_spread_favorite": home_spread_favorite,
        "total_line": total_line,
        "temp": temp,
        "wind": wind,
        "div_game": div_game,
        "market_prob_diff": market_prob_diff,
        "baseline_prob": baseline_prob,
        "market_edge": market_edge
    }
    for col in ADDITIONAL_FEATURE_COLUMNS:
        if col in g:
            row_data[col] = g[col]
    rows.append(row_data)

# --- Add future game rows for prediction ---
future_rows = []
future_games = games[
    (games["home_score"].isna()) & 
    (games["away_score"].isna()) & 
    (games["season"] > 0) & 
    (games["week"] > 0)
]
for _, g in future_games.iterrows():
    h_team, a_team = g["home_team"], g["away_team"]
    s, w = g["season"], g["week"]

    h_elo = get_elo(h_team, s, w)
    a_elo = get_elo(a_team, s, w)
    elo_diff = h_elo - a_elo
    win_prob = 1 / (1 + 10 ** (-elo_diff / 400))

    home_rest = g.get("home_rest")
    away_rest = g.get("away_rest")
    rest_diff = np.nan
    if pd.notna(home_rest) or pd.notna(away_rest):
        rest_diff = (home_rest if pd.notna(home_rest) else 0) - (away_rest if pd.notna(away_rest) else 0)

    home_ml_prob = moneyline_to_prob(g.get("home_moneyline"))
    away_ml_prob = moneyline_to_prob(g.get("away_moneyline"))
    spread_line = g.get("spread_line") if "spread_line" in games.columns else np.nan
    total_line = g.get("total_line") if "total_line" in games.columns else np.nan
    temp = g.get("temp") if "temp" in games.columns else np.nan
    wind = g.get("wind") if "wind" in games.columns else np.nan
    div_game = g.get("div_game") if "div_game" in games.columns else np.nan

    market_prob_diff = np.nan
    if pd.notna(home_ml_prob) and pd.notna(away_ml_prob):
        market_prob_diff = home_ml_prob - away_ml_prob

    spread_abs = np.nan
    home_spread_favorite = np.nan
    if pd.notna(spread_line):
        spread_abs = abs(spread_line)
        home_spread_favorite = 1 if spread_line < 0 else 0

    baseline_prob = home_ml_prob if pd.notna(home_ml_prob) else win_prob
    market_edge = win_prob - baseline_prob if pd.notna(baseline_prob) else np.nan

    future_data = {
        "season": int(s),
        "week": int(w),
        "home_team": h_team,
        "away_team": a_team,
        "elo_home": h_elo,
        "elo_away": a_elo,
        "elo_diff": elo_diff,
        "win_prob": win_prob,
        "home_score": np.nan,
        "away_score": np.nan,
        "target": np.nan,
        "home_rest": home_rest,
        "away_rest": away_rest,
        "rest_diff": rest_diff,
        "home_moneyline_prob": home_ml_prob,
        "away_moneyline_prob": away_ml_prob,
        "spread_line": spread_line,
        "spread_abs": spread_abs,
        "home_spread_favorite": home_spread_favorite,
        "total_line": total_line,
        "temp": temp,
        "wind": wind,
        "div_game": div_game,
        "market_prob_diff": market_prob_diff,
        "baseline_prob": baseline_prob,
        "market_edge": market_edge
    }
    for col in ADDITIONAL_FEATURE_COLUMNS:
        if col in g:
            future_data[col] = g[col]
    future_rows.append(future_data)

rows.extend(future_rows)

df_feat = pd.DataFrame(rows)

# --- Ensure correct types ---
df_feat["season"] = df_feat["season"].astype(int)
df_feat["week"] = df_feat["week"].astype(int)

feature_numeric_cols = [
    "home_rest", "away_rest", "rest_diff", "home_moneyline_prob", "away_moneyline_prob",
    "spread_line", "spread_abs", "home_spread_favorite", "total_line", "temp", "wind",
    "div_game", "market_prob_diff", "market_edge", "baseline_prob"
]
extra_numeric = [
    col for col in df_feat.columns
    if col.startswith(("pbp_", "inj_", "snap_", "line_", "baseline_prob_", "market_prob", "market_edge"))
]
feature_numeric_cols = list(dict.fromkeys(feature_numeric_cols + extra_numeric))
for col in feature_numeric_cols:
    if col in df_feat.columns:
        df_feat[col] = pd.to_numeric(df_feat[col], errors="coerce")

# --- Write output ---
df_feat.to_csv(OUT_FILE, index=False)
df_feat.to_csv(NORM_OUT_FILE, index=False)
print(f"✅ Features saved → {OUT_FILE} ({len(df_feat):,} rows)")
print(f"✅ Normalized feature copy → {NORM_OUT_FILE}")
print(f"ℹ️  Historical games: {len(rows) - len(future_rows):,}, Future games appended for prediction: {len(future_rows):,}")

log_path = Path("logs") / "build_features.md"
log_path.write_text(
    f"# Feature Build Log\n"
    f"- Timestamp: {timestamp}\n"
    f"- Input (games): {RAW_FILE}\n"
    f"- Input (Elo): {ELO_FILE}\n"
    f"- Outputs:\n"
    f"  • {OUT_FILE}\n"
    f"  • {NORM_OUT_FILE}\n"
    f"- Rows: {len(df_feat):,}\n"
    f"- Added numeric features include: moneyline-derived baseline/edge, play-by-play EPA (pbp_*),"
    f" injury aggregates (inj_*), snap usage (snap_*), and sportsbook consensus lines (line_*).\n"
)
print(f"📝 Log written → {log_path}")
