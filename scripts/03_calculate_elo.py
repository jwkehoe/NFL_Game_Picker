#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_calculate_elo.py
───────────────────────────────────────────────
Compute rolling Elo ratings from game outcomes.
Input : data/normalized/nfl_games_all_normalized.csv
Outputs:
  • data/processed/elo_history.csv
  • data/normalized/elo_history_normalized.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt

# --------------------------------------------------------------------- #
RAW_FILE = Path("data/normalized/nfl_games_all_normalized.csv")
OUT_FILE = Path("data/processed/elo_history.csv")
NORM_OUT_FILE = Path("data/normalized/elo_history_normalized.csv")
Path("data/processed").mkdir(parents=True, exist_ok=True)
NORM_OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[{timestamp}] 📘 Loading normalized games → {RAW_FILE}")
try:
    games = pd.read_csv(RAW_FILE)
    print(f"✅ Loaded {len(games):,} rows")
except Exception as e:
    print(f"❌ Failed to read {RAW_FILE}: {e}")
    raise SystemExit(1)

# --- Basic cleanup ---
games.columns = [c.lower() for c in games.columns]
required = ["season", "week", "home_team", "away_team", "home_score", "away_score"]
for col in required:
    if col not in games.columns:
        raise ValueError(f"Missing required column: {col}")

games = games.dropna(subset=["home_team", "away_team", "home_score", "away_score"])
games = games[(games["home_score"] + games["away_score"]) > 0]
games["season"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
games["week"] = pd.to_numeric(games["week"], errors="coerce").astype("Int64")

# --- Elo parameters ---
BASE, K, HOME_ADV = 1500, 20, 65

teams = pd.unique(games[["home_team", "away_team"]].values.ravel("K"))
elo = {t: BASE for t in teams}
elo_records = []

# --- Elo computation ---
games = games.sort_values(["season", "week", "game_date"], na_position="last")

for _, g in games.iterrows():
    h, a = g["home_team"], g["away_team"]
    hs, as_ = g["home_score"], g["away_score"]

    if pd.isna(hs) or pd.isna(as_):
        continue

    Rh, Ra = elo.get(h, BASE), elo.get(a, BASE)
    Eh = 1 / (1 + 10 ** ((Ra - Rh - HOME_ADV) / 400))
    Ea = 1 - Eh

    if hs > as_:
        Sh, Sa = 1, 0
    elif hs < as_:
        Sh, Sa = 0, 1
    else:
        Sh, Sa = 0.5, 0.5

    elo[h] = Rh + K * (Sh - Eh)
    elo[a] = Ra + K * (Sa - Ea)

    elo_records.append({
        "team": h,
        "season": g["season"],
        "week": g["week"],
        "elo_rating": elo[h],
        "opponent": a,
        "is_home": 1
    })
    elo_records.append({
        "team": a,
        "season": g["season"],
        "week": g["week"],
        "elo_rating": elo[a],
        "opponent": h,
        "is_home": 0
    })

df_elo = pd.DataFrame(elo_records)
df_elo["season"] = df_elo["season"].astype(int)
df_elo["week"] = df_elo["week"].astype(int)

# --- Output ---
df_elo.to_csv(OUT_FILE, index=False)
df_elo.to_csv(NORM_OUT_FILE, index=False)
print(f"✅ Elo history saved → {OUT_FILE} ({len(df_elo):,} rows)")
print(f"✅ Normalized Elo copy → {NORM_OUT_FILE}")

log_path = Path("logs") / "calculate_elo.md"
log_path.write_text(
    f"# Elo Calculation Log\n"
    f"- Timestamp: {timestamp}\n"
    f"- Input: {RAW_FILE}\n"
    f"- Outputs:\n"
    f"  • {OUT_FILE}\n"
    f"  • {NORM_OUT_FILE}\n"
    f"- Rows: {len(df_elo):,}\n"
)
print(f"📝 Log written → {log_path}")
