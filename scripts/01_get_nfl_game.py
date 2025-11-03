#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_get_nfl_game.py
───────────────────────────────────────────────
Fetch and normalize all NFL games (historic) from nflverse,
then (if missing) append the 2025 regular-season schedule
from real public sources (no synthetic rows):

  0) Optional: use nfl_data_py (set AUTO_NFL_USE_NFLDATAPY=1)
  1) BigDataBall CSV
  2) FixtureDownload JSON feed
  3) TheSportsDB JSON (public demo key)

Outputs:
  • data/raw/nfl_games_all.csv
  • logs/get_nfl_games.md
"""

import datetime as dt
import io
import json
from pathlib import Path
from typing import List

import pandas as pd
import urllib.request

from nfldatapy_client import (
    NflDataPyUnavailable,
    ScheduleRequest,
    fetch_schedules,
    should_use_nfldatapy,
)

# --------------------------------------------------------------------- #
# Paths
DATA_DIR = Path("data/raw"); DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
OUT_CSV = DATA_DIR / "nfl_games_all.csv"

# --------------------------------------------------------------------- #
# Helpers
def ts():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _ua_request(url: str, timeout: int = 90):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AutoNFL/1.0 (+https://example.com) Python-urllib"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def normalize_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    # lower-case columns, coerce season/week, drop rows missing keys
    df.columns = [c.lower() for c in df.columns]
    if "gameday" in df.columns:
        df = df.rename(columns={"gameday": "game_date"})
    if "gametime" in df.columns:
        df = df.rename(columns={"gametime": "kickoff_time"})
    for c in ("season", "week"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    # ensure core columns exist
    for c in ("home_team", "away_team"):
        if c not in df.columns:
            df[c] = pd.NA
    df = df.dropna(subset=["season", "week", "home_team", "away_team"])
    return df

def make_game_id(row) -> str:
    return f"{row['season']}_{str(row['week']).zfill(2)}_{row['away_team']}_{row['home_team']}"

def pad_2025_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Add columns your downstream expects but these feeds won’t have
    cols_defaults = {
        "home_score": pd.NA,
        "away_score": pd.NA,
        "game_type": "REG",
        "weekday": "",
        "kickoff_time": "",
        "location": "",
        "result": pd.NA,
        "total": pd.NA,
        "overtime": 0,
        "roof": "",
        "surface": "",
        "temp": pd.NA,
        "wind": pd.NA,
    }
    for c, v in cols_defaults.items():
        if c not in df.columns:
            df[c] = v
    # enforce dtypes
    for c in ("home_score", "away_score", "result", "total", "temp", "wind"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def append_if_missing_2025(base_df: pd.DataFrame, df_2025: pd.DataFrame) -> pd.DataFrame:
    if base_df.loc[base_df["season"] == 2025].empty and not df_2025.empty:
        df_2025 = df_2025.copy()
        df_2025["game_id"] = df_2025.apply(make_game_id, axis=1)
        # avoid dupes if any overlap
        combined = pd.concat([base_df, df_2025], ignore_index=True)
        combined.drop_duplicates(subset=["game_id"], inplace=True)
        return combined
    return base_df

# --------------------------------------------------------------------- #
# 1) Primary fetch via nfl_data_py (if available/enabled)
df_all = None
nfldatapy_enabled = should_use_nfldatapy(default=True)
base_source = None

if nfldatapy_enabled:
    current_year = dt.datetime.now().year
    seasons: List[int] = list(range(1999, current_year + 1))
    print(
        f"[{ts()}] 🏈 Fetching NFL schedules via nfl_data_py "
        f"(seasons {seasons[0]}-{seasons[-1]})"
    )
    try:
        df_all = fetch_schedules(
            ScheduleRequest(
                seasons=seasons,
            )
        )
        df_all = normalize_basic_columns(df_all)
        print(f"✅ Loaded {len(df_all):,} rows via nfl_data_py")
        base_source = f"nfl_data_py schedules ({seasons[0]}-{seasons[-1]})"
    except NflDataPyUnavailable as err:
        print(f"⚠️ nfl_data_py unavailable: {err}")
    except Exception as err:  # pragma: no cover - network runtime
        print(f"⚠️ nfl_data_py fetch failed ({err}); falling back to nflverse CSV")
else:
    print("ℹ️ AUTO_NFL_USE_NFLDATAPY=0 → skipping nfl_data_py fetch.")

# --------------------------------------------------------------------- #
# 2) Fallback to historic data from nflverse
if df_all is None:
    URL_NFLVERSE = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    print(f"[{ts()}] 🏈 Fetching NFL data from {URL_NFLVERSE}")

    try:
        raw = _ua_request(URL_NFLVERSE, timeout=90)
        df_all = pd.read_csv(io.BytesIO(raw), low_memory=False)
        df_all = normalize_basic_columns(df_all)
        print(f"✅ Loaded {len(df_all):,} rows from source")
        base_source = URL_NFLVERSE
    except Exception as e:
        print(f"❌ Failed to fetch NFL data: {e}")
        raise SystemExit(1)

# --------------------------------------------------------------------- #
# Early write (so downstream can peek even if 2025 fails)
df_all.to_csv(OUT_CSV, index=False)

# --------------------------------------------------------------------- #
# If 2025 is already present from nflverse, we’re done
if not df_all.loc[df_all["season"] == 2025].empty:
    print("ℹ️  2025 rows already present in nflverse dump; skipping schedule fallbacks.")
    print(f"🗂️  Saved → {OUT_CSV}")
    print(f"📝  Log   → {LOG_DIR / 'get_nfl_games.md'}")
else:
    # ----------------------------------------------------------------- #
    # 2) Try BigDataBall CSV (CSV often behind a pay/login, may 404)
    try:
        url_bdb = "https://www.bigdataball.com/downloads/nfl-2025-schedule.csv"
        print(f"📅 Attempting to fetch 2025 schedule from {url_bdb}")
        df_bdb = pd.read_csv(url_bdb)
        df_bdb = df_bdb.rename(columns={
            "Week": "week",
            "Date": "game_date",
            "Home Team": "home_team",
            "Away Team": "away_team"
        })
        df_bdb["season"] = 2025
        df_bdb = normalize_basic_columns(df_bdb)
        df_bdb = pad_2025_columns(df_bdb)
        df_all = append_if_missing_2025(df_all, df_bdb)
        df_all.to_csv(OUT_CSV, index=False)
        print(f"✅ Appended 2025 schedule — new total {len(df_all):,} rows")
    except Exception as e:
        print(f"⚠️ Could not fetch or append 2025 schedule (BigDataBall): {e}")

    # ----------------------------------------------------------------- #
    # 3) Fallback #1: FixtureDownload JSON feed
    if df_all.loc[df_all["season"] == 2025].empty:
        try:
            print("🔄 Attempting fallback fetch from FixtureDownload.com")
            # Example feed: https://fixturedownload.com/feed/json/nfl-2025
            fd_bytes = _ua_request("https://fixturedownload.com/feed/json/nfl-2025", timeout=90)
            fd_json = json.loads(fd_bytes.decode("utf-8"))

            # Convert list[dict] → DataFrame; expected keys include
            # "Week", "Date", "HomeTeam", "AwayTeam"
            df_fd = pd.DataFrame(fd_json)
            rename_map = {
                "Week": "week",
                "RoundNumber": "week",   # sometimes used
                "Date": "game_date",
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
            }
            df_fd = df_fd.rename(columns=rename_map)
            # prefer "Week", fallback to "RoundNumber"
            if "week" not in df_fd.columns and "RoundNumber" in df_fd.columns:
                df_fd["week"] = df_fd["RoundNumber"]

            df_fd["season"] = 2025
            df_fd = normalize_basic_columns(df_fd)
            df_fd = pad_2025_columns(df_fd)
            df_all = append_if_missing_2025(df_all, df_fd)
            df_all.to_csv(OUT_CSV, index=False)
            print(f"✅ Appended 2025 schedule from FixtureDownload — new total {len(df_all):,} rows")
        except Exception as e:
            print(f"⚠️ Could not fetch or append 2025 schedule (FixtureDownload): {e}")

    # ----------------------------------------------------------------- #
    # 4) Fallback #2: TheSportsDB public JSON (demo key)
    # Docs: https://www.thesportsdb.com/api.php
    # NFL LeagueID commonly 4391; season string "2025"
    if df_all.loc[df_all["season"] == 2025].empty:
        try:
            print("🔄 Attempting fallback fetch from TheSportsDB")
            tsdb_url = "https://www.thesportsdb.com/api/v1/json/1/eventsseason.php?id=4391&s=2025"
            tsdb_bytes = _ua_request(tsdb_url, timeout=90)
            tsdb = json.loads(tsdb_bytes.decode("utf-8"))
            events = tsdb.get("events") or []
            if events:
                df_ts = pd.DataFrame(events)
                # Typical keys: dateEvent, strTime, strHomeTeam, strAwayTeam, intRound
                df_ts = df_ts.rename(columns={
                    "dateEvent": "game_date",
                    "strTime": "kickoff_time",
                    "strHomeTeam": "home_team",
                    "strAwayTeam": "away_team",
                    "intRound": "week",
                })
                df_ts["season"] = 2025
                df_ts = normalize_basic_columns(df_ts)
                df_ts = pad_2025_columns(df_ts)
                df_all = append_if_missing_2025(df_all, df_ts)
                df_all.to_csv(OUT_CSV, index=False)
                print(f"✅ Appended 2025 schedule from TheSportsDB — new total {len(df_all):,} rows")
            else:
                print("ℹ️ TheSportsDB returned no events for 2025.")
        except Exception as e:
            print(f"⚠️ Could not fetch or append 2025 schedule (TheSportsDB): {e}")

# --------------------------------------------------------------------- #
# Log file
log_path = LOG_DIR / "get_nfl_games.md"
base_source_label = base_source or "nfl_data_py (disabled?)"
log_path.write_text(
    f"# NFL Games Ingestion Log\n"
    f"- Timestamp: {ts()}\n"
    f"- Base Source: {base_source_label}\n"
    f"- Output: {OUT_CSV}\n"
    f"- Rows (final): {len(pd.read_csv(OUT_CSV, low_memory=False)):,}\n",
    encoding="utf-8"
)

print(f"🗂️  Saved → {OUT_CSV}")
print(f"📝  Log   → {log_path}")
