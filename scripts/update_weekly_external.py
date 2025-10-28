#!/usr/bin/env python3
"""
Update weekly external datasets (team efficiency, injuries, snap counts, lines)
using nfl_data_py, writing CSVs to data/external/ for 04_build_features.py to merge.

What this prepares
  - Team play-by-play efficiency (offense/defense rates from nfl_data_py pbp)
  - Injury report aggregates (counts/weights by status and by position group)
  - Team snap counts (offense/defense/special teams usage shares)
  - Consensus betting lines (spread, favorite, absolute spread)

How features are used
  04_build_features.py will automatically merge any present external files
  (team_pbp_weekly.csv, injuries_weekly.csv, snap_counts_weekly.csv,
   betting_lines_weekly.csv) by [season, week, team], adding columns with
  prefixes pbp_*, inj_*, snap_*, line_* and their _diff counterparts.

CLI
  --start-season INT      Lower season bound (inclusive)
  --end-season INT        Upper season bound (inclusive)
  --season-types REG,POST Season types to include; default REG

Env vars
  AUTO_NFL_USE_NFLDATAPY=0  Disable fetch (writes empty placeholders)

Examples
  python scripts/update_weekly_external.py
  python scripts/update_weekly_external.py --start-season 2018
  AUTO_NFL_USE_NFLDATAPY=0 python scripts/update_weekly_external.py  # write placeholders
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from nfldatapy_client import (
    BettingLinesRequest,
    InjuryRequest,
    NflDataPyUnavailable,
    SnapCountRequest,
    TeamEfficiencyRequest,
    fetch_injury_aggregates,
    fetch_betting_lines,
    fetch_snap_count_aggregates,
    fetch_team_efficiency,
    should_use_nfldatapy,
)


EXTERNAL_DIR = Path("data/external")
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

TEAM_EFFICIENCY_PATH = EXTERNAL_DIR / "team_pbp_weekly.csv"
INJURIES_PATH = EXTERNAL_DIR / "injuries_weekly.csv"
SNAP_COUNTS_PATH = EXTERNAL_DIR / "snap_counts_weekly.csv"
BETTING_LINES_PATH = EXTERNAL_DIR / "betting_lines_weekly.csv"

TEAM_EFFICIENCY_COLUMNS = [
    "season",
    "week",
    "team",
    "pbp_off_plays",
    "pbp_off_epa_per_play",
    "pbp_off_success_rate",
    "pbp_off_pass_rate",
    "pbp_off_rush_rate",
    "pbp_off_dropback_rate",
    "pbp_off_explosive_rate",
    "pbp_off_late_down_rate",
    "pbp_off_red_zone_rate",
    "pbp_def_plays",
    "pbp_def_epa_per_play_allowed",
    "pbp_def_success_rate_allowed",
    "pbp_def_pass_rate_allowed",
    "pbp_def_rush_rate_allowed",
    "pbp_def_dropback_rate_allowed",
    "pbp_def_explosive_rate_allowed",
    "pbp_def_late_down_rate_allowed",
    "pbp_def_red_zone_rate_allowed",
]

INJURIES_COLUMNS = [
    "season",
    "week",
    "team",
    "inj_players",
    "inj_entries",
    "inj_out",
    "inj_questionable",
    "inj_doubtful",
    "inj_weighted",
    "inj_weight_db",
    "inj_weight_dl",
    "inj_weight_lb",
    "inj_weight_ol",
    "inj_weight_other",
    "inj_weight_qb",
    "inj_weight_rb",
    "inj_weight_st",
    "inj_weight_te",
    "inj_weight_wr",
]

SNAP_COUNTS_COLUMNS = [
    "season",
    "week",
    "team",
    "snap_offense_players",
    "snap_offense_core",
    "snap_defense_players",
    "snap_special_players",
    "snap_offense_share_db",
    "snap_offense_share_dl",
    "snap_offense_share_lb",
    "snap_offense_share_ol",
    "snap_offense_share_other",
    "snap_offense_share_qb",
    "snap_offense_share_rb",
    "snap_offense_share_st",
    "snap_offense_share_te",
    "snap_offense_share_wr",
    "snap_defense_share_db",
    "snap_defense_share_dl",
    "snap_defense_share_lb",
    "snap_defense_share_ol",
    "snap_defense_share_other",
    "snap_defense_share_qb",
    "snap_defense_share_rb",
    "snap_defense_share_st",
    "snap_defense_share_te",
    "snap_defense_share_wr",
    "snap_special_share_db",
    "snap_special_share_dl",
    "snap_special_share_lb",
    "snap_special_share_ol",
    "snap_special_share_other",
    "snap_special_share_qb",
    "snap_special_share_rb",
    "snap_special_share_st",
    "snap_special_share_te",
    "snap_special_share_wr",
]

BETTING_LINES_COLUMNS = [
    "season",
    "week",
    "team",
    "line_consensus_spread",
    "line_consensus_favorite",
    "line_consensus_spread_abs",
]


def determine_seasons(start: int | None, end: int | None) -> List[int]:
    """
    Choose seasons to pull. Prefer using the existing raw games file so we
    only fetch what the project actually needs. Fallback is the full span
    from start→end (default 1999→current year).
    """
    raw_games = Path("data/raw/nfl_games_all.csv")
    seasons: Iterable[int] = []
    if raw_games.exists():
        try:
            df = pd.read_csv(raw_games, usecols=["season"])
            seasons = df["season"].dropna().astype(int).unique().tolist()
        except Exception:
            seasons = []
    if not seasons:
        current_year = dt.datetime.now().year
        start = start or 1999
        end = end or current_year
        seasons = list(range(start, end + 1))
    else:
        seasons = sorted(int(s) for s in seasons if pd.notna(s))
        if start is not None:
            seasons = [s for s in seasons if s >= start]
        if end is not None:
            seasons = [s for s in seasons if s <= end]
    if not seasons:
        raise SystemExit("No seasons available for external fetch.")
    return list(dict.fromkeys(seasons))


def write_placeholder(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch weekly external datasets via nfl_data_py.")
    parser.add_argument("--start-season", type=int, default=None, help="Lower season bound (inclusive).")
    parser.add_argument("--end-season", type=int, default=None, help="Upper season bound (inclusive).")
    parser.add_argument(
        "--season-types",
        default="REG",
        help="Comma-separated season types to include (REG, POST). Default REG.",
    )
    args = parser.parse_args()

    if not should_use_nfldatapy(default=True):
        print("AUTO_NFL_USE_NFLDATAPY disabled → writing empty external CSVs.")
        write_placeholder(TEAM_EFFICIENCY_PATH, TEAM_EFFICIENCY_COLUMNS)
        write_placeholder(INJURIES_PATH, INJURIES_COLUMNS)
        write_placeholder(SNAP_COUNTS_PATH, SNAP_COUNTS_COLUMNS)
        write_placeholder(BETTING_LINES_PATH, BETTING_LINES_COLUMNS)
        return 0

    season_types = [s.strip().upper() for s in args.season_types.split(",") if s.strip()]
    seasons = determine_seasons(args.start_season, args.end_season)
    print(f"Using seasons {seasons[0]} – {seasons[-1]} ({len(seasons)} total); types={season_types}")

    tasks = [
        (
            "team efficiency",
            TEAM_EFFICIENCY_PATH,
            1999,
            None,
            TEAM_EFFICIENCY_COLUMNS,
            lambda season_list: fetch_team_efficiency(
                TeamEfficiencyRequest(seasons=season_list, season_types=season_types)
            ),
        ),
        (
            "injury reports",
            INJURIES_PATH,
            2009,
            2024,
            INJURIES_COLUMNS,
            lambda season_list: fetch_injury_aggregates(
                InjuryRequest(seasons=season_list, season_types=season_types)
            ),
        ),
        (
            "snap counts",
            SNAP_COUNTS_PATH,
            2012,
            None,
            SNAP_COUNTS_COLUMNS,
            lambda season_list: fetch_snap_count_aggregates(
                SnapCountRequest(seasons=season_list, season_types=season_types)
            ),
        ),
        (
            "betting lines",
            BETTING_LINES_PATH,
            2013,
            None,
            BETTING_LINES_COLUMNS,
            lambda season_list: fetch_betting_lines(
                BettingLinesRequest(seasons=season_list, season_types=season_types)
            ),
        ),
    ]

    for label, path, min_season, max_season, columns, loader in tasks:
        print(f"🔄 Fetching {label} → {path}")
        season_subset = [
            s
            for s in seasons
            if s >= min_season and (max_season is None or s <= max_season)
        ]
        if not season_subset:
            print(f"ℹ️ No seasons ≥ {min_season} available for {label}; writing placeholder.")
            write_placeholder(path, columns)
            continue
        try:
            df = loader(season_subset)
            if df.empty:
                write_placeholder(path, columns)
                print(f"ℹ️ {label.title()} not available for selected seasons; placeholder written.")
            else:
                df.to_csv(path, index=False)
                print(f"✅ {label.title()} written ({len(df):,} rows)")
        except NflDataPyUnavailable as err:
            print(f"⚠️ nfl_data_py unavailable for {label}: {err}")
            write_placeholder(path, columns)
        except Exception as err:
            print(f"⚠️ Failed to fetch {label}: {err}")
            write_placeholder(path, columns)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
