#!/usr/bin/env python3
"""
nfldatapy_client.py
────────────────────
Purpose
  Lightweight adapter around nfl_data_py so AutoNFL can optionally source
  schedules and weekly team features without rewriting downstream code.

Guarantees
  - Optional dependency: callers can disable with AUTO_NFL_USE_NFLDATAPY=0
  - Clear failure mode: raises NflDataPyUnavailable on import/feature gaps
  - Schema normalization: returns columns aligned to project expectations

What it fetches
  - Schedules (fetch_schedules) → normalized columns (season, week, game_date, ...)
  - Team efficiency (fetch_team_efficiency) → pbp_off_* / pbp_def_* rates
  - Injuries (fetch_injury_aggregates) → inj_* team aggregates + weighted by position group
  - Snap counts (fetch_snap_count_aggregates) → snap_* shares
  - Betting lines (fetch_betting_lines) → line_consensus_* fields

Usage
  from nfldatapy_client import ScheduleRequest, fetch_schedules
  df = fetch_schedules(ScheduleRequest(seasons=[2024,2025]))

Config
  - AUTO_NFL_USE_NFLDATAPY: truthy/falsey to enable/disable usage
  - Internal batch size for feature fetchers to avoid large-range timeouts

Notes
  - This module does not write to disk; scripts/update_weekly_external.py is the
    owner of writing CSVs into data/external/ for feature merging.
  - nfl_data_py upstream APIs and schema can evolve; we normalize common fields
    and defensively handle missing columns with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Dict

import numpy as np
import pandas as pd

try:
    from nfl_data_py import (
        import_injuries,
        import_pbp_data,
        import_schedules,
        import_snap_counts,
        import_sc_lines,
    )
except Exception:  # pragma: no cover - import guard
    import_schedules = None  # type: ignore[assignment]
    import_pbp_data = None  # type: ignore[assignment]
    import_injuries = None  # type: ignore[assignment]
    import_snap_counts = None  # type: ignore[assignment]
    import_sc_lines = None  # type: ignore[assignment]


class NflDataPyUnavailable(RuntimeError):
    """Raised when nfl_data_py is not importable (missing or incompatible)."""


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "on"}


def should_use_nfldatapy(default: bool = True) -> bool:
    """
    Decide whether the wrapper should attempt to call nfl_data_py.

    Controlled by ``AUTO_NFL_USE_NFLDATAPY`` (truthy/falsey strings).
    When the env var is unset we honour ``default`` so callers can flip
    behaviour without touching the global configuration.
    """
    flag = os.getenv("AUTO_NFL_USE_NFLDATAPY")
    if flag is None:
        return default
    return _env_truthy(flag)


@dataclass
class ScheduleRequest:
    seasons: Sequence[int]
    columns: Sequence[str] | None = None


@dataclass
class TeamEfficiencyRequest:
    seasons: Sequence[int]
    season_types: Sequence[str] = ("REG",)


@dataclass
class InjuryRequest:
    seasons: Sequence[int]
    season_types: Sequence[str] = ("REG",)


@dataclass
class SnapCountRequest:
    seasons: Sequence[int]
    season_types: Sequence[str] = ("REG",)


@dataclass
class BettingLinesRequest:
    seasons: Sequence[int]
    season_types: Sequence[str] = ("REG",)


def _normalize_seasons(seasons: Sequence[int]) -> list[int]:
    uniq: Dict[int, None] = {}
    for value in seasons:
        uniq[int(value)] = None
    normalized = sorted(uniq.keys())
    if not normalized:
        raise ValueError("At least one season must be provided.")
    return normalized


def _require_import() -> None:
    if import_schedules is None:  # pragma: no cover - runtime guard
        raise NflDataPyUnavailable(
            "The nfl_data_py package is not available. "
            "Install `nfl-data-py` (requires pandas<2.0 / numpy<2.0) "
            "or disable AUTO_NFL_USE_NFLDATAPY."
        )


def _require_feature_imports(which: Tuple[str, ...]) -> None:
    missing = []
    if "pbp" in which and import_pbp_data is None:
        missing.append("import_pbp_data")
    if "injuries" in which and import_injuries is None:
        missing.append("import_injuries")
    if "snaps" in which and import_snap_counts is None:
        missing.append("import_snap_counts")
    if "lines" in which and import_sc_lines is None:
        missing.append("import_sc_lines")
    if missing:  # pragma: no cover - runtime guard
        raise NflDataPyUnavailable(
            "nfl_data_py feature fetch unavailable; missing imports: "
            + ", ".join(missing)
        )


def _import_in_batches(importer, seasons: list[int], chunk_size: int = 5) -> pd.DataFrame:
    frames = []
    for idx in range(0, len(seasons), chunk_size):
        subset = seasons[idx : idx + chunk_size]
        if subset:
            frames.append(importer(subset))  # type: ignore[call-arg]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align nfl_data_py schedule output to our expected column names.

    The legacy CSV ingest expects ``season``, ``week`, ``game_type``,
    ``game_date``, ``weekday``, ``kickoff_time``, ``home_team``,
    ``away_team``, ``home_score``, ``away_score``, etc.
    """
    rename_map = {
        "gameday": "game_date",
        "gametime": "kickoff_time",
    }
    df = df.rename(columns=rename_map)

    # Downstream scripts expect Int64 dtypes for season/week.
    for col in ("season", "week"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    score_cols: Iterable[str] = ("home_score", "away_score", "result", "total")
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def fetch_schedules(request: ScheduleRequest) -> pd.DataFrame:
    """
    Fetch schedules for ``request.seasons`` using nfl_data_py and convert
    the output to the schema expected by the AutoNFL normalization flow.
    """
    _require_import()
    seasons = list(dict.fromkeys(int(s) for s in request.seasons))
    if not seasons:
        raise ValueError("ScheduleRequest.seasons must contain at least one season.")

    df = import_schedules(seasons)  # type: ignore[misc]
    df = _normalize_columns(df)

    if request.columns:
        missing = set(request.columns) - set(df.columns)
        if missing:
            raise KeyError(f"nfl_data_py schedules missing columns: {sorted(missing)}")
        df = df.loc[:, request.columns]

    return df


def fetch_team_efficiency(request: TeamEfficiencyRequest) -> pd.DataFrame:
    """
    Aggregate play-by-play data into weekly team efficiency metrics.
    """
    _require_feature_imports(("pbp",))
    seasons = _normalize_seasons(request.seasons)
    pbp = import_pbp_data(seasons)  # type: ignore[misc]
    pbp.columns = [c.lower() for c in pbp.columns]

    if request.season_types:
        pbp = pbp[pbp.get("season_type").isin(request.season_types)]

    for column in ("season", "week"):
        if column in pbp.columns:
            pbp[column] = pd.to_numeric(pbp[column], errors="coerce").astype("Int64")

    def ensure_column(frame: pd.DataFrame, column: str) -> None:
        if column not in frame.columns:
            frame[column] = 0.0

    ensure_column(pbp, "epa")
    ensure_column(pbp, "success")
    ensure_column(pbp, "pass")
    ensure_column(pbp, "rush")
    ensure_column(pbp, "qb_dropback")
    ensure_column(pbp, "yards_gained")
    if "down" not in pbp.columns:
        pbp["down"] = np.nan
    if "yardline_100" not in pbp.columns:
        pbp["yardline_100"] = np.nan
    ensure_column(pbp, "score_differential")
    if "quarter" not in pbp.columns:
        pbp["quarter"] = np.nan

    pbp["pass"] = pbp["pass"].fillna(0.0)
    pbp["rush"] = pbp["rush"].fillna(0.0)
    pbp["qb_dropback"] = pbp["qb_dropback"].fillna(0.0)
    pbp["success"] = pbp["success"].fillna(0.0)
    pbp["epa"] = pbp["epa"].fillna(0.0)
    pbp["yards_gained"] = pbp["yards_gained"].fillna(0.0)
    pbp["score_differential"] = pd.to_numeric(pbp["score_differential"], errors="coerce").fillna(0.0)
    pbp["quarter"] = pd.to_numeric(pbp["quarter"], errors="coerce")

    offense = pbp.loc[pbp["posteam"].notna()].copy()
    offense["team"] = offense["posteam"].astype(str).str.upper()
    offense = offense.loc[offense["team"] != ""]
    if offense.empty:
        return pd.DataFrame(columns=["season", "week", "team"])

    offense["explosive"] = (offense["yards_gained"] >= 20).astype(float)
    offense["late_down"] = offense["down"].isin([3, 4]).astype(float)
    offense["red_zone"] = (offense["yardline_100"] <= 20).astype(float)
    neutral_mask = (
        offense["score_differential"].abs() <= 7
    ) & (
        offense["quarter"].fillna(0).astype(int).isin([1, 2, 3])
    )
    offense_neutral = offense.loc[neutral_mask].copy()

    off_agg = (
        offense.groupby(["season", "week", "team"], as_index=False)
        .agg(
            plays=("play_id", "count"),
            epa_sum=("epa", "sum"),
            success_sum=("success", "sum"),
            pass_sum=("pass", "sum"),
            rush_sum=("rush", "sum"),
            dropbacks=("qb_dropback", "sum"),
            explosive_sum=("explosive", "sum"),
            late_down_sum=("late_down", "sum"),
            red_zone_sum=("red_zone", "sum"),
        )
    )

    off_agg["plays"] = off_agg["plays"].astype(float).replace(0.0, np.nan)
    off_agg["pbp_off_plays"] = off_agg["plays"].fillna(0.0)
    off_agg["pbp_off_epa_per_play"] = off_agg["epa_sum"] / off_agg["plays"]
    off_agg["pbp_off_success_rate"] = off_agg["success_sum"] / off_agg["plays"]
    off_agg["pbp_off_pass_rate"] = off_agg["pass_sum"] / off_agg["plays"]
    off_agg["pbp_off_rush_rate"] = off_agg["rush_sum"] / off_agg["plays"]
    off_agg["pbp_off_dropback_rate"] = off_agg["dropbacks"] / off_agg["plays"]
    off_agg["pbp_off_explosive_rate"] = off_agg["explosive_sum"] / off_agg["plays"]
    off_agg["pbp_off_late_down_rate"] = off_agg["late_down_sum"] / off_agg["plays"]
    off_agg["pbp_off_red_zone_rate"] = off_agg["red_zone_sum"] / off_agg["plays"]

    offense_cols = [
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
    ]
    off_agg = off_agg[offense_cols]

    defense = pbp.loc[pbp["defteam"].notna()].copy()
    defense["team"] = defense["defteam"].astype(str).str.upper()
    defense = defense.loc[defense["team"] != ""]
    defense["explosive"] = (defense["yards_gained"] >= 20).astype(float)
    defense["late_down"] = defense["down"].isin([3, 4]).astype(float)
    defense["red_zone"] = (defense["yardline_100"] <= 20).astype(float)

    def_agg = (
        defense.groupby(["season", "week", "team"], as_index=False)
        .agg(
            plays=("play_id", "count"),
            epa_sum=("epa", "sum"),
            success_sum=("success", "sum"),
            pass_sum=("pass", "sum"),
            rush_sum=("rush", "sum"),
            dropbacks=("qb_dropback", "sum"),
            explosive_sum=("explosive", "sum"),
            late_down_sum=("late_down", "sum"),
            red_zone_sum=("red_zone", "sum"),
        )
    )

    def_agg["plays"] = def_agg["plays"].astype(float).replace(0.0, np.nan)
    def_agg["pbp_def_plays"] = def_agg["plays"].fillna(0.0)
    def_agg["pbp_def_epa_per_play_allowed"] = def_agg["epa_sum"] / def_agg["plays"]
    def_agg["pbp_def_success_rate_allowed"] = def_agg["success_sum"] / def_agg["plays"]
    def_agg["pbp_def_pass_rate_allowed"] = def_agg["pass_sum"] / def_agg["plays"]
    def_agg["pbp_def_rush_rate_allowed"] = def_agg["rush_sum"] / def_agg["plays"]
    def_agg["pbp_def_dropback_rate_allowed"] = def_agg["dropbacks"] / def_agg["plays"]
    def_agg["pbp_def_explosive_rate_allowed"] = def_agg["explosive_sum"] / def_agg["plays"]
    def_agg["pbp_def_late_down_rate_allowed"] = def_agg["late_down_sum"] / def_agg["plays"]
    def_agg["pbp_def_red_zone_rate_allowed"] = def_agg["red_zone_sum"] / def_agg["plays"]

    defense_cols = [
        "season",
        "week",
        "team",
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
    def_agg = def_agg[defense_cols]

    defense_neutral_mask = (
        defense["score_differential"].abs() <= 7
    ) & (
        defense["quarter"].fillna(0).astype(int).isin([1, 2, 3])
    )
    defense_neutral = defense.loc[defense_neutral_mask].copy()

    if not offense_neutral.empty:
        off_neutral = (
            offense_neutral.groupby(["season", "week", "team"], as_index=False)
            .agg(
                plays=("play_id", "count"),
                epa_sum=("epa", "sum"),
                success_sum=("success", "sum"),
            )
        )
        off_neutral["plays"] = off_neutral["plays"].astype(float).replace(0.0, np.nan)
        off_neutral["pbp_off_epa_per_play_neutral"] = off_neutral["epa_sum"] / off_neutral["plays"]
        off_neutral["pbp_off_success_rate_neutral"] = off_neutral["success_sum"] / off_neutral["plays"]
        off_neutral = off_neutral[["season", "week", "team", "pbp_off_epa_per_play_neutral", "pbp_off_success_rate_neutral"]]
        off_agg = off_agg.merge(off_neutral, on=["season", "week", "team"], how="left")
    else:
        off_agg["pbp_off_epa_per_play_neutral"] = np.nan
        off_agg["pbp_off_success_rate_neutral"] = np.nan

    if not defense_neutral.empty:
        def_neutral = (
            defense_neutral.groupby(["season", "week", "team"], as_index=False)
            .agg(
                plays=("play_id", "count"),
                epa_sum=("epa", "sum"),
                success_sum=("success", "sum"),
            )
        )
        def_neutral["plays"] = def_neutral["plays"].astype(float).replace(0.0, np.nan)
        def_neutral["pbp_def_epa_per_play_allowed_neutral"] = def_neutral["epa_sum"] / def_neutral["plays"]
        def_neutral["pbp_def_success_rate_allowed_neutral"] = def_neutral["success_sum"] / def_neutral["plays"]
        def_neutral = def_neutral[["season", "week", "team", "pbp_def_epa_per_play_allowed_neutral", "pbp_def_success_rate_allowed_neutral"]]
        def_agg = def_agg.merge(def_neutral, on=["season", "week", "team"], how="left")
    else:
        def_agg["pbp_def_epa_per_play_allowed_neutral"] = np.nan
        def_agg["pbp_def_success_rate_allowed_neutral"] = np.nan

    merged = pd.merge(
        off_agg,
        def_agg,
        on=["season", "week", "team"],
        how="outer",
        validate="one_to_one",
    )

    merged["season"] = merged["season"].astype("Int64")
    merged["week"] = merged["week"].astype("Int64")
    merged["team"] = merged["team"].astype(str).str.upper()

    metric_cols = [c for c in merged.columns if c.startswith("pbp_")]
    merged = merged.sort_values(["team", "season", "week"]).reset_index(drop=True)
    smoothed = merged.groupby("team", group_keys=False)[metric_cols].apply(
        lambda df: df.shift(1).rolling(window=3, min_periods=1).mean()
    )
    merged[metric_cols] = smoothed
    return merged.sort_values(["season", "week", "team"]).reset_index(drop=True)


STATUS_WEIGHTS = {
    "OUT": 1.0,
    "QUESTIONABLE": 0.3,
    "DOUBTFUL": 0.5,
    "NOTE": 0.0,
    "IR": 1.0,
    "RESERVE/INJURED": 1.0,
}

POSITION_GROUP_MAP = {
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "WR": "WR",
    "TE": "TE",
    "C": "OL",
    "G": "OL",
    "T": "OL",
    "OG": "OL",
    "OT": "OL",
    "OL": "OL",
    "LS": "ST",
    "K": "ST",
    "P": "ST",
    "DT": "DL",
    "DE": "DL",
    "DL": "DL",
    "NT": "DL",
    "EDGE": "DL",
    "LB": "LB",
    "ILB": "LB",
    "OLB": "LB",
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "DB": "DB",
}


def fetch_injury_aggregates(request: InjuryRequest) -> pd.DataFrame:
    """
    Summarize weekly injury report metrics for each team.
    """
    _require_feature_imports(("injuries",))
    seasons = _normalize_seasons(request.seasons)
    inj = _import_in_batches(import_injuries, seasons, chunk_size=4)  # type: ignore[arg-type]
    inj.columns = [c.lower() for c in inj.columns]
    if request.season_types:
        inj = inj[inj.get("game_type").isin(request.season_types)]

    inj["team"] = inj.get("team", "").astype(str).str.upper()
    inj = inj.loc[inj["team"] != ""]
    if inj.empty:
        return pd.DataFrame(columns=["season", "week", "team"])

    inj["season"] = pd.to_numeric(inj.get("season"), errors="coerce").astype("Int64")
    inj["week"] = pd.to_numeric(inj.get("week"), errors="coerce").astype("Int64")
    inj["report_status"] = inj.get("report_status", "").astype(str).str.upper()
    inj["position"] = inj.get("position", "").astype(str).str.strip().str.upper()
    inj["status_weight"] = inj["report_status"].map(STATUS_WEIGHTS).fillna(0.0)
    inj["is_out"] = (inj["report_status"] == "OUT").astype(float)
    inj["is_questionable"] = (inj["report_status"] == "QUESTIONABLE").astype(float)
    inj["is_doubtful"] = (inj["report_status"] == "DOUBTFUL").astype(float)
    inj["pos_group"] = inj["position"].map(POSITION_GROUP_MAP).fillna("OTHER")

    keys = ["season", "week", "team"]

    base = (
        inj.groupby(keys, as_index=False)
        .agg(
            inj_players=("gsis_id", "nunique"),
            inj_entries=("gsis_id", "count"),
            inj_out=("is_out", "sum"),
            inj_questionable=("is_questionable", "sum"),
            inj_doubtful=("is_doubtful", "sum"),
            inj_weighted=("status_weight", "sum"),
        )
    )

    pivot = (
        inj.pivot_table(
            index=keys,
            columns="pos_group",
            values="status_weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        .rename(columns=lambda c: f"inj_weight_{str(c).lower()}")
        .reset_index()
    )

    merged = pd.merge(base, pivot, on=keys, how="left")
    merged["season"] = merged["season"].astype("Int64")
    merged["week"] = merged["week"].astype("Int64")
    merged["team"] = merged["team"].astype(str).str.upper()
    numeric_cols = [c for c in merged.columns if c.startswith("inj_")]
    merged[numeric_cols] = merged[numeric_cols].apply(pd.to_numeric, errors="coerce")
    merged = merged.sort_values(["team", "season", "week"]).reset_index(drop=True)
    smoothed = merged.groupby("team", group_keys=False)[numeric_cols].apply(
        lambda df: df.shift(1).rolling(window=3, min_periods=1).mean()
    )
    merged[numeric_cols] = smoothed
    return merged.sort_values(["season", "week", "team"]).reset_index(drop=True)


def fetch_snap_count_aggregates(request: SnapCountRequest) -> pd.DataFrame:
    """
    Aggregate snap count usage for offense, defense, and special teams.
    """
    _require_feature_imports(("snaps",))
    seasons = _normalize_seasons(request.seasons)
    snaps = _import_in_batches(import_snap_counts, seasons, chunk_size=6)  # type: ignore[arg-type]
    snaps.columns = [c.lower() for c in snaps.columns]
    if request.season_types:
        snaps = snaps[snaps.get("game_type").isin(request.season_types)]

    snaps["team"] = snaps.get("team", "").astype(str).str.upper()
    snaps = snaps.loc[snaps["team"] != ""]
    if snaps.empty:
        return pd.DataFrame(columns=["season", "week", "team"])

    snaps["season"] = pd.to_numeric(snaps.get("season"), errors="coerce").astype("Int64")
    snaps["week"] = pd.to_numeric(snaps.get("week"), errors="coerce").astype("Int64")
    snaps["position"] = snaps.get("position", "").astype(str).str.strip().str.upper()
    snaps["pos_group"] = snaps["position"].map(POSITION_GROUP_MAP).fillna("OTHER")

    for column in ("offense_pct", "defense_pct", "st_pct"):
        if column in snaps.columns:
            snaps[column] = pd.to_numeric(snaps[column], errors="coerce").fillna(0.0)
        else:
            snaps[column] = 0.0

    keys = ["season", "week", "team"]

    base = (
        snaps.groupby(keys, as_index=False)
        .agg(
            snap_offense_players=("offense_pct", lambda s: float((s > 0).sum())),
            snap_offense_core=("offense_pct", lambda s: float((s >= 60).sum())),
            snap_defense_players=("defense_pct", lambda s: float((s > 0).sum())),
            snap_special_players=("st_pct", lambda s: float((s > 0).sum())),
        )
    )

    def share_pivot(value_col: str, prefix: str) -> pd.DataFrame:
        return (
            snaps.pivot_table(
                index=keys,
                columns="pos_group",
                values=value_col,
                aggfunc="sum",
                fill_value=0.0,
            )
            .apply(lambda df: df / 100.0)
            .rename(columns=lambda c: f"{prefix}_{str(c).lower()}")
            .reset_index()
        )

    offense_share = share_pivot("offense_pct", "snap_offense_share")
    defense_share = share_pivot("defense_pct", "snap_defense_share")
    special_share = share_pivot("st_pct", "snap_special_share")

    merged = base
    for frame in (offense_share, defense_share, special_share):
        merged = pd.merge(merged, frame, on=keys, how="left")

    merged["season"] = merged["season"].astype("Int64")
    merged["week"] = merged["week"].astype("Int64")
    merged["team"] = merged["team"].astype(str).str.upper()
    numeric_cols = [c for c in merged.columns if c.startswith("snap_")]
    merged[numeric_cols] = merged[numeric_cols].apply(pd.to_numeric, errors="coerce")
    merged = merged.sort_values(["team", "season", "week"]).reset_index(drop=True)
    smoothed = merged.groupby("team", group_keys=False)[numeric_cols].apply(
        lambda df: df.shift(1).rolling(window=3, min_periods=1).mean()
    )
    merged[numeric_cols] = smoothed
    return merged.sort_values(["season", "week", "team"]).reset_index(drop=True)


def fetch_betting_lines(request: BettingLinesRequest) -> pd.DataFrame:
    """
    Fetch consensus spread lines (sportsbook closing) and reshape into
    team-level features.

    Note: nfl_data_py currently serves lines from 2013 onward.
    """
    _require_feature_imports(("lines",))
    seasons = _normalize_seasons(request.seasons)
    lines = _import_in_batches(import_sc_lines, seasons, chunk_size=6)  # type: ignore[arg-type]
    if lines.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "week",
                "team",
                "line_consensus_spread",
                "line_consensus_favorite",
                "line_consensus_spread_abs",
            ]
        )

    lines.columns = [c.lower() for c in lines.columns]
    lines["team"] = lines.get("side", "").astype(str).str.upper()
    lines = lines.loc[lines["team"] != ""]

    lines["season"] = pd.to_numeric(lines.get("season"), errors="coerce").astype("Int64")
    lines["week"] = pd.to_numeric(lines.get("week"), errors="coerce").astype("Int64")
    lines["line"] = pd.to_numeric(lines.get("line"), errors="coerce")

    if request.season_types and "game_type" in lines.columns:
        lines = lines[lines.get("game_type").isin(request.season_types)]

    subset = lines.loc[:, ["season", "week", "team", "line"]].dropna(subset=["season", "week", "team"])
    subset = subset.drop_duplicates(subset=["season", "week", "team"])

    subset["line_consensus_spread"] = subset["line"]
    subset["line_consensus_favorite"] = (subset["line"] < 0).astype(float)
    subset["line_consensus_spread_abs"] = subset["line"].abs()

    result = subset.drop(columns=["line"]).reset_index(drop=True)
    return result.sort_values(["season", "week", "team"]).reset_index(drop=True)
