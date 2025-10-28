#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_normalize_data.py
───────────────────────────────────────────────
Normalize text fields across all input CSVs.

Inputs:
  • data/raw/nfl_games_all.csv
  • data/processed/elo_history.csv
  • data/features/training_features.csv

Outputs:
  • data/normalized/nfl_games_all_normalized.csv
  • data/normalized/elo_history_normalized.csv
  • data/normalized/training_features_normalized.csv
  • data/mappings/string_normalization.yaml
  • logs/normalize_data.md
"""

import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
import re

# ------------------------------------------------------------
# Setup
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
OUT_DIR = Path("data/normalized")
MAP_DIR = Path("data/mappings")
LOG_DIR = Path("logs")

OUT_DIR.mkdir(parents=True, exist_ok=True)
MAP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAP_PATH = MAP_DIR / "string_normalization.yaml"
LOG_PATH = LOG_DIR / "normalize_data.md"

# ------------------------------------------------------------
def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def normalize_text(s: str) -> str:
    """Clean spacing, punctuation, and casing uniformly."""
    if not isinstance(s, str):
        return s
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-\']+", "", s)
    s = s.upper()
    return s

def normalize_df(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Normalize string fields and update mapping."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).apply(lambda x: mapping.setdefault(x, normalize_text(x)))
    return df

def load_yaml(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def save_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True, allow_unicode=True)

# ------------------------------------------------------------
def process_file(path: Path, out_path: Path, mapping: dict):
    print(f"📘 Normalizing → {path.name}")
    df = pd.read_csv(path)
    norm_df = normalize_df(df, mapping)
    norm_df.to_csv(out_path, index=False)
    print(f"✅ Saved normalized → {out_path} ({len(norm_df):,} rows)")
    return norm_df

# ------------------------------------------------------------
def main():
    mapping = load_yaml(MAP_PATH)

    files = [
        (RAW_DIR / "nfl_games_all.csv", OUT_DIR / "nfl_games_all_normalized.csv"),
        (PROC_DIR / "elo_history.csv", OUT_DIR / "elo_history_normalized.csv"),
        (FEATURES_DIR / "training_features.csv", OUT_DIR / "training_features_normalized.csv"),
    ]

    summary = []
    for src, dst in files:
        if src.exists():
            df = process_file(src, dst, mapping)
            summary.append(f"- {src.name}: {len(df):,} rows → {dst.name}")
        else:
            print(f"⚠️ Missing file: {src}")

    save_yaml(mapping, MAP_PATH)
    log = f"""# Data Normalization Report
- Timestamp: {ts()}
- Files processed: {len(summary)}
- Mapping saved → {MAP_PATH}

## Summary
{chr(10).join(summary)}
"""
    LOG_PATH.write_text(log, encoding="utf-8")
    print(f"📝 Log → {LOG_PATH}")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()