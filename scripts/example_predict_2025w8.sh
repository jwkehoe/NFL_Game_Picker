#!/usr/bin/env bash
# example_predict_2025w8.sh
# ------------------------------------------------------------------
# End-to-end pipeline example to fetch, normalize, engineer features,
# train a model, and generate predictions for the 2025 Week 8 slate.
#
# This is a functional example for a single week (2025 W8). For general use,
# prefer scripts/run_master_pipeline.sh which parameterizes the gates.
#
# Usage:
#   bash scripts/example_predict_2025w8.sh
#
# Requires the project virtual environment at ./venv populated with
# dependencies (run setup_venv.sh if needed).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "❌ Cannot find virtualenv Python at ${PYTHON_BIN}."
  echo "   Run ./setup_venv.sh first."
  exit 1
fi

TRAINGATE="2025W7"
TESTGATE="2025W7"
PREDICTGATE="2025W8"
MODEL_TAG_RAW="2025W7_to_2025W8"
MODEL_TAG="2025w7_to_2025w8"  # normalized tag used by training/prediction outputs

cd "${ROOT_DIR}"

echo "================ STEP 01: Fetch latest NFL games ================"
"${PYTHON_BIN}" scripts/01_get_nfl_game.py

echo "================ STEP 02: Normalize datasets ===================="
"${PYTHON_BIN}" scripts/02_normalize_data.py

echo "================ STEP 03: Recompute Elo ratings ================="
"${PYTHON_BIN}" scripts/03_calculate_elo.py

echo "================ STEP 04: Build modeling features ==============="
"${PYTHON_BIN}" scripts/04_build_features.py

echo "================ STEP 05: Train model for Week 8 ================"
"${PYTHON_BIN}" scripts/05_train_model.py \
  --mode train \
  --traingate "${TRAINGATE}" \
  --testgate "${TESTGATE}" \
  --predictgate "${PREDICTGATE}" \
  --tag "${MODEL_TAG_RAW}"

echo "================ STEP 06: Predict 2025 Week 8 games ============="
"${PYTHON_BIN}" scripts/06_predict_games.py \
  --predictgate "2025 W8" \
  --model_tag "${MODEL_TAG}"

echo "✅ Pipeline complete. Check data/predictions/ for outputs."
