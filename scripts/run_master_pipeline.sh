#!/usr/bin/env bash
# run_master_pipeline.sh
# ------------------------------------------------------------------
# Full AutoNFL workflow:
#   01 Fetch games
#   02 Normalize data
#   03 Recompute Elo
#   04 Build features
#   05 Train model
#   06 Predict upcoming games
#   07 Post-game ground truth (optional, runs by default)
#
# Usage:
#   bash scripts/run_master_pipeline.sh \
#     --traingate 2025W7 \
#     --testgate 2025W7 \
#     --predictgate 2025W8 \
#     [--skip-ground-truth]
#
# Notes:
#   * Requires dependencies installed in ./venv (run setup_venv.sh first).
#   * Model tag defaults to "<traingate>_<testgate>_<predictgate>" and is
#     normalized automatically by the training script; prediction/ground truth
#     use the same normalized tag.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "❌ Virtualenv python not found at ${PYTHON_BIN}."
  echo "   Run ./setup_venv.sh to create the environment."
  exit 1
fi

TRAINGATE="2025W7"
TESTGATE="2025W7"
PREDICTGATE="2025W8"
ROLLING_SEASONS=""
ROLLING_WEEKS=""
INCLUDE_GROUND_TRUTH=true

usage() {
  cat <<'EOF'
Usage: run_master_pipeline.sh [options]
  --traingate YYYYWk     Training cutoff (default 2025W7)
  --testgate YYYYWk      Test gate (default 2025W7)
  --predictgate YYYYWk   Predict gate (default 2025W8)
  --rolling-seasons N    Limit training data to the most recent N seasons
  --rolling-weeks N      Limit training data to the most recent N weeks
  --skip-ground-truth    Skip step 07 (post-game accuracy)
  -h, --help             Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --traingate)
      TRAINGATE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"
      shift 2
      ;;
    --testgate)
      TESTGATE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"
      shift 2
      ;;
    --predictgate)
      PREDICTGATE="$(echo "$2" | tr '[:lower:]' '[:upper:]')"
      shift 2
      ;;
    --rolling-seasons)
      ROLLING_SEASONS="$2"
      shift 2
      ;;
    --rolling-weeks)
      ROLLING_WEEKS="$2"
      shift 2
      ;;
    --skip-ground-truth)
      INCLUDE_GROUND_TRUTH=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

normalize_gate_for_predict() {
  local gate="$1"
  if [[ "${gate}" =~ ^([0-9]{4})W([0-9]{1,2})$ ]]; then
    printf "%s W%s" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
  elif [[ "${gate}" =~ ^([0-9]{4})[[:space:]]W([0-9]{1,2})$ ]]; then
    printf "%s W%s" "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
  else
    echo "❌ Invalid gate format '${gate}'. Expected e.g. 2025W8." >&2
    exit 1
  fi
}

normalize_model_tag() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | tr -cd '[:alnum:]_'
}

PREDICTGATE_DISPLAY="$(normalize_gate_for_predict "${PREDICTGATE}")"
MODEL_TAG_RAW="${TRAINGATE}_${TESTGATE}_${PREDICTGATE}"
MODEL_TAG="$(normalize_model_tag "${MODEL_TAG_RAW}")"

TRAIN_EXTRA_OPTS=()
if [[ -n "${ROLLING_SEASONS}" ]]; then
  TRAIN_EXTRA_OPTS+=(--rolling_seasons "${ROLLING_SEASONS}")
fi
if [[ -n "${ROLLING_WEEKS}" ]]; then
  TRAIN_EXTRA_OPTS+=(--rolling_weeks "${ROLLING_WEEKS}")
fi

echo "🏈 AutoNFL master pipeline"
echo "  Train gate   : ${TRAINGATE}"
echo "  Test gate    : ${TESTGATE}"
echo "  Predict gate : ${PREDICTGATE_DISPLAY}"
echo "  Model tag    : ${MODEL_TAG}"
if [[ -n "${ROLLING_SEASONS}" ]]; then
  echo "  Rolling seasons : ${ROLLING_SEASONS}"
fi
if [[ -n "${ROLLING_WEEKS}" ]]; then
  echo "  Rolling weeks   : ${ROLLING_WEEKS}"
fi
echo ""

cd "${ROOT_DIR}"

step() {
  local label="$1"; shift
  echo "================ ${label} ================"
  "$@"
  echo ""
}

step "STEP 01: Fetch latest NFL games" \
  "${PYTHON_BIN}" scripts/01_get_nfl_game.py

step "STEP 02: Normalize datasets" \
  "${PYTHON_BIN}" scripts/02_normalize_data.py

step "STEP 03: Recompute Elo ratings" \
  "${PYTHON_BIN}" scripts/03_calculate_elo.py

step "STEP 04: Build modeling features" \
  "${PYTHON_BIN}" scripts/04_build_features.py

TRAIN_CMD=(
  "${PYTHON_BIN}" scripts/05_train_model.py
  --mode train
  --traingate "${TRAINGATE}"
  --testgate "${TESTGATE}"
  --predictgate "${PREDICTGATE}"
  --tag "${MODEL_TAG_RAW}"
)
if [[ -n "${ROLLING_SEASONS}" ]]; then
  TRAIN_CMD+=(--rolling_seasons "${ROLLING_SEASONS}")
fi
if [[ -n "${ROLLING_WEEKS}" ]]; then
  TRAIN_CMD+=(--rolling_weeks "${ROLLING_WEEKS}")
fi

step "STEP 05: Train model" \
  "${TRAIN_CMD[@]}"

step "STEP 06: Predict ${PREDICTGATE_DISPLAY} games" \
  "${PYTHON_BIN}" scripts/06_predict_games.py \
    --predictgate "${PREDICTGATE_DISPLAY}" \
    --model_tag "${MODEL_TAG}"

if [[ "${INCLUDE_GROUND_TRUTH}" == true ]]; then
  step "STEP 07: Post-game ground truth for ${PREDICTGATE_DISPLAY}" \
    "${PYTHON_BIN}" scripts/07_post_game_ground_truth.py \
      --model_tag "${MODEL_TAG}" \
      --gate "${PREDICTGATE_DISPLAY}"
else
  echo "⏭️  Skipping ground truth step (requested)."
fi

echo "✅ Master pipeline complete."
