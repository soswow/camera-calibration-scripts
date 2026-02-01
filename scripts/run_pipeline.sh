#!/usr/bin/env bash
set -euo pipefail

# ChArUco calibration pipeline
# Adjust paths and parameters below as needed.

INPUT_FRAMES_DIR="/Users/sasha/hobby/PS3Eye-Driver-MacOS-Silicon/output_png/"
SCORES_JSON="frame_scores.json"
SELECTED_JSON="selected_frames.json"
CALIB_JSON="calibration.json"

# Board parameters (match your generator)
DICTIONARY="DICT_4X4_50"
SQUARES_X=7
SQUARES_Y=10
SQUARE_SIZE=79.14
MARKER_PROP=0.7

# Scoring / selection
MIN_CHARUCO=4
SELECT_COUNT=70
TOP_K=300

# Calibration / pruning
PRUNE_THRESHOLD=1.5

echo "1) Scoring frames..."
python3 scripts/score_charuco_frames.py \
  --input "${INPUT_FRAMES_DIR}" \
  --dictionary "${DICTIONARY}" \
  --squares-x "${SQUARES_X}" \
  --squares-y "${SQUARES_Y}" \
  --square-size "${SQUARE_SIZE}" \
  --marker-proportion "${MARKER_PROP}" \
  --min-charuco "${MIN_CHARUCO}" \
  --output "${SCORES_JSON}"

echo "2) Selecting diverse frames..."
python3 scripts/select_diverse_frames.py \
  --input "${SCORES_JSON}" \
  --output "${SELECTED_JSON}" \
  --select "${SELECT_COUNT}" \
  --top-k "${TOP_K}"

echo "3) Calibrating..."
python3 scripts/calibrate_charuco.py \
  --input "${SELECTED_JSON}" \
  --output "${CALIB_JSON}"

echo "4) Pruning outliers and re-calibrating..."
python3 scripts/calibrate_charuco.py \
  --input "${SELECTED_JSON}" \
  --output "${CALIB_JSON}" \
  --prune-threshold "${PRUNE_THRESHOLD}"

echo "5) Re-calibrating after pruning..."
python3 scripts/calibrate_charuco.py \
  --input "${SELECTED_JSON}" \
  --output "${CALIB_JSON}"

echo "Done."
