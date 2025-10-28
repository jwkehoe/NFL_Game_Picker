## Troubleshooting

Common issues and quick fixes.

—

Missing files or empty outputs

- FileNotFoundError: data/normalized/nfl_games_all_normalized.csv
  - Run in order: 01_get_nfl_game.py → 02_normalize_data.py → 03_calculate_elo.py → 04_build_features.py.

- Missing normalized training features. Run 04_build_features.py...
  - Build features first: python scripts/04_build_features.py.

- No rows found for 'YYYY Ww'
  - Ensure games for that week exist in raw/normalized data, and that 04_build_features.py ran. Double‑check the gate format: '2025 W8'.

—

Model training/prediction issues

- LightGBM complaining about objectives/iterations
  - Use defaults in 05_train_model.py and ensure features are numeric with categorical columns encoded (the script handles this).

- Poor calibration / overconfident probabilities
  - Use the calibrated artifacts produced by 05_train_model.py (tagged runs write calibration_<tag>.json), or enable the Platt‑scaled output in prediction (automatic when calibration file is present).

- Overrides too noisy
  - Tweak edge gating via env vars in scripts/adjustment_utils.py: AUTO_NFL_EDGE_THRESHOLD, AUTO_NFL_EDGE_THRESHOLD_POS, AUTO_NFL_EDGE_THRESHOLD_NEG, AUTO_NFL_MAX_FLIP_BASELINE_OFFSET.

—

Environment

- LightGBM import/build issues (macOS)
  - Ensure compiler toolchain; brew install libomp often resolves runtime errors.

- Version drift
  - Recreate venv and reinstall from requirements.txt. Keep seeds stable (42) for reproducibility.

—

Still stuck?

- Re‑run the full workflow in order and read logs under logs/ (feature build, CV results, predict logs, weekly backtest logs).
- Compare your steps to the examples in WORKFLOW.md and CLI_arguments.md.
