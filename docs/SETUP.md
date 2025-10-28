# Setup

Quick steps to get the project running locally.

—

Prerequisites

- Python 3.10+ (3.11 recommended)
- pip

—

Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Smoke test

```bash
# 1) Fetch games (historic + schedule fallbacks if needed)
python scripts/01_get_nfl_game.py

# 2) Normalize core files
python scripts/02_normalize_data.py

# 3) Compute Elo from normalized games
python scripts/03_calculate_elo.py

# 4) Build features (training + future rows)
python scripts/04_build_features.py
```

You should see outputs under `data/raw`, `data/normalized`, and `data/features`.

If you run into platform‑specific LightGBM issues, ensure your compiler toolchain is available (on macOS, `brew install libomp` may be required). Keep versions pinned via `requirements.txt` for reproducibility.

—

Environment versions and lockfile

- Tested with Python 3.11.9 and pip 25.2.
- Install standard deps: `pip install -r requirements.txt` (pinned versions)
- Exact reproducibility: `pip install -r requirements-lock.txt` (frozen snapshot)

