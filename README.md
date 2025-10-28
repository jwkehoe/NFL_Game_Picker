# AutoNFL

> WARNING — I do not endorse sports wagering. DO NOT USE THIS TOOL FOR BETTING — YOU WILL LOSE!
>
> Professional systems have access to proprietary and paywalled data and advanced analytics. This project is for learning, experimentation, and disciplined evaluation — not betting.

- Start here: `docs/README.md` (testing progression, comparisons, commands)
- Manifesto & guiding principles: `docs/MANIFESTO.md`
- Run tracking log: `docs/model_tracking.md`

## Quick Start

- Build features: `python scripts/04_build_features.py`
- Train Apple: `python scripts/05_train_model.py --mode train --traingate 2025W7 --testgate 2025W7 --predictgate "2025 W8" --tag apple`
- Predict: `python scripts/06_predict_games.py --predictgate "2025 W8" --model_tag apple`
- Backtest: `python scripts/run_weekly_backtest.py`

This repository does not distribute third‑party data. Please fetch data yourself and respect the original providers’ terms.
