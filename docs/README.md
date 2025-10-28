# NFL Game Predictor

A machine learning system for predicting NFL game outcomes, built as a learning platform for practitioners exploring rigorous ML methodology in a competitive, high-variance domain.

## ⚠️ CRITICAL WARNING

**DO NOT USE THIS FOR SPORTS BETTING. YOU WILL LOSE.**

This project is for educational purposes only. Professional sportsbooks have proprietary data, advanced infrastructure, and institutional knowledge that individual bettors cannot match. Sports betting is a losing game for most people.

[Read the full warning and philosophy in the Manifesto →](MANIFESTO.md)

## What This Project Does

This system predicts NFL game winners using:
- **Historical game data** from 1999-present (7,100+ games)
- **ELO rating system** for dynamic team strength
- **Recent form features** (last 3/5/8 games, streaks, point differentials)
- **Situational features** (division games, rest advantages, home field)
- **Machine learning models** (Logistic Regression, XGBoost, Random Forest)

**Current Performance:**
- Accuracy: **72-73%** (matches or slightly trails Vegas favorites)
- Log-loss: **0.571** (beats Vegas on probability calibration)
- Training data: 7,100 completed games
- Test methodology: Walk-forward validation, time-aware splits

This performance represents the ceiling achievable with free, public data sources.

## Project Goals

1. **Educational:** Learn ML best practices through a concrete, measurable problem
2. **Methodological:** Demonstrate rigorous testing, validation, and honest evaluation
3. **Practical:** Win a $20 office football pool (not get rich betting)
4. **Exploratory:** Identify market inefficiencies and contrarian opportunities

## Quick Start

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Installation

```bash
git clone https://github.com/jwkehoe/NFL_Game_Picker.git
cd NFL_Game_Picker
pip install -r requirements.txt
```

### Basic Usage

```python
# Load and prepare data
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load training data (engineered features)
df = pd.read_csv('data/features/training_features.csv')

# Prepare features and target
feature_cols = ['elo_diff', 'home_l5_win_pct', 'away_l5_win_pct',
                'home_rest_days', 'away_rest_days', 'is_division_game']
X = df[feature_cols]
y = df['target']  # 1 = home win, 0 = away win

# Train model
model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
model.fit(X, y)

# Example: make predictions for a specific upcoming week
# (In production, use scripts/06_predict_games.py with --predictgate)
upcoming = df[(df['season'] == 2025) & (df['week'] == 8)]
X_pred = upcoming[feature_cols]
predictions = model.predict_proba(X_pred)[:, 1]  # Probability of home win

# Save model
joblib.dump(model, 'data/models/baseline_logreg.pkl')
```

See TESTING_METHODOLOGY.md and weekly_backtest_report.md for end‑to‑end usage.

## Documentation

| Document | Purpose |
|----------|---------|
| **[MANIFESTO.md](MANIFESTO.md)** | Philosophy, warnings, and guiding principles |
| **[TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md)** | Progressive testing approach (simple → complex) |
| **[METRICS_GUIDE.md](METRICS_GUIDE.md)** | Understanding accuracy, log-loss, and calibration |
| **[SETUP.md](SETUP.md)** | Installation and environment setup |
| **[WORKFLOW.md](WORKFLOW.md)** | Weekly prediction workflow |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common issues and fixes |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Guidelines and standards for contributions |

## Project Structure

```
NFL_Game_Picker/
│
├── data/
│   ├── raw/                      # Raw NFL game data
│   ├── features/                 # Engineered features
│   │   └── training_features.csv         # Engineered training/prediction features
│   └── backups/                  # Database backups
│
├── data/models/                  # Trained models, metadata, artifacts
│
├── scripts/
│   ├── 04_build_features.py      # Feature engineering pipeline
│   ├── 05_train_model.py         # Model training + calibration
│   └── 06_predict_games.py       # Inference for a given week
│
├── docs/                         # Comprehensive documentation
│   ├── README.md
│   ├── MANIFESTO.md
│   ├── TESTING_METHODOLOGY.md
│   ├── METRICS_GUIDE.md
│   ├── CONTRIBUTING.md
│   └── weekly_backtest_report.md
│
├── notebooks/                    # Jupyter notebooks for exploration
├── tests/                        # Unit tests
└── requirements.txt              # Python dependencies
```

## Features

### Engineered Features (25+)

1. **ELO Ratings**
   - Dynamic team strength (updated after each game)
   - Home advantage: +65 ELO points
   - K-factor: 20 (moderate update rate)

2. **Recent Form**
   - Last 3/5/8 game win percentages
   - Point differentials (offensive/defensive strength)
   - Win/loss streaks

3. **Situational**
   - Division games (rivalry effects)
   - Rest advantages (days since last game)
   - Short week flags (Thursday games)
   - Home field advantage

4. **Historical Context**
   - Season/week information
   - Previous head-to-head results

### Model Architecture

**Primary Model:** Logistic Regression with L2 regularization
- Simple, interpretable, fast
- Outperforms complex models on this dataset
- Excellent calibration properties

**Alternative Models Tested:**
- XGBoost (same accuracy, slower)
- Random Forest (same accuracy, worse calibration)
- Neural Networks (overfits, worse generalization)

**Key Insight:** All models converge to similar accuracy (~70-73%) when trained on the same features. Model choice matters less than feature quality.

## Methodology Highlights

### Data Integrity

- **No leakage:** Strict temporal boundaries, no future information in training
- **Clean data:** Validated against known statistics (home win rate: 56.4%)
- **Reproducible:** Fixed random seeds, versioned dependencies

### Validation Approach

1. **Baselines:** Compare against coin flip (50%), home team (56%), market favorite (73%)
2. **Temporal splits:** Train on past, test on future (no shuffling)
3. **Walk-forward validation:** Simulate real-world deployment
4. **Multiple metrics:** Accuracy (picks) + log-loss (probabilities)

### Testing Progression

```
Smoke Test → Baselines → One-Week Holdout → Temporal Split → 
Walk-Forward → Time-Series CV → Calibration → Feature Triage → 
Override Audit → Error Analysis → Robustness → Release
```

[Full testing methodology →](TESTING_METHODOLOGY.md)

## Performance Benchmarks

### Baseline Comparisons

| Strategy | Accuracy | Log-Loss | Notes |
|----------|----------|----------|-------|
| Coin flip | 50% | 0.693 | Absolute floor |
| Home team | 56.4% | 0.637 | Structural advantage |
| ELO only | 66% | 0.610 | Team strength only |
| Market favorite | **73.3%** | **0.5737** | Vegas baseline |
| Apple (calibrated) | 73.0% | **0.5711** | Best probabilities |
| Apple (accuracy) | **73.3%** | 0.5756 | Matches market picks |

### Interpretation

- **Accuracy:** We match or slightly trail the market (73%)
- **Calibration:** We beat the market on probability quality (0.571 vs 0.574)
- **Conclusion:** Useful for probability-based decisions, competitive for binary picks

[Complete metrics guide →](METRICS_GUIDE.md)

## Key Insights & Lessons

### What Works

✅ **Simple models** outperform complex ensembles  
✅ **ELO + recent form** captures most predictive signal  
✅ **Market line** is the strongest single feature (encodes expert knowledge)  
✅ **Time-aware validation** prevents overfitting to noise  
✅ **Flip guards** (minimum edge thresholds) reduce override mistakes

### What Doesn't Work

❌ **Injury data** (already priced into lines) → +0.09% accuracy  
❌ **Weather data** (public APIs lack granularity) → no gain  
❌ **Complex ensembles** (XGBoost + RF + NN) → same accuracy, worse calibration  
❌ **Opening line movement** → noise, no predictive value  
❌ **Aggressive feature triage** → removes signal along with noise

### The Ceiling

With free public data, **70-73% accuracy is the realistic ceiling**. Further gains require:
- Proprietary injury reports (before public disclosure)
- Advanced player tracking data (not publicly available)
- Sharp money line movement (requires paid feeds)
- Institutional infrastructure (continuous updates, fast processing)

Professional syndicates achieve 75%+ by paying for these edges.

## When to Use This System

### ✅ Good Use Cases

- **Office pools:** Weekly winner picks for casual competition
- **Learning ML:** Study rigorous methodology on real problem
- **Probability calibration:** Generate honest uncertainty estimates
- **Market analysis:** Identify games with contrarian value
- **Research:** Baseline for sports prediction research

### ❌ Bad Use Cases

- **Sports betting:** You will lose money. Professional books have better data.
- **Get-rich schemes:** This won't make you wealthy.
- **High-stakes decisions:** Not accurate enough for serious financial risk.

## Contributing

Contributions welcome! Areas for improvement:

1. **Data sources:** Integrate additional free data (weather, coaching)
2. **Feature engineering:** Novel features that capture overlooked signals
3. **Calibration:** Techniques to improve probability estimates
4. **Documentation:** Clarify existing docs, add tutorials
5. **Testing:** Additional validation approaches

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting PRs.

## New User Checklist

- Read MANIFESTO.md to align on purpose and constraints.
- Complete SETUP.md (venv + install).
- Fetch and prepare data: run 01→04 (see WORKFLOW.md).
- Optional: update weekly externals (update_weekly_external.py), then rebuild features.
- Choose gates and train with a clear `--tag` (05_train_model.py).
- Predict the upcoming week (06_predict_games.py) and review logs.
- After games finish, post ground truth (07_post_game_ground_truth.py).
- Compare vs. baselines and calibration (METRICS_GUIDE.md).
- Record the run in model_tracking.md: gates, tag, features, thresholds, metrics, decision.
- Make one change at a time; repeat. Keep artifacts in data/models/ and logs/.

## Utilities

- run_master_pipeline.sh — One-command 01→07 flow with gates. Flags documented in CLI_arguments.md.
- example_predict_2025w8.sh — Functional example for training + predicting 2025 W8.
- run_rolling_ensemble.py — Research: points→win prob ensemble. See CLI_arguments.md.
- update_weekly_external.py — Fetch external datasets into data/external/.

## Topics

- nfl, machine-learning, time-series, lightgbm, elo, calibration, sports-analytics

## Ethical Considerations

This project deliberately:
- ✅ Emphasizes educational value over profit
- ✅ Warns against sports betting throughout
- ✅ Uses only free, public data sources
- ✅ Provides transparent methodology
- ✅ Acknowledges limitations honestly

We do not:
- ❌ Encourage gambling or betting
- ❌ Claim unrealistic performance
- ❌ Promise financial returns
- ❌ Hide methodology or inflate results

## License

MIT License - see [LICENSE](LICENSE) for details.

**Disclaimer:** This software is provided for educational purposes only. The authors are not responsible for any financial losses incurred through misuse. DO NOT USE FOR SPORTS BETTING.

## Citation

If you use this project in research or education, please cite:

```bibtex
@software{nfl_predictor_2024,
  title={NFL Game Predictor: A Learning Platform for Sports Forecasting},
  author={jwkehoe},
  year={2024},
  url={https://github.com/jwkehoe/NFL_Game_Picker}
}
```

## Acknowledgments

- Data sources: ESPN, Pro Football Reference, NFL.com (public APIs)
- Inspiration: Nate Silver's FiveThirtyEight NFL predictions
- Community: r/sportsbook contrarian analysis discussions
- Rating methodology: The Elo rating system was created by Arpad Elo (1950s) for chess. Football-specific adaptations (e.g., home advantage, goal difference, match importance) trace to Bob Runyan’s World Football Elo Ratings (1997). Our implementation is an NFL-tailored variant that builds on that lineage.

## Frequently Asked Questions

**Q: Can I make money with this?**  
A: No. Professional sportsbooks have better data and will beat you over time.

**Q: What accuracy should I expect?**  
A: 70-73% on average, with weekly variance from 55% to 80%.

**Q: Why not use neural networks?**  
A: They overfit on this dataset. Logistic regression performs identically with better interpretability.

**Q: How often should I retrain?**  
A: Weekly during the season. Retrain after each week's games complete.

**Q: Can this beat Vegas?**  
A: On accuracy: rarely. On calibration: sometimes. On long-term profitability: no.

**Q: What's the most important feature?**  
A: Market spread/line (if available). It encodes expert knowledge and sharp money.

See TESTING_METHODOLOGY.md and METRICS_GUIDE.md for deeper dives.

## Contact & Support

- **Issues:** [GitHub Issues](https://github.com/jwkehoe/NFL_Game_Picker/issues)
- **Discussions:** [GitHub Discussions](https://github.com/jwkehoe/NFL_Game_Picker/discussions)

---

## Final Reminder

🚨 **This is a learning tool, not a betting system.**

The NFL is chaotic. Injuries happen. Weather changes. Refs make questionable calls. Teams have bad days. Kickers miss crucial field goals. Even a 73% accuracy model will lose 27% of its picks.

Use this to learn ML. Use this to understand sports forecasting. Use this to challenge yourself against the market.

**Do not use this to bet money you can't afford to lose.**

If you're struggling with gambling addiction, please seek help:
- **National Council on Problem Gambling:** 1-800-522-4700
- **Gamblers Anonymous:** https://www.gamblersanonymous.org/

---

*Built with 🧠 for learning, tested with 📊 for rigor, released with ⚠️ for safety.*
