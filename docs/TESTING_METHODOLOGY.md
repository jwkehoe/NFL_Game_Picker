# Testing Methodology & Best Practices

## Overview

This document outlines a rigorous, progressive testing approach for machine learning models in time-series domains. While focused on NFL prediction, these principles apply to any forecasting problem where temporal order matters and data leakage is a critical concern.

**Philosophy:** Test early, test often, test honestly. Start simple and add complexity only when validated. Every step should increase confidence that your model will perform well on truly unseen data.

## The Testing Pyramid: Simple → Complex

### Level 1: Smoke Tests & Data Sanity

**Purpose:** Verify your pipeline works before investing in model development.

**What to check:**
- Files load without errors
- Schemas are consistent (expected columns present, correct dtypes)
- Time ordering is correct (season/week progression makes sense)
- No obvious data corruption (e.g., scores > 100, dates in the future)

**Leakage prevention:**
- Strip all post-game information from features (home_score, away_score, final_result)
- Verify feature values don't depend on future games
- Check that "last N games" calculations stop before the prediction point

**Expected output:**
```
✓ Loaded 7,100 games
✓ 25 features, all numeric
✓ Target variable balanced (56.4% home wins)
✓ No nulls in critical columns
✓ Date range: 1999-2024
```

**When to use:** Every time you modify data collection or feature engineering.

---

### Level 2: Trivial Baselines

**Purpose:** Establish the floor—the performance any real model must beat.

**Baseline strategies:**

1. **Coin flip (50%):** Random guessing. Any model worse than this is broken.

2. **Home team always wins (56.4%):** Exploits known home-field advantage. Simple but effective.

3. **Market favorite:** Pick the team favored by Vegas odds/spread.
   - Typically achieves ~73% accuracy
   - This is your real target—Vegas is the informed benchmark

4. **ELO only:** Predict based on pure team strength differential.
   - Tests if your rating system captures basic team quality
   - Should beat coin flip, may not beat Vegas
   - Elo rating method by Arpad Elo; football adaptation by Bob Runyan (World Football Elo Ratings, 1997).

**Why this matters:**
If your sophisticated model with 25 features can't beat "pick the home team," something is fundamentally wrong. Fix the pipeline before adding complexity.

**Expected output:**
```
Baseline Results (test set):
- Random: 50.2%
- Home team: 56.4%  
- Market favorite: 73.3%
- ELO only: 65.8%

Target: Beat 73.3% with better calibration
```

---

### Level 3: One-Week Hold-Out

**Purpose:** Quick sanity check with a small, realistic sample.

**Setup:**
- Train on Weeks 1-N
- Test on Week N+1 only
- Use ~12-16 games (typical week size)

**What this tests:**
- Does the model produce reasonable predictions?
- Are probabilities in [0,1]?
- Do predictions align with intuition for obvious mismatches?

**Limitations:**
- Small sample size (high variance)
- One week isn't representative
- Only use for debugging, not final evaluation

**Expected output:**
```
Week 10 predictions (16 games):
- Accuracy: 10/16 (62.5%)
- Mean confidence: 0.68
- Favorites correct: 9/12 (75%)
- Underdogs correct: 1/4 (25%)

→ Model works, proceed to full validation
```

---

### Level 4: Deterministic Temporal Split

**Purpose:** Honest evaluation that respects time boundaries.

**Setup:**
```
Training:   [1999 ────────────────── 2023]
Validation: [2024 first half]
Test:       [2024 second half]
```

**Critical rules:**
- **Never shuffle** time-ordered data randomly
- Train only on data chronologically before the test period
- Don't peek at validation set during feature engineering
- Keep test set completely untouched until final evaluation

**Why this matters:**
Random shuffles allow models to learn from future games, inflating performance artificially. Time-aware splits simulate real prediction conditions.

**Expected output:**
```
2024 First Half (Validation):
- Games: 136
- Accuracy: 71.2%
- Log-loss: 0.5821

2024 Second Half (Test):
- Games: 136  
- Accuracy: 69.8%
- Log-loss: 0.5943

→ Performance holds across time, model is stable
```

---

### Level 5: Walk-Forward Backtest (Full Window)

**Purpose:** Simulate real-world deployment where you retrain weekly on all available history.

**Process:**
```
For each week W in test period:
    1. Train on all games before week W
    2. Generate predictions for week W
    3. Record predictions and actual outcomes
    4. Move to week W+1
    
After all weeks:
    5. Calculate overall metrics across all predictions
```

**What this tests:**
- **Temporal consistency:** Does performance hold week-to-week?
- **Concept drift:** Do older models degrade as the season progresses?
- **Sample size:** Accumulate enough predictions for statistical significance

**Expected output:**
```
Walk-Forward Results (2024 Season, Weeks 1-17):
- Total games: 272
- Overall accuracy: 69.5%
- Weekly accuracy range: 53% to 81%
- Mean log-loss: 0.5711
- Std dev (weekly): 0.087

Week-by-week:
Week  Acc   LL
  1   53%  0.682  (Early season uncertainty)
  2   69%  0.589
  3   75%  0.521
  ...
 17   72%  0.558

→ Stable performance with expected variance
```

**Interpretation:**
- Week-to-week variance is normal (NFL is unpredictable)
- Judge performance over full season, not individual weeks
- Early season (Weeks 1-4) typically has lower accuracy

---

### Level 6: Time-Series Cross-Validation (Lightweight)

**Purpose:** Tune hyperparameters while respecting temporal order.

**Approach:**
```
Fold 1: Train[1999-2020] → Validate[2021]
Fold 2: Train[1999-2021] → Validate[2022]  
Fold 3: Train[1999-2022] → Validate[2023]
Fold 4: Train[1999-2023] → Validate[2024]
```

**When to use:**
- Selecting between 2-3 model types
- Tuning small hyperparameter grids (< 20 combinations)
- Comparing feature sets

**When NOT to use:**
- Exhaustive grid search (too computationally expensive)
- Overfitting hyperparameters to validation folds
- As a substitute for final test set evaluation

**Philosophy:**
Prefer **logically consistent** parameters over hairline improvements. A 0.2% gain that only appears in one fold is likely noise, not signal.

**Expected output:**
```
Hyperparameter CV Results (4 folds):

Config A: C=0.1
  Fold accuracy: [71.2%, 69.8%, 70.5%, 70.1%]
  Mean: 70.4% ± 0.6%

Config B: C=1.0  
  Fold accuracy: [71.5%, 70.1%, 70.8%, 70.3%]
  Mean: 70.7% ± 0.6%

Config C: C=10.0
  Fold accuracy: [71.1%, 69.5%, 70.2%, 69.9%]
  Mean: 70.2% ± 0.7%

→ Select Config B (stable, slight edge)
```

---

### Level 7: Calibration Analysis

**Purpose:** Evaluate if predicted probabilities match observed frequencies.

**Metrics:**

1. **Mean log-loss:** Measures probability quality
   - Lower is better
   - Penalizes confident wrong predictions heavily
   - Formula: `-mean(y*log(p) + (1-y)*log(1-p))`

2. **Calibration curves:** Plot predicted probability bins vs. actual win rates
   ```
   Predicted   Actual
   [0.5-0.6]:  0.58  ✓ Well calibrated
   [0.6-0.7]:  0.64  ✓ Well calibrated  
   [0.7-0.8]:  0.81  ✗ Overconfident
   [0.8-0.9]:  0.79  ✗ Overconfident
   ```

3. **Brier score:** Alternative probability metric
   - Lower is better
   - Formula: `mean((p - y)²)`

**When to apply Platt scaling:**
- If accuracy is good but log-loss is poor
- If calibration curves show systematic bias
- After ensembling (often miscalibrates)

**Platt scaling process:**
```python
from sklearn.calibration import CalibratedClassifierCV

model = LogisticRegression(...)
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrated.fit(X_train, y_train)
```

**Expected output:**
```
Model Calibration Analysis:

Raw Model:
- Accuracy: 73.31%
- Mean log-loss: 0.5756
- Calibration error: 0.047

After Platt Scaling:
- Accuracy: 72.95% (slightly lower)
- Mean log-loss: 0.5711 (improved!)
- Calibration error: 0.031

→ Accept calibrated model for probability-based decisions
```

---

### Level 8: Feature Triage & Ablation Studies

**Purpose:** Identify which features actually contribute to performance.

**Approach:**

1. **Importance ranking:** Train model, extract feature importance
   ```
   Feature               Importance
   spread_line          0.342
   elo_diff             0.198
   home_l5_win_pct      0.091
   away_l5_win_pct      0.087
   ...
   ```

2. **Incremental removal:** Drop lowest-importance features one at a time
   ```
   All 25 features:     69.5% acc, 0.5711 LL
   Top 20 features:     69.5% acc, 0.5709 LL ✓ Simpler is better
   Top 15 features:     69.2% acc, 0.5728 LL ✗ Removed too much
   Top 10 features:     68.1% acc, 0.5891 LL ✗ Underfitting
   ```

3. **Ablation studies:** Remove feature groups to test hypotheses
   - Remove all "recent form" features → accuracy drops to 67%
   - Remove all "rest" features → accuracy drops to 69.3%
   - Remove "division game" indicator → accuracy unchanged

**When to triage:**
- After initial model shows promise
- Before adding more features
- When debugging unexpected predictions

**Warning:** Don't triage too aggressively. Removing features to hit a hairline accuracy gain risks:
- Overfitting to the test set
- Losing calibration
- Breaking robustness

**Expected output:**
```
Feature Triage Results:

Baseline (25 features): 69.5% acc, 0.5711 LL

Remove bottom 5: 69.5% acc, 0.5709 LL ✓ Accept
Remove bottom 10: 69.1% acc, 0.5743 LL ✗ Signal loss  
Remove bottom 15: 67.8% acc, 0.5912 LL ✗ Severe degradation

→ Use top 20 features for production
```

---

### Level 9: Override & Gating Audit

**Purpose:** Understand when your model disagrees with the market, and whether those disagreements are profitable.

**Metrics to track:**

1. **Override rate:** How often does model flip vs. the favorite?
   ```
   Total games: 272
   Model agrees with favorite: 245 (90.1%)
   Model disagrees: 27 (9.9%)
   ```

2. **Override accuracy:** Win rate when model disagrees
   ```
   Games where model flipped: 27
   Model was correct: 12 (44.4%)
   
   → Overrides underperform, add flip guard
   ```

3. **Average edge:** Probability gap when flipping
   ```
   Mean model confidence when flipping: 0.58
   Mean implied market probability: 0.55
   Average edge: 0.03 (too thin!)
   
   → Require minimum 0.08 edge to flip
   ```

**Implementing a flip guard:**
```python
def apply_flip_guard(model_prob, market_prob, min_edge=0.08):
    """Only flip prediction if edge exceeds threshold."""
    edge = abs(model_prob - market_prob)
    
    if edge < min_edge:
        # Edge too thin, defer to market
        return market_prob > 0.5
    else:
        # Strong disagreement, trust model
        return model_prob > 0.5
```

**Expected output:**
```
Override Analysis (with 8% flip guard):

Games where model wants to flip: 27
Games where edge > 8%: 11 (filtered 16)
Accuracy on flips: 6/11 (54.5%)

Result: Reduced noise, preserved signal
→ Use flip guard in production
```

---

### Level 10: Error Cohort Analysis

**Purpose:** Find systematic failure modes to guide improvements.

**Cohorts to analyze:**

1. **By team:**
   ```
   Team    Games  Accuracy  Notes
   KC      16     81.2%     Model understands Chiefs well
   NYG     16     56.2%     Struggles with inconsistent teams
   JAX     16     62.5%     Missing key features for weak teams
   ```

2. **By matchup type:**
   ```
   Type              Games  Accuracy
   Division games    75     68.0%     Slightly worse (rivalry chaos)
   Conference games  120    70.1%     Baseline
   Interconference   77     69.8%     Baseline
   ```

3. **By spread range:**
   ```
   Spread Range    Games  Accuracy
   [-3, 3]        102     61.8%     Toss-ups are hard
   [-7, -3]       68      73.5%     Moderate favorites
   [-14, -7]      55      76.4%     Clear favorites  
   [7+, 14+]      47      78.7%     Blowouts predictable
   ```

4. **By season timing:**
   ```
   Weeks    Games  Accuracy
   1-4      68     58.8%     Early season uncertainty
   5-12     136    72.1%     Best performance
   13-17    68     70.6%     Playoff races create chaos
   ```

**Action items from cohort analysis:**
- **Toss-up games:** Consider abstaining or lowering confidence
- **Division games:** Add historical rivalry records?
- **Early season:** Increase weight on prior season performance
- **Weak teams:** Need features beyond ELO (coaching quality?)

**Expected output:**
```
Error Cohort Insights:

High-error scenarios:
1. Division games in December (56% accuracy)
   → Rivalries + playoff stakes = unpredictable
   
2. Teams with new coaches (61% accuracy)
   → Feature missing: coaching change indicator
   
3. Games after bye weeks (63% accuracy)
   → Rest advantage overestimated in model

Low-error scenarios:
1. Favorites by 10+ points (83% accuracy)
2. Teams on 5+ game win streaks (79% accuracy)
3. Dome teams at home (74% accuracy)

→ Add coaching continuity feature
→ Reduce bye week rest coefficient
→ Consider confidence bands by game type
```

---

### Level 11: Robustness Testing

**Purpose:** Verify the model performs consistently under different conditions.

**Tests:**

1. **Rerun after data refresh:**
   - Update data with latest scores
   - Retrain and retest
   - Confirm performance holds (± 1-2%)

2. **Try neighboring time windows:**
   ```
   Train on 1999-2022, test on 2023: 70.1%
   Train on 1999-2023, test on 2024: 69.5%
   Train on 2000-2023, test on 2024: 69.3%
   
   → Performance stable across windows
   ```

3. **Bootstrap confidence intervals:**
   ```python
   from sklearn.utils import resample
   
   accuracies = []
   for i in range(1000):
       sample = resample(test_data)
       acc = evaluate(model, sample)
       accuracies.append(acc)
   
   ci_95 = (np.percentile(accuracies, 2.5), 
            np.percentile(accuracies, 97.5))
   ```
   
   **Result:** 69.5% accuracy with 95% CI [66.8%, 72.1%]

4. **Adversarial testing:**
   - What if all home teams win? → Accuracy = 56.4%
   - What if all favorites win? → Accuracy = 73.3%
   - What if predictions were random? → Accuracy = 50.2%
   
   **Check:** Model outperforms all trivial strategies ✓

**Expected output:**
```
Robustness Test Results:

Data refresh (10/27/2024):
- Original accuracy: 69.5%
- Refreshed accuracy: 69.7%
- Difference: +0.2% (within noise)

Training window sensitivity:
- 1999-2023: 69.8%
- 2000-2023: 69.3%  
- 2002-2023: 68.9%
- Mean: 69.3% ± 0.5%

Bootstrap 95% CI: [66.8%, 72.1%]

→ Model is robust to data changes
→ Performance estimate: 69.5% ± 2%
```

---

### Level 12: Reproducibility & Release

**Purpose:** Ensure others (including future you) can rebuild the model exactly.

**Requirements:**

1. **Version control:** All code in git with tagged releases
2. **Dependency pinning:** Exact package versions in `requirements.txt`
3. **Random seed fixing:** Set seeds for numpy, sklearn, random
4. **Model serialization:** Save trained model + preprocessing pipeline
5. **Metadata logging:** Track training date, data version, performance
6. **Documentation:** README with setup instructions and examples

**Minimal reproducibility package:**
```
project/
├── data/
│   ├── training_data.csv          (or instructions to download)
│   └── data_version.txt           (source + collection date)
├── models/
│   ├── production_model.pkl       (trained model)
│   └── model_metadata.json        (config + performance)
├── scripts/
│   ├── train.py                   (training script)
│   └── predict.py                 (inference script)
├── requirements.txt               (pinned dependencies)
├── README.md                      (setup + usage)
└── .gitignore                     (exclude large files)
```

**Model metadata example:**
```json
{
  "model_type": "LogisticRegression",
  "sklearn_version": "1.3.0",
  "training_date": "2024-10-27",
  "training_samples": 7100,
  "features": ["elo_diff", "spread_line", ...],
  "test_accuracy": 0.695,
  "test_logloss": 0.5711,
  "hyperparameters": {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": 42
  }
}
```

**Expected output:**
```
Reproducibility Checklist:

✓ Code in git (tagged v1.0)
✓ requirements.txt pinned
✓ Random seed = 42 (fixed)
✓ Model saved with joblib
✓ Metadata JSON created
✓ README with quickstart
✓ Sample predictions included

Test: Fresh clone + setup
  → python train.py → 69.5% accuracy ✓
  → python predict.py → predictions match ✓

→ Release ready
```

---

## Quick Reference: When to Use Which Test

| Stage | Test Type | Sample Size | Purpose |
|-------|-----------|-------------|---------|
| **Development** | Smoke test | All data | Pipeline works |
| **Development** | Trivial baselines | Test set | Set floor |
| **Development** | One-week holdout | ~15 games | Quick validation |
| **Validation** | Temporal split | 50-150 games | Honest performance |
| **Validation** | Walk-forward | Full season | Real-world simulation |
| **Tuning** | Time-series CV | 3-5 folds | Hyperparameter selection |
| **Refinement** | Calibration | Full test set | Probability quality |
| **Refinement** | Feature triage | Full test set | Simplification |
| **Refinement** | Override audit | Games model flips | Risk management |
| **Analysis** | Error cohorts | Subsets | Find failure modes |
| **Release** | Robustness | Multiple conditions | Stability check |
| **Release** | Reproducibility | Fresh environment | Release quality |

---

## Summary: The Testing Mindset

**Progressive complexity:** Start simple (does it run?), end complex (does it generalize?).

**Time awareness:** Always respect temporal order. No shuffling, no peeking ahead.

**Honest evaluation:** Use baselines. Compare to the market. Accept when you've hit the ceiling.

**Incremental changes:** One modification at a time. Measure everything.

**Document failures:** Record what doesn't work. Dead ends teach as much as successes.

**Ship when ready:** Don't chase hairline gains. Stable, validated performance beats fragile over-optimization.

---

*These practices emerged from building, breaking, and rebuilding this system multiple times. They work for NFL prediction. They work for time-series forecasting generally. They work because they're grounded in scientific method: hypothesize, test, measure, iterate.*
