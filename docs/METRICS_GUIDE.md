# Understanding Prediction Metrics

## Overview

Evaluating prediction systems requires understanding multiple metrics that measure different aspects of performance. This guide explains what each metric measures, when to prioritize it, and how to interpret results in the context of NFL game prediction.

**Core principle:** The "best" model depends on your use case. For binary picks in an office pool, accuracy matters most. For probability-based decision making, calibration (log-loss) is critical.

---

## The Two Core Metrics

### Accuracy: "How Often Do We Get It Right?"

**Definition:** The percentage of games where you correctly predicted the winner.

```
Accuracy = (Correct Predictions) / (Total Predictions)
```

**Example:**
```
Games predicted: 100
Correct: 69
Accuracy: 69/100 = 69%
```

**What it measures:**
- Binary correctness: right or wrong, no nuance
- The metric for office pools, survivor leagues, and pick'em contests
- Easy to understand and communicate

**What it doesn't measure:**
- How confident you were in predictions
- Whether you were "close" on losses
- The quality of your probability estimates

**When to prioritize:**
- You need to pick one winner per game
- There's no reward for confidence calibration
- Results are purely binary (win/loss tracking)

**Interpretation guide:**
```
< 50%:  Worse than coin flip (broken model)
≈ 50%:  Random guessing
≈ 56%:  "Always pick home team" baseline
≈ 60%:  Decent model, beats simple baselines
≈ 67%:  Good model, competitive performance
≈ 70%:  Strong model, professional threshold
≈ 73%:  Market favorite baseline (Vegas)
> 75%:  Exceptional (extremely rare without paid data)
```

**Key insight:** Accuracy has a ceiling determined by available data. With free public data, 69-73% is realistic. Professional models with proprietary data may reach 75%, but gains beyond that are rare and expensive.

---

### Log-Loss (Cross-Entropy): "How Good Are Our Probabilities?"

**Definition:** A measure of how well predicted probabilities match actual outcomes. Lower is better.

```
Log-Loss = -mean(y*log(p) + (1-y)*log(1-p))

Where:
- y = actual outcome (1 for home win, 0 for away win)
- p = predicted probability of home team winning
```

**Example:**
```
Game 1: Predict 70% home win → Home wins
  Loss = -log(0.70) = 0.357

Game 2: Predict 80% home win → Away wins  
  Loss = -log(0.20) = 1.609 (large penalty!)

Game 3: Predict 55% home win → Home wins
  Loss = -log(0.55) = 0.598

Mean log-loss = (0.357 + 1.609 + 0.598) / 3 = 0.855
```

**What it measures:**
- **Calibration:** Do predicted probabilities match observed frequencies?
- **Confidence quality:** Are you appropriately confident or overconfident?
- **Probabilistic accuracy:** Not just "right or wrong," but "how right or how wrong"

**What it doesn't measure:**
- Binary accuracy (you can have good log-loss with 50% accuracy if probabilities are honest)
- Practical performance in winner-only contests

**When to prioritize:**
- You're making decisions based on probabilities (bet sizing, confidence weighting)
- You want calibrated uncertainty estimates
- You're building a probability engine, not just a classifier

**Interpretation guide:**
```
> 0.693:  Worse than 50/50 predictions (broken calibration)
≈ 0.693:  Uniform 50/50 predictions (no signal)
≈ 0.600:  Moderate calibration
≈ 0.574:  Good calibration (market baseline)
≈ 0.571:  Strong calibration (beats market)
≈ 0.550:  Excellent calibration (rare)
< 0.500:  Exceptional (extremely rare)
```

**Why it penalizes overconfidence:**
```
Correct prediction at 60% confidence: -log(0.60) = 0.511
Correct prediction at 90% confidence: -log(0.90) = 0.105
Wrong prediction at 90% confidence:   -log(0.10) = 2.303

→ Being confidently wrong is heavily penalized
→ Encourages honest probability estimates
```

**Key insight:** You can have worse accuracy but better log-loss than a baseline. This happens when your probabilities are more honest, even if you pick slightly fewer winners.

---

## The Accuracy vs. Log-Loss Trade-off

Models often face a trade-off between maximizing accuracy and optimizing probability calibration.

### Scenario 1: Accuracy Winner, Calibration Loser

```
Model A (Optimized for accuracy):
- Accuracy: 73.31%
- Mean log-loss: 0.5756
- Strategy: Always confident, aggressive thresholds

Behavior:
- Predicts 85% when model shows 65% → Gets more picks right
- But when wrong, pays heavy log-loss penalty
- Probabilities don't reflect true uncertainty
```

**Use case:** Office pools where only binary picks matter

### Scenario 2: Calibration Winner, Accuracy Loser

```
Model B (Optimized for calibration):
- Accuracy: 72.95%
- Mean log-loss: 0.5711
- Strategy: Honest probabilities, conservative thresholds

Behavior:
- Predicts 65% when internal estimate is 65%
- Slightly fewer correct picks (more coin flips)
- But probabilities match observed frequencies
```

**Use case:** Decision support systems, probability-based strategies

### Scenario 3: The Balanced Sweet Spot

```
Model C (Balanced):
- Accuracy: 73.31% (matches market)
- Mean log-loss: 0.5711 (beats market)
- Strategy: Accurate + calibrated

Behavior:
- Matches market on binary picks
- Provides better probability estimates
- Best of both worlds
```

**Use case:** This is the ideal target for most applications

---

## Comparing Against Baselines

Your model doesn't exist in a vacuum. Performance must be evaluated relative to trivial strategies and sophisticated benchmarks.

### Baseline 1: Coin Flip (50%)

**Strategy:** Flip a coin for every game.

**Expected performance:**
- Accuracy: ≈50%
- Log-loss: 0.693 (assuming 50/50 probabilities)

**Interpretation:**
- **Any model worse than this is broken**
- This is the absolute floor
- Even a trivial "always pick home team" beats this

### Baseline 2: Home Team Advantage (56.4%)

**Strategy:** Always pick the home team to win.

**Expected performance:**
- Accuracy: ≈56.4% (historical home win rate)
- Log-loss: 0.637 (if predicting 56% for all home teams)

**Interpretation:**
- Exploits known structural advantage
- Surprisingly hard to beat with features alone
- If you can't beat this, your features lack signal

### Baseline 3: Market Favorite (≈73%)

**Strategy:** Pick whichever team Vegas favors (based on spread or moneyline).

**Expected performance:**
- Accuracy: ≈73.3%
- Log-loss: ≈0.5737

**Interpretation:**
- **This is your real opponent**
- Vegas aggregates expert knowledge, sharp money, and proprietary data
- Beating this consistently with free data is very difficult
- This is what "professional" performance looks like

### Baseline 4: ELO-Only Model (≈66%)

**Strategy:** Predict based purely on team strength ratings.

**Expected performance:**
- Accuracy: ≈65-67%
- Log-loss: ≈0.610

**Interpretation:**
- Tests if you can capture basic team quality
- Should beat home team advantage
- Won't beat market without additional features

---

## Reference: Current NFL Prediction Performance

Note: “Apple” refers to the calibrated residual LightGBM configuration trained with scripts/05_train_model.py using the tag `--tag apple` (see logs/cv_results_apple.md for a sample CV run).

Based on extensive testing with this system:

### Market Baseline (Vegas Favorite)
```
Games: 272 (2024 season simulation)
Accuracy: 73.31%
Mean log-loss: 0.5737
Weekly variance: 58% to 81%

Strengths:
- Incorporates all public information
- Adjusted for injuries, weather, line movement
- Aggregates expert and sharp bettor knowledge

Weaknesses:
- Susceptible to public bias in some games
- Can be beaten on calibration, rarely on accuracy
```

### Apple System (Best Calibration)
```
Configuration: Optimized for log-loss
Games: 272 (2024 season simulation)
Accuracy: 72.95% (-0.36% vs market)
Mean log-loss: 0.5711 (-0.0026 vs market)
Weekly variance: 55% to 79%

Interpretation:
- Slightly fewer correct picks than market
- But better calibrated probabilities
- More honest uncertainty estimates
- Better for probability-based decisions
```

### Apple System (Tied Accuracy)
```
Configuration: Optimized for accuracy
Games: 272 (2024 season simulation)  
Accuracy: 73.31% (matches market)
Mean log-loss: 0.5756 (+0.0019 vs market)
Weekly variance: 58% to 81%

Interpretation:
- Matches market on binary picks
- Slightly worse probability calibration
- Best for office pools where only picks matter
```

### Apple System (Aggressive Triage)
```
Configuration: Top 6 features, high importance threshold
Games: 272 (2024 season simulation)
Accuracy: 72.60% (-0.71% vs market)
Mean log-loss: 0.5837 (+0.0100 vs market)
Weekly variance: 53% to 78%

Interpretation:
- Removed too much signal
- Lost both accuracy and calibration
- Demonstrates danger of over-simplification
```

---

## Override Analysis: When Models Disagree

A critical metric is understanding what happens when your model disagrees with the market favorite.

### Without Flip Guard

```
Games where model flips: 27 / 272 (9.9%)
Accuracy on flips: 12 / 27 (44.4%)
Average edge: 0.03 (3 percentage points)

Interpretation:
→ Overrides underperform the market
→ Thin edges lead to noise, not signal
→ Need gating mechanism
```

### With 8% Flip Guard

```
Minimum edge required to flip: 8 percentage points
Games passing threshold: 11 / 272 (4.0%)
Accuracy on flips: 6 / 11 (54.5%)
Average edge: 0.12 (12 percentage points)

Interpretation:
→ Higher bar reduces noise
→ Strong disagreements show promise
→ Better risk management
```

**Lesson:** Don't flip against the market unless you have a substantial, principled reason. Thin edges (< 5%) are usually noise.

---

## Calibration Deep Dive

### What is Calibration?

A model is **well-calibrated** if its predicted probabilities match observed frequencies.

**Example of good calibration:**
```
Games predicted at 60% confidence: 100
Actual wins: 61
Calibration error: 1%  ✓
```

**Example of poor calibration (overconfident):**
```
Games predicted at 80% confidence: 100  
Actual wins: 68
Calibration error: 12%  ✗
```

### Calibration Curve

Plot predicted probability bins against actual win rates:

```
Pred Range    Count    Actual    Error
[0.50-0.55]   42       0.52     -0.01  ✓
[0.55-0.60]   68       0.59     +0.01  ✓
[0.60-0.65]   55       0.62     +0.02  ✓
[0.65-0.70]   48       0.71     +0.06  ✗ Underconfident
[0.70-0.75]   35       0.81     +0.11  ✗ Very underconfident
[0.75-0.80]   18       0.83     +0.13  ✗ Very underconfident
[0.80-0.85]   6        0.67     -0.13  ✗ Overconfident

→ Model underconfident on strong picks
→ Model overconfident on very strong picks
```

### Improving Calibration: Platt Scaling

**When to apply:**
- Calibration curve shows systematic bias
- Accuracy is good but log-loss is poor
- After ensembling (often miscalibrates)

**How it works:**
1. Train your base model
2. Fit a logistic regression on validation set predictions
3. Use this to transform probabilities

**Example:**
```python
from sklearn.calibration import CalibratedClassifierCV

base_model = LogisticRegression(C=1.0)
calibrated = CalibratedClassifierCV(
    base_model, 
    method='sigmoid',  # Platt scaling
    cv=5
)

calibrated.fit(X_train, y_train)
probs = calibrated.predict_proba(X_test)
```

**Result:**
```
Before Platt scaling:
- Accuracy: 73.31%
- Log-loss: 0.5756
- Calibration error: 0.047

After Platt scaling:
- Accuracy: 72.95% (-0.36%)
- Log-loss: 0.5711 (-0.0045)  ✓
- Calibration error: 0.031 (-0.016)  ✓

→ Trade slight accuracy loss for better probabilities
```

---

## Brier Score: Alternative Probability Metric

**Definition:** Mean squared error of probability predictions.

```
Brier Score = mean((predicted_prob - actual_outcome)²)
```

**Example:**
```
Game 1: Predict 0.70 → Home wins (1)
  Error = (0.70 - 1)² = 0.09

Game 2: Predict 0.80 → Away wins (0)
  Error = (0.80 - 0)² = 0.64

Mean Brier = (0.09 + 0.64) / 2 = 0.365
```

**Interpretation:**
```
Perfect prediction: 0.00
Random 50/50: 0.25
Poor prediction: > 0.30
```

**Comparison to log-loss:**
- Brier score is more intuitive (squared error)
- Log-loss penalizes confident wrong predictions more heavily
- Both measure calibration, use either consistently

---

## Weekly Variance: Understanding Natural Fluctuations

NFL games are inherently unpredictable. Week-to-week variance is **expected**, not a model failure.

### Typical Weekly Performance

```
Week   Games   Accuracy   Notes
1      16      56.2%      Early season uncertainty
2      16      68.8%      
3      15      73.3%      
4      16      75.0%      Model hitting stride
5      15      66.7%      Variance
6      16      81.2%      Exceptionally good week
7      16      62.5%      Variance
8      15      73.3%      
9      16      68.8%      
10     16      75.0%      
11     15      60.0%      Multiple upsets
12     16      71.9%      
13     16      68.8%      
14     15      73.3%      
15     16      75.0%      
16     16      68.8%      
17     15      73.3%      

Season average: 69.5%
Weekly std dev: 5.8%
```

**Key insights:**
- Individual weeks range from 56% to 81%
- This is **normal** NFL variance, not model failure
- Judge performance over full season
- Weeks 1-4 typically have lower accuracy
- Playoff weeks (15-17) can have higher variance

**Red flags that indicate actual problems:**
- Sustained accuracy < 60% over 4+ weeks
- Log-loss consistently > 0.70
- Performance degrades as season progresses
- Can't beat "always pick home team"

---

## Decision Framework: Which Metric to Optimize?

### Use Case 1: Office Pool (Binary Picks Only)

**Primary metric:** Accuracy  
**Secondary metric:** Log-loss (tie-breaker)

**Strategy:**
- Optimize for accuracy first
- Only consider calibration if tied on accuracy
- Don't sacrifice picks for probability quality

**Model selection:**
```
Model A: 73.31% accuracy, 0.5756 log-loss ✓ Choose this
Model B: 72.95% accuracy, 0.5711 log-loss ✗ Don't choose
Model C: 73.31% accuracy, 0.5711 log-loss ✓ Even better!
```

### Use Case 2: Probability-Based Decision Making

**Primary metric:** Log-loss  
**Secondary metric:** Accuracy (validation only)

**Strategy:**
- Optimize for calibration first
- Accept slight accuracy loss for better probabilities
- Use probabilities to size decisions, manage risk

**Model selection:**
```
Model A: 73.31% accuracy, 0.5756 log-loss ✗ Don't choose
Model B: 72.95% accuracy, 0.5711 log-loss ✓ Choose this
Model C: 73.31% accuracy, 0.5711 log-loss ✓ Even better!
```

### Use Case 3: Contrarian Analysis

**Primary metric:** Override accuracy when edge > threshold  
**Secondary metrics:** Overall accuracy, log-loss

**Strategy:**
- Focus on games where model strongly disagrees with market
- Measure override accuracy separately
- Use flip guard to reduce noise

**Model selection:**
```
Model    Overall Acc    Override Acc (8%+ edge)    Choose?
A        73.31%         44.4% (9.9% of games)      ✗
B        72.95%         54.5% (4.0% of games)      ✓
C        73.31%         61.2% (2.2% of games)      ✓✓

→ Choose C: Matches market on most games, 
   beats market on strong disagreements
```

---

## Practical Guidelines

### 1. Always Compare to Market Baseline

Don't celebrate 70% accuracy in isolation. If the market achieves 73%, you're underperforming.

```
Your accuracy: 70%
Market accuracy: 73%
Verdict: Underperforming, needs improvement
```

### 2. Track Both Metrics

Even if you only care about accuracy, monitor log-loss to:
- Detect overconfidence issues
- Identify calibration drift over time
- Validate probability estimates

### 3. Use Confidence Intervals

A single season's accuracy has variance. Report with confidence bounds:

```
Accuracy: 69.5% [95% CI: 66.8%, 72.1%]
```

This tells you:
- Point estimate: 69.5%
- Plausible range: 67-72%
- True performance could be anywhere in this range

### 4. Separate Train, Validation, Test

```
Training set metrics: 
→ Tell you if the model learned

Validation set metrics:
→ Guide hyperparameter tuning

Test set metrics:
→ Estimate real-world performance

Only report test set metrics as "performance"
```

### 5. Accept the Ceiling

With free public data:
- Accuracy ceiling: ≈70-73%
- Log-loss ceiling: ≈0.571

Further gains require:
- Paid data sources
- Proprietary signals
- Institutional infrastructure

**Don't chase 75%+ accuracy without budget.**

---

## Summary

| Metric | Measures | When Primary | Good Value | Great Value |
|--------|----------|-------------|-----------|------------|
| **Accuracy** | % correct picks | Office pools, binary contests | 67% | 70% |
| **Log-loss** | Probability quality | Decision support, bet sizing | 0.580 | 0.571 |
| **Override Accuracy** | Contrarian picks | Market inefficiency hunting | 55% | 60% |
| **Brier Score** | Squared prob error | Alt to log-loss | 0.20 | 0.18 |

**Golden rules:**
1. Beat the market baseline (73%) or optimize calibration
2. Track both accuracy and log-loss
3. Understand weekly variance is normal
4. Use confidence intervals
5. Accept the ceiling with free data

---

*Metrics are tools, not goals. Choose the right metric for your use case, measure honestly, and don't over-optimize for noise.*
