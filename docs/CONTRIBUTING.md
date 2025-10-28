# Contributing to NFL Game Predictor

Thank you for your interest in contributing! This project exists to explore rigorous ML methodology in a competitive domain, and we welcome contributions that advance that goal.

## 🎯 Project Mission

This is an **educational platform**, not a commercial betting system. Contributions should prioritize:

1. **Learning value:** Does this help others understand ML concepts?
2. **Methodological rigor:** Does this demonstrate best practices?
3. **Honest evaluation:** Does this measure performance fairly?
4. **Ethical use:** Does this discourage gambling and emphasize learning?

## Before You Contribute

### Read the Core Documentation

- **[MANIFESTO.md](MANIFESTO.md)** - Understand the project philosophy and warnings
- **[TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md)** - Learn our validation approach
- **[METRICS_GUIDE.md](METRICS_GUIDE.md)** - Understand how we evaluate performance

### Check Existing Issues

Search [GitHub Issues](https://github.com/jwkehoe/NFL_Game_Picker/issues) to see if your idea has been discussed. If not, open an issue before investing significant time.

## Types of Contributions We Welcome

### 🔍 Data & Features

**Good:**
- New free data sources (with ethical scraping)
- Novel feature engineering approaches
- Data quality improvements

**Process:**
1. Document data source (URL, update frequency, license)
2. Show feature improves validation performance
3. Verify no data leakage (future information)
4. Update feature engineering script

**Example PR:**
```
Title: Add coaching continuity feature

- Source: Pro Football Reference (public)
- Feature: Years with current head coach
- Validation: +0.3% accuracy, -0.002 log-loss
- No leakage: Only uses historical coaching data
```

### 🤖 Models & Algorithms

**Good:**
- Alternative model architectures with evaluation
- Hyperparameter tuning experiments
- Ensemble approaches with calibration analysis

**Not Useful:**
- "I got 85% accuracy" without reproducible code
- Models trained on leaked future data
- Claims without proper validation

**Process:**
1. Implement on consistent train/test split
2. Compare to baseline models (Logistic Regression)
3. Report both accuracy and log-loss
4. Check calibration curve
5. Time the training/inference

**Example PR:**
```
Title: Add LightGBM as alternative to XGBoost

Performance (2024 test set):
- Accuracy: 72.8% (vs 73.0% LogReg baseline)
- Log-loss: 0.5723 (vs 0.5711 LogReg baseline)
- Training time: 12s (vs 45s XGBoost, 3s LogReg)

Conclusion: Slightly worse than baseline, included 
for educational comparison.
```

### 📊 Evaluation & Analysis

**Good:**
- Error analysis on specific game types
- Calibration improvements
- Visualization tools for predictions
- Interpretability methods (SHAP, feature importance)

**Process:**
1. Use existing trained models
2. Analyze on held-out test data only
3. Create reusable visualization/analysis scripts
4. Document findings clearly

**Example PR:**
```
Title: Analysis of division game predictions

Findings:
- Division games: 68.0% accuracy (vs 70.5% overall)
- Rivalry effect: -2.5% accuracy when teams split season series
- Recommendation: Add historical rivalry features

Code: notebooks/division_game_analysis.ipynb
```

### 📚 Documentation

**Good:**
- Tutorial notebooks for beginners
- Clarifications of existing docs
- Real-world use case examples
- FAQ additions

**Process:**
1. Write clearly and concisely
2. Include code examples
3. Test instructions on fresh environment
4. Link to related documentation

**Example PR:**
```
Title: Add tutorial notebook for first-time users

Contents:
- Load data and explore
- Train simple baseline model
- Make predictions for upcoming week
- Interpret results

Target: Complete beginner with Python/pandas knowledge
```

### 🧪 Testing & Validation

**Good:**
- Additional validation approaches
- Unit tests for feature engineering
- Integration tests for full pipeline
- Reproducibility checks

**Process:**
1. Follow existing test structure
2. Use pytest framework
3. Ensure tests are deterministic
4. Document what you're testing

**Example PR:**
```
Title: Add unit tests for ELO calculation

Tests:
- Initial rating assignment
- Home advantage application
- K-factor update logic
- Edge cases (blowouts, upsets)

Coverage: 95% of elo_rating.py
```

## What We Don't Accept

❌ **Gambling-focused features:** "Here's my betting strategy using this model"  
❌ **Proprietary data:** Paywalled sources, insider information  
❌ **Inflated claims:** "My model gets 90% accuracy!" without validation  
❌ **Data leakage:** Models trained on future information  
❌ **Uncommented code:** Submit readable, documented code  
❌ **Breaking changes:** Changes that break existing workflows without discussion

## Contribution Process

### 1. Fork & Clone

```bash
git clone https://github.com/jwkehoe/NFL_Game_Picker.git
cd NFL_Game_Picker
git checkout -b feature/your-feature-name
```

### 2. Set Up Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install pytest black flake8  # Testing dependencies
```

### 3. Make Changes

- Follow existing code style (PEP 8 for Python)
- Add docstrings to functions
- Update relevant documentation
- Write tests if applicable

### 4. Validate Changes

```bash
# Run tests
pytest tests/

# Check code style
flake8 scripts/
black --check scripts/

# Verify time-aware evaluation (no leakage)
python scripts/05_train_model.py --mode train --testgate 2025W7 --predictgate "2025 W8" --tag sanity_check
```

### 5. Commit with Clear Messages

```bash
git add .
git commit -m "Add coaching continuity feature

- Scrapes coaching data from PFR
- Adds years_with_coach feature
- Validates no data leakage
- Improves accuracy by 0.3%"
```

### 6. Push & Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- **Clear title** describing the change
- **Detailed description** of what and why
- **Validation results** (accuracy, log-loss, etc.)
- **Screenshots** if UI changes
- **Breaking changes** clearly marked

## Code Standards

### Python Style

- **PEP 8** compliance (use `black` for auto-formatting)
- **Type hints** for function signatures
- **Docstrings** for all public functions
- **Comments** for complex logic

```python
def calculate_elo_update(
    winner_elo: float,
    loser_elo: float,
    k_factor: int = 20,
    home_advantage: int = 65
) -> tuple[float, float]:
    """
    Calculate ELO rating updates after a game.
    
    Args:
        winner_elo: Current ELO rating of winning team
        loser_elo: Current ELO rating of losing team
        k_factor: Update rate (default: 20)
        home_advantage: ELO bonus for home team (default: 65)
        
    Returns:
        Tuple of (new_winner_elo, new_loser_elo)
        
    Example:
        >>> calculate_elo_update(1500, 1500)
        (1516, 1484)
    """
    # Implementation...
```

### File Organization

```
scripts/
  ├── data_collection.py     # Data scraping and API calls
  ├── feature_engineering.py # Feature calculation
  ├── model_training.py      # Model training logic
  └── prediction.py          # Inference pipeline

tests/
  ├── test_features.py       # Feature engineering tests
  ├── test_models.py         # Model training tests
  └── test_predictions.py    # Prediction pipeline tests

docs/
  ├── MANIFESTO.md           # Philosophy and warnings
  ├── TESTING_METHODOLOGY.md # Validation approach
  └── tutorials/             # Step-by-step guides
```

### Git Practices

- **Small commits:** One logical change per commit
- **Descriptive messages:** Explain what and why, not just what
- **Feature branches:** Never commit directly to `main`
- **Rebase before PR:** Keep history clean

## Pull Request Review Process

### What We Look For

✅ **Correctness:** Does the code work as intended?  
✅ **Testing:** Are there tests? Do they pass?  
✅ **Documentation:** Is the change documented?  
✅ **Style:** Does it follow project conventions?  
✅ **Performance:** Does it maintain or improve performance?  
✅ **Ethics:** Does it align with project mission?

### Timeline

- **Initial review:** Within 3-5 days
- **Follow-up:** Respond to feedback within 1 week
- **Merge:** After approval from 1+ maintainers

### If Your PR is Rejected

Don't take it personally! Common reasons:
- Not aligned with project goals
- Performance degradation without justification
- Data leakage or validation issues
- Insufficient testing or documentation

We'll explain why and suggest alternatives. You can:
- Revise based on feedback
- Open a discussion to clarify goals
- Fork the project for different direction

## Reporting Issues

### Bug Reports

Use this template:

```markdown
**Bug Description**
Clear description of the bug

**Steps to Reproduce**
1. Run command X
2. With data Y
3. Observe error Z

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happened

**Environment**
- OS: Ubuntu 22.04
- Python: 3.10.5
- Package versions: (from pip freeze)

**Additional Context**
Error messages, screenshots, etc.
```

### Feature Requests

Use this template:

```markdown
**Feature Description**
Clear description of proposed feature

**Motivation**
Why is this useful? What problem does it solve?

**Proposed Implementation**
How might this work?

**Alternatives Considered**
Other approaches you've thought about

**Additional Context**
Examples, references, etc.
```

## Recognition

Contributors will be:
- Credited in release notes
- Thanked publicly for significant contributions

Significant contributions may lead to co-authorship on academic papers if this project is published.

## Code of Conduct

### Our Standards

- **Respectful:** Treat all contributors with respect
- **Constructive:** Provide helpful, specific feedback
- **Inclusive:** Welcome contributors of all backgrounds
- **Professional:** Focus on code and ideas, not personal attributes
- **Educational:** Remember this is a learning platform

### Unacceptable Behavior

- **Harassment:** Personal attacks, trolling, insults
- **Promotion:** Advertising gambling services or betting tips
- **Spam:** Irrelevant issues, PRs, or comments
- **Bad faith:** Deliberate misinformation or data leakage

Violations will result in:
1. Warning + request to edit
2. Temporary ban (1 week)
3. Permanent ban

## Questions?

- **General questions:** [GitHub Discussions](https://github.com/jwkehoe/NFL_Game_Picker/discussions)
- **Bug reports:** [GitHub Issues](https://github.com/jwkehoe/NFL_Game_Picker/issues)
- **Private concerns:** open an Issue and tag @jwkehoe

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make this a valuable learning resource! 🎓

Remember: We're here to learn ML methodology, not to get rich betting on football. Keep that spirit in your contributions.
