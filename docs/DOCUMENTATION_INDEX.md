# Documentation Index

Welcome to the NFL Game Predictor documentation! This index will help you navigate to the right document for your needs.

## 📚 Core Documentation

### [README.md](README.md)
**Start here.** Project overview, quick start, and high-level summary.

**Read this if you:**
- Are new to the project
- Want a quick overview of features and performance
- Need installation instructions
- Want to understand project goals

**Key sections:**
- Quick start guide
- Performance benchmarks
- Project structure
- FAQ

---

### [MANIFESTO.md](MANIFESTO.md)
**Philosophy and principles.** Why this project exists, what it's for (and what it's NOT for), and the guiding principles that shaped it.

**Read this if you:**
- Want to understand the project's purpose
- Need the critical warnings about sports betting
- Are curious about methodology and best practices
- Want to apply these principles to other ML projects

**Key sections:**
- Critical warning against sports betting
- Why the NFL as a domain?
- 17 guiding principles for ML projects
- What works, what doesn't, and why

---

### [TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md)
**How to validate models properly.** A progressive, 12-level testing approach from smoke tests to production release.

**Read this if you:**
- Are training or evaluating models
- Want to avoid data leakage
- Need to understand validation strategies
- Are implementing walk-forward backtesting

**Key sections:**
- Testing pyramid (simple → complex)
- Time-aware validation techniques
- Feature triage and ablation studies
- Calibration analysis
- Override audits and error cohorts

---

### [METRICS_GUIDE.md](METRICS_GUIDE.md)
**Understanding and interpreting results.** Deep dive into accuracy, log-loss, calibration, and baseline comparisons.

**Read this if you:**
- Don't understand the difference between accuracy and log-loss
- Want to interpret model performance
- Need to compare against baselines
- Are deciding which metric to optimize

**Key sections:**
- Accuracy vs. log-loss trade-offs
- Baseline comparisons (market, ELO, home team)
- Calibration deep dive
- Weekly variance explanation
- Decision framework for metric selection

---

### [CONTRIBUTING.md](CONTRIBUTING.md)
**How to contribute to the project.** Guidelines, standards, and processes for contributions.

**Read this if you:**
- Want to contribute code or documentation
- Have found a bug to report
- Want to propose a new feature
- Need to understand code standards

**Key sections:**
- Types of welcome contributions
- Code standards and style guide
- Pull request process
- What we don't accept

---

## 🛠️ Practical Guides

### [SETUP.md](SETUP.md)
**Installation and environment configuration.**

### [WORKFLOW.md](WORKFLOW.md)
**Weekly prediction workflow.**

### [CLI_arguments.md](CLI_arguments.md)
**Script flags and examples.**

### [model_tracking.md](model_tracking.md)
**How to record runs and decisions.**

### [DISCUSSIONS.md](DISCUSSIONS.md)
**Welcome + posting guidelines for Discussions.**

### [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
**Common issues and fixes.**

### [weekly_backtest_report.md](weekly_backtest_report.md)
**Season‑long walk‑forward results and workflow notes.**

## 📖 Reading Paths

### Path 1: Complete Beginner

New to the project? Follow this sequence:

1. **[README.md](README.md)** - Understand what this is
2. **[MANIFESTO.md](MANIFESTO.md)** - Understand why this exists and critical warnings
3. **[SETUP.md](SETUP.md)** - Get your environment working
4. **[WORKFLOW.md](WORKFLOW.md)** - Make your first predictions
5. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - When things go wrong

---

### Path 2: ML Practitioner

Experienced in ML but new to sports prediction? Focus here:

1. **[MANIFESTO.md](MANIFESTO.md)** - Learn domain-specific considerations
2. **[TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md)** - Understand validation approach
3. **[METRICS_GUIDE.md](METRICS_GUIDE.md)** - Learn how to evaluate properly
4. **[README.md](README.md)** - Review architecture and current performance

---

### Path 3: Contributor

Want to improve the project? Start here:

1. **[README.md](README.md)** - Understand current capabilities
2. **[MANIFESTO.md](MANIFESTO.md)** - Align with project philosophy
3. **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Learn contribution guidelines
4. **[TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md)** - Validate your changes properly

---

### Path 4: Researcher

Studying sports prediction or ML methodology? Deep dive:

1. **[MANIFESTO.md](MANIFESTO.md)** - Understand principles and limitations
2. **[TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md)** - Study validation techniques
3. **[METRICS_GUIDE.md](METRICS_GUIDE.md)** - Understand evaluation metrics
4. Review code in `scripts/` and `notebooks/`

---

## 🎯 Quick Lookups

### "How do I...?"

| Task | Document | Section |
|------|----------|---------|
| Install the project | Root [README.md](../README.md) | Installation |
| Make predictions | [weekly_backtest_report.md](weekly_backtest_report.md) | Workflow |
| Train a model | [README.md](README.md) | Quick start |
| Understand accuracy | [METRICS_GUIDE.md](METRICS_GUIDE.md) | Accuracy section |
| Add features | [CONTRIBUTING.md](CONTRIBUTING.md) | Data & Features |
| Validate properly | [TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md) | Full guide |
| Understand warnings | [MANIFESTO.md](MANIFESTO.md) | Critical warning |

---

### "What does X mean?"

| Term | Document | Section |
|------|----------|---------|
| Log-loss | [METRICS_GUIDE.md](METRICS_GUIDE.md) | Log-loss section |
| Calibration | [METRICS_GUIDE.md](METRICS_GUIDE.md) | Calibration deep dive |
| Walk-forward | [TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md) | Level 5 |
| Data leakage | [TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md) | Level 1 |
| Flip guard | [METRICS_GUIDE.md](METRICS_GUIDE.md) | Override analysis |
| ELO | [README.md](../README.md) | Features |
| Baseline | [METRICS_GUIDE.md](METRICS_GUIDE.md) | Comparing baselines |

---

### "Why is X happening?"

| Issue | Document | Section |
|-------|----------|---------|
| Accuracy varies by week | [METRICS_GUIDE.md](METRICS_GUIDE.md) | Weekly variance |
| Can't beat Vegas | [MANIFESTO.md](MANIFESTO.md) | Why the NFL |
| Model overfits | [TESTING_METHODOLOGY.md](TESTING_METHODOLOGY.md) | Level 8 |
| Low accuracy early season | [METRICS_GUIDE.md](METRICS_GUIDE.md) | Weekly performance |
| Data loading fails | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Data errors |
| Predictions seem wrong | [WORKFLOW.md](WORKFLOW.md) | Validation |

---

## 📊 Key Concepts Map

```
                    NFL GAME PREDICTOR
                           |
        ┌──────────────────┼──────────────────┐
        |                  |                   |
    PHILOSOPHY         METHODS            EVALUATION
        |                  |                   |
  [MANIFESTO.md]   [TESTING_METHODOLOGY.md]  [METRICS_GUIDE.md]
        |                  |                   |
    - Warnings         - 12 levels         - Accuracy
    - Principles       - Time-aware        - Log-loss
    - Ethics           - Validation        - Calibration
                           |
                    ┌──────┴──────┐
                    |             |
                FEATURES       MODELS
                    |             |
                - ELO         - LogReg
                - Form        - XGBoost
                - Situational - Ensemble
```

---

## 🎓 Learning Objectives by Document

### After reading MANIFESTO.md, you should understand:
- Why this project exists (learning, not betting)
- Critical warnings about sports wagering
- 17 principles for rigorous ML development
- Why the NFL is a good learning domain
- What the data ceiling is with free sources

### After reading TESTING_METHODOLOGY.md, you should be able to:
- Design a progression from simple to complex testing
- Implement time-aware validation splits
- Avoid data leakage in temporal problems
- Conduct walk-forward backtesting
- Analyze error cohorts systematically

### After reading METRICS_GUIDE.md, you should be able to:
- Explain accuracy vs. log-loss trade-offs
- Interpret calibration curves
- Compare against appropriate baselines
- Understand weekly variance
- Choose the right metric for your use case

### After reading README.md, you should be able to:
- Install and run the project
- Understand current performance
- Know the project structure
- Find relevant documentation

### After reading CONTRIBUTING.md, you should be able to:
- Submit a well-structured pull request
- Follow code standards
- Write appropriate tests
- Document changes properly

---

## 🚀 Next Steps After Reading

### If you're a beginner:
1. ✅ Read README + MANIFESTO
2. ✅ Set up environment
3. ✅ Run the example in Quick Start
4. ✅ Make predictions for upcoming week
5. ➡️ Track performance and iterate

### If you're a practitioner:
1. ✅ Read MANIFESTO + TESTING + METRICS
2. ✅ Review existing code and experiments
3. ✅ Try alternative validation approaches
4. ✅ Experiment with new features
5. ➡️ Contribute improvements

### If you're a researcher:
1. ✅ Read all documentation thoroughly
2. ✅ Replicate published results
3. ✅ Identify gaps in methodology
4. ✅ Design experiments to test hypotheses
5. ➡️ Publish findings and share back

---

## 📝 Document Maintenance

These documents are **living documentation** and will evolve as the project develops.

**Last major update:** October 2024  
**Next review scheduled:** January 2025

**How to request documentation updates:**
1. Open an issue tagged `documentation`
2. Describe what's unclear or missing
3. Suggest improvements if you have them

**How to contribute documentation:**
1. Read [CONTRIBUTING.md](../CONTRIBUTING.md)
2. Make changes in a feature branch
3. Submit PR with clear description
4. Link to related issues

---

## 📬 Questions?

- **Clarification questions:** Open a [GitHub Discussion](https://github.com/jwkehoe/NFL_Game_Picker/discussions)
- **Documentation bugs:** Open a [GitHub Issue](https://github.com/jwkehoe/NFL_Game_Picker/issues) tagged `documentation`
- **Private inquiries:** open an Issue and tag @jwkehoe

---

*This documentation index was created to help you navigate the comprehensive documentation efficiently. If something is unclear or hard to find, let us know!*
