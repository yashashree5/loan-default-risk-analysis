# Loan Default Risk Analysis

End-to-end credit risk analysis on 255,347 loan records combining predictive 
modeling, statistical A/B testing, and survival analysis to identify high-risk 
borrowers and quantify default risk.

---

## Business Problem

A lending institution needs to identify which borrowers are likely to default 
before approving loans — and understand WHEN and WHY defaults happen. This 
project answers three questions:
1. **Who** will default? (ML Risk Model)
2. **Does co-signing reduce default?** (A/B Test)
3. **When** do borrowers default? (Survival Analysis)

---

## Key Business Findings

### 1. Unemployed Borrowers Are 43% Riskier
| Employment Type | Default Rate |
|----------------|-------------|
| Unemployed | 13.55% |
| Part-time | 11.97% |
| Self-employed | 11.46% |
| Full-time | 9.46% |

### 2. Young Borrowers Default at 22% — Drops to 13% by Age 37
- Age is the strongest predictor — older borrowers are dramatically more stable
- Rolling SQL analysis confirms danger zone is borrowers under 25

### 3. $265M in Loan Value at Risk From One Segment
- Unemployed + Business loan borrowers represent the highest concentration of risk
- Top 10 risk segments all involve unemployed or part-time borrowers

### 4. Risk Scorecard Separates Borrowers 9x
| Risk Tier | Default Rate |
|-----------|-------------|
| Low | 3.6% |
| Medium | 8.6% |
| High | 17.2% |
| Very High | 32.6% |

---

## A/B Test — Does a Co-Signer Reduce Default?

**Hypothesis:** Borrowers with a co-signer (Treatment) default less than 
those without (Control)

| Group | N | Default Rate |
|-------|---|-------------|
| Control (No Co-Signer) | 127,646 | 12.87% |
| Treatment (Has Co-Signer) | 127,701 | 10.36% |

**Results:**
- Absolute reduction: **2.51%**
- Relative lift: **19.48%**
- Chi-square p-value: **< 0.0001** Statistically significant
- 95% CI: [2.26%, 2.75%] — never crosses zero
- Effect size (Cohen's h): 0.078 (Small but real)
- Power achieved: **1.0** (49.9x minimum required sample)

**Conclusion:** Co-signers statistically and significantly reduce default risk.
Requiring co-signers for high-risk borrowers is a data-backed policy recommendation.

---

## Survival Analysis — When Do Borrowers Default?

Using Kaplan-Meier curves and Cox Proportional Hazards model on 255,347 loans:

**Survival probabilities:**
- Month 12: 97.8% of loans still active
- Month 36: 94.9% still active  
- Month 60: 85.6% still active — 14.4% defaulted by end of term

**Cox Model Hazard Ratios:**
| Factor | Hazard Ratio | Meaning |
|--------|-------------|---------|
| DTI Ratio | 1.24 | 24% higher default risk per unit increase |
| Unemployed | 1.24 | 24% higher default risk vs employed |
| Has Co-Signer | 0.80 | 20% LOWER default risk |
| Interest Rate | 1.06 | 6% higher risk per 1% rate increase |
| Age | 0.97 | 3% lower risk per year older |

**Log-rank test confirms** survival curves differ significantly between 
co-signer groups (p < 0.0001).

---

## Machine Learning Results

| Model | AUC | Default Recall* | Default F1* |
|-------|-----|----------------|------------|
| Logistic Regression | 0.6677 | 0.30 | 0.27 |
| Random Forest | 0.7151 | 0.13 | 0.19 |
| **XGBoost** | **0.7367** | **0.63** | **0.34** |

*At optimized threshold of 0.15

Threshold optimized from 0.5 → 0.15 improving recall from 12% to 63%.

---

## Project Structure
```
loan-default-risk-analysis/
├── data/
│   ├── raw/                      # Original dataset
│   └── processed/                # Cleaned + risk scored data
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory analysis
│   ├── 02_SQL_analysis.ipynb     # Risk segmentation + revenue at risk
│   ├── 03_risk_model.ipynb       # ML models + risk scorecard
│   ├── 04_ab_test.ipynb          # Statistical A/B test + power analysis
│   └── 05_survival_analysis.ipynb # Kaplan-Meier + Cox model
├── visuals/                      # All saved charts
└── README.md
```

## Tech Stack
- **Python** — pandas, numpy, scikit-learn, xgboost, lifelines
- **SQL** — SQLite with window functions and risk segmentation  
- **Statistics** — scipy, statsmodels (chi-square, z-test, power analysis)
- **Survival Analysis** — lifelines (Kaplan-Meier, Cox PH model)
- **Visualization** — matplotlib, seaborn

## Data Source
[Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)
— 255,347 loans, 18 features
