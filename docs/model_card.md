# Model Card: CLV + Churn Risk System

**Version:** 2.0.0
**Author:** Ramesh Shrestha
**Last Updated:** 2026-03-06
**Project:** Customer Lifetime Value with Long-Term Optimization

---

## 1. Model Overview

This document describes the predictive models that form the core of the CLV optimization system:

| Model | Type | Purpose |
|---|---|---|
| **BG/NBD** | Probabilistic (Bayesian) | Forecast future purchase frequency and P(alive) |
| **Gamma-Gamma** | Probabilistic (Bayesian) | Forecast expected average transaction value |
| **Churn Risk (RF)** | Random Forest (auto-selected) | Estimate probability of near-term inactivity |

The churn model is selected via automated multi-model comparison across 4 candidate algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM). As of v2.0, Random Forest was selected with CV ROC AUC = 0.816 ± 0.013.

These models are combined into a single decision policy that selects customers for retention targeting under a budget constraint. RFM segmentation, cohort analysis, and Monte Carlo sensitivity analysis add additional business intelligence layers.

---

## 2. Intended Use

### Primary Use Case
Generate a ranked customer targeting list for retention campaigns under a fixed budget. The list is produced weekly (or on demand) and passed to CRM/marketing teams for execution.

### Intended Users
- Marketing and CRM teams (targeting list consumers)
- Data science and analytics teams (model operators)
- Finance/strategy (ROI review and budget planning)

### Out-of-Scope Uses
- Individual-level credit scoring or financial eligibility decisions
- Real-time scoring (this pipeline is batch; latency is measured in minutes)
- Channels requiring individual causal proof of uplift (use A/B testing first)

---

## 3. Training Data

| Property | Detail |
|---|---|
| **Source** | UCI Online Retail II (public transactional dataset) |
| **Period** | December 2009 – December 2011 |
| **Cutoff Date** | 2011-06-01 (configurable) |
| **Calibration Window** | All transactions before cutoff date |
| **Holdout Window** | 180 days post-cutoff (configurable) |
| **Customers scored** | ~4,933 (post-cleaning) |
| **Geography** | Primarily UK, with international transactions |

### Data Quality Notes
The dataset contains known issues that are handled explicitly:
- Missing customer IDs (~25% of raw rows) — removed in cleaning
- Cancellation invoices (`C`-prefix) — removed in cleaning
- Non-positive quantities and prices — removed in cleaning
- Duplicate rows — deduped in cleaning

---

## 4. CLV Models (BG/NBD + Gamma-Gamma)

### 4.1 Model Description

**BG/NBD** (Beta-Geometric / Negative Binomial Distribution):
- Models the purchase frequency process as Poisson with rate λᵢ ~ Gamma(r, α)
- Models the dropout process as Geometric with dropout probability pᵢ ~ Beta(a, b)
- Outputs: expected number of future purchases, P(alive)

**Gamma-Gamma** (monetary value component):
- Models per-transaction value Mᵢₖ | νᵢ ~ Gamma(p, νᵢ)
- Customer heterogeneity: νᵢ ~ Gamma(q, γ)
- Requires ≥ 2 repeat purchases for fitting
- Outputs: expected average invoice value per customer

**CLV Approximation:**
```
CLV_i(H) ≈ E[N_i(H)] × E[μ_i] × discount_factor
discount_factor = 1 / (1 + d_daily)^H
```

### 4.2 Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `penalizer_coef` (BG/NBD) | 0.001 | Light L2 regularization for stability |
| `penalizer_coef` (GG) | 0.001 | Light L2 regularization for stability |
| `discount_rate_annual` | 10% | Standard WACC proxy |
| `clv_horizon_days` | 180 | 6-month forward planning window |

### 4.3 Evaluation

**Primary metric:** Decile lift — average holdout revenue should increase monotonically with predicted CLV decile.
**Secondary metric:** Spearman rank correlation (ρ) between predicted CLV and actual holdout revenue.

| Metric | Observed Value |
|---|---|
| Spearman ρ (CLV vs holdout revenue) | 0.57 |
| Top-decile avg holdout revenue | £5,339 |
| Lift: Decile 10 vs Decile 1 | 22× |
| Top decile vs population mean | 10× |

The 95% bootstrap confidence intervals on decile lift (500 resamples) are included in `reports/tables/clv_decile_lift.csv` to quantify statistical uncertainty.

---

## 5. Churn Risk Model

### 5.1 Model Selection (Automated Multi-Model Comparison)

Version 2.0 introduces automated model selection across 4 candidate algorithms. Each is evaluated under 5-fold stratified cross-validation; the winner (best mean CV ROC AUC) is automatically selected and used for all downstream scoring.

| Model | CV ROC AUC | CV Avg Precision | Notes |
|---|---|---|---|
| **Random Forest** | **0.816 ± 0.013** | **0.883 ± 0.007** | **Winner — selected** |
| Logistic Regression | 0.810 ± 0.015 | 0.882 ± 0.009 | Strong linear baseline |
| XGBoost | 0.807 ± 0.014 | 0.878 ± 0.008 | Gradient boosting |
| LightGBM | 0.792 ± 0.014 | 0.866 ± 0.008 | Fastest; slightly lower AUC |

**Holdout performance (Random Forest):**
- ROC AUC = 0.831
- Average Precision = 0.878
- CV → holdout generalization gap: +0.015 AUC (no overfitting)

### 5.2 Model Description (Random Forest)

**Algorithm:** Random Forest (sklearn `RandomForestClassifier`)
**Preprocessing:** Median imputation only (tree-based models are scale-invariant)
**Class weighting:** `class_weight="balanced"` (handles churn/non-churn imbalance)
**Random state:** 42 (fully reproducible)

**Churn Label Definition:**
```
churn_label_i = 1  if  no purchase in (cutoff, cutoff + inactivity_days]
churn_label_i = 0  otherwise
```

This is an **inactivity-based proxy** — explicit churn events are not available in this dataset.

### 5.3 Features

| Feature | Description | SHAP Direction |
|---|---|---|
| `recency_days` | Days since last purchase | ↑ value → ↑ churn (top driver) |
| `tenure_days` | Days since first purchase | ↑ value → varies (non-linear) |
| `n_invoices` | Number of distinct invoices | ↑ value → ↓ churn |
| `total_revenue` | Cumulative spend | ↑ value → ↓ churn |
| `avg_order_value` | Mean spend per invoice | ↑ value → ↓ churn (moderate) |
| `revenue_last_30d` | Revenue in trailing 30 days | ↑ value → ↓ churn |
| `revenue_last_90d` | Revenue in trailing 90 days | ↑ value → ↓ churn (strong) |
| `rev_30_to_90_ratio` | Recent vs medium-term revenue | ↑ value → ↓ churn (momentum) |

**SHAP Interpretation:** Feature directions above are derived from SHAP values computed via `TreeExplainer` on the selected Random Forest model. `recency_days` has the highest mean |SHAP| value and is the dominant predictor.

### 5.4 Evaluation Strategy (Time-Safe)

Training and evaluation use time-based splits to prevent data leakage:

```
Train cutoff:  C          (features built at C; labels from (C, C + 180d])
Test  cutoff:  C + 60d    (same features; labels from (C+60d, C+240d])
```

This mimics production: the model is trained on an earlier cohort and validated on a later one.

**Additionally:** 5-fold stratified cross-validation on the training data provides stability estimates (mean ± std AUC) for model comparison.

### 5.5 Evaluation Artifacts

| Artifact | Location |
|---|---|
| Model comparison table | `reports/tables/churn_model_comparison.csv` |
| Feature importance (SHAP) | `reports/tables/churn_feature_importance.csv` |
| ROC + PR curves | `reports/figures/churn_roc_pr_curves.png` |
| Calibration curve | `reports/figures/churn_calibration_curve.png` |
| SHAP summary plot | `reports/figures/churn_shap_summary.png` |
| Model comparison bar chart | `reports/figures/churn_model_comparison.png` |

### 5.6 Calibration

The Random Forest model was evaluated with a reliability diagram (calibration curve). The model outputs probability estimates suitable for use in business calculations (expected gain = CLV × churn_prob × effectiveness).

---

## 6. RFM Customer Segmentation

### 6.1 Methodology

Customers are scored on Recency, Frequency, and Monetary value using quartile ranking (1–4 scale). RFM scores are combined using a priority rule matrix to assign named segments.

**Segment definitions (priority-ordered):**

| Segment | Recency | Frequency | Description |
|---|---|---|---|
| Champions | R=4 | F=4 | Bought recently, buy often, highest spenders |
| Loyal Customers | R≥3 | F≥3 | Regular buyers with high engagement |
| Potential Loyalists | R≥3 | F≤2 | Recent buyers who haven't yet built frequency |
| New Customers | R=4 | F=1 | First-time buyers acquired recently |
| At Risk | R≤2 | F≥3 | Were Champions/Loyal; haven't bought recently |
| Cant Lose Them | R=1 | F=4 | High-frequency buyers who've gone silent |
| Hibernating | R≤2 | F≤2 | Infrequent + inactive |
| Lost | R=1 | F=1 | Very inactive, low frequency |

### 6.2 Segment Results

| Segment | Customers | % Customers | Avg CLV | Avg Churn Prob | % Revenue |
|---|---|---|---|---|---|
| Champions | 681 | 13.8% | £2,436 | 13% | 55.8% |
| Loyal Customers | 1,049 | 21.3% | £761 | 36% | 22.3% |
| Potential Loyalists | 407 | 8.3% | −£145 | 49% | 2.1% |
| New Customers | 99 | 2.0% | −£2,992 | 49% | 0.3% |
| At Risk | 694 | 14.1% | £309 | 64% | 10.3% |
| Cant Lose Them | 42 | 0.9% | £302 | 62% | 1.5% |
| Hibernating | 952 | 19.3% | −£803 | 69% | 4.0% |
| Lost | 1,009 | 20.5% | −£438 | 86% | 3.7% |

**Key insight:** 35.1% of customers (Champions + Loyal) generate 78.1% of all revenue. The At Risk + Cant Lose Them segments represent 11.8% of revenue that is at elevated churn risk (62–64% churn probability).

---

## 7. Budget Optimization Model

### 7.1 Formulation

This is not a learned model but a deterministic optimization:

**Objective:** Maximize total expected net gain subject to budget constraint.

```
maximize  Σᵢ xᵢ · net_gain_i
s.t.      Σᵢ xᵢ · cost_i ≤ B
          xᵢ ∈ {0, 1}
```

**Economic proxy:**
```
expected_benefit_i = CLV_i × churn_prob_i × retention_effectiveness
net_gain_i         = expected_benefit_i - cost_i
```

**Solver:** PuLP (CBC exact integer programming) with greedy fallback.

### 7.2 Assumptions

| Assumption | Value | Source |
|---|---|---|
| Retention effectiveness (η) | 10% | Industry assumption; should be estimated via A/B testing |
| Unit cost per customer | £2.00 | Configurable; represents marginal campaign contact cost |
| CLV horizon | 180 days | Forward-looking planning window |

### 7.3 Sensitivity Analysis (Monte Carlo)

1,000 Monte Carlo simulations vary two key assumptions simultaneously:
- `retention_effectiveness` ~ Uniform(5%, 25%)
- `unit_cost` ~ Uniform(£1, £5)

| Budget | Median ROI | 90% CI (p5–p95) |
|---|---|---|
| £500 | ~18× | 10×–26× |
| £1,000 | ~18× | 10×–26× |
| £2,000 | ~18× | 10×–26× |
| £5,000 | ~14× | 8×–20× |

The ROI range holds robust even under pessimistic assumptions (5% effectiveness, £5/customer cost), demonstrating the resilience of the targeting policy.

---

## 8. Limitations and Risks

### Statistical Limitations
- **Non-causal estimates**: CLV and churn models are predictive, not causal. Expected gains represent potential impact — actual impact depends on intervention quality.
- **Distributional shift**: Model performance may degrade if customer behavior changes significantly (seasonality, competitive entry, economic shocks).
- **Inactivity proxy**: Churn is operationalized as inactivity. This may mis-classify customers who buy infrequently by nature (seasonal buyers).
- **SHAP on observational data**: SHAP values reflect correlation, not causation. `recency_days` being the top SHAP feature does not mean reducing recency will mechanically reduce churn.

### Business Limitations
- **Retention effectiveness is assumed**: The 10% effectiveness rate is a placeholder. This should be replaced with empirically estimated uplift from controlled experiments.
- **Homogeneous cost structure**: All customers have the same intervention cost. Real campaigns have variable costs by channel.
- **No channel modeling**: Email, SMS, and call treatments are not differentiated.

### Fairness Considerations
- The model does not use demographic features (age, gender, location beyond country). Targeting is based solely on behavioral signals.
- High-value customers in certain geographic regions may receive systematically more attention — operators should monitor geographic distribution of targeted customers.

---

## 9. Monitoring Recommendations

For production deployment, the following monitoring should be implemented:

| Signal | Metric | Frequency |
|---|---|---|
| Data quality | Missing customer_id rate, duplicate rate | Each batch run |
| Feature drift | PSI (Population Stability Index) per feature | Weekly |
| Score drift | Mean/std of CLV and churn scores over time | Weekly |
| Model performance | AUC on any available outcome labels | Monthly |
| Business outcome | Actual retention rate of targeted cohort | Campaign cycle |
| Segment drift | % of customers per RFM segment | Monthly |

**Trigger thresholds:** Retrain if PSI > 0.25 for any feature, or if AUC drops > 5 percentage points from baseline.

**Model selection re-run:** Re-evaluate all 4 candidate models whenever retraining is triggered. The winning algorithm may change as data evolves.

---

## 10. Production Readiness Checklist

| Item | Status |
|---|---|
| Time-safe train/test split | ✅ Implemented |
| Reproducible random seeds | ✅ `random_state=42` |
| MLflow experiment tracking | ✅ Implemented |
| Cross-validation (training stability) | ✅ 5-fold stratified CV |
| Multi-model comparison | ✅ 4 algorithms, auto-selection |
| SHAP feature importance | ✅ TreeExplainer + beeswarm |
| Model calibration | ✅ Reliability diagram |
| Bootstrap confidence intervals | ✅ 95% CI on decile lift |
| RFM customer segmentation | ✅ 8 named segments |
| Cohort retention analysis | ✅ Monthly acquisition cohorts |
| Monte Carlo sensitivity analysis | ✅ 1,000 simulations, tornado chart |
| Unit tests | ✅ Core functions covered |
| Config-driven pipeline | ✅ YAML configs |
| CLI-driven scripts | ✅ All steps |
| A/B testing integration | ❌ Requires controlled experiment |
| Real-time serving | ❌ Batch-only (weekly cadence) |
| Channel-specific modeling | ❌ Planned future work |

---

## 11. Citation and References

- Fader, P.S., Hardie, B.G.S., & Lee, K.L. (2005). *"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model*. Marketing Science.
- Fader, P.S. & Hardie, B.G.S. (2013). *The Gamma-Gamma Model of Monetary Value*. SSRN Working Paper.
- Lundberg, S.M. & Lee, S.I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS. (SHAP)
- Lifetimes library: Cameron Davidson-Pilon et al. [github.com/CamDavidsonPilon/lifetimes](https://github.com/CamDavidsonPilon/lifetimes)

---

*This model card follows the framework proposed by Mitchell et al. (2019), "Model Cards for Model Reporting", FAccT.*
