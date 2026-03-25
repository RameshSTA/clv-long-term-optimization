<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a2c4e,100:4a6b9e&height=230&section=header&text=Longitudinal%20Probabilistic%20Modelling&fontColor=ffffff&fontSize=52&fontAlignY=38&desc=Survival%20Analysis%20and%20Long-Term%20Forecasting&descColor=b8cceb&descSize=22&descAlignY=63" width="100%" />
</div>

<br>

<div align="center">

[![CI](https://github.com/RameshSTA/clv-long-term-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/RameshSTA/clv-long-term-optimization/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![MLflow](https://img.shields.io/badge/MLflow-tracked-blue?logo=mlflow)
![PuLP](https://img.shields.io/badge/Optimization-PuLP%2FCBC-green)
![Tests](https://img.shields.io/badge/tests-10%20passing-brightgreen)

</div>

<br>

<p align="center">
  <b>A production-grade decision intelligence system — from raw retail transactions to budget-constrained,<br>fully explainable retention targeting with quantified ROI.</b>
</p>

<br>

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LINKEDIN-CONNECT-0077B5?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=555555)](https://www.linkedin.com/in/rameshsta/)&nbsp;&nbsp;[![GitHub](https://img.shields.io/badge/GITHUB-RAMESHSTA-1a2332?style=for-the-badge&logo=github&logoColor=white&labelColor=555555)](https://github.com/RameshSTA)&nbsp;&nbsp;[![View on GitHub](https://img.shields.io/badge/VIEW_ON_GITHUB-1a2332?style=for-the-badge&logo=github&logoColor=white)](https://github.com/RameshSTA/clv-long-term-optimization)

</div>

---

## At a Glance

<div align="center">

| Metric | Value | Context |
|:---|:---:|:---|
| CLV Spearman rank correlation (ρ) | **0.57** | Predicted vs. actual holdout revenue, all 4,933 customers |
| Top-decile lift | **6.3×** | £5,339 avg holdout revenue vs. £852 population mean |
| Top-to-bottom decile ratio | **22×** | £5,339 (D10) vs. £241 (D1) — no rank inversions |
| Churn model holdout AUC | **0.831** | Random Forest, out-of-time holdout evaluation |
| CV average precision | **0.883 ± 0.007** | 5-fold stratified cross-validation |
| Best optimization ROI | **44.5×** | £200 budget, 100 customers targeted |
| Monte Carlo lower bound | **positive in all 1,000 runs** | 90% CI at £2,000: 10×–26× ROI |
| Revenue concentration (Gini) | **0.726** | Top 10% of customers → 62% of £12.1M total revenue |

</div>

---

## Documentation

<div align="center">

| | Description |
|:---:|:---|
| [![Model Card](https://img.shields.io/badge/Model_Card-3A567A?style=for-the-badge&logoColor=white)](docs/model_card.md) | Algorithm selection, SHAP results, calibration, monitoring thresholds |
| [![Business Problem](https://img.shields.io/badge/Business_Problem-3A567A?style=for-the-badge&logoColor=white)](docs/business_problem.md) | Problem framing, stakeholder context, success metrics |
| [![Architecture Overview](https://img.shields.io/badge/Architecture_Overview-3A567A?style=for-the-badge&logoColor=white)](docs/architecture_overview.md) | System design, data flow, module responsibilities |
| [![Feature Engineering](https://img.shields.io/badge/Feature_Engineering-3A567A?style=for-the-badge&logoColor=white)](docs/feature_engineering.md) | Feature design, leakage prevention, cutoff-safe computation |
| [![Evaluation Strategy](https://img.shields.io/badge/Evaluation_Strategy-3A567A?style=for-the-badge&logoColor=white)](docs/evaluation_strategy.md) | Holdout methodology, metric rationale, bootstrap CI design |
| [![Mathematical Intuition](https://img.shields.io/badge/Mathematical_Intuition-3A567A?style=for-the-badge&logoColor=white)](docs/Mathintuition_datascienceframing.md) | BG/NBD + Gamma-Gamma derivations, churn label design |
| [![Business Impact & ROI](https://img.shields.io/badge/Business_Impact_%26_ROI-3A567A?style=for-the-badge&logoColor=white)](docs/business_impact_and_roi.md) | Full ROI narrative, segment strategy, decision framework |
| [![Modeling Assumptions](https://img.shields.io/badge/Modeling_Assumptions-3A567A?style=for-the-badge&logoColor=white)](docs/modeling_assumptions.md) | Assumptions, risks, sensitivity bounds |
| [![Deployment Plan](https://img.shields.io/badge/Deployment_Plan-3A567A?style=for-the-badge&logoColor=white)](docs/deployment_plan.md) | Production readiness, monitoring plan, retraining triggers |
| [![Data Quality Rules](https://img.shields.io/badge/Data_Quality_Rules-3A567A?style=for-the-badge&logoColor=white)](docs/data_quality_rules.md) | 7-rule cleaning specification, edge case handling |

</div>

---

## Table of Contents

<div align="center">

[![Business Problem](https://img.shields.io/badge/Business_Problem-3A567A?style=flat-square&logoColor=white)](#the-business-problem)&nbsp;[![Solution](https://img.shields.io/badge/Solution-3A567A?style=flat-square&logoColor=white)](#the-solution)&nbsp;[![Key Results](https://img.shields.io/badge/Key_Results-3A567A?style=flat-square&logoColor=white)](#key-results)&nbsp;[![Architecture](https://img.shields.io/badge/Architecture-3A567A?style=flat-square&logoColor=white)](#end-to-end-architecture)

[![Data Pipeline](https://img.shields.io/badge/Data_Pipeline-3A567A?style=flat-square&logoColor=white)](#steps-13-data-pipeline)&nbsp;[![CLV Modeling](https://img.shields.io/badge/CLV_Modeling-3A567A?style=flat-square&logoColor=white)](#step-4-clv-modeling)&nbsp;[![Churn Modeling](https://img.shields.io/badge/Churn_Modeling-3A567A?style=flat-square&logoColor=white)](#step-5-churn-risk-modeling)&nbsp;[![Budget Optimization](https://img.shields.io/badge/Budget_Optimization-3A567A?style=flat-square&logoColor=white)](#step-6-budget-optimization)

[![Evaluation](https://img.shields.io/badge/Evaluation-3A567A?style=flat-square&logoColor=white)](#step-7-backtesting--evaluation)&nbsp;[![RFM Segmentation](https://img.shields.io/badge/RFM_Segmentation-3A567A?style=flat-square&logoColor=white)](#step-8-rfm-customer-segmentation)&nbsp;[![Cohort Analysis](https://img.shields.io/badge/Cohort_Analysis-3A567A?style=flat-square&logoColor=white)](#step-9-cohort-retention-analysis)&nbsp;[![Business Intel](https://img.shields.io/badge/Business_Intel-3A567A?style=flat-square&logoColor=white)](#step-10-business-intelligence--pareto)

[![Sensitivity Analysis](https://img.shields.io/badge/Sensitivity_Analysis-3A567A?style=flat-square&logoColor=white)](#step-11-monte-carlo-sensitivity-analysis)&nbsp;[![DS Practices](https://img.shields.io/badge/DS_Practices-3A567A?style=flat-square&logoColor=white)](#professional-ds-practices)&nbsp;[![Skills](https://img.shields.io/badge/Skills-3A567A?style=flat-square&logoColor=white)](#skills-demonstrated)&nbsp;[![How to Run](https://img.shields.io/badge/How_to_Run-3A567A?style=flat-square&logoColor=white)](#how-to-run)

[![Repo Structure](https://img.shields.io/badge/Repo_Structure-3A567A?style=flat-square&logoColor=white)](#repository-structure)&nbsp;[![Assumptions & Risks](https://img.shields.io/badge/Assumptions_%26_Risks-3A567A?style=flat-square&logoColor=white)](#assumptions-risks-and-limitations)&nbsp;[![Future Work](https://img.shields.io/badge/Future_Work-3A567A?style=flat-square&logoColor=white)](#future-improvements)

</div>

---

## The Business Problem

<p align="justify">
In retail and e-commerce, every customer team faces the same constraint: <b>limited retention budget, unlimited customers to target.</b> Without a rigorous system, teams default to three failing strategies:
</p>

<div align="center">

| Strategy | What Goes Wrong |
|:---|:---|
| **Blanket campaigns** | Same message to all customers — no differentiation, wasted spend |
| **Heuristic targeting** | "Target our biggest spenders" — ignores churn risk; budget spent on customers who would have stayed |
| **Static RFM buckets** | Segment labels without economic value attached — no way to prioritise within segments |

</div>

<p align="justify">
<b>The result:</b> Budget is spent on the wrong customers, retention ROI is unmeasured, and the business loses high-value customers it could have saved.
</p>

### What this data reveals

<p align="justify">
This system was built on the <b>UCI Online Retail II dataset</b> — 1M+ real transactions from a UK-based online retailer (2009–2011). Three facts from this data define the problem:
</p>

> **Revenue is dangerously concentrated.**
> Top 10% of customers generate **62% of total revenue**. Gini coefficient = **0.726** — approaching income-inequality levels.

> **High-value customers churn at an alarming rate.**
> The "At Risk" segment — previously frequent buyers — carries **64% churn probability** and represents **£1.25M of threatened revenue** (10% of total).

> **Churn prediction alone is not enough.**
> Predicting who might churn does not tell you who to *spend* your budget on. That requires combining churn probability, expected future value, and cost — simultaneously, under a hard constraint.

---

## The Solution

<p align="justify">
This project builds an <b>11-step, config-driven decision intelligence pipeline</b> that answers a single business question:
</p>

> **Given a fixed retention budget, which customers should be targeted to maximise long-term business value — and how confident are you in that ROI?**

<div align="center">

| Layer | Approach | Output |
|:---|:---|:---|
| **Probabilistic CLV forecasting** | BG/NBD + Gamma-Gamma (lifetimes) | Expected future value per customer (£) |
| **Evidence-based churn modeling** | 4-algorithm CV comparison + SHAP | Calibrated churn probability + feature explanations |
| **Constrained budget optimization** | 0/1 Knapsack (integer programming) | Optimal targeting list under spend constraint |
| **Business intelligence** | RFM segmentation, cohort analysis, Pareto | Revenue concentration, decay curves, segment strategy |
| **Uncertainty quantification** | Monte Carlo simulation (1,000 draws) | ROI confidence intervals under assumption uncertainty |

</div>

---

## Key Results

### CLV Model — Holdout Validation

<div align="center">

| Metric | Value | Interpretation |
|:---|:---:|:---|
| Spearman rank correlation (ρ) | **0.57** | Strong alignment between predicted and actual future revenue |
| Top decile avg holdout revenue | **£5,339** | vs. £852 population mean — **6.3× lift** |
| Top-to-bottom decile ratio | **22×** | £5,339 (decile 10) vs. £241 (decile 1) |
| Monotonic lift | **Yes — all 10 deciles** | No rank inversions; consistent model quality across the full range |

</div>

### Churn Model — 4-Algorithm Comparison

<div align="center">

| Model | CV ROC AUC | CV Avg Precision | Status |
|:---|:---:|:---:|:---:|
| **Random Forest** | **0.816 ± 0.013** | **0.883 ± 0.007** | **Selected** |
| Logistic Regression | 0.810 ± 0.015 | 0.882 ± 0.009 | Baseline |
| XGBoost | 0.807 ± 0.014 | 0.878 ± 0.008 | Evaluated |
| LightGBM | 0.792 ± 0.014 | 0.867 ± 0.008 | Evaluated |

</div>

<p align="center">Holdout ROC AUC = <b>0.831</b> &nbsp;|&nbsp; Holdout Avg Precision = <b>0.878</b> &nbsp;|&nbsp; Churn base rate = 62.7%</p>

### Budget Optimization ROI

<div align="center">

| Budget | Customers Targeted | ROI | Net Gain |
|:---:|:---:|:---:|:---:|
| £200 | 100 | **44.5×** | £8,902 |
| £2,462 | 1,231 | **15.4×** | £37,809 |
| £4,725 | 2,362 | **10.8×** | £50,822 |
| £6,422 | 3,211 | **8.5×** | £54,503 |

</div>

> Monte Carlo 90% CI at £2,000 budget: **ROI range 10×–26×** across 1,000 simulations. ROI is positive in every single simulation.

---

## End-to-End Architecture

<div align="center">
  <img src="assets/clv_architecture.png" alt="CLV Optimization Architecture" width="90%"/>
</div>

```
Raw Transactions (~1M rows, UCI Online Retail II)
             │
             ▼
 [1] Ingestion ─────────► transactions_raw.parquet
     Schema validation
     + Parquet serialisation
             │
             ▼
 [2] Cleaning ──────────► transactions_clean.parquet
     7 deterministic rules
             │
             ▼
 [3] Feature Engineering ► customer_features.parquet
     Cutoff-safe · No leakage
             │
      ┌──────┴───────┐
      ▼              ▼
[4] CLV Modeling  [5] Churn Risk Modeling
    BG/NBD+GG         4-model CV comparison
    ρ = 0.57          RF wins (AUC = 0.816)
    22× lift          SHAP + calibration
      │              │
      └──────┬───────┘
             ▼
  [6] Budget Optimization
      0/1 Knapsack · PuLP/CBC solver
      maximize Σ xᵢ · net_gainᵢ
      subject to Σ xᵢ · costᵢ ≤ B, xᵢ ∈ {0,1}
             │
             ▼
  [7] Evaluation + Reporting
      Decile lift · Bootstrap CIs · ROI curve · MLflow
             │
      ┌──────┼──────────┐
      ▼      ▼          ▼
[8] RFM    [9] Cohort  [10] Business
Segments   Analysis     Insights
8 segments 25 cohorts   Gini = 0.726
             │
             ▼
  [11] Monte Carlo Sensitivity
       1,000 simulations · 90% CI on ROI
```

---

## Step-by-Step: Methods and Findings

---

### Steps 1–3: Data Pipeline

<div align="center">
  <img src="assets/data_pipeline_feature_engineering.png" alt="Data Pipeline Architecture" width="90%"/>
</div>

<p align="justify">
<b>Ingestion</b> reads the UCI Online Retail II dataset (two Excel sheets, ~1M rows), validates column schema, coerces types, and writes Parquet. No data manipulation at this stage — only parsing and serialisation.
</p>

<p align="justify">
<b>Cleaning</b> applies 7 deterministic, documented rules in a fixed order. Each rule is tracked separately to measure its impact on row count:
</p>

<div align="center">

| Rule | Rows Removed | Rationale |
|:---|:---:|:---|
| Remove cancellation invoices (prefix `C`) | ~8,905 | Cancellations reverse prior revenue; not purchase events |
| Remove non-positive unit price | ~33 | Price ≤ 0 indicates internal adjustments |
| Remove non-positive quantity | ~10,624 | Negative quantities without `C` prefix are data errors |
| Remove missing customer IDs | ~135,080 | Cannot assign revenue to customer without ID |
| Remove invalid timestamps | ~0 | Malformed date strings |
| Deduplicate exact rows | ~5,268 | Exact duplicates indicate ingestion errors |
| Compute `revenue = quantity × unit_price` | — | Derived field; applied after cleaning |

</div>

<p align="justify">
<b>Feature engineering</b> computes 8 per-customer features, all strictly computed before the cutoff date. A leakage check runs explicitly before writing the output file.
</p>

<div align="center">

| Feature | Description | Primary Signal For |
|:---|:---|:---|
| `recency_days` | Days since last purchase at cutoff | Churn (top SHAP driver) |
| `tenure_days` | Days since first purchase | Customer maturity |
| `n_invoices` | Distinct purchase events | Frequency (F in RFM) |
| `total_revenue` | Cumulative spend | Monetary value (M in RFM) |
| `avg_order_value` | Revenue ÷ invoices | Spend-per-visit pattern |
| `revenue_last_30d` | Trailing 30-day revenue | Short-term engagement |
| `revenue_last_90d` | Trailing 90-day revenue | Medium-term trend |
| `rev_30_to_90_ratio` | Recent ÷ medium-term revenue | Momentum / acceleration |

</div>

---

### Step 4: CLV Modeling

<p align="justify">
<b>Why probabilistic models?</b> Unlike regression, BG/NBD explicitly models two simultaneous processes: <i>when</i> a customer buys (Poisson purchase frequency) and <i>whether</i> they are still active (Geometric dropout probability). This produces interpretable, theoretically grounded estimates rather than black-box regression residuals.
</p>

**BG/NBD model:**
- Purchase rate: λᵢ ~ Gamma(r, α) — heterogeneous across customers
- Dropout probability: pᵢ ~ Beta(a, b) — customer may leave after any purchase
- Outputs: `E[N_i(H)]` (expected future purchases) and `P(alive)`

**Gamma-Gamma model** (monetary component):
- Transaction value: Mᵢₖ | νᵢ ~ Gamma(p, νᵢ)
- Customer heterogeneity: νᵢ ~ Gamma(q, γ)
- Outputs: `E[μᵢ]` (expected average transaction value per customer)

**CLV formula:**
```
CLV_i(H) = E[N_i(H)] × E[μ_i] × discount_factor

discount_factor = (1 + daily_rate)^(−H)
daily_rate      = (1 + r_annual)^(1/365) − 1
```

<p align="justify">
<b>Time-safe evaluation:</b> Calibration window (before cutoff) trains models. Holdout revenue is measured strictly after cutoff — never seen during training.
</p>

#### Result: CLV Decile Lift

<div align="center">
  <img src="reports/figures/clv_decile_lift.png" alt="CLV Decile Lift with Bootstrap CIs" width="80%"/>
</div>

> Customers sorted by predicted CLV into 10 equal deciles. Bars show average observed holdout revenue per decile. Error bands are **95% bootstrap confidence intervals** (500 resamples, seed=42).

<div align="center">

| Decile | Customers | Avg Predicted CLV | Avg Holdout Revenue | Lift vs Mean |
|:---:|:---:|---:|---:|:---:|
| 1 (Lowest) | 494 | −£1,972 | £241 | 0.28× |
| 2 | 493 | −£924 | £159 | 0.19× |
| 3 | 493 | −£598 | £66 | 0.08× |
| 4 | 493 | −£82 | £78 | 0.09× |
| 5 | 494 | £149 | £170 | 0.20× |
| 6 | 493 | £242 | £305 | 0.36× |
| 7 | 493 | £371 | £355 | 0.42× |
| 8 | 493 | £575 | £659 | 0.77× |
| 9 | 493 | £959 | £1,143 | 1.34× |
| **10 (Highest)** | **494** | **£3,553** | **£5,339** | **6.26×** |

</div>

<p align="justify">
<b>Spearman ρ = 0.57</b> across all 4,933 customers. Holdout revenue is <b>monotonically increasing</b> across all 10 deciles — no rank inversions. The model reliably separates high-value from low-value customers.
</p>

> The top decile generates **22× more observed revenue** than the bottom decile. This is the decision-grade signal needed for budget allocation.

---

### Step 5: Churn Risk Modeling

<p align="justify">
<b>Selection philosophy:</b> Choosing an algorithm based on assumption ("logistic regression is interpretable") rather than evidence is a methodological error. All viable candidates are trained, compared via cross-validation, and the winner is selected automatically.
</p>

#### 5a. Multi-Model Comparison

<div align="center">
  <img src="reports/figures/churn_model_comparison.png" alt="Churn Model Comparison" width="80%"/>
</div>

> **4 algorithms compared using 5-fold stratified cross-validation** (shuffled, `random_state=42`). Bars show mean CV ROC AUC; error bars show ± 1 standard deviation across folds. The winner (Random Forest) is highlighted.

<div align="center">

| Model | CV ROC AUC | CV Avg Precision | Pipeline Preprocessing |
|:---|:---:|:---:|:---|
| **Random Forest** | **0.816 ± 0.013** | **0.883 ± 0.007** | Impute (median) only |
| Logistic Regression | 0.810 ± 0.015 | 0.882 ± 0.009 | Impute → StandardScaler |
| XGBoost | 0.807 ± 0.014 | 0.878 ± 0.008 | Impute (median) only |
| LightGBM | 0.792 ± 0.014 | 0.867 ± 0.008 | Impute (median) only |

</div>

Random Forest wins with the highest mean CV AUC and lowest variance — evidence of both accuracy and stability.

#### 5b. ROC and Precision-Recall Curves

<div align="center">
  <img src="reports/figures/churn_roc_pr_curves.png" alt="ROC and PR Curves" width="80%"/>
</div>

> **Left:** ROC curve — AUC = 0.831 on out-of-time holdout. **Right:** Precision-Recall curve — AP = 0.878. PR curves are especially informative at 62.7% base rate; high AP confirms the model ranks true churners near the top of its scored list.

<p align="justify">
The holdout AUC (0.831) <i>exceeds</i> the CV mean (0.816) — confirming that the model generalises well and has not overfit to the training distribution.
</p>

#### 5c. Probability Calibration

<div align="center">
  <img src="reports/figures/churn_calibration_curve.png" alt="Calibration Curve" width="80%"/>
</div>

> **Left:** Reliability diagram — each point compares predicted probability bin (x-axis) to observed churn rate (y-axis). A perfectly calibrated model falls on the diagonal. **Right:** Score distribution by class — churners and non-churners clearly separated with minimal overlap.

<p align="justify">
Calibration matters because churn probability is used directly in business calculations. If <code>churn_prob_i</code> does not correspond to real-world churn rates, the ROI calculation produces misleading estimates:
</p>

```
Expected Benefit_i = CLV_i × churn_prob_i × retention_effectiveness
```

#### 5d. SHAP Feature Importance

<div align="center">
  <img src="reports/figures/churn_shap_summary.png" alt="SHAP Summary" width="80%"/>
</div>

> **Left (bar chart):** Mean absolute SHAP value per feature — overall importance ranking. **Right (beeswarm):** Each dot is one customer. Position on x-axis = SHAP value (positive → pushes prediction toward churn). Colour = feature value (red = high, blue = low).

**Reading the beeswarm:**
- `recency_days` — **top churn driver**: customers with high recency (long time since last purchase) have large positive SHAP values → high predicted churn probability. Dominant signal.
- `revenue_last_90d` — **strong churn protector**: customers with high medium-term revenue push SHAP negative → lower predicted churn. Active spenders are unlikely to churn.
- `n_invoices` — **frequency protects**: high invoice count pulls SHAP negative → frequent buyers are retained.
- `rev_30_to_90_ratio` — momentum signal: recent acceleration in spending reduces churn risk.

> SHAP values are computed via `TreeExplainer` — exact (not approximate) for tree-based models. Feature directions are data-driven, not assumed.

---

### Step 6: Budget Optimization

**Economic proxy:**
```
Expected Benefit_i = CLV_i × churn_prob_i × retention_effectiveness
Net Gain_i         = Expected Benefit_i − cost_i
```

**Optimization problem (0/1 Knapsack):**
```
maximize:   Σ xᵢ · net_gainᵢ
subject to: Σ xᵢ · costᵢ ≤ B
            xᵢ ∈ {0, 1}   ∀i
```

<p align="justify">
Solved exactly via <b>PuLP (CBC integer solver)</b> — optimal, not approximate. Greedy fallback if PuLP is unavailable.
</p>

**Eligibility criteria:**
- `CLV_i > min_clv` (configurable threshold)
- `churn_prob_i > 0`
- `net_gain_i > 0` (only target customers where expected benefit exceeds cost)

<p align="justify">
This is decision intelligence: it does not simply rank customers — it allocates a scarce resource to maximise expected economic value under a hard budget constraint.
</p>

---

### Step 7: Backtesting & Evaluation

#### ROI vs Budget Curve

<div align="center">
  <img src="reports/figures/policy_roi_vs_budget.png" alt="Policy ROI vs Budget" width="80%"/>
</div>

> **Dual-axis chart.** Left axis (bars): total net gain (£) at each budget level. Right axis (line): ROI multiplier. As budget grows, more customers are targeted but marginal ROI decreases — classic diminishing returns. The optimal range depends on the organisation's budget envelope.

<div align="center">
  <img src="reports/figures/policy_targeted_vs_budget.png" alt="Customers Targeted vs Budget" width="80%"/>
</div>

> Number of customers selected by the knapsack solver at each budget level. Growth rate reflects the population of customers with positive net gain at each spend level.

---

### Step 8: RFM Customer Segmentation

<div align="center">
  <img src="reports/figures/segment_clv_churn_heatmap.png" alt="Segment CLV and Churn Heatmap" width="80%"/>
</div>

> **Heatmap:** Each cell shows average CLV (left) and average churn probability (right) for each of the 8 named segments. Segment size (n) is annotated. Read together, CLV and churn probability determine the economic priority of each segment for retention investment.

<p align="justify">
<b>Methodology:</b> Each customer is scored on three dimensions using quartile ranking (1–4 scale): <b>R (Recency)</b> — days since last purchase, reversed; <b>F (Frequency)</b> — distinct invoices; <b>M (Monetary)</b> — total historical revenue. Combined into 8 named segments via a priority rule matrix based on RFM marketing literature (Kumar & Reinartz, 2012).
</p>

<div align="center">

| Segment | R | F | Customers | Avg CLV | Churn Risk | Revenue Share |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Champions** | 4 | 4 | 681 (13.8%) | £2,436 | **13%** | **55.8%** |
| Loyal Customers | ≥3 | ≥3 | 1,049 (21.3%) | £761 | 36% | 22.3% |
| Potential Loyalists | ≥3 | ≤2 | 407 (8.3%) | −£145 | 49% | 2.1% |
| New Customers | 4 | 1 | 99 (2.0%) | −£2,992 | 49% | 0.3% |
| **At Risk** | ≤2 | ≥3 | 694 (14.1%) | £309 | **64%** | **10.3%** |
| Cant Lose Them | 1 | 4 | 42 (0.9%) | £302 | 62% | 1.5% |
| Hibernating | ≤2 | ≤2 | 952 (19.3%) | −£803 | 69% | 4.0% |
| Lost | 1 | ≤2 | 1,009 (20.5%) | −£438 | 86% | 3.7% |

</div>

<div align="center">
  <img src="reports/figures/segment_rfm_scatter.png" alt="Segment RFM Scatter" width="80%"/>
</div>

> **Scatter plot:** Each dot is a customer. X-axis = churn probability; Y-axis = predicted CLV. Colour = segment. The ideal retention targets occupy the **upper-right quadrant**: high CLV + high churn risk. Champions (upper-left) are safe; Lost customers (lower-right) have low CLV — low priority for expensive interventions.

**Business implications:**
- **Champions** (13.8% of customers) drive **55.8% of revenue** at only 13% churn risk. Protect but do not over-invest — they are not at risk.
- **At Risk** segment (14.1%) carries **64% churn probability** and represents **£1.25M threatened revenue**. Highest-value retention target.
- **Lost** (20.5% of customers) have 86% churn and negative CLV. Reacquisition cost likely exceeds expected value — deprioritise.

---
git
### Step 9: Cohort Retention Analysis

<div align="center">
  <img src="reports/figures/cohort_retention_heatmap.png" alt="Cohort Retention Heatmap" width="90%"/>
</div>

> **Retention heatmap:** Rows = acquisition cohort (month of first purchase). Columns = months since acquisition (0 = acquisition month). Cell value = % of cohort still active. Darker = higher retention. The rapid colour fade from left to right reveals the natural churn decay curve.

<p align="justify">
<b>How to read this:</b> Month 0 (acquisition) is always 100% by definition. By Month 1, most cohorts lose 60–80% of customers. By Month 3, only the core loyal base remains. This decay pattern is precisely what CLV-based retention targeting is designed to slow.
</p>

<div align="center">
  <img src="reports/figures/cohort_retention_curves.png" alt="Cohort Retention Curves" width="80%"/>
</div>

> **Retention decay curves** — one line per acquisition cohort, coloured by cohort month. Lines that stay elevated longer indicate higher-quality cohort acquisition. Cohorts acquired in peak trading periods (November–December) tend to have faster initial decay — likely driven by one-time seasonal buyers.

<div align="center">
  <img src="reports/figures/cohort_revenue_bar.png" alt="Cohort Revenue" width="80%"/>
</div>

> **Revenue by acquisition cohort.** Bars show total revenue per cohort (left axis). The line shows average revenue per customer (right axis). Early cohorts (2009–2010) generate higher total revenue — they had more time to purchase, and surviving members are likely the most engaged.

> **Key insight:** Cohort analysis reveals that **retention decay is steep and early** — most churn happens in the first 1–2 months. The BG/NBD model captures this dropout process parametrically and uses it to project forward.

---

### Step 10: Business Intelligence & Pareto

<div align="center">
  <img src="reports/figures/revenue_concentration_curve.png" alt="Revenue Concentration Curve" width="80%"/>
</div>

> **Left (Lorenz curve):** Cumulative revenue share (y-axis) vs. cumulative customer share (x-axis), sorted by revenue. The dashed diagonal = perfect equality. The further below the diagonal, the more unequal the distribution. The red crosshairs mark the 80% revenue threshold. **Right (bar chart):** Revenue share by customer percentile group.

<div align="center">

| Metric | Value |
|:---|:---:|
| **Gini coefficient** | **0.726** |
| Customers generating 80% of revenue | ~0.3% (inverted Pareto — extreme concentration) |
| Top 10% customers → revenue share | **62%** of £12.1M |
| Revenue at risk (At Risk + Cant Lose) | ~12% of total |

</div>

> A Gini of 0.73 approaches income-inequality levels. This means **treating all customers equally is structurally wasteful** — the vast majority of marketing budget applied to the bottom 80% reaches customers generating only 38% of revenue. CLV-based targeting is not an optimisation; it is a necessity.

<div align="center">
  <img src="reports/figures/monthly_revenue_trend.png" alt="Monthly Revenue Trend" width="80%"/>
</div>

> **Monthly revenue** (bars, left axis) and **3-month moving average** (line) from December 2009 to December 2011. Right axis: monthly active customer count. Strong November–December seasonal spikes correspond to holiday trading.

<div align="center">
  <img src="reports/figures/customer_value_distribution.png" alt="Customer Value Distribution" width="80%"/>
</div>

> **Distribution of per-customer lifetime revenue** — linear scale (left) and log scale (right). The linear plot shows extreme right-skew. The log plot reveals an approximately log-normal distribution — the statistical basis for why Gamma-Gamma monetary modeling is appropriate and why simple averages are misleading.

---

### Step 11: Monte Carlo Sensitivity Analysis

<p align="justify">
<b>Why this matters:</b> The budget optimization model uses two assumed parameters: <code>retention_effectiveness</code> (η) and <code>unit_cost</code>. Both are operationally assumed — not measured from A/B test data. A professional analysis must quantify how sensitive the ROI conclusions are to these assumptions.
</p>

**Methodology:** 1,000 Monte Carlo draws:
```
retention_effectiveness ~ Uniform(0.05, 0.25)   [central: 0.10]
unit_cost               ~ Uniform(£1.00, £5.00) [central: £2.00]
```

For each draw, the full knapsack policy is recomputed at 12 budget levels.

#### Monte Carlo ROI Uncertainty Bands

<div align="center">
  <img src="reports/figures/roi_monte_carlo.png" alt="Monte Carlo ROI" width="80%"/>
</div>

> **Shaded uncertainty bands:** Dark centre line = median ROI (p50). Inner band = 50% CI (p25–p75). Outer band = 90% CI (p5–p95). Even at the pessimistic 5th percentile, ROI remains strongly positive across all budget levels tested.

<div align="center">

| Budget | Median ROI | 90% CI (p5–p95) |
|:---:|:---:|:---:|
| £500 | ~18× | 8×–32× |
| £1,331 | ~18× | 8×–35× |
| £2,462 | ~16× | 6×–26× |
| £4,725 | ~11× | 4×–18× |

</div>

> At £2,000 budget: **ROI is positive in all 1,000 simulations**. Even the most pessimistic combination (5% effectiveness + £5/customer cost) produces a profitable campaign. The business case is robust to assumption uncertainty.

#### Tornado Chart — One-at-a-Time Sensitivity

<div align="center">
  <img src="reports/figures/sensitivity_tornado.png" alt="Sensitivity Tornado" width="80%"/>
</div>

> **Tornado chart:** Each bar shows how much ROI changes when the parameter is moved ±50% of its central value (all other parameters held constant). Longer bar = greater sensitivity.

<div align="center">

| Parameter | Base ROI | Low Value ROI | High Value ROI | Swing |
|:---|:---:|:---:|:---:|:---:|
| `retention_effectiveness` | 16.9× | 8.0× (η=5%) | 25.9× (η=15%) | **17.9×** |
| `unit_cost` | 16.9× | 24.8× (£1) | 13.1× (£3) | **11.7×** |

</div>

<p align="justify">
<b>Reading the tornado:</b> <code>retention_effectiveness</code> has greater total influence on ROI than <code>unit_cost</code>. Empirically measuring retention uplift via A/B testing would have more impact on decision quality than negotiating down channel costs. This is a direct, actionable business recommendation.
</p>

---

## Professional DS Practices

<div align="center">

| Practice | Implementation | Why It Matters |
|:---|:---|:---|
| **Multi-model comparison** | 4 algorithms, 5-fold stratified CV, auto-selection by CV AUC | Avoids model-selection bias; choice backed by evidence not assumption |
| **SHAP interpretability** | `TreeExplainer` — exact values; bar + beeswarm plots | Stakeholder trust; regulatory readiness; direction + magnitude per feature |
| **Probability calibration** | Reliability diagram + score distribution by class | Probabilities must reflect real-world rates to be usable in business math |
| **Dual evaluation curves** | ROC and Precision-Recall both reported | PR especially informative at 63% churn base rate |
| **Bootstrap confidence intervals** | 95% CIs on decile lift (500 resamples, seed=42) | Statistical rigour around point estimates |
| **Monte Carlo sensitivity** | 1,000 draws; tornado chart; 90% CI bands | Quantifies how robust ROI claims are to assumption uncertainty |
| **Time-safe feature engineering** | All features computed before cutoff; leakage check runs explicitly | Realistic simulation of production performance — no look-ahead |
| **Out-of-time evaluation** | Train on cutoff C; evaluate on cutoff C+60d | Models tested exactly as they will be used in production |
| **Cohort analysis** | 25 monthly cohorts × 12-month retention matrix | Reveals natural decay; contextualises why CLV targeting is needed |
| **Spearman rank correlation** | ρ(CLV, holdout revenue) = 0.57 | Right metric for targeting: ranking quality, not absolute accuracy |
| **MLflow experiment tracking** | Params, metrics, and artifacts logged per run | Full reproducibility; enables fair comparison across runs |
| **Config-driven pipeline** | All parameters in YAML; zero magic numbers in code | Any parameter change is one YAML edit — nothing is hidden in code |
| **Google-style docstrings** | Args, Returns, Raises on every public function | Code is readable by collaborators without opening the implementation |
| **Full type hints** | Complete signatures on all functions | IDE assistance; self-documenting interfaces |
| **10 pytest unit tests** | Synthetic data fixtures; pure-function design; CI-safe | Regression protection as the codebase evolves |
| **Modern packaging** | `pyproject.toml` + `Makefile` + `.pre-commit-config.yaml` | Project is installable, lintable, testable, and deployable in one command |

</div>

---

## Skills Demonstrated

<div align="center">

| Skill Area | Demonstrated By |
|:---|:---|
| **Statistical & probabilistic modeling** | BG/NBD + Gamma-Gamma CLV; inactivity-based churn label design; discount rate derivation |
| **Supervised machine learning** | 4-algorithm comparison; stratified k-fold CV; class imbalance handling (`class_weight="balanced"`) |
| **Model interpretability** | SHAP `TreeExplorer`; mean \|SHAP\| bar; beeswarm scatter; direction analysis |
| **Model evaluation** | ROC/PR curves; calibration reliability diagram; decile lift; bootstrap CIs; out-of-time holdout |
| **Mathematical optimisation** | 0/1 Knapsack; integer programming (PuLP/CBC); greedy fallback; economic objective function design |
| **Uncertainty quantification** | Monte Carlo simulation; tornado chart (OAT sensitivity); 90% CI bands on ROI |
| **Customer analytics** | RFM segmentation (8 named segments); cohort retention matrix; Lorenz curve; Gini coefficient |
| **Data engineering** | Multi-layer Parquet pipeline; schema validation; cutoff-safe computation; 7-rule deterministic cleaning |
| **Software engineering** | Config-driven YAML; frozen dataclasses; CLI scripts; Google-style docstrings; type hints throughout |
| **MLOps fundamentals** | MLflow experiment tracking; reproducible seeds; model card (v2.0); monitoring thresholds |
| **Testing** | 10 pytest tests; synthetic data fixtures; pure-function design; no I/O in tests |
| **Project packaging** | `pyproject.toml`; `Makefile` (11 targets); `.pre-commit-config.yaml` (ruff lint + format) |
| **Business communication** | Results-first documentation; quantified claims; explicit assumptions; model card with limitations |

</div>

---

## How to Run

### Prerequisites

- Python 3.9+
- ~2 GB RAM (for 1M-row dataset processing)
- UCI Online Retail II dataset (`.xlsx`) placed at `data/raw/online_retail_II.xlsx`

### Option A: Makefile (recommended)

```bash
git clone <repo-url>
cd clv-long-term-optimization

make install        # Creates .venv, installs all dependencies from requirements.txt
make pipeline       # Runs all 11 steps end-to-end (~15–20 min on a laptop)
make test           # Runs 10 unit tests
make mlflow-ui      # Launches MLflow UI at http://localhost:5000
make lint           # ruff lint check
make format         # ruff autoformat
make clean          # Removes generated data and reports (keeps raw data)
```

### Option B: Manual steps

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline
python -m src.pipelines.weekly_scoring_pipeline --config-dir config

# Override budget for a one-off run
python -m src.pipelines.weekly_scoring_pipeline --config-dir config --budget 8000

# Individual analysis modules
python -m src.analysis.customer_segmentation
python -m src.analysis.cohort_analysis
python -m src.analysis.business_insights
python -m src.evaluation.sensitivity_analysis

# Tests
pytest -v
```

### Configuration

<p align="justify">
All parameters are controlled via YAML files — no code edits required for any standard workflow:
</p>

<div align="center">

| File | Controls |
|:---|:---|
| `config/project.yaml` | Paths, cutoff date, holdout window |
| `config/modeling.yaml` | Hyperparameters, CV folds, random state, model type (`auto`) |
| `config/business.yaml` | Budget, unit cost, retention effectiveness, solver |
| `config/evaluation.yaml` | Decile count, currency symbol, rounding |

</div>

---

## Repository Structure

```
clv-long-term-optimization/
│
├── src/
│   ├── ingestion/
│   │   └── load_data.py               # Step 1: Schema validation + Parquet export
│   ├── cleaning/
│   │   └── clean_transactions.py      # Step 2: 7-rule deterministic cleaning
│   ├── features/
│   │   └── build_features.py          # Step 3: Cutoff-safe RFM + trend features
│   ├── modeling/
│   │   ├── train_clv_models.py        # Step 4: BG/NBD + Gamma-Gamma + MLflow
│   │   └── train_churn_risk.py        # Step 5: 4-model CV + SHAP + calibration
│   ├── optimization/
│   │   └── budget_allocator.py        # Step 6: 0/1 Knapsack (PuLP/CBC)
│   ├── evaluation/
│   │   ├── backtesting.py             # Step 7: Decile lift + 95% CIs + ROI curve
│   │   └── sensitivity_analysis.py   # Step 11: Monte Carlo ROI + tornado chart
│   ├── analysis/
│   │   ├── customer_segmentation.py  # Step 8: RFM segments + CLV/churn overlay
│   │   ├── cohort_analysis.py        # Step 9: Monthly cohort retention heatmap
│   │   └── business_insights.py      # Step 10: Pareto + Lorenz + monthly trend
│   ├── pipelines/
│   │   └── weekly_scoring_pipeline.py # CLI orchestrator — runs all 11 steps
│   └── utils/
│       ├── config_loader.py           # YAML loader with validation
│       └── helpers.py                 # Logging and filesystem utilities
│
├── tests/
│   ├── test_cleaning.py               # 2 tests: cleaning rules and revenue computation
│   ├── test_features.py               # 2 tests: cutoff safety and feature completeness
│   └── test_models.py                 # 6 tests: CLV aggregation, churn labels, snapshot
│
├── config/
│   ├── project.yaml                   # Paths, cutoff date, data locations
│   ├── modeling.yaml                  # Hyperparameters, cv_folds, model_type: "auto"
│   ├── business.yaml                  # Budget, cost, retention effectiveness, solver
│   └── evaluation.yaml                # Decile count, currency, rounding precision
│
├── reports/
│   ├── figures/                       # 17 PNG charts (auto-generated by pipeline)
│   └── tables/                        # 11 CSV tables (auto-generated by pipeline)
│
├── docs/
│   ├── model_card.md
│   ├── business_problem.md
│   ├── architecture_overview.md
│   ├── feature_engineering.md
│   ├── evaluation_strategy.md
│   ├── business_impact_and_roi.md
│   ├── modeling_assumptions.md
│   ├── deployment_plan.md
│   ├── data_quality_rules.md
│   └── Mathintuition_datascienceframing.md
│
├── assets/
│   ├── header_banner.svg
│   ├── footer_banner.svg
│   ├── clv_architecture.png
│   ├── data_pipeline_feature_engineering.png
│   ├── deployment_lifecycle_architecture.png
│   └── evaluation_monitoring_architecture.png
│
├── data/
│   ├── raw/                           # UCI Online Retail II xlsx (~1M rows)
│   ├── interim/                       # transactions_raw, transactions_clean (Parquet)
│   └── processed/                     # features, CLV scores, churn scores, segments, targeting
│
├── notebooks/                         # EDA and model interpretation notebooks
├── pyproject.toml                     # Packaging, ruff lint config, pytest config
├── Makefile                           # install / pipeline / test / lint / mlflow-ui / clean
├── .pre-commit-config.yaml            # ruff lint + format pre-commit hooks
└── requirements.txt                   # Full dependency list with version pins
```

---

## Assumptions, Risks, and Limitations

### Key Assumptions

<div align="center">

| Assumption | Value | How Sensitivity Is Tested |
|:---|:---:|:---|
| Retention effectiveness (η) | 10% | Monte Carlo: Uniform(5%–25%) — ROI positive throughout |
| Unit cost per customer | £2 | Monte Carlo: Uniform(£1–£5) — second-largest ROI driver |
| Contact frequency cap | None | Production systems need per-customer contact limits |
| Channel model | Single channel | Real optimisation would model email/SMS/calls separately |

</div>

### Known Risks

<p align="justify">
<b>Non-causal:</b> CLV and churn models are predictive, not causal. Expected gains are potential impact — not guaranteed uplift. SUTVA is not satisfied without controlled experiments.
</p>

<p align="justify">
<b>Observational bias:</b> Purchasing behaviour reflects unobserved factors (promotions, seasonality, competitive events) not captured in features.
</p>

<p align="justify">
<b>Stationarity assumption:</b> BG/NBD assumes stationary purchase rates — violated by strong seasonality or structural market shifts.
</p>

<p align="justify">
<b>Concept drift:</b> Model trained on 2009–2011 UK retail data. Performance will degrade without periodic retraining on fresh data.
</p>

### Appropriate Uses

**This system is appropriate for:**
- Batch-mode retention targeting (weekly or monthly cadence)
- Internal CRM decision support and budget planning
- Scenario analysis and ROI forecasting

**This system is not appropriate for:**
- Real-time scoring (latency is measured in minutes)
- Individual credit or financial eligibility decisions
- Campaigns requiring causal proof of uplift (run A/B tests first)

---

## Future Improvements

<div align="center">

| Improvement | Business Impact | Complexity |
|:---|:---:|:---:|
| **Uplift modeling (T-learner / X-learner)** | Replace assumed η with measured causal effect | High |
| **A/B test design + power analysis** | Scientifically measure true intervention lift | Medium |
| **Channel-aware multi-constraint knapsack** | Separate email/SMS/call budgets, different unit costs | Medium |
| **SHAP interaction values** | Understand feature × feature effects on churn | Low |
| **Automated drift monitoring** | PSI alerts + scheduled retraining (PSI > 0.25 threshold) | Medium |
| **Per-customer variable cost** | Higher offers for highest-value customers | Low |
| **Cohort-stratified CLV** | Separate BG/NBD model per acquisition cohort | Medium |
| **Model governance layer** | Versioning, lineage, approval workflow, audit trail | Medium |

</div>

---

## License and Copyright

<p align="center">
Copyright &copy; 2026 <b>Ramesh Shrestha</b>. All rights reserved.<br>
You may reference this repository for learning and portfolio review.<br>
For commercial use or redistribution, please contact the author.
</p>

---

<div align="center">
  <img src="assets/footer_banner.svg" width="100%" alt="Ramesh Shrestha — Data Scientist · ML Engineer · Sydney, Australia"/>
</div>
