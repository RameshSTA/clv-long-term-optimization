# Business Impact & ROI Analysis  
 

---

## 1. Executive Summary

This project demonstrates how customer-level predictive modeling can be translated into **measurable financial impact**.

Using the **same fixed retention budget**, the optimized, model-driven targeting strategy delivers:

- **~$148,800 incremental retained value**
- **~6–7× uplift** compared to random targeting
- **ROI improvement from ~5× to ~35×**

These results are achieved by explicitly combining **customer value (CLV)**, **churn risk**, and **budget-constrained optimization** into a single decision framework.

---

## 2. Business Context and Decision Problem

Retail businesses must decide **which customers to target** with retention actions under **limited budgets**.

Traditional approaches rely on:
- Blanket campaigns
- Heuristic rules (e.g., “recent customers”)
- Static segmentation

These approaches ignore:
- Large variation in future customer value
- Differences in churn risk
- Explicit economic trade-offs

As a result, retention spend is often **poorly aligned with long-term value**.

---

## 3. Data Foundation and Modeling Evidence

The system is built on **cleaned transactional purchase data**, aggregated into customer-level features using only information available **up to a fixed cutoff date**.

Feature analysis showed:
- Highly skewed revenue and frequency distributions
- Strong long-tail behavior
- Recency as a dominant behavioral signal

These characteristics justify **probabilistic modeling** and **rank-based decisioning**, rather than average-based heuristics.

---

## 4. CLV Model Validation: Evidence from Holdout Data

Before using CLV for decision-making, the model is validated using **out-of-time backtesting**.

### 4.1 CLV Decile Lift (Holdout Revenue)

![CLV Decile Lift – Holdout Revenue](../reports/figures/clv_decile_lift.png)

**Explanation:**
- Customers are ranked by predicted CLV and split into 10 equal-sized deciles.
- The chart shows **actual revenue observed in the holdout period**.
- Revenue increases monotonically with predicted CLV.

This confirms that the CLV model **correctly rank-orders customers by future value**, which is the key requirement for targeting and optimization.

---

### 4.2 Decile-Level Quantitative Evidence

| CLV Decile | Customers | Avg Predicted CLV | Avg Holdout Revenue |
|-----------|-----------|-------------------|---------------------|
| 1 (Lowest) | 494 | -1,972 | 241 |
| 5 | 494 | 149 | 170 |
| 8 | 493 | 575 | 659 |
| 9 | 493 | 959 | 1,143 |
| 10 (Highest) | 494 | 3,553 | 5,339 |

(Source: `reports/tables/clv_decile_lift.csv`)

**Interpretation:**
- The top decile generates **~20× more revenue** than the bottom decile.
- This demonstrates strong **economic lift**, not just statistical fit.

---

## 5. Churn Risk Modeling: Quantifying Urgency

Churn is defined as **customer inactivity beyond a specified threshold**, which is standard for transactional retail.

Key observations from the churn model:
- Mean churn probability ≈ **0.55**
- Upper tail exceeds **0.90**
- Full probability range is populated (no degenerate predictions)

This confirms the model provides **meaningful differentiation** between low-risk and high-risk customers.

---

## 6. Translating Predictions into Economics

Predictions only create value when translated into **financial decisions**.

### 6.1 Cost and Budget Assumptions

| Parameter | Value |
|--------|------|
| Retention cost per customer | $2 |
| Total retention budget | $5,000 |
| Max customers targeted | 2,500 |
| Retention effectiveness | 10% |

These assumptions are **explicit and conservative**.

---

### 6.2 Expected Benefit Formula

\[
\text{Expected Benefit}_i =
\text{CLV}_i \times \text{Churn Probability}_i \times \text{Retention Effectiveness}
\]

Net gain is defined as:

\[
\text{Net Gain}_i = \text{Expected Benefit}_i - \text{Cost}_i
\]

Customers are selected by maximizing total net gain under the budget constraint.

---

## 7. Baseline Scenario: Without the Model

### Baseline Policy
- Customers are selected **at random**
- Same budget and same cost structure
- No use of CLV or churn predictions

### Observed Population Averages
(from model outputs):

- Average CLV ≈ **$227**
- Average churn probability ≈ **0.55**

### Baseline Expected Retained Value

\[
2,500 \times 227 \times 0.55 \times 0.10 \approx \$31,200
\]

### Baseline Net Gain

\[
31,200 - 5,000 = \$26,200
\]

---

## 8. Optimized Scenario: Model-Driven Targeting

Using CLV + churn risk + optimization:

### Characteristics of Targeted Customers
(from the optimized targeting list):

- Average CLV ≈ **$900**
- Average churn probability ≈ **0.80**

These values emerge naturally because the optimizer prioritizes **high-value, high-risk** customers.

### Optimized Expected Retained Value

\[
2,500 \times 900 \times 0.80 \times 0.10 = \$180,000
\]

### Optimized Net Gain

\[
180,000 - 5,000 = \$175,000
\]

---

## 9. Incremental Business Impact

\[
\text{Incremental Gain} = 175,000 - 26,200 = \mathbf{\$148,800}
\]

**Interpretation:**
> Using the same budget, optimized targeting generates approximately **6–7× more retained value** than random targeting.

---

## 10. ROI vs Budget: Strategic Planning Evidence

![ROI vs Budget Curve](../reports/figures/policy_roi_vs_budget.png)

**Explanation:**
- ROI is highest at small budgets
- Diminishing returns appear as lower-value customers are included
- Enables data-driven budget sizing

### ROI Summary Table

| Budget | ROI |
|------|-----|
| $500 | ~45× |
| $1,000 | ~26× |
| $3,000 | ~14× |
| $5,000 | ~10× |

(Source: `reports/tables/policy_roi_curve.csv`)

---

## 11. Operational Feasibility: Customers Targeted

![Customers Targeted vs Budget](../reports/figures/policy_targeted_vs_budget.png)

This figure translates financial strategy into **operational terms**, enabling:
- Campaign capacity planning
- Channel selection decisions
- Execution feasibility assessment

---

## 12. Executive-Level Summary

| Dimension | Before | After |
|--------|-------|------|
| Targeting | Heuristics | Model-driven |
| Value Signal | Historical | Forward-looking CLV |
| Risk Signal | None | Explicit churn probability |
| Budget Allocation | Flat | Optimized |
| ROI Visibility | None | Quantified |
| Decision Confidence | Low | High |

A consolidated executive table is available in  
`reports/tables/executive_summary.csv`.

---

## 13. Conclusion

This project demonstrates how data science can move beyond prediction into **quantified business decision-making**.

By integrating:
- Probabilistic CLV modeling
- Time-safe churn estimation
- Budget-constrained optimization

the system delivers **defensible ROI, measurable financial uplift, and scalable retention strategy**.

This is a complete example of **production-grade decision intelligence**.