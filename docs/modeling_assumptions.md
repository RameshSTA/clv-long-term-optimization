# Model Assumptions & Limitations  
## Customer Lifetime Value (CLV) with Long-Term Optimization

---

## 0. Purpose of This Document

This document explicitly states the **assumptions underlying all models and decisions** in the CLV, churn, and optimization pipeline.

Its goals are to:
- make assumptions transparent and reviewable
- explain why each assumption is reasonable in practice
- document known limitations and risks
- define mitigation strategies used by real data science teams

This document is critical for:
- stakeholder trust
- auditability
- production readiness
- future model evolution

---

## 1. Why Assumptions Matter in Data Science

All models are **simplifications of reality**.  
Ignoring assumptions does not remove them — it merely hides them.

Professional data science requires:
- explicitly stating assumptions
- validating them where possible
- monitoring their impact over time

This project prioritizes **decision robustness over theoretical perfection**.

---

## 2. Data-Level Assumptions

### Assumption 2.1 — Transactions represent real purchases

**Statement:**  
After cleaning, each transaction reflects a genuine customer purchase.

**Why needed:**  
CLV and churn models assume observed transactions are meaningful behavioral signals.

**Risk if violated:**  
Returns, cancellations, or duplicates inflate or distort value estimates.

**Mitigation:**  
Strict data quality rules remove:
- cancellations
- negative quantities
- zero or negative prices
- duplicate rows

---

### Assumption 2.2 — Customer identifiers are stable

**Statement:**  
Each customer_id consistently represents the same real-world customer.

**Why needed:**  
CLV and churn are defined at the customer level.

**Risk if violated:**  
Merged or fragmented customer identities lead to incorrect frequency and value estimates.

**Mitigation:**  
Rows without customer_id are excluded.  
In production, identity resolution would be handled upstream.

---

### Assumption 2.3 — Transaction timestamps are accurate

**Statement:**  
Invoice timestamps correctly reflect purchase timing.

**Why needed:**  
Recency, tenure, churn labeling, and evaluation depend on time ordering.

**Risk if violated:**  
Incorrect timestamps cause data leakage or mislabeling.

**Mitigation:**  
Rows with missing or invalid timestamps are excluded.  
Time-based splits are enforced strictly.

---

## 3. Feature Engineering Assumptions

### Assumption 3.1 — Past behavior predicts future behavior

**Statement:**  
Historical purchasing patterns contain information about future value and churn risk.

**Why needed:**  
All predictive models rely on this behavioral continuity.

**Risk if violated:**  
Sudden structural changes (pricing, competition, product mix) reduce model accuracy.

**Mitigation:**  
- Use rolling retraining windows
- Monitor feature drift
- Re-evaluate performance periodically

---

### Assumption 3.2 — Time-safe features prevent leakage

**Statement:**  
All features are computed using data available at the decision cutoff date.

**Why needed:**  
Models must operate under the same information constraints in training and production.

**Risk if violated:**  
Inflated offline performance that collapses in production.

**Mitigation:**  
Explicit cutoff date enforced in all feature pipelines and evaluations.

---

## 4. CLV Model Assumptions (BG/NBD + Gamma-Gamma)

### Assumption 4.1 — Purchase behavior follows a stochastic process

**Statement:**  
While active, customers purchase randomly over time with a stable rate.

**Why needed:**  
BG/NBD models purchase frequency as a Poisson process.

**Risk if violated:**  
Strong seasonality or promotional spikes can reduce model fit.

**Mitigation:**  
- Evaluate using out-of-time validation
- Interpret CLV as expected value, not exact prediction
- Adjust horizon to match business cadence

---

### Assumption 4.2 — Customers eventually become inactive

**Statement:**  
Each customer has a non-zero probability of permanently dropping out.

**Why needed:**  
BG/NBD explicitly models customer “death.”

**Risk if violated:**  
Subscription-like businesses with guaranteed renewals may violate this assumption.

**Mitigation:**  
This model is appropriate for **non-contractual retail**, which matches the dataset.

---

### Assumption 4.3 — Heterogeneity across customers

**Statement:**  
Customers differ in purchase rate and dropout probability.

**Why needed:**  
Gamma and Beta priors capture population heterogeneity.

**Risk if violated:**  
Homogeneous populations reduce the benefit of probabilistic modeling.

**Mitigation:**  
Model still degrades gracefully to simpler behavior.

---

### Assumption 4.4 — Monetary value is independent of purchase timing

**Statement:**  
Average transaction value is independent of purchase frequency and recency.

**Why needed:**  
Gamma-Gamma requires conditional independence between spend and timing.

**Risk if violated:**  
Heavy discounters or bulk buyers may violate independence.

**Mitigation:**  
- Fit monetary model only for repeat customers
- Validate CLV ranking empirically
- Use CLV primarily for ordering, not absolute forecasting

---

## 5. Churn Modeling Assumptions

### Assumption 5.1 — Churn can be proxied by inactivity

**Statement:**  
A customer who does not purchase within a defined window is considered churned.

**Why needed:**  
Explicit churn labels do not exist in transactional retail data.

**Risk if violated:**  
Seasonal customers may be misclassified as churned.

**Mitigation:**  
- Choose inactivity window aligned with business cycle
- Validate using historical behavior
- Treat churn probability as a risk score, not a binary truth

---

### Assumption 5.2 — Behavioral features capture churn risk

**Statement:**  
Recency, frequency, spend, and trends encode churn signals.

**Why needed:**  
Churn models rely on observable behavior.

**Risk if violated:**  
External drivers (competition, life events) are unobserved.

**Mitigation:**  
Use churn probabilities probabilistically, not deterministically.

---

### Assumption 5.3 — Logistic regression is sufficient

**Statement:**  
A linear decision boundary is adequate for ranking churn risk.

**Why needed:**  
Interpretability, stability, and robustness.

**Risk if violated:**  
Highly nonlinear churn patterns may not be captured.

**Mitigation:**  
Logistic regression is used as a **baseline**; more complex models can replace it if justified.

---

## 6. Economic Modeling Assumptions

### Assumption 6.1 — Retention effectiveness is constant

**Statement:**  
Retention actions reduce churn probability by a fixed proportion.

**Why needed:**  
No experimental data is available to estimate individual uplift.

**Risk if violated:**  
Actual effectiveness varies by customer and offer type.

**Mitigation:**  
- Treat effectiveness as a scenario parameter
- Run sensitivity analysis
- Replace with uplift models in production

---

### Assumption 6.2 — Costs are known and fixed

**Statement:**  
Each retention action has a known cost.

**Why needed:**  
Optimization requires comparable costs.

**Risk if violated:**  
Hidden costs (operational load, brand impact) are not modeled.

**Mitigation:**  
Costs are treated conservatively; optimization remains valid under monotonic cost increases.

---

## 7. Optimization Assumptions

### Assumption 7.1 — Decisions are binary

**Statement:**  
Each customer is either targeted or not targeted.

**Why needed:**  
Simplifies optimization to a knapsack problem.

**Risk if violated:**  
Multi-level offers or intensity decisions are not modeled.

**Mitigation:**  
Binary decision is a first-order approximation; multi-action extensions are possible.

---

### Assumption 7.2 — Budget is a hard constraint

**Statement:**  
Total retention cost cannot exceed a fixed budget.

**Why needed:**  
Reflects real financial constraints.

**Risk if violated:**  
Soft budgets or flexible spend rules are not modeled.

**Mitigation:**  
Optimization can be re-run under different budget scenarios.

---

## 8. Evaluation Assumptions

### Assumption 8.1 — Historical future revenue is a proxy for value

**Statement:**  
Holdout revenue reflects realized customer value.

**Why needed:**  
True CLV is unobservable at evaluation time.

**Risk if violated:**  
External shocks affect future revenue.

**Mitigation:**  
Use ranking-based metrics and multiple horizons.

---

### Assumption 8.2 — Simulated policy evaluation approximates real impact

**Statement:**  
Expected benefit approximates real intervention outcomes.

**Why needed:**  
No actual campaign data is available.

**Risk if violated:**  
Simulated ROI differs from real ROI.

**Mitigation:**  
Explicitly state simulation limits and recommend A/B testing in production.

---

## 9. Summary of Assumptions by Component

| Component | Key Assumption |
|---------|----------------|
Data | Cleaned transactions represent real behavior |
Features | Past behavior predicts future behavior |
CLV | Stochastic purchase + dropout process |
Churn | Inactivity approximates churn |
Economics | Fixed effectiveness and costs |
Optimization | Binary decisions under budget |
Evaluation | Holdout revenue reflects value |

---

## 10. Final Note

These assumptions are **explicit design choices**, not hidden weaknesses.

They reflect:
- common industry practice
- constraints of transactional retail data
- a preference for interpretability and robustness

In a production environment, assumptions should be:
- monitored
- stress-tested
- replaced with empirical estimates when data becomes available