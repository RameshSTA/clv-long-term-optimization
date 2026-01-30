# Evaluation Strategy  
## Customer Lifetime Value (CLV) with Long-Term Optimization

---

## 0. Purpose of This Document

This document defines the **evaluation framework** used to assess the quality, reliability, and business impact of the CLV, churn, and optimization system.

The goal of evaluation in this project is **not** to maximize isolated model metrics, but to answer a more important question:

> Does this system lead to better business decisions under real-world constraints?

This document explains:
- what components are evaluated
- how evaluation is performed in a time-safe manner
- which metrics matter to the business
- how to interpret results responsibly
- what evaluation cannot tell us

---

## 1. Evaluation Philosophy

### 1.1 Evaluation is decision-focused

Traditional model evaluation focuses on prediction accuracy.  
This project focuses on **decision quality**.

A model with slightly lower accuracy but better economic outcomes is preferred over a highly accurate model that leads to poor decisions.

---

### 1.2 Time realism is mandatory

All evaluations must respect the **temporal structure of the data**.

At evaluation time:
- the model must only use information available at the decision date
- outcomes must come from future data

This mirrors real deployment conditions.

---

### 1.3 Ranking quality matters more than point accuracy

In retention prioritization:
- we care about *who* is ranked above whom
- absolute prediction values are secondary

Therefore, evaluation emphasizes **ranking-based metrics** and **economic curves**.

---

## 2. Evaluation Components Overview

The system is evaluated at four levels:

1. **Data quality validation**
2. **CLV model evaluation**
3. **Churn model evaluation**
4. **Decision policy (optimization) evaluation**

Each level answers a different question and serves a different stakeholder.

---

## 3. Data Quality Evaluation

### 3.1 Objective

Ensure that downstream evaluations are based on **trustworthy inputs**.

If data quality is poor, model evaluation is meaningless.

---

### 3.2 Checks performed

Before any modeling evaluation, the following checks are enforced:

- No cancellation invoices remain
- No negative or zero revenue
- No missing customer identifiers
- No missing or invalid timestamps
- No duplicate purchase events
- Logical consistency (tenure ≥ recency)

---

### 3.3 Reporting

We report:
- row counts before and after each cleaning rule
- percentage of data removed per rule
- final dataset size

This ensures transparency and auditability.

---

## 4. CLV Model Evaluation

### 4.1 What we are evaluating

The CLV model is evaluated on its ability to **rank customers by future value**, not to predict exact monetary amounts.

The key question is:

> Do customers predicted to have higher CLV actually generate more revenue in the future?

---

### 4.2 Time-based evaluation setup

1. Select a **cutoff date** \(T\)
2. Fit CLV models using data up to \(T\)
3. Compute predicted CLV over a fixed horizon \(H\)
4. Observe actual revenue generated in \((T, T+H]\)

This produces an **out-of-time evaluation**.

---

### 4.3 Primary CLV evaluation metrics

#### 4.3.1 Decile lift analysis

Customers are:
1. sorted by predicted CLV
2. split into 10 equal-sized groups (deciles)

For each decile, we compute:
- average actual holdout revenue
- total actual holdout revenue

**Interpretation:**
- A good CLV model shows monotonic increase in revenue from lowest to highest decile
- Strong separation between top and bottom deciles indicates useful ranking

---

#### 4.3.2 Rank correlation (Spearman)

We compute Spearman rank correlation between:
- predicted CLV ranking
- actual future revenue ranking

**Interpretation:**
- High correlation indicates consistent ordering
- Absolute values are less important than stability and direction

---

### 4.4 What we do NOT over-interpret

- Exact CLV dollar amounts  
- Small differences between adjacent ranks  

CLV is treated as an **expected value signal**, not a guaranteed forecast.

---

## 5. Churn Model Evaluation

### 5.1 What we are evaluating

The churn model is evaluated on its ability to **rank customers by risk of inactivity**, not to perfectly classify churned vs non-churned customers.

---

### 5.2 Labeling and time safety

- Features are computed at cutoff date \(T\)
- Churn labels are defined using future inactivity in \((T, T+L]\)

This ensures no leakage.

---

### 5.3 Primary churn evaluation metrics

#### 5.3.1 ROC-AUC

Measures the probability that a randomly chosen churned customer is ranked higher than a non-churned one.

**Interpretation:**
- Values above 0.7 indicate reasonable separation
- Used as a sanity check, not a final success metric

---

#### 5.3.2 Average Precision (AP)

Useful when churn is imbalanced.

**Interpretation:**
- Higher AP means high-risk customers are concentrated near the top of the ranking

---

### 5.4 Business interpretation

The churn model’s output is:
- a **risk score**, not a binary truth
- an input to economic decision-making, not a standalone action trigger

---

## 6. Combined Economic Signal Evaluation

### 6.1 From predictions to economics

For each customer, we compute:

- CLV (future value)
- churn probability (risk)
- expected benefit
- net gain after cost

Evaluation must confirm that **economic signals behave sensibly**.

---

### 6.2 Sanity checks

We validate that:
- customers with high CLV and high churn risk have high expected benefit
- customers with low CLV or low churn risk have low expected benefit
- negative net gain customers exist and are filtered out

This ensures predictions translate into rational economics.

---

## 7. Optimization (Policy) Evaluation

### 7.1 Why policy evaluation is required

Evaluating models independently is insufficient.

The real business outcome depends on:
- how predictions are combined
- how budget constraints are enforced
- which customers are ultimately targeted

This is evaluated at the **policy level**.

---

### 7.2 Simulation-based evaluation

Because no real campaign outcomes exist, we use **policy simulation**.

For a range of budgets \(B\):

1. Solve the optimization problem
2. Select customers under budget
3. Compute:
   - total expected benefit
   - total cost
   - total net gain
   - ROI

---

### 7.3 ROI vs budget curve

We plot:
- ROI as a function of budget
- net gain as a function of budget

**Interpretation:**
- ROI typically decreases as budget increases (diminishing returns)
- Business can choose budget at a point that balances scale and efficiency

---

### 7.4 Targeted vs untargeted comparison

We compare:
- optimized targeting
- naive strategies (e.g. random, top-CLV only)

**Interpretation:**
- Optimized strategy should dominate baselines across most budget levels

---

## 8. Sensitivity Analysis

### 8.1 Retention effectiveness sensitivity

Because effectiveness is assumed, we evaluate outcomes under multiple values (e.g. 10%, 20%, 30%).

**Interpretation:**
- Directional conclusions should remain stable
- Absolute ROI may change

---

### 8.2 Budget sensitivity

We examine:
- very small budgets (high precision)
- moderate budgets (balanced)
- large budgets (diminishing returns)

This helps stakeholders understand trade-offs.

---

## 9. What This Evaluation Cannot Prove

This evaluation does **not** prove:
- true causal impact of retention actions
- exact uplift from interventions
- long-term brand effects

These require **controlled experiments**.

---

## 10. Recommended Production Evaluation (Next Step)

In a real deployment, we recommend:

1. A/B testing:
   - control: existing strategy
   - treatment: optimized targeting

2. Measure:
   - incremental revenue
   - incremental margin
   - churn reduction
   - cost per retained customer

3. Feed experimental results back into:
   - retention effectiveness estimates
   - future optimization runs

---

## 11. Summary

This evaluation strategy ensures that:

- models are validated using time-safe methods
- performance is judged by business-relevant outcomes
- decisions are evaluated economically, not just statistically
- limitations are explicitly acknowledged

The evaluation framework aligns with how professional data science teams assess decision intelligence systems in real organizations.