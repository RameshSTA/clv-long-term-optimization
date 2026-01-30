# Feature Engineering Standard  
## Customer Lifetime Value (CLV) & Churn Modeling (Professional Specification)

---

## 0. Purpose of This Document

This document defines **how raw transactional data is transformed into customer-level features** suitable for:

- CLV modeling (BG/NBD + Gamma-Gamma)
- churn risk modeling
- decision optimization and evaluation

It explains:
- what features are created
- why each feature exists (business + math intuition)
- how time safety is enforced
- how features support downstream models

This document represents the **contract between raw data and models**.

---

## 1. Feature Engineering Philosophy

Feature engineering is not about creating as many features as possible.  
It is about **encoding customer behavior in a way that matches the assumptions of the models and the business decision**.

The guiding principles are:

1. **Customer-level representation**  
   All features are aggregated per customer.

2. **Time safety**  
   No feature may use information after the cutoff date.

3. **Behavioral meaning**  
   Every feature must correspond to a real business concept.

4. **Stability over complexity**  
   Prefer robust, interpretable features over fragile, high-variance signals.

---

## 2. Input Data (Post-Cleaning)

Feature engineering operates on **clean transaction data** with the following guarantees:

- one row per valid line item
- positive quantity and price
- valid customer identifiers
- valid timestamps
- derived revenue field

The cleaned dataset is treated as the **source of truth**.

---

## 3. Time Reference and Cutoff Date

### 3.1 Cutoff date definition

The **cutoff date** \(T\) represents the moment when a business decision is made.

All features must satisfy:
\[
\text{feature}_i = f(\{t_{ij}, m_{ij} \mid t_{ij} \le T\})
\]

No information from \(t > T\) may influence features.

---

### 3.2 Why time safety is critical

If future transactions leak into features:
- CLV appears unrealistically accurate
- churn prediction becomes trivial
- evaluation results are invalid
- production performance collapses

Time safety ensures **train = production conditions**.

---

## 4. Feature Categories Overview

Features are grouped into four categories:

1. **Core behavioral features** (RFM-style)
2. **Temporal aggregation features**
3. **Derived ratio and intensity features**
4. **Eligibility and stability features**

Each category is described below.

---

## 5. Core Behavioral Features (Foundational)

These features encode long-term customer behavior and are essential for CLV and churn models.

---

### 5.1 Recency (recency_days)

**Definition:**
\[
\text{recency}_i = T - t_{i,\text{last}}
\]

**Business intuition:**  
Customers who purchased recently are more engaged and less likely to churn.

**Model intuition:**  
Recency is a strong signal in:
- BG/NBD (probability alive)
- churn classification

**Constraints:**
- recency ≥ 0
- computed using last purchase ≤ cutoff

---

### 5.2 Frequency (n_invoices)

**Definition:**
\[
\text{frequency}_i = \text{number of distinct invoices up to } T
\]

**Business intuition:**  
Frequent buyers are more habitual and predictable.

**Model intuition:**  
Frequency directly influences BG/NBD expected purchase counts.

---

### 5.3 Monetary Value (total_revenue)

**Definition:**
\[
\text{total\_revenue}_i = \sum_{j: t_{ij} \le T} m_{ij}
\]

**Business intuition:**  
Captures historical customer worth.

**Model intuition:**  
Used directly or indirectly to estimate average order value.

---

### 5.4 Average Order Value (avg_order_value)

**Definition:**
\[
\text{AOV}_i = \frac{\text{total\_revenue}_i}{\text{n\_invoices}_i}
\]

**Business intuition:**  
Distinguishes high-spend vs low-spend customers.

**Model intuition:**  
Primary input to Gamma-Gamma monetary modeling.

---

### 5.5 Tenure (tenure_days)

**Definition:**
\[
\text{tenure}_i = T - t_{i,\text{first}}
\]

**Business intuition:**  
Long-tenure customers are often more stable.

**Model intuition:**  
Provides context for interpreting recency and frequency.

---

## 6. Temporal Aggregation Features (Short-Term Signals)

These features capture **recent changes in behavior**, which are especially important for churn risk.

---

### 6.1 Revenue in last 30 days (revenue_last_30d)

**Definition:**
\[
\sum_{j: T-30 < t_{ij} \le T} m_{ij}
\]

**Business intuition:**  
Measures very recent engagement.

**Churn relevance:**  
Sharp drops often precede churn.

---

### 6.2 Revenue in last 90 days (revenue_last_90d)

**Definition:**
\[
\sum_{j: T-90 < t_{ij} \le T} m_{ij}
\]

**Business intuition:**  
Smooths short-term noise while capturing medium-term trends.

---

## 7. Derived Ratio and Intensity Features

These features normalize behavior across customers and help models compare customers fairly.

---

### 7.1 Recent-to-medium spend ratio (rev_30_to_90_ratio)

**Definition:**
\[
\text{ratio}_i =
\begin{cases}
\frac{\text{revenue\_last\_30d}_i}{\text{revenue\_last\_90d}_i}, & \text{if revenue\_last\_90d}_i > 0 \\
0, & \text{otherwise}
\end{cases}
\]

**Business intuition:**  
Detects acceleration or decline in spending.

**Model intuition:**  
Ratio features are scale-invariant and highlight behavior changes.

---

## 8. Feature Stability and Eligibility Rules

### 8.1 Minimum data requirements

Certain models require minimum behavioral history:

- Gamma-Gamma requires ≥ 2 purchases
- Churn modeling requires non-missing core features

Customers not meeting requirements are:
- excluded from specific models
- handled explicitly downstream

---

### 8.2 Missing value handling

Rules:
- core behavioral features must not be missing
- ratio features default to zero when denominator is zero
- no imputation using future data

---

## 9. Feature Scaling and Preparation

### 9.1 Scaling rationale

Churn models (logistic regression) benefit from standardized inputs.

**Standardization:**
\[
X' = \frac{X - \mu}{\sigma}
\]

This improves:
- numerical stability
- coefficient interpretability

CLV models operate on raw counts and monetary values and do not require scaling.

---

## 10. Feature Output Contract

The final feature table must satisfy:

- one row per customer
- stable schema across runs
- no leakage beyond cutoff date
- non-negative values where expected

**Canonical output:**
`data/processed/customer_features.parquet`

---

## 11. Validation Checks

After feature engineering, we enforce:

- recency ≥ 0
- tenure ≥ recency
- n_invoices ≥ 1
- total_revenue ≥ 0
- no duplicate customer_id rows

These checks prevent silent corruption.

---

## 12. How Feature Engineering Supports the Full System

Feature engineering acts as the bridge between:

- raw transactional data  
- probabilistic CLV models  
- churn risk models  
- economic optimization  

Well-designed features ensure:
- CLV estimates are stable
- churn probabilities are meaningful
- optimization decisions are economically sound

---

## 13. Summary

This feature engineering standard ensures that:

- customer behavior is represented faithfully
- time leakage is impossible
- features align with business intuition
- downstream models receive clean, interpretable inputs

These practices reflect how professional data science teams build customer-level modeling systems for production use.