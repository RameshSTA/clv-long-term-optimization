# Data Quality Rules & Cleaning Standard  
## Customer Lifetime Value (CLV) with Long-Term Optimization (Production-Style)

---

## 0. Purpose of This Document

This document defines the **data quality rules**, **cleaning logic**, and **validation checks** used to convert raw transactional data into a reliable “purchase event” dataset for:

- CLV modeling (BG/NBD + Gamma-Gamma)
- churn-risk labeling
- optimization and evaluation

It is written as **audit-ready documentation** that a real data team would maintain so that:
- rules are explicit (not implicit)
- decisions are reproducible
- stakeholders can trust results
- future engineers can maintain the pipeline

---

## 1. What “Good Data” Means for This Project

CLV and churn systems depend on accurately capturing **purchase behavior**.  
For this project, “good data” means:

1. Each row represents a **valid purchase line** or is correctly excluded  
2. Each purchase can be assigned to a **known customer**  
3. Dates are correct and comparable across time  
4. Revenue and quantity values are economically meaningful  
5. Duplicates and cancellations do not distort spending  
6. The final dataset is stable enough for modeling and evaluation

---

## 2. Raw Data Reality (Why Cleaning is Required)

Real transactional datasets often contain issues such as:
- cancellations represented as invoices beginning with a prefix (e.g., "C")
- returns recorded as negative quantities
- free items or data errors recorded as zero price
- missing customer identifiers (especially in guest checkouts)
- duplicate rows due to system re-exports or joins
- inconsistent data types and date formats

If we train CLV models on this raw data, the models will learn patterns that reflect **accounting artifacts**, not real customer behavior.

---

## 3. Data Model: From Line Items to Purchase Events

### 3.1 Line-item data (raw)
Raw datasets typically record one row per product line purchased, such as:

- invoice number
- stock code
- quantity
- unit price
- timestamp
- customer ID

### 3.2 Purchase events (model-ready)
CLV models require “purchase events”:
- one event per invoice per customer
- event timestamp
- total invoice revenue

We therefore convert line items → invoice-level purchases:

\[
\text{invoice\_revenue} = \sum_{line \in invoice} (quantity \times unit\_price)
\]

This transformation reduces noise and aligns the dataset with the modeling assumptions.

---

## 4. Canonical Schema (Required Columns)

For this project, the cleaned dataset must contain at minimum:

| Column | Type | Purpose |
|---|---|---|
| customer_id | integer/string | entity identifier (who) |
| invoice | string | purchase event grouping |
| invoice_dt | datetime | purchase timestamp (when) |
| quantity | numeric | units purchased |
| unit_price | numeric | price per unit |
| revenue | numeric | derived line revenue = quantity × unit_price |
| country | string | geography context (optional but useful) |

---

## 5. Data Quality Rules (Business + Mathematical Justification)

Each rule below includes:
- the rule itself
- why it matters
- what happens if ignored

---

### Rule 1 — Remove cancellation invoices

**Rule:**  
Exclude rows where invoice identifiers indicate cancellation (e.g., invoice starts with `"C"`).

**Why this matters:**  
Cancellations are not purchases; they reverse purchases.  
If included, they distort frequency, recency, and monetary estimates.

**Model impact if ignored:**
- BG/NBD sees false extra “transactions”
- Gamma-Gamma sees incorrect spending behavior
- CLV becomes unstable or negative

**Example rationale:**  
A canceled order should not increase customer value.

---

### Rule 2 — Exclude non-positive quantities

**Rule:**  
Remove rows with:
- quantity ≤ 0

**Why this matters:**  
Negative quantities are often returns or reversals.  
Zero quantity is typically an error.

**Model impact if ignored:**
- creates negative or zero revenue
- breaks monetary distribution assumptions (Gamma-Gamma)
- invalidates invoice-level revenue calculations

---

### Rule 3 — Exclude non-positive unit prices

**Rule:**  
Remove rows with:
- unit_price ≤ 0

**Why this matters:**  
Prices cannot be negative in normal sales.  
Zero prices may represent:
- freebies
- data entry errors
- internal adjustments

**Model impact if ignored:**
- drags monetary estimates downward
- artificially increases transaction counts without value
- distorts value segmentation

---

### Rule 4 — Require valid customer identifiers

**Rule:**  
Remove rows where:
- customer_id is null or missing

**Why this matters:**  
CLV is defined per customer.  
If the customer is unknown, the transaction cannot be attributed to any entity.

**Model impact if ignored:**
- customer-level features become inconsistent
- purchase counts and revenue can’t be assigned
- model training becomes noisy and misleading

**Business note:**  
In production, you may handle guest checkouts via alternative identifiers.  
In this dataset, we remove them to preserve correctness.

---

### Rule 5 — Require valid transaction timestamps

**Rule:**  
Remove rows with missing or invalid invoice_dt.

**Why this matters:**  
Recency, tenure, and time-based splits depend on timestamps.

**Model impact if ignored:**
- cannot compute recency and calibration window
- breaks evaluation logic
- causes leakage or incorrect labeling

---

### Rule 6 — Remove exact duplicate rows

**Rule:**  
If two rows are identical across all fields, keep only one.

**Why this matters:**  
Duplicates often occur from export errors or merges.

**Model impact if ignored:**
- inflates frequency and revenue
- overestimates CLV

---

### Rule 7 — Standardize data types

**Rule:**  
Ensure:
- invoice is string
- customer_id is integer-like or stable string
- invoice_dt is datetime
- quantity and unit_price are numeric

**Why this matters:**  
Type instability creates subtle bugs and incorrect comparisons (especially with dates).

**Model impact if ignored:**
- incorrect sorting of timestamps
- broken grouping and aggregation
- inconsistent customer identity

---

### Rule 8 — Create and validate derived revenue

**Rule:**  
Create:
\[
revenue = quantity \times unit\_price
\]

And enforce:
- revenue > 0

**Why this matters:**  
Revenue is the monetary target used throughout CLV and evaluation.

**Model impact if ignored:**
- inconsistent monetary signals
- incorrect holdout revenue computation
- unstable CLV ranking evaluation

---

## 6. Invoice-Level Aggregation Rules

After cleaning line items, we convert to purchase events:

### Rule 9 — Aggregate line items into invoice events
For each (customer_id, invoice):

- invoice_dt = minimum timestamp in that invoice
- invoice_revenue = sum of revenue across line items

**Why this matters:**  
CLV models are typically fit on “purchase events.”  
Invoice-level aggregation matches the idea of “a shopping visit” or order.

---

## 7. Validation Checks (Data Contracts)

After cleaning, we enforce the following checks:

### 7.1 Integrity checks
- No cancellation invoices remain
- quantity > 0 for all rows
- unit_price > 0 for all rows
- customer_id not null
- invoice_dt not null

### 7.2 Revenue checks
- revenue > 0 for all rows
- invoice-level revenue is non-negative and sensible

### 7.3 Duplicate checks
- no exact duplicates

### 7.4 Time checks
- invoice_dt range is within expected bounds
- no future timestamps beyond dataset max

These checks prevent silent corruption.

---

## 8. Outputs and Storage Standards

### 8.1 Intermediate outputs
- `data/interim/transactions_raw.parquet`  
  Raw standardized schema (no cleaning assumptions)

- `data/interim/transactions_clean.parquet`  
  Clean line-item data with derived revenue

### 8.2 Processed outputs
- Invoice-level purchase events used for CLV models  
  (produced inside modeling scripts)

---

## 9. Auditability and Reproducibility

A production-quality pipeline must allow:
- reproducing the exact dataset given the same input
- explaining why rows were removed
- tracking how much data was excluded by each rule

Recommended logging outputs:
- row counts before and after each rule
- percent removed by each rule
- summary table of removals

This helps:
- debugging
- governance
- stakeholder trust

---

## 10. Practical Intuition: How Cleaning Connects to CLV Mathematics

CLV models assume transaction data represents **real purchases**. Cleaning ensures:

1. **Frequency is real**  
   - cancellations and duplicates do not inflate transaction counts

2. **Monetary value is real**  
   - negative or zero revenue does not distort spend distributions

3. **Recency and tenure are correct**  
   - timestamps are valid and consistent

4. **Customer identity is stable**  
   - purchases are attributable to the correct customer

When these assumptions hold, probabilistic models like BG/NBD and Gamma-Gamma become reliable and interpretable.

---

## 11. Summary

This cleaning standard ensures we produce a dataset where:

- transactions represent real customer purchase behavior
- time is reliable for recency/tenure and evaluation splits
- monetary values are meaningful
- the final dataset is stable for CLV modeling, churn modeling, and optimization

These rules match how real data science teams define “model-ready” transaction data for customer analytics.