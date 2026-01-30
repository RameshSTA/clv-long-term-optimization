# Math Intuition & Data Science Framing  
## Customer Lifetime Value (CLV) with Long-Term Optimization (Detailed)

---

## 0. Purpose of This Document

This document explains, from a **data scientist’s** perspective:

1. How the business problem is translated into data science and mathematical problems  
2. The math intuition behind each model and component  
3. How we solve the full system end-to-end in a way that is time-safe and decision-focused  

The project is structured as a **decision system**:

**Transactions → Features → CLV (value) + Churn risk (risk) → Economic benefit → Budget optimization → Evaluation**

---

## 1. Translating Business Questions into Mathematical Objectives

### Business question
> “Who should we target with limited retention budget to maximize long-term value?”

### Data science reframing
We need to compute, for each customer \(i\), a **decision** \(x_i \in \{0,1\}\) such that:

- \(x_i = 1\): we target customer \(i\) (send offer, call, etc.)
- \(x_i = 0\): we do not target customer \(i\)

The business goal becomes:

> Choose \(x_i\) to maximize total expected return subject to a budget constraint.

This requires:

1. an estimate of future value if the customer stays active (CLV)
2. an estimate of risk of losing that value (churn probability)
3. an estimate of benefit gained by intervention (retention effectiveness)
4. an optimization step to select customers under limited budget

---

## 2. Data Model and Notation (Transactions)

We observe line-item transactions. After cleaning and aggregation, we represent data as **purchase events** (invoice-level):

For customer \(i\), we observe purchase times:
\[
t_{i1}, t_{i2}, \dots, t_{in_i}
\]
and monetary values (invoice revenue):
\[
m_{i1}, m_{i2}, \dots, m_{in_i}
\]

We define:

- **Recency**: time since last purchase
  \[
  r_i = T - t_{in_i}
  \]
- **Frequency**: number of repeat purchases
  \[
  x_i = n_i - 1
  \]
- **Age / Tenure**: length of observation window
  \[
  T_i = T - t_{i1}
  \]
where \(T\) is the cutoff date (decision date).

These are the core sufficient statistics used by classical CLV models.

---

## 3. Why Time-Safe Framing Matters (No Data Leakage)

We always split time into:

- **Calibration window**: data used to build features and models  
- **Holdout window**: future data used only for evaluation

If we let features include holdout transactions, we leak future information and overestimate performance.

Mathematically, when predicting outcomes after \(T\):

- Features must depend only on:
  \[
  \{t_{ij}, m_{ij}\ |\ t_{ij} \le T\}
  \]
- Labels and evaluation depend on:
  \[
  \{t_{ij}, m_{ij}\ |\ t_{ij} > T\}
  \]

This is crucial for real-world deployment: you never have future data when scoring customers.

---

## 4. CLV Modeling: Mathematical Intuition

CLV is the expected discounted value of a customer’s future purchases over a horizon \(H\):

\[
CLV_i(H) = \mathbb{E}\left[\sum_{k=1}^{N_i(H)} \frac{M_{ik}}{(1+d)^{\tau_{ik}}}\right]
\]

Where:
- \(N_i(H)\) is the random number of future purchases in horizon \(H\)
- \(M_{ik}\) is monetary value per purchase
- \(d\) is discount rate (time value of money)
- \(\tau_{ik}\) is time until the \(k\)-th future purchase

We decompose the problem into two parts:
1. **Purchase frequency** → BG/NBD model
2. **Monetary value** → Gamma-Gamma model

This separation is standard and practical.

---

## 5. BG/NBD Model (Purchase Frequency)

### 5.1 What BG/NBD is trying to model
Customers buy repeatedly over time, but some “drop out” (become inactive).  
BG/NBD models:

- how often customers buy while active
- the probability they are still “alive”

### 5.2 Generative assumptions (intuition)

1. While alive, customer purchases follow a **Poisson process** with rate \(\lambda_i\):
\[
N(t)\ |\ \lambda_i \sim \text{Poisson}(\lambda_i t)
\]

2. Customers are heterogeneous:
\[
\lambda_i \sim \text{Gamma}(r, \alpha)
\]
Gamma prior captures that some customers buy frequently and others rarely.

3. After each purchase, customer may drop out with probability \(p_i\):
\[
p_i \sim \text{Beta}(a, b)
\]
Beta distribution captures heterogeneity in dropout tendency.

This combination yields closed-form formulas for:
- expected future purchases
- probability alive

### 5.3 Sufficient statistics used
BG/NBD uses:
- \(x_i\) = number of repeat purchases
- \(t_{x_i}\) = time of last purchase
- \(T_i\) = total observation time

### 5.4 What we get from BG/NBD
We estimate:
- \(\mathbb{E}[N_i(H)]\): expected number of purchases in the next \(H\) days
- \(P(\text{alive}_i)\): probability customer is still active at cutoff

These outputs directly answer:
> “Who is likely to purchase again?”

---

## 6. Gamma-Gamma Model (Monetary Value)

### 6.1 What Gamma-Gamma is trying to model
Given a customer purchases, how much do they spend on average?

We model mean transaction value per customer:
\[
\mu_i = \mathbb{E}[M_{ik}]
\]

Assumption (common, but important):
- the spending distribution is independent of purchase timing/frequency (conditional on being a repeat customer)

### 6.2 Generative intuition
1. Transaction values for customer \(i\) are Gamma distributed:
\[
M_{ik}\ |\ \nu_i \sim \text{Gamma}(p, \nu_i)
\]

2. Customer heterogeneity:
\[
\nu_i \sim \text{Gamma}(q, \gamma)
\]

This leads to a closed-form estimate for expected average spend:
\[
\mathbb{E}[\mu_i\ |\ \text{transactions}]
\]

### 6.3 Practical note
Gamma-Gamma is typically fit only for customers with at least 2 repeat purchases to ensure stability.

---

## 7. CLV Construction (Combining Frequency + Monetary)

A practical CLV estimate over horizon \(H\) is:

\[
CLV_i(H) \approx \mathbb{E}[N_i(H)] \times \mathbb{E}[\mu_i] \times \text{discounting}
\]

Discounting uses a daily discount rate \(d_{\text{daily}}\) derived from annual rate \(d_{\text{annual}}\):

\[
d_{\text{daily}} = (1 + d_{\text{annual}})^{1/365} - 1
\]

This produces:
- a single monetary score per customer: \(CLV_i(H)\)

---

## 8. Churn Risk Modeling: Mathematical Intuition

### 8.1 Why churn is tricky in transactional data
In many retail businesses, churn is not explicit. A customer can “leave” silently.

So we define churn operationally as **inactivity**.

### 8.2 Churn labeling definition
Given cutoff \(T\), inactivity threshold \(L\) days:

\[
y_i =
\begin{cases}
1 & \text{if no purchase in } (T, T+L] \\
0 & \text{otherwise}
\end{cases}
\]

This turns a missing business label into a measurable label using future behavior.

### 8.3 Predictive framing
We now have a supervised learning dataset:

- Input features \(X_i\): computed using transactions up to \(T\)
- Label \(y_i\): computed using transactions after \(T\)

We want:
\[
\hat{p}_i = P(y_i = 1\ |\ X_i)
\]

### 8.4 Logistic regression baseline (intuition)

Logistic regression estimates:
\[
\hat{p}_i = \sigma(\beta_0 + \beta^T X_i)
\]
where:
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Interpretation:
- Each feature contributes linearly to the log-odds of churn.
- Coefficients are interpretable: sign and magnitude show direction and strength.

We standardize features because:
- it stabilizes optimization
- it makes coefficients comparable

---

## 9. Turning Predictions into Economics

Predictions alone are not the business decision.  
We need a “value of acting” estimate.

### 9.1 Value at risk
A high-risk customer with high CLV has large value at risk.

Approximate expected value at risk:

\[
\text{value\_at\_risk}_i = CLV_i(H) \times \hat{p}_i
\]

### 9.2 Adding retention effectiveness
If an intervention prevents churn with probability \(\eta\), then expected benefit is:

\[
\text{benefit}_i = CLV_i(H) \times \hat{p}_i \times \eta
\]

Where \(\eta \in [0,1]\) is the assumed effectiveness.

### 9.3 Net gain
If intervention cost is \(c_i\), then:

\[
\text{net\_gain}_i = \text{benefit}_i - c_i
\]

If net gain is negative, the rational decision is not to intervene.

---

## 10. Budget Optimization: Knapsack Problem

### 10.1 Business decision variable
\[
x_i \in \{0,1\}
\]

### 10.2 Optimization objective
Maximize total net gain:

\[
\max_{x} \sum_{i=1}^{N} x_i \cdot \text{net\_gain}_i
\]

### 10.3 Budget constraint
\[
\sum_{i=1}^{N} x_i \cdot c_i \le B
\]

This is a classical **0/1 knapsack**.

### 10.4 Why optimization (not just ranking)
Ranking by net gain gives a heuristic solution.  
Optimization provides the best feasible subset under budget constraints (especially if costs vary).

We implemented:
- exact solution via integer programming (PuLP) or
- greedy approximation for speed

---

## 11. Evaluation: Mathematical View

### 11.1 CLV evaluation (ranking usefulness)
The core goal of CLV in business is ranking: we want top-ranked customers to produce more future revenue.

We evaluate using:
- **decile lift**
- **Spearman rank correlation**

#### Spearman rank correlation
Compute correlation between ranks:

\[
\rho = \text{corr}(\text{rank}(CLV_i), \text{rank}(\text{holdout\_revenue}_i))
\]

High \(\rho\) means the model orders customers correctly.

#### Decile lift
Split customers into 10 groups by predicted CLV and compute average holdout revenue by decile.  
A good model shows monotonic increase from decile 1 → 10.

### 11.2 Churn evaluation (ranking)
We use AUC/AP to ensure the model can separate churn vs non-churn, but business value depends on targeting decisions.

### 11.3 Policy evaluation (economic curves)
Because we do not have actual campaign outcomes, we evaluate the policy by simulation:

For each budget \(B\):
- select customers under policy
- compute total expected benefit and net gain
- compute ROI:

\[
ROI(B) = \frac{\text{benefit}(B) - \text{cost}(B)}{\text{cost}(B)}
\]

A typical curve shows diminishing returns (ROI decreases as budget expands).

---

## 12. Why This Solves the Real Business Problem

The project is structured around the real decision:

1. **CLV** estimates future value (\(CLV_i\))
2. **Churn model** estimates risk (\(\hat{p}_i\))
3. Combine into **expected benefit** (\(CLV_i \cdot \hat{p}_i \cdot \eta\))
4. Solve a **budget constrained optimization**
5. Produce an actionable target list with economic justification
6. Evaluate with time-safe ranking and ROI curves

This is how real companies move from “analytics” to “decision intelligence”.

---

## 13. What Would Change in Production (Next-Level Realism)

In a real business setting, we would replace assumptions with measured effects:

- Estimate \(\eta\) using A/B tests (uplift / causal inference)
- Allow \(c_i\) to vary by offer type and customer segment
- Incorporate operational constraints (channel quotas, contact rules)
- Monitor data drift and score drift

But the structure of the decision system remains the same.

---
# Glossary of Concepts (Data Science & Mathematical Perspective)

This glossary explains all key terms used in the project from both a **business** and **mathematical / data science** perspective. It is intended to make the system understandable to analysts, data scientists, and stakeholders.

---

## Customer Lifetime Value (CLV)

**Business meaning:**  
The total expected monetary value a customer will generate for the business in the future.

**Mathematical meaning:**  
The expected discounted sum of future transaction values over a fixed horizon.

\[
CLV_i(H) = \mathbb{E}\left[\sum_{k=1}^{N_i(H)} \frac{M_{ik}}{(1+d)^{\tau_{ik}}}\right]
\]

Where:
- \(N_i(H)\) is the random number of future purchases
- \(M_{ik}\) is the value of each purchase
- \(d\) is the discount rate
- \(\tau_{ik}\) is the time until each purchase

---

## Recency

**Business meaning:**  
How recently a customer has purchased. Customers who purchased recently are usually more engaged.

**Mathematical definition:**  
\[
\text{Recency}_i = T - t_{i,\text{last}}
\]
Where \(T\) is the cutoff date.

Lower recency values generally indicate higher engagement.

---

## Frequency

**Business meaning:**  
How often a customer purchases.

**Mathematical definition:**  
\[
\text{Frequency}_i = n_i - 1
\]
Where \(n_i\) is the total number of purchases observed.

Frequency is a core signal of habitual behavior.

---

## Monetary Value

**Business meaning:**  
How much a customer spends per transaction on average.

**Mathematical definition:**  
\[
\mu_i = \mathbb{E}[M_{ik}]
\]

Estimated using the Gamma-Gamma model.

---

## Tenure (Customer Age)

**Business meaning:**  
How long the customer has been active with the business.

**Mathematical definition:**  
\[
\text{Tenure}_i = T - t_{i,\text{first}}
\]

Longer tenure often indicates customer maturity and stability.

---

## Cutoff Date

**Business meaning:**  
The date at which the business makes a decision (e.g., who to target).

**Data science meaning:**  
A strict boundary separating:
- **past data** (used for features and training)
- **future data** (used only for labels and evaluation)

This prevents data leakage.

---

## Calibration Window

**Definition:**  
The historical time window used to estimate model parameters.

All features and model fitting must use only data within this window.

---

## Holdout Window

**Definition:**  
A future time window used **only for evaluation**, never for training.

Used to measure:
- actual future revenue
- churn behavior

---

## BG/NBD Model

**Full name:**  
Beta-Geometric / Negative Binomial Distribution model

**Purpose:**  
Estimate how frequently a customer will purchase and whether they are still active.

**Core intuition:**  
- Customers purchase randomly over time while alive
- Each customer has their own purchase rate
- After each purchase, a customer may permanently drop out

**Key outputs:**
- Expected future number of purchases
- Probability the customer is still “alive”

---

## Probability Alive (p_alive)

**Meaning:**  
The probability that a customer has not permanently dropped out as of the cutoff date.

This is **not churn probability**, but rather a latent activity probability.

---

## Gamma-Gamma Model

**Purpose:**  
Estimate expected average transaction value per customer.

**Key assumption:**  
Transaction value is independent of purchase frequency (conditional on customer).

Used to model spending heterogeneity across customers.

---

## Discount Rate

**Business meaning:**  
Future money is worth less than present money.

**Mathematical use:**  
Discount future cash flows when computing CLV.

Daily discount rate:
\[
d_{\text{daily}} = (1 + d_{\text{annual}})^{1/365} - 1
\]

---

## Churn

**Business meaning:**  
A customer who stops purchasing.

**Data science proxy (used here):**  
A customer is considered churned if no purchase occurs within a specified inactivity window after the cutoff date.

---

## Inactivity Window

**Definition:**  
The number of days without a purchase after which a customer is considered churned.

Example:
- 90 days for fast-moving retail
- 180+ days for seasonal categories

---

## Churn Probability

**Mathematical meaning:**  
\[
\hat{p}_i = P(\text{churn}_i = 1 \mid X_i)
\]

Estimated using a supervised classification model.

---

## Logistic Regression

**Purpose:**  
Estimate churn probability from behavioral features.

**Mathematical form:**
\[
\hat{p}_i = \frac{1}{1 + e^{-(\beta_0 + \beta^T X_i)}}
\]

**Why used:**
- Interpretable
- Stable
- Well-suited for tabular business data

---

## Value at Risk (Customer)

**Meaning:**  
The expected future value that could be lost if the customer churns.

\[
\text{Value at Risk}_i = CLV_i \times \hat{p}_i
\]

---

## Retention Effectiveness

**Meaning:**  
The probability that a retention action successfully prevents churn.

**Important:**  
This is an assumption in this project due to lack of experimental data.

In production, this should be estimated using A/B testing.

---

## Expected Benefit

**Definition:**  
The expected monetary value preserved by intervening.

\[
\text{Expected Benefit}_i = CLV_i \times \hat{p}_i \times \eta
\]

Where \(\eta\) is retention effectiveness.

---

## Net Gain

**Business meaning:**  
How much value the business gains after accounting for intervention cost.

\[
\text{Net Gain}_i = \text{Expected Benefit}_i - \text{Cost}_i
\]

Only customers with positive net gain should be targeted.

---

## Budget Constraint

**Definition:**  
A hard limit on total intervention cost:

\[
\sum_i x_i \cdot \text{Cost}_i \le B
\]

Where \(B\) is the total budget.

---

## Knapsack Optimization

**Purpose:**  
Select a subset of customers that maximizes total net gain under a budget constraint.

**Decision variable:**
\[
x_i \in \{0,1\}
\]

This converts predictions into executable business decisions.

---

## Decile Lift

**Meaning:**  
A ranking-based evaluation metric.

Customers are sorted by predicted score and split into 10 equal-sized groups (deciles).  
A strong model shows increasing actual outcomes from lowest to highest decile.

---

## Spearman Rank Correlation

**Purpose:**  
Measure how well predicted rankings align with actual future outcomes.

\[
\rho = \text{corr}(\text{rank}(y), \text{rank}(\hat{y}))
\]

Used when absolute prediction accuracy is less important than ordering.

---

## ROI (Return on Investment)

**Definition:**  
\[
ROI = \frac{\text{Benefit} - \text{Cost}}{\text{Cost}}
\]

Primary metric used by business leadership to assess effectiveness of retention spend.

---

## Policy Evaluation

**Meaning:**  
Evaluation of the **decision strategy**, not just individual models.

Assesses:
- ROI vs budget
- diminishing returns
- operational feasibility

---

## Decision Intelligence

**Definition:**  
A system that combines:
- data
- models
- economics
- optimization

to produce **actionable business decisions**, not just predictions.

This project is an example of a decision intelligence system.

---
