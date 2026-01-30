# Business Problem: Customer Lifetime Value (CLV) with Long-Term Optimization 

---

## 1. Executive Overview

Australian retail and e-commerce businesses operate in an environment of rising customer acquisition costs, increasing operational expenses, and limited population scale. As a result, **retaining existing customers has become significantly more important than acquiring new ones**.

The core objective of this project is to design a **data-driven decision system** that helps a business allocate limited retention resources (discounts, vouchers, offers, service prioritization) to the customers where those resources will generate the highest long-term return.

This is **not only a prediction problem**.  
It is a **decision optimization problem under uncertainty and constraints**.

---

## 2. Australian Market Context

### 2.1 Why this problem matters in Australia

In the Australian market, several structural factors increase the importance of precise retention targeting:

- High digital advertising competition leading to elevated customer acquisition costs (CAC)
- High logistics, shipping, and return costs due to geography and distance
- Smaller total addressable market compared to the US or Europe
- High customer price sensitivity in many retail categories
- Increasing operational costs (warehousing, labor, last-mile delivery)

These factors mean that **blanket discounts or untargeted retention campaigns can easily destroy margin**.

---

### 2.2 Typical real-world use cases

Australian businesses use retention prioritization systems for:

- Loyalty program offers and tier upgrades
- Win-back campaigns (discounts, free shipping)
- Personalized retention emails and SMS campaigns
- Customer service prioritization
- Proactive outreach to high-value at-risk customers

In all cases, the business must decide **who to target and who not to target**.

---

## 3. Core Business Problem Definition

The business has:
- A large customer base with heterogeneous purchasing behavior
- A fixed retention budget per campaign (e.g. AUD 5,000)
- Limited operational capacity to contact customers

The business asks the following questions:

1. Which customers are expected to generate high future value?
2. Which customers are likely to stop purchasing soon?
3. Which customers are worth spending retention resources on?
4. How should limited budget be allocated to maximize return?
5. How can this decision be justified with data and evidence?

This problem cannot be solved by ranking customers by revenue alone.

---

## 4. Business Decision and Output

### 4.1 Decision timing

The decision is typically made:
- weekly or fortnightly in fast-moving retail
- monthly in lower-frequency retail
- before major campaigns (EOFY, Black Friday, Boxing Day)

### 4.2 Final output

The final output of the system is a **customer-level decision table**, for example:

| customer_id | churn_probability | clv_horizon | expected_benefit | cost | net_gain | recommended_action | priority_rank |
|------------|-------------------|-------------|------------------|------|----------|--------------------|---------------|

This output is directly consumable by CRM and marketing systems.

---

## 5. Business Constraints

### 5.1 Budget constraint (hard)

Retention actions have real costs:
- Discounts reduce margin
- Vouchers and free shipping incur direct costs
- Customer service outreach consumes labor time

The total spend must not exceed the allocated budget.

---

### 5.2 Capacity constraint

Even with budget available:
- Marketing systems have throughput limits
- Call centers and service teams have limited capacity
- Over-contacting customers increases unsubscribe risk

---

### 5.3 Data constraints

Typical transactional datasets do not include:
- Explicit churn events (customers do not formally cancel)
- Measured effectiveness of past retention actions
- Complete demographic information

Therefore:
- Churn must be inferred from inactivity
- Retention effectiveness must be explicitly assumed
- Evaluation must be carefully designed

---

### 5.4 Regulatory and reputational constraints

In the Australian context:
- Overuse of discounts can train customers to wait for promotions
- Excessive communication can damage brand trust
- Retention decisions must be explainable and defensible

---

## 6. Success Metrics

### 6.1 Primary business success metrics

1. **Expected Net Gain**  
   Net Gain = Expected Retained Value − Retention Cost

2. **Return on Investment (ROI)**  
   ROI = (Expected Benefit − Cost) / Cost

3. **Long-term revenue preserved**

These metrics reflect actual business impact.

---

### 6.2 Secondary metrics

- Targeting efficiency (share of customers with positive net gain)
- Budget utilization efficiency
- Stability of customer prioritization over time
- Operational feasibility (number of customers targeted)

---

### 6.3 Model performance metrics (supporting only)

- CLV ranking quality (decile lift, rank correlation)
- Churn model ranking quality (ROC-AUC, Average Precision)

These metrics validate the models but do not define business success on their own.

---

## 7. Risks and Limitations

### 7.1 Churn definition risk
Inactivity-based churn may misclassify seasonal customers.

**Mitigation:**  
Use business-appropriate inactivity windows and validate using historical backtesting.

---

### 7.2 CLV uncertainty and outliers
A small number of customers can dominate value estimates.

**Mitigation:**  
Monitor tails, cap values for reporting, preserve raw values for decisioning.

---

### 7.3 Lack of causal intervention data
Retention effectiveness is assumed rather than measured.

**Mitigation:**  
Treat the solution as a decision policy prototype and recommend A/B testing in production.

---

### 7.4 Data leakage risk
Using future information inflates performance.

**Mitigation:**  
Strict cutoff date; all features are computed using only pre-cutoff data.

---

## 8. Translating the Business Problem into Data Science Problems

The business problem is decomposed into **five connected data science tasks**.

---

## 9. Data Cleaning and Event Definition

### Business need
Ensure that only valid, meaningful purchases are used for modeling.

### Data science tasks
- Remove cancellation invoices
- Remove negative quantities and non-positive prices
- Remove missing customer identifiers
- Standardize timestamps
- Create consistent revenue definition
- Aggregate line items into invoice-level purchase events

### Why this matters
CLV models assume transactions represent real customer purchase behavior.

---

## 10. Feature Engineering (Customer Behavior Representation)

### Business need
Represent each customer’s behavior compactly and meaningfully.

### Core behavioral concepts
- **Recency:** how recently the customer purchased
- **Frequency:** how often the customer purchases
- **Monetary value:** how much the customer spends
- **Tenure:** how long the customer has been active

### Data science requirement
All features must be **time-safe**, computed only using data available at the decision date.

---

## 11. CLV Modeling (Future Value Estimation)

### Business question
“How much value will this customer generate in the future?”

### Data science framing
Estimate expected future number of purchases and expected spend per purchase over a fixed horizon.

### Modeling approach
- **BG/NBD model** to estimate purchase frequency and probability the customer is still active
- **Gamma-Gamma model** to estimate average transaction value

### Why probabilistic models
- Handle irregular purchase behavior
- Work well with sparse transactional data
- Provide expected values rather than point predictions

---

## 12. Churn Risk Modeling (Inactivity-Based)

### Business question
“Which customers are at risk of stopping purchases?”

### Churn definition
A customer is considered churned if no purchase occurs within a defined inactivity window after the cutoff date.

### Modeling approach
- Create time-safe labels using post-cutoff behavior
- Train a supervised classification model
- Output churn probability for each customer

### Model choice rationale
A logistic regression baseline is used due to:
- interpretability
- stability
- suitability for tabular behavioral data

---

## 13. Economic Value Combination

### Business need
Combine value and risk into a single prioritization signal.

### Expected benefit formulation
Expected Retained Value =  CLV × Churn Probability × Retention Effectiveness
This converts predictions into an **economic quantity**.

---

## 14. Budget-Constrained Optimization

### Business question
“Given limited budget, who exactly should we target?”

### Data science framing
Each customer has:
- a cost
- an expected benefit

### Optimization problem
- Objective: maximize total expected net gain
- Constraint: total cost ≤ budget
- Decision variable: target or not target

This is formulated as a **knapsack-style optimization problem**.

---

## 15. Evaluation Strategy

### 15.1 CLV evaluation
- Use out-of-time holdout revenue
- Compare predicted CLV ranking to actual future revenue
- Report decile lift and rank correlation

### 15.2 Churn model evaluation
- Assess ranking quality using standard classification metrics
- Interpret metrics cautiously due to class imbalance

### 15.3 Policy evaluation
- Simulate ROI vs budget curves
- Evaluate diminishing returns
- Assess operational feasibility

---

## 16. Business Impact

### 16.1 Operational impact
- Moves targeting from intuition-based to data-driven
- Produces actionable, customer-level decisions
- Improves budget discipline and transparency

### 16.2 Financial impact (in real deployment)
- Higher ROI on retention spend
- Reduced discount waste
- Improved repeat purchase rates
- Better customer lifetime profitability

### 16.3 Measuring real impact
In production, impact should be validated using:
- controlled experiments (A/B testing)
- incremental revenue and margin uplift
- churn reduction metrics

---

## 17. Final Summary

This project builds a **decision intelligence system** that:

1. Cleans raw transactional data into reliable purchase events  
2. Represents customer behavior in a time-safe manner  
3. Estimates long-term value probabilistically  
4. Estimates churn risk without explicit labels  
5. Optimizes retention decisions under budget constraints  
6. Evaluates outcomes using business-aligned metrics  

The result is a practical, explainable, and economically grounded solution aligned with how real data science teams operate in the Australian retail market.
