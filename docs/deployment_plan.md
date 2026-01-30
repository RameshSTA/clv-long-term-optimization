# Deployment Plan  
## Customer Lifetime Value (CLV) with Long-Term Optimization

---

## 0. Purpose of This Document

This document describes **how the CLV, churn, and optimization system is deployed, operated, and monitored in a real production environment**.

It answers the following questions:
- How does this move from notebooks to production?
- How are models executed on a schedule?
- How are outputs delivered to the business?
- How is performance monitored and maintained?
- How does the system evolve safely over time?

This plan reflects **industry-standard data science deployment practices**.

---

## 1. Deployment Philosophy

### 1.1 Separation of concerns

The system is divided into clear layers:

- **Exploration layer**: notebooks (analysis, validation)
- **Production layer**: versioned Python modules and pipelines
- **Delivery layer**: data outputs consumed by business systems
- **Monitoring layer**: quality, performance, and drift checks

Notebooks are **not** production artifacts.  
All production logic lives in `src/`.

---

### 1.2 Batch-first design

This system is designed as a **batch decision pipeline**, not a real-time API, because:

- retention decisions are periodic (weekly/monthly)
- CLV and churn do not change minute-by-minute
- batch processing is simpler, cheaper, and more robust

Real-time extensions are possible but not required initially.

---

## 2. Deployment Architecture Overview

### 2.1 High-level flow

1. Raw transactional data arrives
2. Data is cleaned and validated
3. Customer features are generated at cutoff date
4. CLV and churn models are applied
5. Economic optimization is solved
6. Results are exported for business use
7. Monitoring checks are logged

---

### 2.2 Logical architecture

Raw Data
↓
Ingestion & Cleaning
↓
Feature Engineering
↓
Model Scoring (CLV + Churn)
↓
Economic Calculation
↓
Budget Optimization
↓
Decision Output (CSV / Parquet)
↓
Monitoring & Logging

---

## 3. Production Components

### 3.1 Code structure

Key production modules:

- `src/ingestion/`  
  Load and standardize raw data

- `src/cleaning/`  
  Apply data quality rules and validations

- `src/features/`  
  Generate customer-level features (time-safe)

- `src/modeling/`  
  CLV models, churn models, scoring logic

- `src/optimization/`  
  Budget-constrained decision optimization

- `src/pipelines/`  
  End-to-end orchestration scripts

- `src/evaluation/`  
  Offline evaluation and reporting logic

---

### 3.2 Configuration management

All environment-specific values are stored in configuration files:

- cutoff date
- modeling horizon
- inactivity window
- budget amount
- assumed retention effectiveness
- file paths

Configuration is externalized to:src/config/config.yaml
This allows:
- reproducible runs
- scenario analysis
- safe parameter changes without code edits

---

## 4. Execution Mode

### 4.1 Scheduled batch execution

The pipeline is designed to run on a fixed schedule, for example:

- weekly retention run
- monthly strategic planning run

Execution is triggered via:
- cron job
- workflow orchestrator (e.g. Airflow, Prefect)
- CI/CD runner (in smaller setups)

---

### 4.2 Pipeline entry point

The main production entry point is:src/pipelines/weekly_scoring_pipeline.py
This script:
- loads configuration
- executes each pipeline stage in order
- performs validation checks
- writes final outputs

---

## 5. Model Lifecycle Management

### 5.1 Training vs scoring

Two modes exist:

1. **Training mode**
   - fit CLV and churn models
   - save model artifacts
   - log metadata

2. **Scoring mode**
   - load trained models
   - score latest customer features
   - generate decisions

Training is done periodically; scoring happens more frequently.

---

### 5.2 Model versioning

Models are versioned by:
- training date
- cutoff date
- dataset snapshot
- configuration hash

Model artifacts are stored with metadata to ensure traceability.

---

## 6. Output Delivery

### 6.1 Primary business outputs

The pipeline produces a **customer-level decision file** containing:

- customer_id
- CLV estimate
- churn probability
- expected benefit
- cost
- net gain
- recommended action
- priority rank

Output formats:
- CSV (for CRM ingestion)
- Parquet (for analytics and auditing)

---

### 6.2 Downstream consumption

Outputs are consumed by:
- marketing automation platforms
- CRM systems
- analytics dashboards
- finance and strategy teams

The data scientist does **not** execute campaigns; they provide the decision input.

---

## 7. Monitoring and Quality Control

### 7.1 Data quality monitoring

At each run, the pipeline logs:
- number of rows ingested
- number of rows removed by each cleaning rule
- feature distribution summaries

Alerts are triggered if:
- row counts drop unexpectedly
- key distributions shift materially

---

### 7.2 Model monitoring

Monitored indicators include:
- average predicted CLV
- churn probability distribution
- correlation between CLV and churn
- share of customers with positive net gain

Large deviations indicate potential data drift or model degradation.

---

### 7.3 Decision monitoring

We monitor:
- number of customers targeted
- total budget consumed
- expected ROI

This ensures decisions remain economically rational over time.

---

## 8. Logging and Auditability

Each pipeline run produces:
- execution timestamp
- configuration snapshot
- model versions used
- summary statistics
- output file references

This enables:
- reproducibility
- compliance reviews
- post-hoc analysis

---

## 9. Failure Handling and Recovery

### 9.1 Fail-fast design

The pipeline fails immediately if:
- critical data quality checks fail
- required inputs are missing
- optimization is infeasible

This prevents silent propagation of errors.

---

### 9.2 Recovery strategy

On failure:
- no outputs are published
- logs indicate failure reason
- pipeline can be re-run after issue resolution

No partial or corrupted outputs are allowed downstream.

---

## 10. Security and Access Control

Recommended practices:
- restrict write access to raw data
- restrict edit access to production code
- log all executions
- separate development and production environments

---

## 11. Scaling Considerations

### 11.1 Data volume scaling

The pipeline scales linearly with:
- number of transactions
- number of customers

Batch processing and aggregation ensure feasibility at large scale.

---

### 11.2 Model complexity scaling

CLV and churn models are computationally efficient and suitable for:
- tens of thousands
- hundreds of thousands
- millions of customers (with batching)

---

## 12. Production Validation (First Deployment)

Before full rollout:
1. Run pipeline on historical periods
2. Compare outputs with known business intuition
3. Validate ROI curves
4. Review with stakeholders
5. Launch controlled pilot

---

## 13. Continuous Improvement

Post-deployment enhancements may include:
- replacing assumed retention effectiveness with experimental uplift
- segment-specific costs and actions
- richer churn models
- causal modeling extensions

The deployment architecture supports incremental evolution without redesign.

---

## 14. Summary

This deployment plan ensures that the CLV and optimization system:

- moves cleanly from research to production
- runs reliably on a schedule
- produces actionable business outputs
- is auditable, monitorable, and maintainable
- aligns with real-world data science operations

It reflects how professional data science teams deploy decision intelligence systems in practice.