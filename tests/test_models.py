# tests/test_models.py
"""
Unit tests for CLV and churn modeling components.

Design notes
------------
- All tests use synthetic data so they run without requiring the full dataset.
- Tests are scoped to pure-function behavior (no I/O, no file system).
- Time-safe labeling is validated explicitly (cutoff boundary respected).
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from src.modeling.train_clv_models import (
    aggregate_to_invoice_level,
    annual_to_daily_discount_rate,
)
from src.modeling.train_churn_risk import (
    build_snapshot,
    compute_churn_labels,
)


def test_aggregate_to_invoice_level_sums_revenue_and_one_row_per_invoice() -> None:
    """
    Invoice-level aggregation should:
    - return one row per (customer_id, invoice)
    - sum revenue across line items for the same invoice
    """
    tx = pd.DataFrame(
        {
            "customer_id": [1, 1, 1],
            "invoice": ["A1", "A1", "A2"],
            "invoice_dt": pd.to_datetime(["2011-01-01", "2011-01-01", "2011-02-01"]),
            "revenue": [10.0, 5.0, 7.0],
        }
    )

    inv = aggregate_to_invoice_level(tx)

    assert inv.shape[0] == 2  # A1 and A2
    a1 = inv.loc[inv["invoice"] == "A1"].iloc[0]
    assert abs(float(a1["revenue"]) - 15.0) < 1e-9


def test_annual_to_daily_discount_rate_reasonable() -> None:
    """
    Daily discount rate derived from 10% annual should be small and positive.
    """
    r_daily = annual_to_daily_discount_rate(0.10)
    assert 0 < r_daily < 0.001


def test_annual_to_daily_discount_rate_zero_is_zero() -> None:
    """Zero annual rate should produce zero daily rate."""
    assert annual_to_daily_discount_rate(0.0) == 0.0


def test_compute_churn_labels_inactivity_definition() -> None:
    """
    compute_churn_labels produces a label for every customer in `customers`.

    - Customer 1 purchases 9 days after cutoff  -> churn = 0  (within inactivity window)
    - Customer 2 purchases 106 days after cutoff -> churn = 1  (exceeds 90-day threshold)
    - Customer 3 purchases after the horizon end -> churn = 1  (no observed activity)
    """
    cutoff = pd.Timestamp("2011-06-01")
    horizon_days = 180
    inactivity_days = 90

    # cutoff + 180 days = 2011-11-28
    inv = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "invoice": ["A1", "B1", "C1"],
            "invoice_dt": pd.to_datetime(["2011-06-10", "2011-09-15", "2011-12-01"]),
            "revenue": [10.0, 10.0, 10.0],
        }
    )

    customers = pd.Series([1, 2, 3])

    labels = compute_churn_labels(
        inv=inv,
        cutoff=cutoff,
        horizon_days=horizon_days,
        inactivity_days=inactivity_days,
        customers=customers,
    )

    # All three customers must be present in the output.
    assert set(labels["customer_id"].tolist()) == {1, 2, 3}

    # Customer 1: purchases 9 days after cutoff -> not churned
    c1 = labels.loc[labels["customer_id"] == 1].iloc[0]
    assert int(c1["churn_label"]) == 0, "Customer 1 should not be churned (bought within 9 days)"

    # Customer 2: purchases 106 days after cutoff -> churned (> 90-day threshold)
    c2 = labels.loc[labels["customer_id"] == 2].iloc[0]
    assert int(c2["churn_label"]) == 1, "Customer 2 should be churned (106 days > 90-day threshold)"

    # Customer 3: purchases after horizon end -> no activity observed -> churned
    c3 = labels.loc[labels["customer_id"] == 3].iloc[0]
    assert int(c3["churn_label"]) == 1, "Customer 3 should be churned (outside horizon window)"


def test_compute_churn_labels_respects_cutoff_boundary() -> None:
    """
    A purchase exactly on the cutoff date should NOT count as post-cutoff activity.
    Only purchases strictly after the cutoff are used for labeling.
    """
    cutoff = pd.Timestamp("2011-06-01")

    inv = pd.DataFrame(
        {
            "customer_id": [1, 1],
            "invoice": ["A1", "A2"],
            # A1 is exactly at cutoff (should not count), A2 is post-cutoff
            "invoice_dt": pd.to_datetime(["2011-06-01", "2011-06-05"]),
            "revenue": [100.0, 10.0],
        }
    )

    labels = compute_churn_labels(
        inv=inv,
        cutoff=cutoff,
        horizon_days=90,
        inactivity_days=30,
        customers=pd.Series([1]),
    )

    # A2 (June 5) is 4 days after cutoff -> within 30-day inactivity threshold -> not churned
    c1 = labels.loc[labels["customer_id"] == 1].iloc[0]
    assert int(c1["churn_label"]) == 0


def test_build_snapshot_attaches_labels_and_defaults_missing_to_churned() -> None:
    """
    build_snapshot should correctly join features and labels.
    Customers present in features but absent from labels must default to churn_label=1
    (they had no observable post-cutoff activity within the horizon).
    """
    features = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "recency_days": [10, 20, 30],
            "tenure_days": [100, 200, 300],
            "n_invoices": [2, 3, 1],
            "total_revenue": [50.0, 60.0, 10.0],
            "avg_order_value": [25.0, 20.0, 10.0],
            "revenue_last_30d": [10.0, 0.0, 0.0],
            "revenue_last_90d": [50.0, 60.0, 10.0],
            "rev_30_to_90_ratio": [0.2, 0.0, 0.0],
        }
    )

    # Labels only contain customers 1 and 2; customer 3 is absent -> should be churned
    labels = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "first_purchase_post_cutoff": pd.to_datetime(["2011-06-10", "2011-07-01"]),
            "days_to_next_purchase": [9, 30],
            "churn_label": [0, 0],
        }
    )

    cutoff = pd.Timestamp("2011-06-01")
    framed = build_snapshot(
        features=features,
        labels=labels,
        cutoff=cutoff,
        horizon_days=180,
        inactivity_days=90,
    )

    # All 3 customers should appear in the output
    assert len(framed) == 3

    # Customer 3, missing from labels, should receive churn_label = 1
    c3 = framed.loc[framed["customer_id"] == 3].iloc[0]
    assert int(c3["churn_label"]) == 1, "Unlabeled customer must default to churn_label=1"

    # Customers 1 and 2 retain their original labels
    c1 = framed.loc[framed["customer_id"] == 1].iloc[0]
    assert int(c1["churn_label"]) == 0

    # Metadata columns must be attached
    assert "cutoff_date" in framed.columns
    assert "prediction_horizon_days" in framed.columns
    assert "churn_inactivity_days" in framed.columns
