# tests/test_models.py
from __future__ import annotations

import pandas as pd

from src.modeling.train_clv_models import (
    aggregate_to_invoice_level,
    annual_to_daily_discount_rate,
)
from src.modeling.train_churn_risk import (
    build_training_frame,
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


def test_compute_churn_labels_inactivity_definition() -> None:
    """
    compute_churn_labels returns labels only for customers with at least one post-cutoff purchase
    within the horizon window. Customers outside the horizon will not be present and are handled
    later (filled as churned) by build_training_frame.
    """
    cutoff = pd.Timestamp("2011-06-01")
    horizon_days = 180
    inactivity_days = 90

    inv = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "invoice": ["A1", "B1", "C1"],
            "invoice_dt": pd.to_datetime(["2011-06-10", "2011-09-15", "2011-12-01"]),
            "revenue": [10.0, 10.0, 10.0],
        }
    )

    labels = compute_churn_labels(
        inv=inv,
        cutoff=cutoff,
        horizon_days=horizon_days,
        inactivity_days=inactivity_days,
    )

    # Customer 1 purchases 9 days after cutoff -> not churn
    c1 = labels.loc[labels["customer_id"] == 1].iloc[0]
    assert int(c1["churn_label"]) == 0

    # Customer 2 purchases 106 days after cutoff -> churn ( > 90 days)
    c2 = labels.loc[labels["customer_id"] == 2].iloc[0]
    assert int(c2["churn_label"]) == 1

    # Customer 3 is outside the horizon -> not present in labels output
    assert (labels["customer_id"] == 3).sum() == 0


def test_build_training_frame_missing_label_becomes_churned() -> None:
    """
    If a customer has no post-cutoff purchase within the horizon window, they will be missing
    from the labels table and should be labeled churned when joined.
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

    # labels only contains customers 1 and 2
    labels = pd.DataFrame(
        {
            "customer_id": [1, 2],
            "first_purchase_post_cutoff": pd.to_datetime(["2011-06-10", "2011-07-01"]),
            "days_to_next_purchase": [9, 30],
            "churn_label": [0, 0],
        }
    )

    framed = build_training_frame(features, labels)
    c3 = framed.loc[framed["customer_id"] == 3].iloc[0]
    assert int(c3["churn_label"]) == 1