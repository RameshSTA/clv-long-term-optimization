# tests/test_features.py
from __future__ import annotations

import pandas as pd

from src.features.build_features import build_customer_features


def test_build_customer_features_one_row_per_customer() -> None:
    """
    Feature engineering should:
    - return one row per customer
    - produce expected feature columns
    - produce non-negative, sensible values
    """
    cutoff = pd.Timestamp("2011-06-01")

    tx = pd.DataFrame(
        {
            "customer_id": [1, 1, 2, 2, 2],
            "invoice": ["A1", "A2", "B1", "B2", "B3"],
            "invoice_dt": pd.to_datetime(
                ["2011-01-01", "2011-05-01", "2011-02-01", "2011-03-01", "2011-05-15"]
            ),
            "revenue": [10.0, 20.0, 5.0, 7.0, 9.0],
        }
    )

    features = build_customer_features(tx, cutoff)

    expected_cols = {
        "customer_id",
        "recency_days",
        "tenure_days",
        "n_invoices",
        "total_revenue",
        "avg_order_value",
        "revenue_last_30d",
        "revenue_last_90d",
        "rev_30_to_90_ratio",
    }
    assert expected_cols.issubset(set(features.columns))

    assert features["customer_id"].nunique() == len(features)
    assert (features["recency_days"] >= 0).all()
    assert (features["tenure_days"] >= 0).all()
    assert (features["n_invoices"] > 0).all()
    assert (features["total_revenue"] >= 0).all()


def test_build_customer_features_respects_cutoff() -> None:
    """
    Transactions on or after the cutoff must not influence feature values.
    """
    cutoff = pd.Timestamp("2011-06-01")

    tx = pd.DataFrame(
        {
            "customer_id": [1, 1],
            "invoice": ["A1", "A2"],
            "invoice_dt": pd.to_datetime(["2011-05-01", "2011-07-01"]),
            "revenue": [10.0, 9999.0],
        }
    )

    features = build_customer_features(tx, cutoff)
    row = features.loc[features["customer_id"] == 1].iloc[0]

    assert abs(float(row["total_revenue"]) - 10.0) < 1e-9
