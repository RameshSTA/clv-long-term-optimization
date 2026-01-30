# tests/test_cleaning.py
from __future__ import annotations

import pandas as pd

from src.cleaning.clean_transactions import clean_transactions


def test_cleaning_removes_invalid_rows() -> None:
    """
    The cleaner should remove:
    - cancellation invoices (invoice starts with 'C')
    - non-positive quantity
    - non-positive price
    - missing invoice_dt
    """
    df = pd.DataFrame(
        {
            "invoice": ["C1", "2", "3"],
            "stock_code": ["A", "B", "C"],
            "description": ["x", "y", "z"],
            "quantity": [-1, 1, 2],
            "invoice_dt": pd.to_datetime(["2011-01-01", "2011-01-02", None]),
            "unit_price": [10.0, 0.0, 5.0],
            "customer_id": [123, 456, 789],
            "country": ["UK", "UK", "UK"],
            "source_sheet": ["s", "s", "s"],
        }
    )

    out = clean_transactions(df)
    assert len(out) == 0


def test_cleaning_creates_revenue_and_keeps_valid_rows() -> None:
    """Cleaner should keep valid rows and create a correct revenue column."""
    df = pd.DataFrame(
        {
            "invoice": ["1", "2"],
            "stock_code": ["A", "B"],
            "description": ["x", "y"],
            "quantity": [2, 3],
            "invoice_dt": pd.to_datetime(["2011-01-01", "2011-01-02"]),
            "unit_price": [10.0, 5.0],
            "customer_id": [123, 456],
            "country": ["UK", "UK"],
            "source_sheet": ["s", "s"],
        }
    )

    out = clean_transactions(df)

    assert len(out) == 2
    assert "revenue" in out.columns
    assert out["revenue"].tolist() == [20.0, 15.0]
    assert (out["quantity"] > 0).all()
    assert (out["unit_price"] > 0).all()
    assert out["customer_id"].notna().all()
    assert out["invoice_dt"].notna().all()