"""
Data cleaning for transactional purchase data.

This module takes the standardized raw transaction dataset produced by the ingestion
step and applies business-justified cleaning rules to produce a clean, analysis-ready
transaction table suitable for:
- CLV modeling
- churn / inactivity modeling
- feature engineering

Scope and responsibilities
--------------------------
- This step standardizes the dataset into "valid purchase events" only.
- This step does not aggregate to customer level (feature engineering happens later).
- This step does not encode modeling labels (labeling happens in modeling steps).

Cleaning philosophy
-------------------
We apply deterministic rules that remove records that cannot represent a valid purchase
event in this business context. Each rule is separately measurable so stakeholders can
understand the impact of cleaning (rows removed, customers affected, and revenue impact).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CleaningConfig:
    """Configuration for the cleaning pipeline."""
    input_path: Path
    output_path: Path
    log_level: str


CANONICAL_COLUMNS = (
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_dt",
    "unit_price",
    "customer_id",
    "country",
    "source_sheet",
)


def setup_logging(level: str) -> None:
    """
    Configure console logging.

    Parameters
    ----------
    level:
        Logging level name (e.g., 'INFO', 'DEBUG').

    Raises
    ------
    ValueError
        If an invalid log level is provided.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: Sequence[str] | None = None) -> CleaningConfig:
    """
    Parse CLI arguments into a CleaningConfig.

    Parameters
    ----------
    argv:
        Optional argument list. If None, argparse reads from sys.argv.

    Returns
    -------
    CleaningConfig
        Parsed configuration for cleaning.
    """
    parser = argparse.ArgumentParser(description="Clean transactional purchase data.")
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/interim/transactions_raw.parquet",
        help="Path to standardized raw transactions (Parquet).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/interim/transactions_clean.parquet",
        help="Path to write cleaned transactions (Parquet).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args(args=argv)

    return CleaningConfig(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        log_level=args.log_level,
    )


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure that the parent directory of `path` exists.

    Parameters
    ----------
    path:
        Target file path whose parent directory should exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def assert_required_schema(df: pd.DataFrame) -> None:
    """
    Validate that the raw-standardized input has the expected canonical columns.

    Parameters
    ----------
    df:
        Input DataFrame (output of ingestion step).

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input data does not match expected schema. "
            f"Missing columns: {missing}. Found: {list(df.columns)}"
        )


def profile_dataset(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute key quality indicators for a transaction dataset.

    Parameters
    ----------
    df:
        Transaction DataFrame.

    Returns
    -------
    Dict[str, float]
        Simple profiling metrics.
    """
    # Note: duplicates are computed on full rows by design.
    return {
        "rows": float(len(df)),
        "unique_customers": float(df["customer_id"].nunique(dropna=True)),
        "missing_customer_id_rate": float(df["customer_id"].isna().mean()),
        "missing_invoice_dt_rate": float(df["invoice_dt"].isna().mean()),
        "nonpositive_price_rate": float((df["unit_price"].fillna(0) <= 0).mean()),
        "nonpositive_quantity_rate": float((df["quantity"].fillna(0) <= 0).mean()),
        "duplicate_row_rate": float(df.duplicated().mean()),
    }


def compute_revenue(df: pd.DataFrame) -> pd.Series:
    """
    Compute revenue as quantity * unit_price.

    Assumptions
    -----------
    - quantity is positive integer after cleaning.
    - unit_price is positive numeric after cleaning.

    Parameters
    ----------
    df:
        Cleaned DataFrame.

    Returns
    -------
    pd.Series
        Revenue per row.
    """
    # Cast to stable numeric types to avoid unexpected object dtype arithmetic.
    qty = df["quantity"].astype("int64")
    price = df["unit_price"].astype("float64")
    return qty * price


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply business-driven cleaning rules to standardized transactional data.

    Rules implemented
    -----------------
    1) Remove cancellations: invoices starting with 'C' represent cancellations/returns.
    2) Keep only positive quantities: quantity <= 0 cannot represent a valid purchase event.
    3) Keep only positive prices: unit_price <= 0 cannot represent a valid purchase event.
    4) Require customer_id: customer-level modeling requires customer identifiers.
    5) Require invoice_dt: time-based modeling requires valid timestamps.
    6) Drop exact duplicate rows: removes repeated entries with identical fields.
    7) Compute revenue explicitly for downstream modeling.

    Parameters
    ----------
    df:
        Standardized raw transaction dataset (from ingestion).

    Returns
    -------
    pd.DataFrame
        Cleaned transaction dataset containing valid purchase events only.
    """
    assert_required_schema(df)

    cleaned = df.copy()

    # Rule 1: Remove cancellations (invoice codes starting with 'C').
    # Online Retail II commonly uses 'C' prefix for cancelled invoices.
    is_cancellation = cleaned["invoice"].astype("string").str.startswith("C", na=False)
    cleaned = cleaned.loc[~is_cancellation]

    # Rule 2: Require positive quantity.
    cleaned = cleaned.loc[cleaned["quantity"].notna()]
    cleaned = cleaned.loc[cleaned["quantity"] > 0]

    # Rule 3: Require positive unit price.
    cleaned = cleaned.loc[cleaned["unit_price"].notna()]
    cleaned = cleaned.loc[cleaned["unit_price"] > 0]

    # Rule 4: Require customer identifier.
    cleaned = cleaned.loc[cleaned["customer_id"].notna()]

    # Rule 5: Require timestamp.
    cleaned = cleaned.loc[cleaned["invoice_dt"].notna()]

    # Rule 6: Remove exact duplicate rows.
    cleaned = cleaned.drop_duplicates()

    # Rule 7: Compute revenue.
    cleaned = cleaned.assign(revenue=compute_revenue(cleaned))

    # Optional: deterministic ordering is helpful for diffing and auditing.
    cleaned = cleaned.sort_values(["customer_id", "invoice_dt", "invoice"], kind="mergesort")

    return cleaned.reset_index(drop=True)


def cleaning_impact(before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, float]:
    """
    Compute before/after cleaning impact summary.

    Parameters
    ----------
    before:
        Raw-standardized DataFrame before cleaning.
    after:
        Cleaned DataFrame after rules are applied.

    Returns
    -------
    Dict[str, float]
        Summary impact metrics for logging and reporting.
    """
    rows_before = len(before)
    rows_after = len(after)
    removed = rows_before - rows_after

    return {
        "rows_before": float(rows_before),
        "rows_after": float(rows_after),
        "rows_removed": float(removed),
        "pct_removed": float(removed / max(rows_before, 1)),
        "customers_before": float(before["customer_id"].nunique(dropna=True)),
        "customers_after": float(after["customer_id"].nunique(dropna=True)),
        "revenue_after_sum": float(after["revenue"].sum()) if "revenue" in after.columns else float("nan"),
    }


def write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write a DataFrame to Parquet.

    Parameters
    ----------
    df:
        DataFrame to write.
    output_path:
        Destination path.
    """
    ensure_parent_dir(output_path)
    LOGGER.info("Writing cleaned transactions to '%s' (rows=%d)", str(output_path), df.shape[0])
    df.to_parquet(output_path, index=False)
    LOGGER.info("Write complete")


def run(cfg: CleaningConfig) -> None:
    """
    Execute the cleaning pipeline.

    Parameters
    ----------
    cfg:
        CleaningConfig for the current run.

    Raises
    ------
    FileNotFoundError
        If the input parquet file does not exist.
    """
    setup_logging(cfg.log_level)

    if not cfg.input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {cfg.input_path}. "
            "Run Step 1 ingestion first."
        )

    LOGGER.info("Reading standardized raw transactions from '%s'", str(cfg.input_path))
    raw_df = pd.read_parquet(cfg.input_path)
    LOGGER.info("Raw dataset shape=%s", raw_df.shape)
    LOGGER.info("Raw profiling metrics: %s", profile_dataset(raw_df))

    clean_df = clean_transactions(raw_df)

    LOGGER.info("Clean dataset shape=%s", clean_df.shape)
    LOGGER.info("Cleaning impact summary: %s", cleaning_impact(raw_df, clean_df))
    LOGGER.info("Clean profiling metrics: %s", profile_dataset(clean_df))

    write_parquet(clean_df, cfg.output_path)

    LOGGER.info("Cleaning step completed successfully")


def main() -> None:
    """CLI entry point."""
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()