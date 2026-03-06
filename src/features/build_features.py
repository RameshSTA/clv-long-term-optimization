"""
Customer-level feature engineering.

This module aggregates cleaned transactional purchase data into a single row per
customer using only information available up to a defined cutoff date. The resulting
feature table is designed to be directly usable for:

- CLV modeling (BG/NBD + Gamma-Gamma)
- churn / inactivity modeling
- customer segmentation and targeting
- downstream decision optimization

Design principles
-----------------
- Time safety: Features are computed strictly from transactions before the cutoff date.
- Reproducibility: A single cutoff date fully determines the feature snapshot.
- Interpretability: Features are domain-aligned (RFM, tenure, recent spend windows).
- Stability: Handles customers with no recent purchases via explicit zero-fill rules
  after aggregation (not before).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for customer feature generation."""

    input_path: Path
    output_path: Path
    cutoff_date: pd.Timestamp
    log_level: str


REQUIRED_INPUT_COLUMNS = (
    "customer_id",
    "invoice",
    "invoice_dt",
    "revenue",
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


def parse_args(argv: Sequence[str] | None = None) -> FeatureConfig:
    """
    Parse CLI arguments into a FeatureConfig.

    Parameters
    ----------
    argv:
        Optional argument list. If None, argparse reads from sys.argv.

    Returns
    -------
    FeatureConfig
        Parsed configuration for feature generation.
    """
    parser = argparse.ArgumentParser(
        description="Build customer-level features from cleaned transactions."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/interim/transactions_clean.parquet",
        help="Path to cleaned transactions (Parquet).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/customer_features.parquet",
        help="Path to write customer-level features (Parquet).",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2011-06-01",
        help="Cutoff date (YYYY-MM-DD) for feature construction.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args(args=argv)

    cutoff = pd.to_datetime(args.cutoff_date)

    return FeatureConfig(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        cutoff_date=cutoff,
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
    Assert the input DataFrame contains required columns.

    Parameters
    ----------
    df:
        Clean transactions DataFrame.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input transactions do not contain required columns. "
            f"Missing={missing}. Found={list(df.columns)}"
        )


def _sum_revenue_in_window(df: pd.DataFrame, cutoff: pd.Timestamp, window_days: int) -> pd.Series:
    """
    Sum revenue per customer over a trailing window ending at cutoff.

    Parameters
    ----------
    df:
        Transactions filtered to invoice_dt < cutoff.
    cutoff:
        Cutoff timestamp.
    window_days:
        Trailing window size in days.

    Returns
    -------
    pd.Series
        Revenue summed per customer for the trailing window.
    """
    days_before_cutoff = (cutoff - df["invoice_dt"]).dt.days
    in_window = days_before_cutoff <= window_days
    return df.loc[in_window].groupby("customer_id")["revenue"].sum()


def build_customer_features(transactions: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Aggregate transactions into customer-level, time-safe features.

    The function:
    - filters transactions strictly before cutoff
    - computes RFM-like metrics and tenure
    - computes trailing revenue windows (30d, 90d)
    - derives simple trend ratios
    - applies carefully chosen missing-value handling after aggregation

    Parameters
    ----------
    transactions:
        Clean transactional purchase data (output of cleaning step).
    cutoff:
        Cutoff date; only transactions with invoice_dt < cutoff are used.

    Returns
    -------
    pd.DataFrame
        Customer-level feature table, one row per customer.

    Raises
    ------
    ValueError
        If no transactions exist before cutoff or if sanity checks fail.
    """
    assert_required_schema(transactions)

    df = transactions.loc[transactions["invoice_dt"] < cutoff].copy()
    if df.empty:
        raise ValueError(
            "No transactions found before cutoff date. Check cutoff_date or input data."
        )

    min_dt = df["invoice_dt"].min()
    max_dt = df["invoice_dt"].max()
    LOGGER.info("Feature window: invoice_dt from %s to %s (cutoff=%s)", min_dt, max_dt, cutoff)

    # Core purchase dates per customer.
    last_purchase = df.groupby("customer_id")["invoice_dt"].max()
    first_purchase = df.groupby("customer_id")["invoice_dt"].min()

    # Purchase frequency: unique invoices per customer.
    n_invoices = df.groupby("customer_id")["invoice"].nunique()

    # Monetary: sum of revenue per customer.
    total_revenue = df.groupby("customer_id")["revenue"].sum()

    features = pd.DataFrame(
        {
            "recency_days": (cutoff - last_purchase).dt.days,
            "tenure_days": (cutoff - first_purchase).dt.days,
            "n_invoices": n_invoices,
            "total_revenue": total_revenue,
        }
    )

    # Average order value (AOV). Guard against divide-by-zero (should not happen for valid customers).
    features["avg_order_value"] = features["total_revenue"] / features["n_invoices"].replace(
        0, pd.NA
    )

    # Trailing revenue windows (these may be missing for customers with no purchases in the window).
    rev_30 = _sum_revenue_in_window(df, cutoff=cutoff, window_days=30)
    rev_90 = _sum_revenue_in_window(df, cutoff=cutoff, window_days=90)

    features["revenue_last_30d"] = rev_30
    features["revenue_last_90d"] = rev_90

    # Trend ratio: how much of 90-day revenue comes from last 30 days.
    # If revenue_last_90d is 0 or missing, define ratio as 0 (no recent contribution).
    denom = features["revenue_last_90d"].fillna(0.0)
    numer = features["revenue_last_30d"].fillna(0.0)
    features["rev_30_to_90_ratio"] = (numer / denom.replace(0.0, pd.NA)).fillna(0.0)

    # Fill window-based missing values with zeros (represents inactivity in those windows).
    features["revenue_last_30d"] = features["revenue_last_30d"].fillna(0.0)
    features["revenue_last_90d"] = features["revenue_last_90d"].fillna(0.0)

    # If AOV is missing due to any unexpected zero invoices, fill with 0.0 for robustness.
    features["avg_order_value"] = features["avg_order_value"].fillna(0.0)

    # Sanity checks: these guard against leakage or incorrect filtering.
    if (features["recency_days"] < 0).any():
        raise ValueError("Sanity check failed: negative recency_days detected.")

    if (features["n_invoices"] <= 0).any():
        raise ValueError("Sanity check failed: customers with n_invoices <= 0 detected.")

    if (features["total_revenue"] < 0).any():
        raise ValueError("Sanity check failed: negative total_revenue detected after cleaning.")

    return features.reset_index()


def profile_features(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute lightweight profiling metrics for the customer feature table.

    Parameters
    ----------
    df:
        Customer feature table.

    Returns
    -------
    Dict[str, float]
        Profiling metrics.
    """
    return {
        "rows": float(len(df)),
        "unique_customers": float(df["customer_id"].nunique(dropna=True)),
        "avg_recency_days": float(df["recency_days"].mean()),
        "avg_n_invoices": float(df["n_invoices"].mean()),
        "avg_total_revenue": float(df["total_revenue"].mean()),
        "pct_zero_revenue_last_30d": float((df["revenue_last_30d"] == 0).mean()),
    }


def write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write customer features to Parquet.

    Parameters
    ----------
    df:
        Feature table to write.
    output_path:
        Destination path.
    """
    ensure_parent_dir(output_path)
    LOGGER.info("Writing customer features to '%s' (rows=%d)", str(output_path), df.shape[0])
    df.to_parquet(output_path, index=False)
    LOGGER.info("Write complete")


def run(cfg: FeatureConfig) -> None:
    """
    Execute the feature engineering pipeline.

    Parameters
    ----------
    cfg:
        FeatureConfig for the current run.

    Raises
    ------
    FileNotFoundError
        If the cleaned transactions file does not exist.
    """
    setup_logging(cfg.log_level)

    if not cfg.input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {cfg.input_path}. Run Step 2 cleaning first."
        )

    LOGGER.info("Reading cleaned transactions from '%s'", str(cfg.input_path))
    transactions = pd.read_parquet(cfg.input_path)
    LOGGER.info("Transactions shape=%s", transactions.shape)

    features = build_customer_features(transactions=transactions, cutoff=cfg.cutoff_date)

    LOGGER.info("Customer feature table shape=%s", features.shape)
    LOGGER.info("Feature profiling metrics: %s", profile_features(features))

    write_parquet(features, cfg.output_path)

    LOGGER.info("Feature engineering completed successfully")


def main() -> None:
    """CLI entry point."""
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
