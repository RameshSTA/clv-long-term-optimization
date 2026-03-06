# src/analysis/cohort_analysis.py
"""
Monthly cohort retention analysis.

Methodology
-----------
Cohort analysis groups customers by their acquisition month (first purchase month)
and tracks what percentage of each cohort made at least one purchase in each
subsequent month. This reveals:

  - Organic retention curves per acquisition cohort
  - Whether newer cohorts retain better/worse than older ones
  - The "natural" churn decay pattern in the business
  - Revenue contribution and stability per cohort

Cohort retention is the foundation of any serious CLV business case: it shows
whether CLV-based interventions are needed and how quickly value decays without action.

Outputs
-------
- reports/tables/cohort_retention.csv — cohort × month retention matrix
- reports/tables/cohort_revenue.csv — cohort revenue totals
- reports/figures/cohort_retention_heatmap.png — retention heatmap (% of cohort)
- reports/figures/cohort_revenue_bar.png — revenue by acquisition cohort
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Configure root-level console logging.

    Args:
        level: A string log level (e.g. ``"INFO"``, ``"DEBUG"``).
            Unrecognised values fall back to ``INFO``.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the cohort retention analysis module.

    Args:
        argv: Optional list of CLI tokens.  Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed :class:`argparse.Namespace` with ``transactions_path``,
        ``max_cohort_months``, ``min_cohort_size``, and ``log_level``.
    """
    parser = argparse.ArgumentParser(description="Monthly cohort retention analysis.")
    parser.add_argument("--transactions-path", default="data/interim/transactions_clean.parquet")
    parser.add_argument(
        "--max-cohort-months",
        type=int,
        default=12,
        help="Maximum number of months to track after acquisition.",
    )
    parser.add_argument(
        "--min-cohort-size",
        type=int,
        default=10,
        help="Minimum customers in a cohort to include in analysis.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def build_cohort_frame(txn: pd.DataFrame) -> pd.DataFrame:
    """
    Build cohort assignment and purchase month columns.

    Cohort = customer's first purchase month (YYYY-MM period).
    Purchase month = month of each transaction.

    Returns one row per (customer_id, purchase_month) pair.
    """
    txn = txn.copy()
    txn["purchase_month"] = txn["invoice_dt"].dt.to_period("M")

    # Acquisition cohort = first purchase month per customer
    first_purchase = txn.groupby("customer_id")["purchase_month"].min().rename("cohort_month")
    txn = txn.merge(first_purchase.reset_index(), on="customer_id", how="left")

    # Months since acquisition (0 = acquisition month)
    txn["months_since_acquisition"] = (
        txn["purchase_month"].dt.to_timestamp() - txn["cohort_month"].dt.to_timestamp()
    ).dt.days // 30

    return txn


def compute_retention_matrix(
    txn: pd.DataFrame,
    max_months: int = 12,
    min_cohort_size: int = 10,
) -> pd.DataFrame:
    """
    Compute cohort retention matrix: % of each cohort active in each month.

    Rows = acquisition cohort (period).
    Columns = months since acquisition (0, 1, 2, ..., max_months).
    Values = % of cohort who made at least one purchase in that month.
    """
    cohort_frame = build_cohort_frame(txn)

    # Cohort sizes (number of unique customers per acquisition cohort)
    cohort_sizes = (
        cohort_frame.groupby("cohort_month")["customer_id"].nunique().rename("cohort_size")
    )

    # Filter small cohorts
    valid_cohorts = cohort_sizes[cohort_sizes >= min_cohort_size].index

    # Monthly active customers per cohort
    monthly_active = (
        cohort_frame[cohort_frame["cohort_month"].isin(valid_cohorts)]
        .groupby(["cohort_month", "months_since_acquisition"])["customer_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_id": "active_customers"})
    )

    # Pivot to matrix shape
    retention = monthly_active.pivot(
        index="cohort_month",
        columns="months_since_acquisition",
        values="active_customers",
    ).reindex(columns=range(max_months + 1))

    # Merge cohort sizes
    retention = retention.join(cohort_sizes)

    # Convert to percentages
    for col in range(max_months + 1):
        if col in retention.columns:
            retention[col] = (retention[col] / retention["cohort_size"] * 100).round(1)

    retention = retention.drop(columns=["cohort_size"])
    retention.index = retention.index.astype(str)

    return retention


def compute_cohort_revenue(txn: pd.DataFrame, min_cohort_size: int = 10) -> pd.DataFrame:
    """Compute total and average revenue per acquisition cohort."""
    cohort_frame = build_cohort_frame(txn)
    cohort_sizes = cohort_frame.groupby("cohort_month")["customer_id"].nunique()
    valid_cohorts = cohort_sizes[cohort_sizes >= min_cohort_size].index

    rev = (
        cohort_frame[cohort_frame["cohort_month"].isin(valid_cohorts)]
        .groupby("cohort_month")
        .agg(
            total_revenue=("revenue", "sum"),
            n_customers=("customer_id", "nunique"),
            n_transactions=("invoice_dt", "count"),
        )
        .reset_index()
    )
    rev["avg_revenue_per_customer"] = (rev["total_revenue"] / rev["n_customers"]).round(2)
    rev["cohort_month"] = rev["cohort_month"].astype(str)
    return rev


def plot_cohort_heatmap(retention: pd.DataFrame, output_path: Path) -> None:
    """
    Seaborn heatmap of cohort retention rates (% active by month).
    """
    # Limit to first 12 months, last 15 cohorts for readability
    cols = [c for c in retention.columns if c <= 12]
    plot_data = retention[cols].tail(15)

    fig, ax = plt.subplots(figsize=(14, max(5, len(plot_data) * 0.5)))

    mask = plot_data.isna()
    sns.heatmap(
        plot_data,
        annot=True,
        fmt=".0f",
        mask=mask,
        cmap="YlOrRd_r",
        linewidths=0.3,
        linecolor="white",
        vmin=0,
        vmax=100,
        ax=ax,
        cbar_kws={"label": "% of Cohort Active", "shrink": 0.6},
        annot_kws={"size": 8},
    )

    ax.set_xlabel("Months Since First Purchase", fontsize=11)
    ax.set_ylabel("Acquisition Cohort (Month)", fontsize=11)
    ax.set_title(
        "Monthly Cohort Retention Rates\n(% of acquisition cohort who purchased in each month)",
        fontsize=12,
        fontweight="bold",
    )

    # Highlight acquisition month column
    ax.add_patch(
        plt.Rectangle((0, 0), 1, len(plot_data), fill=False, edgecolor="blue", linewidth=2)
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Cohort retention heatmap → '%s'", str(output_path))


def plot_cohort_revenue(cohort_rev: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of total revenue by acquisition cohort."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cohorts = cohort_rev["cohort_month"].values
    x = np.arange(len(cohorts))

    # Total revenue
    ax = axes[0]
    ax.bar(x, cohort_rev["total_revenue"].values / 1000, color="#1565C0", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total Revenue (£ thousands)", fontsize=10)
    ax.set_title("Total Revenue by Acquisition Cohort", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}k"))
    ax.grid(axis="y", alpha=0.3)

    # Avg revenue per customer
    ax2 = axes[1]
    ax2.bar(x, cohort_rev["avg_revenue_per_customer"].values, color="#C62828", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cohorts, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Avg Revenue per Customer (£)", fontsize=10)
    ax2.set_title("Average Revenue per Customer by Cohort", fontsize=11, fontweight="bold")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Cohort Revenue Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Cohort revenue chart → '%s'", str(output_path))


def plot_retention_curves(retention: pd.DataFrame, output_path: Path) -> None:
    """Line plot of retention curves — one line per cohort."""
    cols = sorted([c for c in retention.columns if isinstance(c, int) and c <= 9])
    plot_data = retention[cols].dropna(thresh=3)

    if len(plot_data) == 0:
        LOGGER.warning("No cohort data for retention curves.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap("Blues", len(plot_data) + 3)

    for i, (cohort, row) in enumerate(plot_data.iterrows()):
        values = row.dropna()
        if len(values) < 2:
            continue
        ax.plot(
            values.index,
            values.values,
            "o-",
            color=cmap(i + 2),
            linewidth=1.5,
            markersize=4,
            alpha=0.8,
            label=str(cohort),
        )

    ax.set_xlabel("Months Since First Purchase", fontsize=11)
    ax.set_ylabel("% of Cohort Active", fontsize=11)
    ax.set_title("Cohort Retention Curves", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title="Cohort", framealpha=0.9
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Retention curves → '%s'", str(output_path))


def run(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    setup_logging(args.log_level)

    txn_path = Path(args.transactions_path)
    if not txn_path.exists():
        raise FileNotFoundError(f"Transactions not found: {txn_path}. Run cleaning step first.")

    LOGGER.info("Loading transactions from '%s'...", str(txn_path))
    txn = pd.read_parquet(txn_path)
    LOGGER.info(
        "Transactions: %d rows, %d unique customers", len(txn), txn["customer_id"].nunique()
    )

    LOGGER.info("Computing cohort retention matrix (max %d months)...", args.max_cohort_months)
    retention = compute_retention_matrix(
        txn,
        max_months=args.max_cohort_months,
        min_cohort_size=args.min_cohort_size,
    )
    LOGGER.info("Cohorts included: %d", len(retention))

    # Save tables
    retention_path = Path("reports/tables/cohort_retention.csv")
    retention_path.parent.mkdir(parents=True, exist_ok=True)
    retention.to_csv(retention_path)
    LOGGER.info("Cohort retention matrix → '%s'", str(retention_path))

    cohort_rev = compute_cohort_revenue(txn, min_cohort_size=args.min_cohort_size)
    rev_path = Path("reports/tables/cohort_revenue.csv")
    cohort_rev.to_csv(rev_path, index=False)
    LOGGER.info("Cohort revenue → '%s'", str(rev_path))

    # Figures
    plot_cohort_heatmap(retention, Path("reports/figures/cohort_retention_heatmap.png"))
    plot_cohort_revenue(cohort_rev, Path("reports/figures/cohort_revenue_bar.png"))
    plot_retention_curves(retention, Path("reports/figures/cohort_retention_curves.png"))

    LOGGER.info("Cohort analysis complete.")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
