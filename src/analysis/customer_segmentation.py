# src/analysis/customer_segmentation.py
"""
RFM-based customer segmentation with CLV and churn risk overlay.

Methodology
-----------
RFM (Recency, Frequency, Monetary) segmentation assigns each customer a score
on three dimensions using quartile ranking:

  R (Recency)   : Days since last purchase. Lower is better → reversed quartile.
  F (Frequency) : Number of distinct invoices. Higher is better.
  M (Monetary)  : Total historical revenue. Higher is better.

Each dimension is scored 1–4 (4 = best). Combined into segment labels using
a rule matrix based on RFM marketing literature (Kumar & Reinartz, 2012).

Segment labels and business meaning:
  Champions         : R=4, F=4 → high value, recent, active
  Loyal Customers   : F≥3, R≥3 → consistent purchasers
  Potential Loyalists: R≥3, F=2 → recent but not yet habitual
  New Customers     : R=4, F=1 → just started, retention priority
  At Risk           : R≤2, F≥3 → used to buy often; now inactive
  Cant Lose Them    : R=1, F=4 → previously top customer, now gone
  Hibernating       : R=2, F≤2 → low engagement both ways
  Lost              : R=1, F≤2 → dormant and low historical value

Outputs
-------
- data/processed/customer_segments.parquet — one row per customer
- reports/tables/segment_summary.csv — business summary by segment
- reports/figures/segment_clv_churn_heatmap.png — CLV vs churn by segment
- reports/figures/segment_distribution.png — customer count and revenue by segment
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

REQUIRED_FEATURES = {"customer_id", "recency_days", "n_invoices", "total_revenue"}
REQUIRED_CLV = {"customer_id", "clv_horizon"}
REQUIRED_CHURN = {"customer_id", "churn_probability"}


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
    """Parse CLI arguments for the customer segmentation module.

    Args:
        argv: Optional list of CLI tokens.  Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed :class:`argparse.Namespace` with ``features_path``,
        ``clv_path``, ``churn_path``, ``output_path``, and ``log_level``.
    """
    parser = argparse.ArgumentParser(description="RFM customer segmentation.")
    parser.add_argument("--features-path", default="data/processed/customer_features.parquet")
    parser.add_argument("--clv-path", default="data/processed/customer_clv_scores.parquet")
    parser.add_argument("--churn-path", default="data/processed/customer_churn_risk_scores.parquet")
    parser.add_argument("--output-path", default="data/processed/customer_segments.parquet")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _quartile_rank(series: pd.Series, ascending: bool = True, n: int = 4) -> pd.Series:
    """Rank a series into n equal-count bins (1 = worst, n = best)."""
    ranked = pd.qcut(series.rank(method="first"), q=n, labels=False, duplicates="drop")
    ranked = ranked.fillna(0).astype(int) + 1  # shift to 1-indexed
    if not ascending:
        ranked = n + 1 - ranked
    return ranked.clip(1, n)


def compute_rfm_scores(features: pd.DataFrame) -> pd.DataFrame:
    """
    Assign R, F, M quartile scores (1–4) to each customer.

    R: lower recency_days is better (more recent) → reversed quartile.
    F: higher n_invoices is better.
    M: higher total_revenue is better.
    """
    df = features[["customer_id", "recency_days", "n_invoices", "total_revenue"]].copy()
    df["R"] = _quartile_rank(df["recency_days"], ascending=False)   # lower recency = better
    df["F"] = _quartile_rank(df["n_invoices"], ascending=True)
    df["M"] = _quartile_rank(df["total_revenue"], ascending=True)
    df["rfm_score"] = df["R"] + df["F"] + df["M"]
    return df


def assign_segment(row: pd.Series) -> str:
    """
    Map RFM scores to a named business segment.

    Priority rules are applied top-down; first match wins.
    """
    R, F, M = int(row["R"]), int(row["F"]), int(row["M"])

    if R == 4 and F == 4:
        return "Champions"
    if R == 1 and F == 4:
        return "Cant Lose Them"
    if R <= 2 and F >= 3:
        return "At Risk"
    if R >= 3 and F >= 3:
        return "Loyal Customers"
    if R >= 3 and F == 2:
        return "Potential Loyalists"
    if R == 4 and F == 1:
        return "New Customers"
    if R == 1 and F <= 2:
        return "Lost"
    return "Hibernating"


# Segment display order and colors for consistent plotting
SEGMENT_ORDER = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "New Customers",
    "At Risk",
    "Cant Lose Them",
    "Hibernating",
    "Lost",
]

SEGMENT_COLORS = {
    "Champions":           "#1565C0",
    "Loyal Customers":     "#1E88E5",
    "Potential Loyalists": "#42A5F5",
    "New Customers":       "#80D8FF",
    "At Risk":             "#FF8F00",
    "Cant Lose Them":      "#E65100",
    "Hibernating":         "#78909C",
    "Lost":                "#37474F",
}


def build_segments(features: pd.DataFrame, clv: pd.DataFrame, churn: pd.DataFrame) -> pd.DataFrame:
    """Combine RFM scores with CLV and churn risk into a single segment frame."""
    rfm = compute_rfm_scores(features)
    rfm["segment"] = rfm.apply(assign_segment, axis=1)

    clv_cols = clv[["customer_id", "clv_horizon", "p_alive"]].rename(
        columns={"clv_horizon": "clv_score", "p_alive": "probability_alive"}
    )
    churn_cols = churn[["customer_id", "churn_probability", "risk_band"]]

    merged = rfm.merge(clv_cols, on="customer_id", how="left")
    merged = merged.merge(churn_cols, on="customer_id", how="left")
    merged["segment"] = pd.Categorical(merged["segment"], categories=SEGMENT_ORDER, ordered=True)

    return merged


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute business-level summary statistics per segment."""
    summary = (
        df.groupby("segment", observed=True)
        .agg(
            n_customers=("customer_id", "count"),
            pct_customers=("customer_id", lambda x: 100 * len(x) / len(df)),
            avg_clv=("clv_score", "mean"),
            median_clv=("clv_score", "median"),
            avg_churn_prob=("churn_probability", "mean"),
            avg_recency_days=("recency_days", "mean"),
            avg_n_invoices=("n_invoices", "mean"),
            avg_revenue=("total_revenue", "mean"),
            total_revenue=("total_revenue", "sum"),
            pct_revenue=("total_revenue", lambda x: 100 * x.sum() / df["total_revenue"].sum()),
        )
        .reset_index()
    )
    summary = summary.round(2)
    return summary


def plot_segment_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Bubble heatmap: segment vs average CLV (color) and churn probability (size)."""
    summary = segment_summary(df)
    segs = [s for s in SEGMENT_ORDER if s in summary["segment"].values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: avg CLV and churn prob by segment (bar with secondary axis)
    ax = axes[0]
    x = np.arange(len(segs))
    seg_data = summary.set_index("segment").reindex(segs)

    colors = [SEGMENT_COLORS.get(s, "#90A4AE") for s in segs]
    bars = ax.bar(x, seg_data["avg_clv"].values, color=colors, edgecolor="white", linewidth=0.5)

    ax2 = ax.twinx()
    ax2.plot(x, seg_data["avg_churn_prob"].values * 100, "o--", color="#E53935",
             linewidth=2, markersize=7, label="Avg churn prob (%)")
    ax2.set_ylabel("Avg Churn Probability (%)", fontsize=10, color="#E53935")
    ax2.tick_params(axis="y", labelcolor="#E53935")
    ax2.set_ylim(0, 100)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n") for s in segs], fontsize=9)
    ax.set_ylabel("Average CLV Score (£)", fontsize=10)
    ax.set_title("Average CLV and Churn Risk by Segment", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 2: customer count and % revenue
    ax3 = axes[1]
    width = 0.4
    x2 = np.arange(len(segs))
    ax3.bar(x2 - width / 2, seg_data["n_customers"].values, width=width,
            color=colors, edgecolor="white", label="# Customers")
    ax4 = ax3.twinx()
    ax4.bar(x2 + width / 2, seg_data["pct_revenue"].values, width=width,
            color=[c + "99" for c in [SEGMENT_COLORS.get(s, "#90A4AE") for s in segs]],
            edgecolor="white", label="% Revenue")
    ax4.set_ylabel("% Total Revenue", fontsize=10)

    ax3.set_xticks(x2)
    ax3.set_xticklabels([s.replace(" ", "\n") for s in segs], fontsize=9)
    ax3.set_ylabel("Number of Customers", fontsize=10)
    ax3.set_title("Segment Size and Revenue Contribution", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    fig.suptitle("RFM Customer Segmentation — Business Overview", fontsize=13, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Segment heatmap → '%s'", str(output_path))


def plot_rfm_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter: Churn Probability vs CLV Score, colored by segment."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for seg in SEGMENT_ORDER:
        subset = df[df["segment"] == seg]
        if len(subset) == 0:
            continue
        ax.scatter(
            subset["churn_probability"],
            subset["clv_score"],
            label=f"{seg} (n={len(subset)})",
            color=SEGMENT_COLORS.get(seg, "#90A4AE"),
            alpha=0.6,
            s=20,
            linewidths=0,
        )

    # Annotation quadrants
    ax.axhline(df["clv_score"].median(), color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(0.02, ax.get_ylim()[1] * 0.97, "Low churn risk\nHigh CLV → Retain",
            fontsize=8, color="green", va="top")
    ax.text(0.75, ax.get_ylim()[1] * 0.97, "High churn risk\nHigh CLV → Priority target",
            fontsize=8, color="red", va="top")

    ax.set_xlabel("Churn Probability", fontsize=11)
    ax.set_ylabel("CLV Score (£)", fontsize=11)
    ax.set_title("Customer Segments: CLV vs Churn Risk",
                 fontsize=12, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("RFM scatter → '%s'", str(output_path))


def run(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    setup_logging(args.log_level)

    for p, name in [
        (args.features_path, "customer_features"),
        (args.clv_path, "customer_clv_scores"),
        (args.churn_path, "customer_churn_risk_scores"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required input missing: {p}. Run the full pipeline first.")

    LOGGER.info("Loading inputs...")
    features = pd.read_parquet(args.features_path)
    clv = pd.read_parquet(args.clv_path)
    churn = pd.read_parquet(args.churn_path)

    LOGGER.info("Building RFM segments (n=%d customers)...", len(features))
    segments = build_segments(features, clv, churn)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments.to_parquet(output_path, index=False)
    LOGGER.info("Customer segments → '%s'", str(output_path))

    summary = segment_summary(segments)
    summary_path = Path("reports/tables/segment_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Segment summary → '%s'", str(summary_path))

    LOGGER.info("Segment distribution:\n%s", summary[["segment", "n_customers",
                "pct_customers", "avg_clv", "avg_churn_prob"]].to_string(index=False))

    plot_segment_heatmap(segments, Path("reports/figures/segment_clv_churn_heatmap.png"))
    plot_rfm_scatter(segments, Path("reports/figures/segment_rfm_scatter.png"))

    LOGGER.info("Customer segmentation complete.")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
