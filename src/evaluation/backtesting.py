"""
Step 7: Evaluation and reporting for CLV + churn risk + budget targeting.

This script produces:
1) Out-of-time CLV ranking evaluation using holdout revenue:
   - decile lift table
   - rank correlation
2) Policy simulation evaluation for the targeting list:
   - ROI vs budget curve
   - expected benefit vs cost
3) Report artifacts:
   - CSV tables saved to reports/tables/
   - PNG figures saved to reports/figures/

Usage (from project root):
    python -m src.evaluation.backtesting \
        --transactions-path data/interim/transactions_clean.parquet \
        --clv-path data/processed/customer_clv_scores.parquet \
        --risk-path data/processed/customer_churn_risk_scores.parquet \
        --targeting-path data/processed/targeting_list.parquet \
        --cutoff-date 2011-06-01 \
        --holdout-days 180 \
        --retention-effectiveness 0.10 \
        --reports-dir reports \
        --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    transactions_path: str
    clv_path: str
    risk_path: str
    targeting_path: str
    cutoff_date: pd.Timestamp
    holdout_days: int
    retention_effectiveness: float
    reports_dir: str
    log_level: str


def setup_logging(level: str) -> None:
    """Configure root-level console logging for this script.

    Args:
        level: A string log level (e.g. ``"INFO"``, ``"DEBUG"``, ``"WARNING"``).

    Raises:
        ValueError: If ``level`` is not a recognised Python logging level name.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_dir(path: str) -> None:
    """Create *path* and all intermediate directories if they do not exist.

    Args:
        path: Directory path to create.  Existing directories are left
            untouched (idempotent).
    """
    os.makedirs(path, exist_ok=True)


def parse_args() -> EvalConfig:
    """Parse CLI arguments and return a validated :class:`EvalConfig`.

    All arguments have sensible defaults so the script can be invoked without
    flags. Production runs should supply explicit paths to avoid relying on
    the working-directory convention.

    Returns:
        A frozen :class:`EvalConfig` dataclass with all parameters resolved
        to their correct Python types.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate CLV and targeting policy; generate reports."
    )
    parser.add_argument(
        "--transactions-path",
        type=str,
        default="data/interim/transactions_clean.parquet",
        help="Path to cleaned transactions (Parquet).",
    )
    parser.add_argument(
        "--clv-path",
        type=str,
        default="data/processed/customer_clv_scores.parquet",
        help="Path to CLV scores (Parquet).",
    )
    parser.add_argument(
        "--risk-path",
        type=str,
        default="data/processed/customer_churn_risk_scores.parquet",
        help="Path to churn risk scores (Parquet).",
    )
    parser.add_argument(
        "--targeting-path",
        type=str,
        default="data/processed/targeting_list.parquet",
        help="Path to targeting list (Parquet).",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2011-06-01",
        help="Cutoff date used for backtesting (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=180,
        help="Holdout window length in days.",
    )
    parser.add_argument(
        "--retention-effectiveness",
        type=float,
        default=0.10,
        help="Same assumption used in budget allocation (0..1).",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory for saving tables and figures.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    args = parser.parse_args()

    return EvalConfig(
        transactions_path=args.transactions_path,
        clv_path=args.clv_path,
        risk_path=args.risk_path,
        targeting_path=args.targeting_path,
        cutoff_date=pd.to_datetime(args.cutoff_date),
        holdout_days=int(args.holdout_days),
        retention_effectiveness=float(args.retention_effectiveness),
        reports_dir=args.reports_dir,
        log_level=args.log_level,
    )


def aggregate_to_invoice_level(txn: pd.DataFrame) -> pd.DataFrame:
    """
    Convert line-item transactions to invoice-level events.

    Holdout revenue should be computed at invoice-level to match CLV modeling assumptions.
    """
    required_cols = {"customer_id", "invoice", "invoice_dt", "revenue"}
    missing = required_cols - set(txn.columns)
    if missing:
        raise ValueError(f"Missing required columns for invoice aggregation: {sorted(missing)}")

    inv = (
        txn.groupby(["customer_id", "invoice"], as_index=False)
        .agg(
            invoice_dt=("invoice_dt", "min"),
            revenue=("revenue", "sum"),
        )
        .sort_values(["customer_id", "invoice_dt", "invoice"])
        .reset_index(drop=True)
    )
    return inv


def compute_holdout_actuals(
    inv: pd.DataFrame,
    cutoff: pd.Timestamp,
    holdout_days: int,
) -> pd.DataFrame:
    """
    Compute actual holdout outcomes for evaluation.

    Holdout window: [cutoff, cutoff + holdout_days]
    """
    end = cutoff + pd.Timedelta(days=holdout_days)
    mask = (inv["invoice_dt"] >= cutoff) & (inv["invoice_dt"] <= end)
    holdout = inv.loc[mask].copy()

    actuals = holdout.groupby("customer_id", as_index=False).agg(
        holdout_transactions=("invoice", "nunique"),
        holdout_revenue=("revenue", "sum"),
    )
    return actuals


def spearman_rank_corr(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Spearman rank correlation (without scipy dependency).

    Returns NaN if insufficient variability.
    """
    if x.nunique() <= 1 or y.nunique() <= 1:
        return float("nan")

    rx = x.rank(method="average")
    ry = y.rank(method="average")

    cov = ((rx - rx.mean()) * (ry - ry.mean())).mean()
    std = rx.std(ddof=0) * ry.std(ddof=0)
    return float(cov / std) if std != 0 else float("nan")


def bootstrap_decile_ci(
    df_eval: pd.DataFrame,
    clv_col: str = "clv_horizon",
    revenue_col: str = "holdout_revenue",
    n_bootstrap: int = 500,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for average holdout revenue per CLV decile.

    Bootstrap procedure:
    - Resample customers with replacement `n_bootstrap` times.
    - For each resample, compute average holdout revenue per decile.
    - Report empirical CI from the percentile method.

    This quantifies the uncertainty in the decile lift chart and is crucial
    for communicating that the ranking benefit is statistically reliable.

    Parameters
    ----------
    df_eval:
        Evaluation DataFrame with columns clv_col, revenue_col, customer_id.
    clv_col:
        Column name for predicted CLV.
    revenue_col:
        Column name for actual holdout revenue.
    n_bootstrap:
        Number of bootstrap resamples.
    ci_level:
        Confidence level (e.g., 0.95 for 95% CI).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Decile table with ci_lower and ci_upper columns for avg_holdout_revenue.
    """
    rng = np.random.default_rng(random_state)
    n = len(df_eval)
    alpha = 1.0 - ci_level

    # Assign deciles on the full dataset using consistent ranks.
    work = df_eval.copy()
    work[clv_col] = work[clv_col].fillna(0.0)
    work[revenue_col] = work[revenue_col].fillna(0.0)
    work["_rank"] = work[clv_col].rank(method="first")
    work["decile"] = (pd.qcut(work["_rank"], 10, labels=False) + 1).astype(int)

    # Bootstrap: resample rows, keep decile assignments from full dataset.
    boot_means: dict[int, list[float]] = {d: [] for d in range(1, 11)}
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sample = work.iloc[sample_idx]
        for d in range(1, 11):
            vals = sample.loc[sample["decile"] == d, revenue_col]
            boot_means[d].append(float(vals.mean()) if len(vals) > 0 else 0.0)

    ci_rows = []
    for d in range(1, 11):
        arr = np.array(boot_means[d])
        ci_rows.append(
            {
                "decile": d,
                "ci_lower": float(np.percentile(arr, 100 * alpha / 2)),
                "ci_upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
                "ci_level": ci_level,
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(ci_rows)


def clv_decile_lift_table(
    df_eval: pd.DataFrame,
    clv_col: str = "clv_horizon",
    revenue_col: str = "holdout_revenue",
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Produce a decile lift table showing how predicted CLV ranks relate to actual holdout revenue.

    Includes 95% bootstrap confidence intervals on avg_holdout_revenue.
    Decile 10 = highest predicted CLV.

    Parameters
    ----------
    df_eval:
        Evaluation DataFrame with customer_id, clv_col, revenue_col.
    clv_col:
        Predicted CLV column.
    revenue_col:
        Actual holdout revenue column.
    n_bootstrap:
        Bootstrap resamples for CI estimation.
    random_state:
        Random seed for bootstrap.

    Returns
    -------
    pd.DataFrame
        Decile lift table with lift metric and bootstrap CIs.
    """
    work = df_eval.copy()
    work[clv_col] = work[clv_col].fillna(0.0)
    work[revenue_col] = work[revenue_col].fillna(0.0)

    # qcut can fail if many ties; use rank to stabilize bins
    work["_rank"] = work[clv_col].rank(method="first")
    work["decile"] = (pd.qcut(work["_rank"], 10, labels=False) + 1).astype(int)

    table = (
        work.groupby("decile", as_index=False)
        .agg(
            customers=("customer_id", "nunique"),
            avg_pred_clv=(clv_col, "mean"),
            avg_holdout_revenue=(revenue_col, "mean"),
            total_holdout_revenue=(revenue_col, "sum"),
        )
        .sort_values("decile")
    )

    # Lift relative to overall mean revenue
    overall_mean = float(work[revenue_col].mean())
    table["lift_vs_overall_mean"] = (
        table["avg_holdout_revenue"] / overall_mean if overall_mean > 0 else np.nan
    )

    # Bootstrap confidence intervals
    ci_df = bootstrap_decile_ci(
        df_eval=work,
        clv_col=clv_col,
        revenue_col=revenue_col,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    table = table.merge(ci_df[["decile", "ci_lower", "ci_upper"]], on="decile", how="left")
    return table


def plot_decile_lift(table: pd.DataFrame, out_path: str) -> None:
    """
    Save a decile lift plot with 95% bootstrap confidence intervals.

    The CI band illustrates statistical reliability of the ranking signal.
    A monotonically increasing curve with non-overlapping CIs across deciles
    is strong evidence of model utility for targeting decisions.

    Parameters
    ----------
    table:
        Output of clv_decile_lift_table (must contain ci_lower, ci_upper).
    out_path:
        File path for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        table["decile"],
        table["avg_holdout_revenue"],
        marker="o",
        color="#2563EB",
        linewidth=2,
        label="Avg Holdout Revenue",
    )

    if "ci_lower" in table.columns and "ci_upper" in table.columns:
        ax.fill_between(
            table["decile"],
            table["ci_lower"],
            table["ci_upper"],
            alpha=0.20,
            color="#2563EB",
            label="95% Bootstrap CI",
        )

    ax.axhline(
        table["avg_holdout_revenue"].mean(),
        color="#6B7280",
        linestyle="--",
        linewidth=1,
        label="Population Mean",
    )

    ax.set_xlabel("Predicted CLV Decile (1=Lowest, 10=Highest)", fontsize=11)
    ax.set_ylabel("Average Holdout Revenue (£)", fontsize=11)
    ax.set_title(
        "CLV Model Validation: Holdout Revenue by Predicted CLV Decile",
        fontsize=12,
        fontweight="bold",
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_policy_frame(
    clv: pd.DataFrame,
    risk: pd.DataFrame,
    unit_cost: float,
    retention_effectiveness: float,
) -> pd.DataFrame:
    """
    Build per-customer economics for policy simulation.

    expected_benefit = clv_horizon * churn_probability * retention_effectiveness
    net_gain = expected_benefit - unit_cost
    """
    if "customer_id" not in clv.columns or "customer_id" not in risk.columns:
        raise ValueError("Both CLV and risk tables must contain customer_id.")
    if "clv_horizon" not in clv.columns:
        raise ValueError("CLV table must contain clv_horizon.")
    if "churn_probability" not in risk.columns:
        raise ValueError("Risk table must contain churn_probability.")

    df = clv[["customer_id", "clv_horizon"]].merge(
        risk[["customer_id", "churn_probability"]],
        on="customer_id",
        how="inner",
    )

    df["cost"] = float(unit_cost)
    df["expected_benefit"] = (
        df["clv_horizon"].clip(lower=0.0)
        * df["churn_probability"].clip(lower=0.0, upper=1.0)
        * float(retention_effectiveness)
    )
    df["net_gain"] = df["expected_benefit"] - df["cost"]

    # A policy should not target negative net gain customers
    df = df.loc[df["net_gain"] > 0].copy()
    df = df.sort_values("net_gain", ascending=False).reset_index(drop=True)
    return df


def roi_curve(
    policy_df: pd.DataFrame,
    budgets: list[float],
) -> pd.DataFrame:
    """
    Compute ROI and totals under varying budgets using greedy selection on net_gain.

    This produces a smooth evaluation curve and is standard for reporting.

    ROI = (total_expected_benefit - total_cost) / total_cost
    """
    rows = []
    for b in budgets:
        if b <= 0:
            continue

        # Greedy selection by descending net gain
        spent = 0.0
        benefit = 0.0
        targeted = 0

        for _, r in policy_df.iterrows():
            c = float(r["cost"])
            if spent + c <= b:
                spent += c
                benefit += float(r["expected_benefit"])
                targeted += 1
            else:
                continue

        net = benefit - spent
        roi = (net / spent) if spent > 0 else 0.0

        rows.append(
            {
                "budget": float(b),
                "customers_targeted": float(targeted),
                "total_cost": float(spent),
                "total_expected_benefit": float(benefit),
                "total_net_gain": float(net),
                "roi": float(roi),
            }
        )

    return pd.DataFrame(rows)


def plot_roi_curve(curve_df: pd.DataFrame, out_path: str) -> None:
    """
    Save ROI vs budget plot with diminishing-returns annotation.

    A well-behaved policy shows monotonically decreasing ROI as budget expands:
    early spend targets the highest net-gain customers; later spend targets
    progressively lower-value customers.

    Parameters
    ----------
    curve_df:
        Output of roi_curve (must have budget, roi, total_expected_benefit).
    out_path:
        File path for the saved figure.
    """
    fig, ax1 = plt.subplots(figsize=(9, 5))

    color_roi = "#2563EB"
    ax1.plot(
        curve_df["budget"], curve_df["roi"], marker="o", color=color_roi, linewidth=2, label="ROI"
    )
    ax1.set_xlabel("Retention Budget (£)", fontsize=11)
    ax1.set_ylabel("Return on Investment (×)", fontsize=11, color=color_roi)
    ax1.tick_params(axis="y", labelcolor=color_roi)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}×"))
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    ax2 = ax1.twinx()
    color_benefit = "#16A34A"
    ax2.plot(
        curve_df["budget"],
        curve_df["total_expected_benefit"],
        marker="s",
        color=color_benefit,
        linewidth=2,
        linestyle="--",
        label="Expected Benefit",
    )
    ax2.set_ylabel("Total Expected Benefit (£)", fontsize=11, color=color_benefit)
    ax2.tick_params(axis="y", labelcolor=color_benefit)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.9)

    ax1.set_title(
        "Policy Simulation: ROI and Expected Benefit vs Retention Budget",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_targeted_vs_budget(curve_df: pd.DataFrame, out_path: str) -> None:
    """
    Save customers targeted vs budget plot.

    This chart supports operational planning: how many customers can be
    contacted for a given spend level?

    Parameters
    ----------
    curve_df:
        Output of roi_curve (must have budget, customers_targeted).
    out_path:
        File path for the saved figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(curve_df["budget"], curve_df["customers_targeted"], alpha=0.15, color="#7C3AED")
    ax.plot(
        curve_df["budget"],
        curve_df["customers_targeted"],
        marker="o",
        color="#7C3AED",
        linewidth=2,
        label="Customers Targeted",
    )

    ax.set_xlabel("Retention Budget (£)", fontsize=11)
    ax.set_ylabel("Customers Targeted", fontsize=11)
    ax.set_title(
        "Policy Simulation: Customers Targeted vs Retention Budget", fontsize=12, fontweight="bold"
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    """CLI entry point for Step 7: evaluation and report generation.

    Orchestrates the full evaluation workflow:

    1. Validates that all input files (transactions, CLV scores, churn scores,
       targeting list) are present on disk.
    2. Builds the CLV decile-lift table with 95 % bootstrap confidence
       intervals and saves it to ``reports/tables/``.
    3. Produces the dual-axis ROI curve (net gain + ROI vs budget) and
       targeted-customer vs budget plots, saved to ``reports/figures/``.
    4. Logs a human-readable summary to stdout.

    Invoked by ``python -m src.evaluation.backtesting`` or via the pipeline
    orchestrator (Step 7 of the weekly scoring pipeline).
    """
    cfg = parse_args()
    setup_logging(cfg.log_level)

    for pth in (cfg.transactions_path, cfg.clv_path, cfg.risk_path, cfg.targeting_path):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"Required input not found: {pth}")

    # Prepare report directories
    tables_dir = os.path.join(cfg.reports_dir, "tables")
    figs_dir = os.path.join(cfg.reports_dir, "figures")
    ensure_dir(tables_dir)
    ensure_dir(figs_dir)

    # Load required data
    LOGGER.info("Reading cleaned transactions: %s", cfg.transactions_path)
    txn = pd.read_parquet(cfg.transactions_path)

    LOGGER.info("Reading CLV scores: %s", cfg.clv_path)
    clv = pd.read_parquet(cfg.clv_path)

    LOGGER.info("Reading churn risk scores: %s", cfg.risk_path)
    risk = pd.read_parquet(cfg.risk_path)

    LOGGER.info("Reading targeting list: %s", cfg.targeting_path)
    targeting = pd.read_parquet(cfg.targeting_path)

    # Compute holdout actuals
    inv = aggregate_to_invoice_level(txn)
    actuals = compute_holdout_actuals(inv, cfg.cutoff_date, cfg.holdout_days)

    # CLV evaluation frame
    eval_df = clv[["customer_id", "clv_horizon"]].merge(actuals, on="customer_id", how="left")
    eval_df["holdout_transactions"] = eval_df["holdout_transactions"].fillna(0).astype(int)
    eval_df["holdout_revenue"] = eval_df["holdout_revenue"].fillna(0.0)

    # CLV ranking quality
    rho = spearman_rank_corr(eval_df["clv_horizon"], eval_df["holdout_revenue"])
    LOGGER.info("CLV ranking quality (Spearman rank correlation vs holdout revenue): %.4f", rho)

    # Decile lift (with 95% bootstrap confidence intervals)
    LOGGER.info("Computing decile lift table with bootstrap CIs (n_bootstrap=500)...")
    decile_table = clv_decile_lift_table(eval_df, n_bootstrap=500, random_state=42)
    decile_csv = os.path.join(tables_dir, "clv_decile_lift.csv")
    decile_table.to_csv(decile_csv, index=False)
    LOGGER.info("Wrote decile lift table: %s", decile_csv)

    # Plot decile lift
    decile_fig = os.path.join(figs_dir, "clv_decile_lift.png")
    plot_decile_lift(decile_table, decile_fig)
    LOGGER.info("Wrote decile lift figure: %s", decile_fig)

    # Policy simulation evaluation
    # Infer unit cost from targeting list if available; otherwise default to 2.0
    unit_cost = float(targeting["cost"].median()) if "cost" in targeting.columns else 2.0

    policy_df = build_policy_frame(
        clv=clv,
        risk=risk,
        unit_cost=unit_cost,
        retention_effectiveness=cfg.retention_effectiveness,
    )

    # Budget grid for curves
    # Use a sensible grid based on observed scale
    max_budget = float(min(20000.0, max(100.0, unit_cost * min(len(policy_df), 10000))))
    budgets = list(np.linspace(unit_cost * 100, max_budget, 12))

    curve = roi_curve(policy_df, budgets)
    curve_csv = os.path.join(tables_dir, "policy_roi_curve.csv")
    curve.to_csv(curve_csv, index=False)
    LOGGER.info("Wrote policy ROI curve table: %s", curve_csv)

    # Plots
    roi_fig = os.path.join(figs_dir, "policy_roi_vs_budget.png")
    plot_roi_curve(curve, roi_fig)
    LOGGER.info("Wrote ROI vs budget figure: %s", roi_fig)

    tgt_fig = os.path.join(figs_dir, "policy_targeted_vs_budget.png")
    plot_targeted_vs_budget(curve, tgt_fig)
    LOGGER.info("Wrote targeted vs budget figure: %s", tgt_fig)

    # Save a compact executive summary table (single-row)
    exec_summary = pd.DataFrame(
        [
            {
                "cutoff_date": str(cfg.cutoff_date.date()),
                "holdout_days": cfg.holdout_days,
                "spearman_corr_clv_vs_holdout_revenue": rho,
                "customers_scored_clv": float(clv["customer_id"].nunique()),
                "customers_scored_risk": float(risk["customer_id"].nunique()),
                "unit_cost_assumed": unit_cost,
                "retention_effectiveness_assumed": cfg.retention_effectiveness,
            }
        ]
    )
    exec_csv = os.path.join(tables_dir, "executive_summary.csv")
    exec_summary.to_csv(exec_csv, index=False)
    LOGGER.info("Wrote executive summary: %s", exec_csv)

    LOGGER.info("Step 7 completed successfully")


if __name__ == "__main__":
    main()
