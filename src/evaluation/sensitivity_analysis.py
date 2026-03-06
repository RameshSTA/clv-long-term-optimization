# src/evaluation/sensitivity_analysis.py
"""
Monte Carlo sensitivity analysis on retention ROI assumptions.

Why this matters
----------------
The budget optimization model uses two fixed assumptions:
  - retention_effectiveness: % of targeted customers who are successfully retained
  - unit_cost: cost per customer contacted

Both are operationally assumed, not measured (no A/B test data in this public dataset).
A professional analysis must quantify how sensitive the ROI conclusions are to
these assumptions — and what the uncertainty bounds look like.

Methodology
-----------
Monte Carlo simulation (1,000 draws):
  - retention_effectiveness ~ Uniform(0.05, 0.25)  [central: 0.10]
  - unit_cost               ~ Uniform(1.0, 5.0)    [central: 2.0]

For each draw, recompute the full knapsack policy at N budget levels.
Collect ROI distribution per budget point → 5th / 50th / 95th percentiles.

Tornado chart (one-at-a-time sensitivity):
  Vary each parameter ±50% of its central value, holding others constant.
  Show which assumption has the largest impact on ROI at the reference budget.

Outputs
-------
- reports/tables/sensitivity_summary.csv — percentile bands per budget
- reports/figures/roi_monte_carlo.png — ROI uncertainty bands
- reports/figures/sensitivity_tornado.png — tornado chart by parameter
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

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
    """Parse CLI arguments for the Monte Carlo sensitivity analysis module.

    Args:
        argv: Optional list of CLI tokens.  Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed :class:`argparse.Namespace` with ``clv_path``, ``churn_path``,
        ``n_simulations``, ``n_budget_points``, ``base_effectiveness``,
        ``base_unit_cost``, ``reference_budget``, ``random_state``, and
        ``log_level``.
    """
    parser = argparse.ArgumentParser(description="Monte Carlo sensitivity analysis on ROI.")
    parser.add_argument("--clv-path", default="data/processed/customer_clv_scores.parquet")
    parser.add_argument("--churn-path", default="data/processed/customer_churn_risk_scores.parquet")
    parser.add_argument("--n-simulations", type=int, default=1000,
                        help="Number of Monte Carlo draws.")
    parser.add_argument("--n-budget-points", type=int, default=12,
                        help="Number of budget levels to evaluate.")
    parser.add_argument("--base-effectiveness", type=float, default=0.10,
                        help="Central retention effectiveness assumption.")
    parser.add_argument("--base-unit-cost", type=float, default=2.0,
                        help="Central unit cost per customer (£).")
    parser.add_argument("--reference-budget", type=float, default=2000.0,
                        help="Reference budget for tornado chart.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def _compute_roi_at_budget(
    clv: np.ndarray,
    churn_prob: np.ndarray,
    effectiveness: float,
    unit_cost: float,
    budget: float,
) -> Tuple[float, float, int]:
    """
    Greedy knapsack allocation at a single budget level.

    Returns (roi, net_gain, n_targeted).
    """
    expected_benefit = clv * churn_prob * effectiveness
    net_gain = expected_benefit - unit_cost

    # Only consider customers with positive net gain
    eligible = net_gain > 0
    if not eligible.any() or budget <= 0:
        return 0.0, 0.0, 0

    # Greedy: sort by benefit/cost ratio, select until budget exhausted
    eligible_idx = np.where(eligible)[0]
    ratios = expected_benefit[eligible_idx] / unit_cost
    sorted_idx = eligible_idx[np.argsort(ratios)[::-1]]

    total_cost = 0.0
    total_benefit = 0.0
    n_targeted = 0

    for i in sorted_idx:
        if total_cost + unit_cost > budget:
            break
        total_cost += unit_cost
        total_benefit += expected_benefit[i]
        n_targeted += 1

    if total_cost == 0:
        return 0.0, 0.0, 0

    net = total_benefit - total_cost
    roi = net / total_cost
    return float(roi), float(net), int(n_targeted)


def monte_carlo_simulation(
    clv: np.ndarray,
    churn_prob: np.ndarray,
    budget_levels: np.ndarray,
    n_simulations: int,
    effectiveness_range: Tuple[float, float],
    cost_range: Tuple[float, float],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation over uncertainty in effectiveness and unit cost.

    Returns DataFrame with columns:
      budget, roi_p5, roi_p50, roi_p95,
      net_gain_p5, net_gain_p50, net_gain_p95
    """
    rng = np.random.default_rng(random_state)
    effectivenesses = rng.uniform(*effectiveness_range, size=n_simulations)
    unit_costs = rng.uniform(*cost_range, size=n_simulations)

    # Shape: (n_simulations, n_budget_levels) for ROI and net_gain
    roi_matrix = np.zeros((n_simulations, len(budget_levels)))
    net_matrix = np.zeros((n_simulations, len(budget_levels)))

    for sim_idx in range(n_simulations):
        eff = effectivenesses[sim_idx]
        cost = unit_costs[sim_idx]
        for b_idx, budget in enumerate(budget_levels):
            roi, net, _ = _compute_roi_at_budget(clv, churn_prob, eff, cost, budget)
            roi_matrix[sim_idx, b_idx] = roi
            net_matrix[sim_idx, b_idx] = net

    rows = []
    for b_idx, budget in enumerate(budget_levels):
        roi_col = roi_matrix[:, b_idx]
        net_col = net_matrix[:, b_idx]
        rows.append({
            "budget": float(budget),
            "roi_p5": float(np.percentile(roi_col, 5)),
            "roi_p50": float(np.percentile(roi_col, 50)),
            "roi_p95": float(np.percentile(roi_col, 95)),
            "net_gain_p5": float(np.percentile(net_col, 5)),
            "net_gain_p50": float(np.percentile(net_col, 50)),
            "net_gain_p95": float(np.percentile(net_col, 95)),
        })

    return pd.DataFrame(rows)


def tornado_sensitivity(
    clv: np.ndarray,
    churn_prob: np.ndarray,
    reference_budget: float,
    base_params: dict,
    variation: float = 0.5,
) -> pd.DataFrame:
    """
    One-at-a-time (OAT) sensitivity analysis at the reference budget.

    For each parameter, compute ROI at (base - variation%, base, base + variation%).
    Returns a DataFrame suitable for tornado chart plotting.
    """
    params = {
        "retention_effectiveness": base_params["effectiveness"],
        "unit_cost": base_params["unit_cost"],
    }

    base_roi, _, _ = _compute_roi_at_budget(
        clv, churn_prob,
        params["retention_effectiveness"],
        params["unit_cost"],
        reference_budget,
    )

    rows = []
    for param_name, base_val in params.items():
        low_val = base_val * (1 - variation)
        high_val = base_val * (1 + variation)

        if param_name == "unit_cost":
            low_val = max(low_val, 0.1)

        roi_low, _, _ = _compute_roi_at_budget(
            clv, churn_prob,
            low_val if param_name == "retention_effectiveness" else params["retention_effectiveness"],
            low_val if param_name == "unit_cost" else params["unit_cost"],
            reference_budget,
        )
        roi_high, _, _ = _compute_roi_at_budget(
            clv, churn_prob,
            high_val if param_name == "retention_effectiveness" else params["retention_effectiveness"],
            high_val if param_name == "unit_cost" else params["unit_cost"],
            reference_budget,
        )

        rows.append({
            "parameter": param_name,
            "base_value": base_val,
            "low_value": low_val,
            "high_value": high_val,
            "roi_at_base": base_roi,
            "roi_at_low": roi_low,
            "roi_at_high": roi_high,
            "roi_swing": abs(roi_high - roi_low),
        })

    return pd.DataFrame(rows).sort_values("roi_swing", ascending=True)


def plot_monte_carlo_roi(mc_df: pd.DataFrame, base_df: pd.DataFrame, output_path: Path) -> None:
    """ROI vs budget with Monte Carlo uncertainty bands."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: ROI bands
    ax = axes[0]
    budgets = mc_df["budget"].values
    ax.fill_between(budgets, mc_df["roi_p5"], mc_df["roi_p95"],
                    alpha=0.2, color="#1565C0", label="5th–95th percentile (uncertainty)")
    ax.fill_between(budgets, mc_df["roi_p5"], mc_df["roi_p50"],
                    alpha=0.3, color="#1565C0")
    ax.plot(budgets, mc_df["roi_p50"], color="#1565C0", linewidth=2.5,
            label="Median ROI")

    if base_df is not None and len(base_df) > 0:
        ax.plot(base_df["budget"], base_df["roi"], color="#E53935", linewidth=1.5,
                linestyle="--", label="Baseline (central assumptions)")

    ax.set_xlabel("Retention Budget (£)", fontsize=11)
    ax.set_ylabel("Return on Investment (×)", fontsize=11)
    ax.set_title("ROI vs Budget — Monte Carlo Uncertainty\n"
                 "(retention_effectiveness ~ Uniform(5%–25%), cost ~ Uniform(£1–£5))",
                 fontsize=10, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}×"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Net gain bands
    ax2 = axes[1]
    ax2.fill_between(budgets, mc_df["net_gain_p5"] / 1000, mc_df["net_gain_p95"] / 1000,
                     alpha=0.2, color="#43A047", label="5th–95th percentile")
    ax2.plot(budgets, mc_df["net_gain_p50"] / 1000, color="#43A047", linewidth=2.5,
             label="Median net gain")
    ax2.set_xlabel("Retention Budget (£)", fontsize=11)
    ax2.set_ylabel("Expected Net Gain (£ thousands)", fontsize=11)
    ax2.set_title("Net Gain vs Budget — Uncertainty Bands", fontsize=11, fontweight="bold")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}k"))
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle("Monte Carlo Sensitivity Analysis — Retention ROI Under Assumption Uncertainty",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Monte Carlo ROI chart → '%s'", str(output_path))


def plot_tornado(tornado_df: pd.DataFrame, reference_budget: float, output_path: Path) -> None:
    """Tornado chart: one-at-a-time parameter sensitivity at reference budget."""
    fig, ax = plt.subplots(figsize=(9, max(4, len(tornado_df) * 1.5)))

    base_roi = tornado_df.iloc[-1]["roi_at_base"]  # last row has largest swing
    param_labels = [p.replace("_", " ").title() for p in tornado_df["parameter"]]

    for i, (_, row) in enumerate(tornado_df.iterrows()):
        low_delta = row["roi_at_low"] - row["roi_at_base"]
        high_delta = row["roi_at_high"] - row["roi_at_base"]

        ax.barh(i, high_delta, left=0, color="#1565C0", alpha=0.75, height=0.5,
                label="Parameter +50%" if i == 0 else "")
        ax.barh(i, low_delta, left=0, color="#C62828", alpha=0.75, height=0.5,
                label="Parameter −50%" if i == 0 else "")

    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_yticks(range(len(tornado_df)))
    ax.set_yticklabels([p.replace("_", " ").title() for p in tornado_df["parameter"]], fontsize=11)
    ax.set_xlabel("Change in ROI from baseline (×)", fontsize=11)
    ax.set_title(
        f"Sensitivity Tornado Chart\nROI impact of ±50% parameter variation "
        f"at £{reference_budget:,.0f} budget",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # Base ROI annotation
    ax.text(0.98, 0.02, f"Baseline ROI: {base_roi:.1f}×",
            transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Tornado chart → '%s'", str(output_path))


def run(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    setup_logging(args.log_level)

    for p in [args.clv_path, args.churn_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required input missing: {p}. Run the full pipeline first.")

    LOGGER.info("Loading CLV and churn scores...")
    clv_df = pd.read_parquet(args.clv_path)
    churn_df = pd.read_parquet(args.churn_path)

    merged = clv_df[["customer_id", "clv_horizon"]].merge(
        churn_df[["customer_id", "churn_probability"]], on="customer_id", how="inner"
    )
    merged = merged.dropna(subset=["clv_horizon", "churn_probability"])
    clv_arr = merged["clv_horizon"].values
    churn_arr = merged["churn_probability"].values
    LOGGER.info("Customers for simulation: %d", len(merged))

    # Budget range: from min cost to cover all eligible
    max_budget = len(merged) * args.base_unit_cost
    budget_levels = np.linspace(
        args.base_unit_cost * 10,
        min(max_budget * 0.7, 8000),
        args.n_budget_points,
    )

    LOGGER.info("Running Monte Carlo simulation: %d draws × %d budget levels...",
                args.n_simulations, args.n_budget_points)
    mc_df = monte_carlo_simulation(
        clv=clv_arr,
        churn_prob=churn_arr,
        budget_levels=budget_levels,
        n_simulations=args.n_simulations,
        effectiveness_range=(0.05, 0.25),
        cost_range=(1.0, 5.0),
        random_state=args.random_state,
    )

    # Baseline ROI curve (central assumptions) for comparison
    base_rows = []
    for budget in budget_levels:
        roi, net, n = _compute_roi_at_budget(
            clv_arr, churn_arr,
            args.base_effectiveness, args.base_unit_cost, budget,
        )
        base_rows.append({"budget": budget, "roi": roi, "net_gain": net, "n_targeted": n})
    base_df = pd.DataFrame(base_rows)

    # Save sensitivity table
    sensitivity_path = Path("reports/tables/sensitivity_summary.csv")
    sensitivity_path.parent.mkdir(parents=True, exist_ok=True)
    mc_df.to_csv(sensitivity_path, index=False)
    LOGGER.info("Sensitivity summary → '%s'", str(sensitivity_path))

    # Tornado analysis
    base_params = {"effectiveness": args.base_effectiveness, "unit_cost": args.base_unit_cost}
    tornado_df = tornado_sensitivity(
        clv=clv_arr, churn_prob=churn_arr,
        reference_budget=args.reference_budget,
        base_params=base_params,
        variation=0.5,
    )
    tornado_path = Path("reports/tables/sensitivity_tornado.csv")
    tornado_df.to_csv(tornado_path, index=False)

    LOGGER.info("Tornado analysis at £%.0f budget:\n%s",
                args.reference_budget,
                tornado_df[["parameter", "roi_at_low", "roi_at_base", "roi_at_high"]].to_string(index=False))

    # Plots
    plot_monte_carlo_roi(mc_df, base_df, Path("reports/figures/roi_monte_carlo.png"))
    plot_tornado(tornado_df, args.reference_budget, Path("reports/figures/sensitivity_tornado.png"))

    LOGGER.info("Sensitivity analysis complete.")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
