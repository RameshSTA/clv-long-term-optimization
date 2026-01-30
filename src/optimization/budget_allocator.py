"""
 Budget optimization (targeting policy) for long-term value.

Business objective
------------------
Given a fixed retention/engagement budget, select a set of customers to target in order
to maximize expected long-term value.

Because this public dataset does not contain interventions (offers, calls) nor their
causal impact, this module implements a defensible policy prototype that is commonly
used before controlled experimentation:

- prioritize valuable customers (high CLV)
- prioritize at-risk customers (high churn probability)
- allocate spend under a budget constraint

Economic proxy
--------------
We approximate expected benefit of targeting customer i as:

    expected_benefit_i = CLV_i * churn_prob_i * retention_effectiveness

where retention_effectiveness is an explicit assumption (0..1) representing the average
fraction of churn-related value loss prevented by the intervention.

We define:

    net_gain_i = expected_benefit_i - cost_i

Optimization problem
--------------------
0/1 knapsack:

    maximize   sum_i x_i * net_gain_i
    subject to sum_i x_i * cost_i <= budget
               x_i ∈ {0, 1}

Solvers
-------
- pulp  : exact integer programming solution using PuLP (recommended).
- greedy: baseline heuristic using net_gain / cost ranking.

Outputs
-------
- Targeting list with recommended action ("target" / "no_action")
- Priority ranking and economics per customer
- Summary metrics logged (cost, expected benefit, net gain)
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd

try:
    import pulp
except Exception:  # pragma: no cover
    pulp = None


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BudgetConfig:
    """Configuration for budget allocation."""
    clv_path: Path
    risk_path: Path
    output_path: Path
    budget: float
    unit_cost: float
    retention_effectiveness: float
    min_clv: float
    solver: str
    log_level: str


REQUIRED_CLV_COLUMNS = {"customer_id", "clv_horizon"}
REQUIRED_RISK_COLUMNS = {"customer_id", "churn_probability", "risk_band"}


def setup_logging(level: str) -> None:
    """
    Configure console logging.

    Parameters
    ----------
    level:
        Logging level name.

    Raises
    ------
    ValueError
        If the log level is invalid.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory exists for a file path.

    Parameters
    ----------
    path:
        Target output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> BudgetConfig:
    """
    Parse CLI args into a BudgetConfig.

    Parameters
    ----------
    argv:
        Optional argument list. If None, argparse reads from sys.argv.

    Returns
    -------
    BudgetConfig
        Parsed configuration.
    """
    parser = argparse.ArgumentParser(description="Allocate retention budget based on CLV and churn risk.")
    parser.add_argument(
        "--clv-path",
        type=str,
        default="data/processed/customer_clv_scores.parquet",
        help="Path to customer CLV scores (Parquet).",
    )
    parser.add_argument(
        "--risk-path",
        type=str,
        default="data/processed/customer_churn_risk_scores.parquet",
        help="Path to customer churn risk scores (Parquet).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/targeting_list.parquet",
        help="Path to write the targeting list (Parquet).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        required=True,
        help="Total retention budget available.",
    )
    parser.add_argument(
        "--unit-cost",
        type=float,
        default=2.0,
        help="Per-customer intervention cost (constant).",
    )
    parser.add_argument(
        "--retention-effectiveness",
        type=float,
        default=0.10,
        help="Assumed average fraction of churn-related value loss prevented (0..1).",
    )
    parser.add_argument(
        "--min-clv",
        type=float,
        default=0.0,
        help="Minimum CLV eligibility threshold.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="pulp",
        choices=["pulp", "greedy"],
        help="Optimization solver: pulp (exact) or greedy (baseline).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args(args=argv)

    return BudgetConfig(
        clv_path=Path(args.clv_path),
        risk_path=Path(args.risk_path),
        output_path=Path(args.output_path),
        budget=float(args.budget),
        unit_cost=float(args.unit_cost),
        retention_effectiveness=float(args.retention_effectiveness),
        min_clv=float(args.min_clv),
        solver=args.solver,
        log_level=args.log_level,
    )


def validate_config(cfg: BudgetConfig) -> None:
    """
    Validate configuration values for correctness and safety.

    Parameters
    ----------
    cfg:
        Budget configuration.

    Raises
    ------
    ValueError
        If any configuration values are invalid.
    """
    if cfg.budget <= 0:
        raise ValueError("budget must be > 0.")
    if cfg.unit_cost <= 0:
        raise ValueError("unit_cost must be > 0.")
    if not (0.0 <= cfg.retention_effectiveness <= 1.0):
        raise ValueError("retention_effectiveness must be between 0 and 1.")
    if cfg.min_clv < 0:
        raise ValueError("min_clv must be >= 0.")


def _assert_columns_present(df: pd.DataFrame, required: set[str], name: str) -> None:
    """
    Validate required columns exist.

    Parameters
    ----------
    df:
        DataFrame to validate.
    required:
        Required column names.
    name:
        Friendly name used in error messages.
    """
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}. Found: {list(df.columns)}")


def load_inputs(clv_path: Path, risk_path: Path) -> pd.DataFrame:
    """
    Load and merge CLV and churn risk scores.

    Parameters
    ----------
    clv_path:
        Path to CLV scores table.
    risk_path:
        Path to churn risk scores table.

    Returns
    -------
    pd.DataFrame
        Merged table with: customer_id, clv_horizon, churn_probability, risk_band.
    """
    if not clv_path.exists():
        raise FileNotFoundError(f"CLV file not found: {clv_path}")
    if not risk_path.exists():
        raise FileNotFoundError(f"Risk file not found: {risk_path}")

    clv = pd.read_parquet(clv_path)
    risk = pd.read_parquet(risk_path)

    _assert_columns_present(clv, REQUIRED_CLV_COLUMNS, name="CLV input")
    _assert_columns_present(risk, REQUIRED_RISK_COLUMNS, name="Risk input")

    merged = clv[["customer_id", "clv_horizon"]].merge(
        risk[["customer_id", "churn_probability", "risk_band"]],
        on="customer_id",
        how="inner",
        validate="one_to_one",
    )

    merged["customer_id"] = merged["customer_id"].astype("int64")
    return merged


def build_economics(df: pd.DataFrame, cfg: BudgetConfig) -> pd.DataFrame:
    """
    Build economics required for optimization.

    Eligibility rules
    -----------------
    - clv_horizon >= min_clv
    - churn_probability > 0

    Benefit proxy
    -------------
    expected_benefit = clv_horizon * churn_probability * retention_effectiveness

    Net gain
    --------
    net_gain = expected_benefit - cost

    Parameters
    ----------
    df:
        Merged CLV + risk DataFrame.
    cfg:
        BudgetConfig.

    Returns
    -------
    pd.DataFrame
        DataFrame with cost, expected_benefit, net_gain, eligible flags.
    """
    out = df.copy()
    out["cost"] = float(cfg.unit_cost)

    out["expected_benefit"] = (
        out["clv_horizon"].clip(lower=0.0)
        * out["churn_probability"].clip(lower=0.0, upper=1.0)
        * float(cfg.retention_effectiveness)
    )
    out["net_gain"] = out["expected_benefit"] - out["cost"]

    out = out.loc[out["clv_horizon"] >= float(cfg.min_clv)].copy()
    out = out.loc[out["churn_probability"] > 0].copy()

    # Strict eligibility: only customers with positive net gain should be targeted
    # under a profit-maximizing assumption.
    out["eligible"] = out["net_gain"] > 0

    return out


def solve_knapsack_pulp(df: pd.DataFrame, budget: float) -> pd.Series:
    """
    Solve the 0/1 knapsack problem using PuLP.

    Parameters
    ----------
    df:
        DataFrame containing eligible candidates with cost and net_gain.
        Must include boolean column 'eligible'.
    budget:
        Total budget available.

    Returns
    -------
    pd.Series
        Boolean Series aligned to df.index indicating selected customers.

    Raises
    ------
    RuntimeError
        If PuLP is unavailable.
    """
    if pulp is None:
        raise RuntimeError(
            "PuLP is not installed/available. Install with: pip install pulp or use --solver greedy."
        )

    candidates = df.loc[df["eligible"]].copy()
    selected = pd.Series(False, index=df.index)

    if candidates.empty:
        return selected

    prob = pulp.LpProblem("RetentionBudgetAllocation", pulp.LpMaximize)

    x = {idx: pulp.LpVariable(f"x_{idx}", cat="Binary") for idx in candidates.index}

    prob += pulp.lpSum([x[idx] * float(candidates.loc[idx, "net_gain"]) for idx in candidates.index])
    prob += pulp.lpSum([x[idx] * float(candidates.loc[idx, "cost"]) for idx in candidates.index]) <= float(budget)

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        LOGGER.warning("PuLP did not find an optimal solution. Status: %s", pulp.LpStatus[status])

    for idx in candidates.index:
        selected.loc[idx] = bool(pulp.value(x[idx]) == 1)

    return selected


def solve_greedy(df: pd.DataFrame, budget: float) -> pd.Series:
    """
    Greedy allocation baseline using net_gain per unit cost.

    Parameters
    ----------
    df:
        DataFrame containing candidates and 'eligible' column.
    budget:
        Total budget available.

    Returns
    -------
    pd.Series
        Boolean Series aligned to df.index indicating selection.
    """
    selected = pd.Series(False, index=df.index)

    candidates = df.loc[df["eligible"]].copy()
    if candidates.empty:
        return selected

    candidates["gain_per_cost"] = candidates["net_gain"] / candidates["cost"]
    candidates = candidates.sort_values("gain_per_cost", ascending=False)

    spent = 0.0
    for idx, row in candidates.iterrows():
        cost = float(row["cost"])
        if spent + cost <= float(budget):
            selected.loc[idx] = True
            spent += cost

    return selected


def summarize_allocation(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute summary metrics for the chosen targeting policy.

    Parameters
    ----------
    df:
        DataFrame containing economics and selection columns.

    Returns
    -------
    Dict[str, float]
        Summary metrics (costs, expected benefits, and averages).
    """
    targeted = df.loc[df["target"]].copy()

    return {
        "customers_total": float(len(df)),
        "customers_eligible": float(df["eligible"].sum()),
        "customers_targeted": float(len(targeted)),
        "total_cost": float(targeted["cost"].sum()) if len(targeted) else 0.0,
        "total_expected_benefit": float(targeted["expected_benefit"].sum()) if len(targeted) else 0.0,
        "total_net_gain": float(targeted["net_gain"].sum()) if len(targeted) else 0.0,
        "avg_churn_probability_targeted": float(targeted["churn_probability"].mean()) if len(targeted) else 0.0,
        "avg_clv_targeted": float(targeted["clv_horizon"].mean()) if len(targeted) else 0.0,
    }


def run(cfg: BudgetConfig) -> None:
    """
    Execute the budget allocation policy.

    Parameters
    ----------
    cfg:
        BudgetConfig for the current run.
    """
    setup_logging(cfg.log_level)
    validate_config(cfg)

    merged = load_inputs(cfg.clv_path, cfg.risk_path)
    LOGGER.info("Merged CLV + risk dataset shape=%s", merged.shape)

    economics = build_economics(merged, cfg)
    LOGGER.info(
        "Eligibility summary: eligible=%d / total=%d",
        int(economics["eligible"].sum()),
        int(len(economics)),
    )

    if cfg.solver == "pulp":
        selection = solve_knapsack_pulp(economics, cfg.budget)
    else:
        selection = solve_greedy(economics, cfg.budget)

    economics["target"] = selection.values

    # Priority ranking among targeted customers by net_gain descending
    economics["priority_rank"] = 0
    economics.loc[economics["target"], "priority_rank"] = (
        economics.loc[economics["target"], "net_gain"].rank(method="first", ascending=False).astype(int)
    )

    economics["recommended_action"] = np.where(economics["target"], "target", "no_action")

    summary = summarize_allocation(economics)
    LOGGER.info("Allocation summary: %s", summary)

    # Persist targeting list
    out_cols = [
        "customer_id",
        "recommended_action",
        "priority_rank",
        "risk_band",
        "churn_probability",
        "clv_horizon",
        "cost",
        "expected_benefit",
        "net_gain",
    ]

    targeting = economics.loc[:, out_cols].sort_values(
        ["recommended_action", "priority_rank"],
        ascending=[True, True],
    )

    ensure_parent_dir(cfg.output_path)
    targeting.to_parquet(cfg.output_path, index=False)
    LOGGER.info("Targeting list written to '%s' (rows=%d)", str(cfg.output_path), len(targeting))

    LOGGER.info("Step 6 completed successfully")


def main() -> None:
    """CLI entry point."""
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()