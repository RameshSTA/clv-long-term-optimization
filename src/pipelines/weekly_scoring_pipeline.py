"""
Weekly batch scoring pipeline (config-driven).

Runs the full 10-step workflow end-to-end using parameters defined in config files:

1)  Ingestion
2)  Cleaning
3)  Feature engineering
4)  CLV modeling (BG/NBD + Gamma-Gamma)
5)  Churn risk modeling (multi-model comparison + SHAP)
6)  Budget allocation (0/1 Knapsack)
7)  Evaluation reporting (decile lift + bootstrap CIs + ROI curve)
8)  Customer segmentation (RFM segments + CLV/churn overlay)
9)  Cohort analysis (monthly retention heatmap)
10) Business insights (Pareto + revenue concentration + monthly trend)
11) Sensitivity analysis (Monte Carlo ROI uncertainty)

Usage (from project root):
    python -m src.pipelines.weekly_scoring_pipeline --config-dir config

Optional overrides:
    python -m src.pipelines.weekly_scoring_pipeline --config-dir config --budget 8000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from src.utils.config_loader import load_configs


@dataclass(frozen=True)
class PipelineArgs:
    config_dir: str
    budget: float | None


def run(cmd: list[str]) -> None:
    """Execute a subprocess command and raise immediately on non-zero exit.

    Using ``check=False`` and inspecting ``returncode`` (rather than
    ``check=True``) lets us emit a cleaner error message that includes the
    full command string for easier debugging.

    Args:
        cmd: The command and its arguments as a list of strings, e.g.
            ``[sys.executable, "-m", "src.ingestion.load_data", "--input-path", "..."]``.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero return code.
    """
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def parse_args() -> PipelineArgs:
    """Parse pipeline-level CLI overrides.

    The pipeline is primarily config-driven via YAML files.  CLI arguments
    here serve only as lightweight overrides (e.g. one-off budget changes)
    without needing to edit configuration files.

    Returns:
        A frozen :class:`PipelineArgs` dataclass with ``config_dir`` and an
        optional ``budget`` override.
    """
    p = argparse.ArgumentParser(description="Run weekly CLV pipeline end-to-end (config-driven).")
    p.add_argument("--config-dir", type=str, default="config", help="Directory containing YAML config files.")
    p.add_argument("--budget", type=float, default=None, help="Override total_budget from business.yaml")
    args = p.parse_args()
    return PipelineArgs(config_dir=args.config_dir, budget=args.budget)


def main() -> None:
    """CLI entry point: orchestrate the full 11-step CLV pipeline.

    Reads all parameters from YAML config files (project, modeling, business)
    and launches each pipeline step as an isolated subprocess via
    :func:`run`.  This design means each step is independently
    restartable and produces its own log output.

    Steps executed in order:

    1.  Ingestion — raw Excel → ``transactions_raw.parquet``
    2.  Cleaning — 7-rule business logic → ``transactions_clean.parquet``
    3.  Feature engineering — cutoff-safe RFM + trend features
    4.  CLV modeling — BG/NBD + Gamma-Gamma + MLflow tracking
    5.  Churn risk — multi-model comparison (4 algorithms) + SHAP
    6.  Budget allocation — 0/1 Knapsack (PuLP/CBC)
    7.  Evaluation — decile lift + 95 % bootstrap CIs + ROI curve
    8.  Customer segmentation — RFM segments + CLV/churn overlay
    9.  Cohort analysis — monthly acquisition cohort retention matrix
    10. Business insights — Pareto analysis + Lorenz curve + monthly trend
    11. Sensitivity analysis — Monte Carlo ROI under assumption uncertainty

    Raises:
        RuntimeError: If any pipeline step exits with a non-zero return code.
            The failing step is identified in the error message so execution
            can be resumed from that point.
    """
    args = parse_args()
    cfgs = load_configs(args.config_dir)

    project = cfgs["project"]
    modeling = cfgs["modeling"]
    business = cfgs["business"]

    cutoff_date = project["dates"]["cutoff_date"]
    holdout_days = int(project["dates"]["holdout_days"])
    clv_horizon_days = int(project["dates"]["clv_horizon_days"])

    churn_inactivity_days = int(modeling["churn_model"]["churn_inactivity_days"])
    prediction_horizon_days = int(modeling["churn_model"]["prediction_horizon_days"])
    churn_model_type = str(modeling["churn_model"].get("model_type", "logistic"))

    total_budget = float(args.budget) if args.budget is not None else float(business["budget"]["total_budget"])
    unit_cost = float(business["budget"]["unit_cost_per_customer"])
    retention_effectiveness = float(business["retention"]["assumed_effectiveness"])
    min_clv = float(business["eligibility"]["min_clv"])
    solver = str(business["solver"].get("type", "pulp"))

    # Ingestion
    run([
        sys.executable, "-m", "src.ingestion.load_data",
        "--input-path", project["paths"]["input_excel"],
        "--output-path", project["paths"]["interim_raw"],
    ])

    # Cleaning
    run([
        sys.executable, "-m", "src.cleaning.clean_transactions",
        "--input-path", project["paths"]["interim_raw"],
        "--output-path", project["paths"]["interim_clean"],
    ])

    # Features
    run([
        sys.executable, "-m", "src.features.build_features",
        "--input-path", project["paths"]["interim_clean"],
        "--output-path", project["paths"]["features"],
        "--cutoff-date", cutoff_date,
    ])

    # CLV
    run([
        sys.executable, "-m", "src.modeling.train_clv_models",
        "--transactions-path", project["paths"]["interim_clean"],
        "--output-path", project["paths"]["clv_scores"],
        "--cutoff-date", cutoff_date,
        "--holdout-days", str(holdout_days),
        "--clv-horizon-days", str(clv_horizon_days),
        "--discount-rate-annual", str(modeling["clv_model"]["discount_rate_annual"]),
        "--penalizer-bgnbd", str(modeling["clv_model"]["penalizer_bgnbd"]),
        "--penalizer-gg", str(modeling["clv_model"]["penalizer_gg"]),
    ])

    # Churn
    run([
        sys.executable, "-m", "src.modeling.train_churn_risk",
        "--transactions-path", project["paths"]["interim_clean"],
        "--features-path", project["paths"]["features"],
        "--output-scores-path", project["paths"]["churn_scores"],
        "--output-model-path", project["paths"]["churn_model"],
        "--cutoff-date", cutoff_date,
        "--prediction-horizon-days", str(prediction_horizon_days),
        "--churn-inactivity-days", str(churn_inactivity_days),
        "--eval-gap-days", str(int(modeling["churn_model"].get("eval_gap_days", 60))),
        "--model-type", churn_model_type,
    ])

    # Budget allocation
    run([
        sys.executable, "-m", "src.optimization.budget_allocator",
        "--clv-path", project["paths"]["clv_scores"],
        "--risk-path", project["paths"]["churn_scores"],
        "--output-path", project["paths"]["targeting_list"],
        "--budget", str(total_budget),
        "--unit-cost", str(unit_cost),
        "--retention-effectiveness", str(retention_effectiveness),
        "--min-clv", str(min_clv),
        "--solver", solver,
    ])

    # Evaluation / reporting
    run([
        sys.executable, "-m", "src.evaluation.backtesting",
        "--cutoff-date", cutoff_date,
        "--holdout-days", str(holdout_days),
        "--retention-effectiveness", str(retention_effectiveness),
    ])

    # Customer segmentation (RFM + CLV + churn overlay)
    run([
        sys.executable, "-m", "src.analysis.customer_segmentation",
        "--features-path", project["paths"]["features"],
        "--clv-path", project["paths"]["clv_scores"],
        "--churn-path", project["paths"]["churn_scores"],
        "--output-path", project["paths"].get("customer_segments", "data/processed/customer_segments.parquet"),
    ])

    # Cohort analysis
    run([
        sys.executable, "-m", "src.analysis.cohort_analysis",
        "--transactions-path", project["paths"]["interim_clean"],
    ])

    # Business insights (Pareto, revenue concentration, monthly trend)
    run([
        sys.executable, "-m", "src.analysis.business_insights",
        "--transactions-path", project["paths"]["interim_clean"],
        "--features-path", project["paths"]["features"],
    ])

    # Sensitivity analysis (Monte Carlo ROI)
    run([
        sys.executable, "-m", "src.evaluation.sensitivity_analysis",
        "--clv-path", project["paths"]["clv_scores"],
        "--churn-path", project["paths"]["churn_scores"],
        "--base-effectiveness", str(retention_effectiveness),
        "--base-unit-cost", str(unit_cost),
    ])


if __name__ == "__main__":
    main()