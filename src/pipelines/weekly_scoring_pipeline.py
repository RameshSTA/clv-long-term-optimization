"""
Weekly batch scoring pipeline (config-driven).

Runs the full workflow end-to-end using parameters defined in config files:

1) Ingestion
2) Cleaning
3) Feature engineering
4) CLV modeling
5) Churn risk modeling
6) Budget allocation
7) Evaluation reporting

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
    """
    Run a command and fail fast with a helpful error message.
    """
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def parse_args() -> PipelineArgs:
    p = argparse.ArgumentParser(description="Run weekly CLV pipeline end-to-end (config-driven).")
    p.add_argument("--config-dir", type=str, default="config", help="Directory containing YAML config files.")
    p.add_argument("--budget", type=float, default=None, help="Override total_budget from business.yaml")
    args = p.parse_args()
    return PipelineArgs(config_dir=args.config_dir, budget=args.budget)


def main() -> None:
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
        "--discount-rate", str(modeling["clv_model"]["discount_rate_annual"]),
        "--penalizer-bgnbd", str(modeling["clv_model"]["penalizer_bgnbd"]),
        "--penalizer-gg", str(modeling["clv_model"]["penalizer_gg"]),
    ])

    # Churn
    run([
        sys.executable, "-m", "src.modeling.train_churn_risk",
        "--transactions-path", project["paths"]["interim_clean"],
        "--features-path", project["paths"]["features"],
        "--output-scores-path", project["paths"]["churn_scores"],
        "--cutoff-date", cutoff_date,
        "--prediction-horizon-days", str(prediction_horizon_days),
        "--churn-inactivity-days", str(churn_inactivity_days),
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

    # Evaluation / reporting (if your module exists)
    run([
        sys.executable, "-m", "src.evaluation.backtesting",
        "--cutoff-date", cutoff_date,
        "--holdout-days", str(holdout_days),
        "--retention-effectiveness", str(retention_effectiveness),
    ])


if __name__ == "__main__":
    main()