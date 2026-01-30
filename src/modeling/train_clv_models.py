"""
 CLV modeling (BG/NBD + Gamma-Gamma) with time-based evaluation.

Purpose
-------
This module trains probabilistic customer lifetime value (CLV) models using the
`lifetimes` library:

- BG/NBD (BetaGeoFitter): models repeat purchase behavior and customer "alive" probability.
- Gamma-Gamma (GammaGammaFitter): models monetary value per transaction.

We implement an out-of-time evaluation using a calibration/holdout split:
- calibration window: transactions strictly before cutoff_date
- holdout window:     transactions in (cutoff_date, holdout_end]

This prevents information leakage and reflects how CLV models are validated in practice.

Outputs
-------
- Customer-level CLV score table (Parquet):
  customer_id, p_alive, expected purchases, expected value, discounted CLV, and metadata.
- A lightweight decile evaluation is logged for interpretability.
  (If you want, we can also persist evaluation tables to reports/ as a next step.)

"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CLVConfig:
    """Configuration for CLV model training and scoring."""
    transactions_path: Path
    output_path: Path
    cutoff_date: pd.Timestamp
    holdout_days: int
    clv_horizon_days: int
    discount_rate_annual: float
    penalizer_coef_bgnbd: float
    penalizer_coef_gg: float
    log_level: str


REQUIRED_TXN_COLUMNS = {"customer_id", "invoice", "invoice_dt", "revenue"}


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


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure the parent directory exists for a file path.

    Parameters
    ----------
    path:
        Target file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> CLVConfig:
    """
    Parse CLI args into a CLVConfig.

    Parameters
    ----------
    argv:
        Optional argument list. If None, argparse reads from sys.argv.

    Returns
    -------
    CLVConfig
        Parsed configuration.
    """
    parser = argparse.ArgumentParser(description="Train CLV models and score customers.")
    parser.add_argument(
        "--transactions-path",
        type=str,
        default="data/interim/transactions_clean.parquet",
        help="Path to cleaned transactions (Parquet).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/customer_clv_scores.parquet",
        help="Where to write CLV scores (Parquet).",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2011-06-01",
        help="Calibration/holdout split date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=180,
        help="Holdout window size in days for backtesting.",
    )
    parser.add_argument(
        "--clv-horizon-days",
        type=int,
        default=180,
        help="Forecast horizon in days for CLV scoring (e.g., 180 for 6 months).",
    )
    parser.add_argument(
        "--discount-rate-annual",
        type=float,
        default=0.10,
        help="Annual discount rate (e.g., 0.10 for 10% annual).",
    )
    parser.add_argument(
        "--penalizer-bgnbd",
        type=float,
        default=0.001,
        help="Penalizer coefficient for BG/NBD fitter.",
    )
    parser.add_argument(
        "--penalizer-gg",
        type=float,
        default=0.001,
        help="Penalizer coefficient for Gamma-Gamma fitter.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args(args=argv)

    return CLVConfig(
        transactions_path=Path(args.transactions_path),
        output_path=Path(args.output_path),
        cutoff_date=pd.to_datetime(args.cutoff_date),
        holdout_days=args.holdout_days,
        clv_horizon_days=args.clv_horizon_days,
        discount_rate_annual=args.discount_rate_annual,
        penalizer_coef_bgnbd=args.penalizer_bgnbd,
        penalizer_coef_gg=args.penalizer_gg,
        log_level=args.log_level,
    )


def _assert_columns_present(df: pd.DataFrame, required: set[str], name: str) -> None:
    """
    Validate required columns exist.

    Parameters
    ----------
    df:
        DataFrame to validate.
    required:
        Set of required column names.
    name:
        Friendly name for error context.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}. Found: {list(df.columns)}")


def aggregate_to_invoice_level(txn: pd.DataFrame) -> pd.DataFrame:
    """
    Convert line-item transactions to invoice-level purchase events.

    Rationale
    ---------
    Lifetimes models assume each row corresponds to a single transaction event.
    In Online Retail II, invoices can contain multiple line items; therefore we
    aggregate to one event per invoice with summed revenue.

    Parameters
    ----------
    txn:
        Cleaned line-item transactions.

    Returns
    -------
    pd.DataFrame
        Invoice-level events (customer_id, invoice, invoice_dt, revenue).
    """
    _assert_columns_present(txn, REQUIRED_TXN_COLUMNS, name="transactions")

    inv = (
        txn.groupby(["customer_id", "invoice"], as_index=False)
        .agg(
            invoice_dt=("invoice_dt", "min"),
            revenue=("revenue", "sum"),
        )
        .sort_values(["customer_id", "invoice_dt", "invoice"], kind="mergesort")
        .reset_index(drop=True)
    )
    return inv


def infer_holdout_end(inv: pd.DataFrame, cutoff: pd.Timestamp, holdout_days: int) -> pd.Timestamp:
    """
    Determine holdout end date given a cutoff and desired holdout window length.

    If the dataset does not extend to cutoff + holdout_days, we shorten the holdout window
    to the maximum available date to avoid empty evaluation.

    Parameters
    ----------
    inv:
        Invoice-level events.
    cutoff:
        Calibration period end (split point).
    holdout_days:
        Desired holdout window size in days.

    Returns
    -------
    pd.Timestamp
        Observation period end for the calibration/holdout split.
    """
    max_dt = pd.to_datetime(inv["invoice_dt"]).max()
    desired_end = cutoff + pd.Timedelta(days=holdout_days)
    end = min(desired_end, max_dt)

    if end <= cutoff:
        raise ValueError(
            "Holdout end is not after cutoff. Choose an earlier cutoff date or check invoice_dt parsing."
        )
    return end


def build_calibration_holdout(inv: pd.DataFrame, cutoff: pd.Timestamp, holdout_end: pd.Timestamp) -> pd.DataFrame:
    """
    Build customer summary table required for BG/NBD training and holdout evaluation.

    Parameters
    ----------
    inv:
        Invoice-level events.
    cutoff:
        End of calibration period.
    holdout_end:
        End of observation period (calibration + holdout).

    Returns
    -------
    pd.DataFrame
        Customer-level summary with lifetimes conventions:
        frequency_cal, recency_cal, T_cal, frequency_holdout, etc.
    """
    summary = calibration_and_holdout_data(
        transactions=inv,
        customer_id_col="customer_id",
        datetime_col="invoice_dt",
        calibration_period_end=cutoff,
        observation_period_end=holdout_end,
        freq="D",
    ).reset_index()

    expected = {"customer_id", "frequency_cal", "recency_cal", "T_cal", "frequency_holdout"}
    missing = expected - set(summary.columns)
    if missing:
        raise ValueError(f"Unexpected lifetimes output; missing: {sorted(missing)}")

    return summary


def compute_holdout_actuals(inv: pd.DataFrame, cutoff: pd.Timestamp, holdout_end: pd.Timestamp) -> pd.DataFrame:
    """
    Compute actual holdout outcomes for evaluation.

    Parameters
    ----------
    inv:
        Invoice-level events.
    cutoff:
        Holdout window start.
    holdout_end:
        Holdout window end.

    Returns
    -------
    pd.DataFrame
        Actuals per customer in holdout:
        holdout_transactions, holdout_revenue.
    """
    mask = (inv["invoice_dt"] > cutoff) & (inv["invoice_dt"] <= holdout_end)
    holdout = inv.loc[mask].copy()

    actuals = (
        holdout.groupby("customer_id", as_index=False)
        .agg(
            holdout_transactions=("invoice", "nunique"),
            holdout_revenue=("revenue", "sum"),
        )
    )
    return actuals


def eval_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute simple regression-style metrics for holdout prediction.

    Parameters
    ----------
    y_true:
        Actual values.
    y_pred:
        Predicted values.

    Returns
    -------
    Dict[str, float]
        MAE, RMSE, and MAPE (MAPE computed only where y_true != 0).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))) if np.any(nonzero) else float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape}


def annual_to_daily_discount_rate(r_annual: float) -> float:
    """
    Convert an annual discount rate to an approximate daily rate.

    We assume a 365-day year:
      r_daily = (1 + r_annual)^(1/365) - 1

    Parameters
    ----------
    r_annual:
        Annual discount rate.

    Returns
    -------
    float
        Daily discount rate.
    """
    if r_annual < 0:
        raise ValueError("discount_rate_annual must be non-negative.")
    return (1.0 + r_annual) ** (1.0 / 365.0) - 1.0


def fit_bgnbd(summary: pd.DataFrame, penalizer: float) -> BetaGeoFitter:
    """
    Fit a BG/NBD model on calibration summary statistics.

    Parameters
    ----------
    summary:
        Calibration/holdout summary table with frequency_cal, recency_cal, T_cal.
    penalizer:
        Penalizer coefficient for regularization.

    Returns
    -------
    BetaGeoFitter
        Fitted BG/NBD model.
    """
    model = BetaGeoFitter(penalizer_coef=penalizer)
    model.fit(
        frequency=summary["frequency_cal"],
        recency=summary["recency_cal"],
        T=summary["T_cal"],
    )
    return model


def fit_gamma_gamma(inv: pd.DataFrame, cutoff: pd.Timestamp, penalizer: float) -> Tuple[GammaGammaFitter, float]:
    """
    Fit a Gamma-Gamma model for monetary value.

    Lifetimes recommends fitting Gamma-Gamma on customers with repeat purchases.

    Parameters
    ----------
    inv:
        Invoice-level events.
    cutoff:
        End of calibration window.
    penalizer:
        Penalizer coefficient.

    Returns
    -------
    Tuple[GammaGammaFitter, float]
        Fitted Gamma-Gamma model and fallback global mean invoice revenue.
    """
    inv_cal = inv.loc[inv["invoice_dt"] < cutoff].copy()

    monetary = (
        inv_cal.groupby("customer_id", as_index=False)
        .agg(
            frequency=("invoice", "nunique"),
            monetary_value=("revenue", "mean"),
        )
    )

    monetary = monetary.loc[monetary["frequency"] > 1].copy()
    if monetary.empty:
        raise ValueError(
            "No customers with >1 transaction in calibration window. "
            "Gamma-Gamma cannot be fit. Choose an earlier cutoff date."
        )

    gg = GammaGammaFitter(penalizer_coef=penalizer)
    gg.fit(monetary["frequency"], monetary["monetary_value"])

    global_mean_value = float(inv_cal["revenue"].mean())
    return gg, global_mean_value


def score_customers(
    inv: pd.DataFrame,
    summary: pd.DataFrame,
    bgnbd: BetaGeoFitter,
    gg: GammaGammaFitter,
    global_mean_value: float,
    cfg: CLVConfig,
    holdout_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Produce customer-level CLV scores at the cutoff date.

    The score table includes:
    - p_alive at cutoff
    - expected number of purchases over clv_horizon_days
    - expected average invoice value (Gamma-Gamma)
    - discounted CLV approximation over horizon
    - metadata for auditability

    Parameters
    ----------
    inv:
        Invoice-level events.
    summary:
        Calibration/holdout customer summary.
    bgnbd:
        Fitted BG/NBD model.
    gg:
        Fitted Gamma-Gamma model.
    global_mean_value:
        Fallback mean invoice value for customers lacking monetary history.
    cfg:
        CLVConfig.
    holdout_end:
        Holdout end date used for evaluation split.

    Returns
    -------
    pd.DataFrame
        Customer-level CLV score table.
    """
    horizon_days = cfg.clv_horizon_days
    daily_discount = annual_to_daily_discount_rate(cfg.discount_rate_annual)

    scores = summary[["customer_id", "frequency_cal", "recency_cal", "T_cal", "frequency_holdout"]].copy()

    scores["p_alive"] = bgnbd.conditional_probability_alive(
        frequency=scores["frequency_cal"],
        recency=scores["recency_cal"],
        T=scores["T_cal"],
    )

    scores["exp_purchases_horizon"] = bgnbd.conditional_expected_number_of_purchases_up_to_time(
        t=horizon_days,
        frequency=scores["frequency_cal"],
        recency=scores["recency_cal"],
        T=scores["T_cal"],
    )

    # Average invoice value in calibration (used as input to Gamma-Gamma expectation).
    inv_cal = inv.loc[inv["invoice_dt"] < cfg.cutoff_date].copy()
    avg_value_cal = inv_cal.groupby("customer_id")["revenue"].mean().rename("avg_value_cal")

    scores = scores.merge(avg_value_cal.reset_index(), on="customer_id", how="left")
    scores["avg_value_cal"] = scores["avg_value_cal"].fillna(global_mean_value).clip(lower=0.0)

    # Predict expected average profit per transaction.
    # Gamma-Gamma requires a frequency and observed monetary value proxy.
    scores["exp_avg_value"] = gg.conditional_expected_average_profit(
        scores["frequency_cal"].clip(lower=0),
        scores["avg_value_cal"],
    )

    # Discounted CLV approximation over the horizon.
    # For short horizons this is adequate and transparent:
    # CLV ≈ E[N(horizon)] * E[value] discounted by the horizon length.
    discount_factor = 1.0 / ((1.0 + daily_discount) ** horizon_days)
    scores["clv_horizon"] = scores["exp_purchases_horizon"] * scores["exp_avg_value"] * discount_factor

    # Metadata
    scores["cutoff_date"] = cfg.cutoff_date
    scores["holdout_end_date"] = holdout_end
    scores["holdout_days"] = int((holdout_end - cfg.cutoff_date).days)
    scores["clv_horizon_days"] = horizon_days
    scores["discount_rate_annual"] = cfg.discount_rate_annual
    scores["penalizer_coef_bgnbd"] = cfg.penalizer_coef_bgnbd
    scores["penalizer_coef_gg"] = cfg.penalizer_coef_gg

    return scores


def build_decile_evaluation(eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a decile-level evaluation table for stakeholder interpretability.

    Parameters
    ----------
    eval_df:
        Customer-level table with clv_horizon, holdout_revenue, holdout_transactions.

    Returns
    -------
    pd.DataFrame
        Decile table with mean predicted CLV and realized holdout outcomes.
    """
    # Highest predicted CLV should correspond to higher realized holdout revenue on average.
    tmp = eval_df.sort_values("clv_horizon", ascending=False).copy()

    # qcut on rank is robust to many ties.
    tmp["decile"] = pd.qcut(tmp["clv_horizon"].rank(method="first"), 10, labels=False) + 1

    return (
        tmp.groupby("decile", as_index=False)
        .agg(
            customers=("customer_id", "nunique"),
            avg_pred_clv=("clv_horizon", "mean"),
            avg_holdout_revenue=("holdout_revenue", "mean"),
            avg_holdout_txn=("holdout_transactions", "mean"),
        )
        .sort_values("decile")
    )


def run(cfg: CLVConfig) -> None:
    """
    Execute CLV model training, time-based evaluation, and scoring.

    Parameters
    ----------
    cfg:
        CLVConfig for the current run.
    """
    setup_logging(cfg.log_level)

    if not cfg.transactions_path.exists():
        raise FileNotFoundError(
            f"Missing cleaned transactions: {cfg.transactions_path}. Run Step 2 cleaning first."
        )

    LOGGER.info("Reading cleaned transactions from '%s'", str(cfg.transactions_path))
    txn = pd.read_parquet(cfg.transactions_path)
    LOGGER.info("Transactions shape=%s", txn.shape)

    inv = aggregate_to_invoice_level(txn)
    LOGGER.info("Invoice-level transactions shape=%s", inv.shape)

    holdout_end = infer_holdout_end(inv, cfg.cutoff_date, cfg.holdout_days)
    LOGGER.info("Using cutoff_date=%s and holdout_end=%s", cfg.cutoff_date.date(), holdout_end.date())

    summary = build_calibration_holdout(inv, cfg.cutoff_date, holdout_end)
    LOGGER.info("Calibration/holdout summary shape=%s", summary.shape)

    # Actual outcomes in holdout for evaluation
    actuals = compute_holdout_actuals(inv, cfg.cutoff_date, holdout_end)
    eval_df = summary.merge(actuals, on="customer_id", how="left")
    eval_df["holdout_transactions"] = eval_df["holdout_transactions"].fillna(0).astype(int)
    eval_df["holdout_revenue"] = eval_df["holdout_revenue"].fillna(0.0)

    # Fit models
    bgnbd = fit_bgnbd(summary, penalizer=cfg.penalizer_coef_bgnbd)
    gg, global_mean_value = fit_gamma_gamma(inv, cutoff=cfg.cutoff_date, penalizer=cfg.penalizer_coef_gg)

    # Holdout transaction count prediction for evaluation
    holdout_length_days = int((holdout_end - cfg.cutoff_date).days)
    pred_holdout_txn = bgnbd.conditional_expected_number_of_purchases_up_to_time(
        t=holdout_length_days,
        frequency=summary["frequency_cal"],
        recency=summary["recency_cal"],
        T=summary["T_cal"],
    )
    txn_metrics = eval_predictions(
        y_true=eval_df["holdout_transactions"].values,
        y_pred=np.asarray(pred_holdout_txn),
    )
    LOGGER.info("Holdout transaction prediction metrics: %s", txn_metrics)

    # Score customers at cutoff
    scores = score_customers(
        inv=inv,
        summary=summary,
        bgnbd=bgnbd,
        gg=gg,
        global_mean_value=global_mean_value,
        cfg=cfg,
        holdout_end=holdout_end,
    )

    # Decile evaluation for interpretability
    eval_for_deciles = eval_df.merge(scores[["customer_id", "clv_horizon"]], on="customer_id", how="left")
    deciles = build_decile_evaluation(eval_for_deciles)
    LOGGER.info("Decile evaluation (1=lowest,10=highest):\n%s", deciles.to_string(index=False))

    # Persist scores
    ensure_parent_dir(cfg.output_path)
    LOGGER.info("Writing CLV scores to '%s' (rows=%d)", str(cfg.output_path), len(scores))
    scores.to_parquet(cfg.output_path, index=False)

    LOGGER.info("Step 4 completed successfully")


def main() -> None:
    """CLI entry point."""
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()