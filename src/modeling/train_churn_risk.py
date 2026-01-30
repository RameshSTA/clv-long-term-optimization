# src/modeling/train_churn_risk.py
"""
Churn risk modeling (inactivity-based) with time-safe labeling and evaluation.

Context
-------
In this public transactional dataset we do not observe explicit churn events (e.g., an
account cancellation). A common operational proxy is inactivity-based churn:

A customer is considered "churned" if they do not make a purchase within a fixed number
of days after a decision cutoff date.

This script:
1) Reads cleaned transactions
2) Reads customer features computed at a cutoff date
3) Creates churn labels using ONLY post-cutoff activity (time-safe; no leakage).
4) Trains a churn risk model (baseline logistic regression; optional CLI arg preserved).
5) Evaluates on an out-of-time holdout set within a bounded horizon.
6) Writes churn probabilities + risk bands to Parquet and saves the model artifact.

Evaluation strategy (time-safe)
-------------------------------
We use an out-of-time split based on cutoff dates:

- Train cutoff: C
- Test cutoff:  C + eval_gap_days

For each cutoff snapshot, we label churn using only activity in (cutoff, cutoff + horizon].
This prevents leakage and mimics production: train using an earlier cutoff and evaluate
using a later cutoff.

Important note about features
-----------------------------
A production system would rebuild features for the test cutoff snapshot. In this project,
features are built at a single cutoff (Step 3). For evaluation, we keep the same feature
snapshot but shift the labeling cutoff forward (time-safe). This is conservative and still
useful for model sanity checking and monitoring.

CLI compatibility note
----------------------
The weekly pipeline passes --model-type logistic. This script accepts --model-type for
backwards compatibility even though only "logistic" is implemented.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChurnConfig:
    """Configuration for churn model training and scoring."""
    transactions_path: Path
    features_path: Path
    output_scores_path: Path
    output_model_path: Path
    cutoff_date: pd.Timestamp
    prediction_horizon_days: int
    churn_inactivity_days: int
    eval_gap_days: int
    model_type: str
    log_level: str


FEATURE_COLUMNS: Tuple[str, ...] = (
    "recency_days",
    "tenure_days",
    "n_invoices",
    "total_revenue",
    "avg_order_value",
    "revenue_last_30d",
    "revenue_last_90d",
    "rev_30_to_90_ratio",
)

REQUIRED_TXN_COLUMNS = {"customer_id", "invoice", "invoice_dt", "revenue"}
REQUIRED_FEATURE_COLUMNS = {"customer_id", *FEATURE_COLUMNS}


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


def parse_args(argv: Sequence[str] | None = None) -> ChurnConfig:
    """
    Parse CLI args into a ChurnConfig.

    Parameters
    ----------
    argv:
        Optional argument list. If None, argparse reads from sys.argv.

    Returns
    -------
    ChurnConfig
        Parsed configuration.
    """
    parser = argparse.ArgumentParser(description="Train churn risk model and score customers.")
    parser.add_argument(
        "--transactions-path",
        type=str,
        default="data/interim/transactions_clean.parquet",
        help="Path to cleaned transactions (Parquet).",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="data/processed/customer_features.parquet",
        help="Path to customer features table (Parquet) built at cutoff date.",
    )
    parser.add_argument(
        "--output-scores-path",
        type=str,
        default="data/processed/customer_churn_risk_scores.parquet",
        help="Path to write churn risk scores (Parquet).",
    )
    parser.add_argument(
        "--output-model-path",
        type=str,
        default="data/processed/churn_risk_model.joblib",
        help="Path to write trained model artifact.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default="2011-06-01",
        help="Cutoff date for feature construction (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--prediction-horizon-days",
        type=int,
        default=180,
        help="How far ahead to observe for labeling and evaluation.",
    )
    parser.add_argument(
        "--churn-inactivity-days",
        type=int,
        default=90,
        help="Inactivity threshold (days) defining churn.",
    )
    parser.add_argument(
        "--eval-gap-days",
        type=int,
        default=60,
        help=(
            "Gap between train cutoff and test cutoff (days). "
            "Used for out-of-time evaluation."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="logistic",
        choices=["logistic"],
        help="Model type to train. (Currently only logistic is supported.)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args(args=argv)

    return ChurnConfig(
        transactions_path=Path(args.transactions_path),
        features_path=Path(args.features_path),
        output_scores_path=Path(args.output_scores_path),
        output_model_path=Path(args.output_model_path),
        cutoff_date=pd.to_datetime(args.cutoff_date),
        prediction_horizon_days=int(args.prediction_horizon_days),
        churn_inactivity_days=int(args.churn_inactivity_days),
        eval_gap_days=int(args.eval_gap_days),
        model_type=str(args.model_type),
        log_level=str(args.log_level),
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
    Aggregate line-item transactions into invoice-level purchase events.

    Rationale
    ---------
    Online Retail II contains line items. For churn labeling we want "purchase events"
    that represent a customer's shopping occasion. Invoices are a stable proxy for that:
    one purchase event per invoice with revenue summed across items.

    Parameters
    ----------
    txn:
        Cleaned transaction line items.

    Returns
    -------
    pd.DataFrame
        Invoice-level events with columns: customer_id, invoice, invoice_dt, revenue.
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


def compute_churn_labels(
    inv: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_days: int,
    inactivity_days: int,
    customers: pd.Series,
) -> pd.DataFrame:
    """
    Compute inactivity-based churn labels in a time-safe manner.

    Definition
    ----------
    For each customer in the population at cutoff:
    - Look for their first purchase after cutoff within (cutoff, cutoff + horizon_days].
    - If no purchase occurs within inactivity_days, label churn = 1, else churn = 0.
    - If no purchase is observed within the horizon window, churn = 1.

    Parameters
    ----------
    inv:
        Invoice-level events.
    cutoff:
        Cutoff timestamp.
    horizon_days:
        Maximum look-ahead window for observing activity after cutoff.
    inactivity_days:
        Inactivity threshold that defines churn.
    customers:
        Series of customer_ids present in the feature snapshot (population).

    Returns
    -------
    pd.DataFrame
        Labels table with one row per customer_id in `customers` and columns:
        customer_id, first_purchase_post_cutoff, days_to_next_purchase, churn_label.
    """
    if horizon_days <= 0:
        raise ValueError("prediction_horizon_days must be positive.")
    if inactivity_days <= 0:
        raise ValueError("churn_inactivity_days must be positive.")
    if inactivity_days > horizon_days:
        raise ValueError("churn_inactivity_days must be <= prediction_horizon_days.")

    obs_end = cutoff + pd.Timedelta(days=horizon_days)

    post = inv.loc[(inv["invoice_dt"] > cutoff) & (inv["invoice_dt"] <= obs_end)].copy()
    first_post = post.groupby("customer_id")["invoice_dt"].min().rename("first_purchase_post_cutoff")

    # Use Int64 to preserve nullability safely; cast to int64 where needed after validation.
    labels = pd.DataFrame({"customer_id": pd.Series(customers).astype("Int64")})
    labels = labels.merge(first_post.reset_index(), on="customer_id", how="left")

    labels["days_to_next_purchase"] = (labels["first_purchase_post_cutoff"] - cutoff).dt.days

    # If no post-cutoff purchase observed within horizon, days_to_next_purchase is NaN -> churn=1.
    labels["churn_label"] = (
        labels["days_to_next_purchase"].isna()
        | (labels["days_to_next_purchase"] > inactivity_days)
    ).astype(int)

    labels["days_to_next_purchase"] = labels["days_to_next_purchase"].fillna(np.inf)

    # Return stable schema
    out = labels[["customer_id", "first_purchase_post_cutoff", "days_to_next_purchase", "churn_label"]].copy()
    out["customer_id"] = out["customer_id"].astype("int64")
    return out


def build_snapshot(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon_days: int,
    inactivity_days: int,
) -> pd.DataFrame:
    """
    Join a feature snapshot with churn labels and attach metadata.

    Parameters
    ----------
    features:
        Customer feature table built at the snapshot cutoff date.
    labels:
        Time-safe churn labels for the same customer population.
    cutoff:
        Snapshot cutoff.
    horizon_days:
        Labeling horizon in days.
    inactivity_days:
        Inactivity threshold in days.

    Returns
    -------
    pd.DataFrame
        Training/evaluation frame including features and churn_label.
    """
    df = features.merge(labels, on="customer_id", how="inner").copy()
    df["cutoff_date"] = cutoff
    df["prediction_horizon_days"] = horizon_days
    df["churn_inactivity_days"] = inactivity_days
    return df


def make_model(feature_cols: Tuple[str, ...]) -> Pipeline:
    """
    Build the churn model pipeline.

    We use a transparent and production-friendly baseline:
    - median imputation (robust to outliers)
    - standard scaling
    - logistic regression with class_weight balanced

    Parameters
    ----------
    feature_cols:
        Names of numeric feature columns.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Preprocessing + classifier pipeline.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, list(feature_cols))],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute ranking-based metrics appropriate for churn targeting.

    Parameters
    ----------
    y_true:
        Ground truth labels.
    y_prob:
        Predicted churn probabilities.

    Returns
    -------
    Dict[str, float]
        Dictionary containing ROC AUC, Average Precision, and base rate.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) <= 1:
        return {
            "roc_auc": float("nan"),
            "avg_precision": float("nan"),
            "churn_base_rate": float(np.mean(y_true)),
        }

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "churn_base_rate": float(np.mean(y_true)),
    }


def assign_risk_band(prob: pd.Series) -> pd.Series:
    """
    Convert churn probability into coarse risk bands for stakeholder usability.

    Parameters
    ----------
    prob:
        Series of churn probabilities in [0, 1].

    Returns
    -------
    pd.Series
        Categorical risk band labels.
    """
    bins = [0.0, 0.33, 0.66, 1.0]
    labels = ["low", "medium", "high"]
    return pd.cut(prob.clip(0, 1), bins=bins, labels=labels, include_lowest=True)


def train_and_evaluate(
    inv: pd.DataFrame,
    features: pd.DataFrame,
    cfg: ChurnConfig,
) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame]:
    """
    Train churn model on a train cutoff snapshot and evaluate on a later cutoff snapshot.

    Returns
    -------
    model:
        Trained sklearn pipeline.
    metrics:
        Evaluation metrics computed on the out-of-time window.
    scored:
        Customer-level scores table at the training cutoff.
    """
    _assert_columns_present(features, REQUIRED_FEATURE_COLUMNS, name="features")

    if cfg.model_type != "logistic":
        raise ValueError(f"Unsupported model_type: {cfg.model_type}")

    customers = features["customer_id"]

    # Training labels at the configured cutoff.
    train_labels = compute_churn_labels(
        inv=inv,
        cutoff=cfg.cutoff_date,
        horizon_days=cfg.prediction_horizon_days,
        inactivity_days=cfg.churn_inactivity_days,
        customers=customers,
    )
    train_frame = build_snapshot(
        features=features,
        labels=train_labels,
        cutoff=cfg.cutoff_date,
        horizon_days=cfg.prediction_horizon_days,
        inactivity_days=cfg.churn_inactivity_days,
    )

    X_train = train_frame.loc[:, list(FEATURE_COLUMNS)]
    y_train = train_frame["churn_label"].astype(int)

    model = make_model(FEATURE_COLUMNS)
    model.fit(X_train, y_train)

    # Out-of-time evaluation using a later cutoff for labeling (time-safe).
    test_cutoff = cfg.cutoff_date + pd.Timedelta(days=cfg.eval_gap_days)
    test_labels = compute_churn_labels(
        inv=inv,
        cutoff=test_cutoff,
        horizon_days=cfg.prediction_horizon_days,
        inactivity_days=cfg.churn_inactivity_days,
        customers=customers,
    )
    test_frame = build_snapshot(
        features=features,
        labels=test_labels,
        cutoff=test_cutoff,
        horizon_days=cfg.prediction_horizon_days,
        inactivity_days=cfg.churn_inactivity_days,
    )

    X_test = test_frame.loc[:, list(FEATURE_COLUMNS)]
    y_test = test_frame["churn_label"].astype(int)

    y_test_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_probabilities(y_test.values, y_test_prob)

    # Scores output at training cutoff (what would be used operationally).
    y_train_prob = model.predict_proba(X_train)[:, 1]
    scored = pd.DataFrame(
        {
            "customer_id": train_frame["customer_id"].astype("int64"),
            "cutoff_date": cfg.cutoff_date,
            "prediction_horizon_days": cfg.prediction_horizon_days,
            "churn_inactivity_days": cfg.churn_inactivity_days,
            "churn_probability": y_train_prob,
        }
    )
    scored["risk_band"] = assign_risk_band(scored["churn_probability"]).astype("string")

    return model, metrics, scored


def write_outputs(scores: pd.DataFrame, model: Pipeline, cfg: ChurnConfig) -> None:
    """
    Persist churn scores and model artifact.

    Parameters
    ----------
    scores:
        Customer-level churn scores.
    model:
        Trained model pipeline.
    cfg:
        Configuration containing output paths.
    """
    ensure_parent_dir(cfg.output_scores_path)
    scores.to_parquet(cfg.output_scores_path, index=False)
    LOGGER.info("Churn risk scores written to '%s' (rows=%d)", str(cfg.output_scores_path), len(scores))

    ensure_parent_dir(cfg.output_model_path)
    dump(model, cfg.output_model_path)
    LOGGER.info("Model artifact written to '%s'", str(cfg.output_model_path))


def run(cfg: ChurnConfig) -> None:
    """
    Execute churn labeling, model training, evaluation, and scoring.

    Parameters
    ----------
    cfg:
        ChurnConfig for the current run.
    """
    setup_logging(cfg.log_level)

    if not cfg.transactions_path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {cfg.transactions_path}. "
            "Ensure Step 2 (cleaning) was completed."
        )
    if not cfg.features_path.exists():
        raise FileNotFoundError(
            f"Required input file not found: {cfg.features_path}. "
            "Ensure Step 3 (features) was completed."
        )

    LOGGER.info("Reading cleaned transactions from '%s'", str(cfg.transactions_path))
    txn = pd.read_parquet(cfg.transactions_path)
    LOGGER.info("Transactions shape=%s", txn.shape)

    LOGGER.info("Reading customer features from '%s'", str(cfg.features_path))
    features = pd.read_parquet(cfg.features_path)
    LOGGER.info("Features shape=%s", features.shape)

    inv = aggregate_to_invoice_level(txn)
    LOGGER.info("Invoice-level events shape=%s", inv.shape)

    model, metrics, scores = train_and_evaluate(inv=inv, features=features, cfg=cfg)
    LOGGER.info(
        "Churn model evaluation (out-of-time labeling, eval_gap_days=%d): %s",
        cfg.eval_gap_days,
        metrics,
    )

    write_outputs(scores=scores, model=model, cfg=cfg)
    LOGGER.info("Step 5 completed successfully")


def main() -> None:
    """CLI entry point."""
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()