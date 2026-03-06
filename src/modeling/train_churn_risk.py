# src/modeling/train_churn_risk.py
"""
Churn risk modeling with multi-model comparison, SHAP, and calibration analysis.

Context
-------
In this public transactional dataset we do not observe explicit churn events (e.g., an
account cancellation). A common operational proxy is inactivity-based churn:

A customer is considered "churned" if they do not make a purchase within a fixed number
of days after a decision cutoff date.

This script:
1) Reads cleaned transactions and customer features computed at a cutoff date.
2) Creates churn labels using ONLY post-cutoff activity (time-safe; no leakage).
3) Compares multiple churn models (Logistic Regression, Random Forest, XGBoost, LightGBM)
   using stratified k-fold cross-validation.
4) Selects the best model by CV ROC AUC and retrains it on full training data.
5) Evaluates on an out-of-time holdout set.
6) Generates SHAP values for model interpretability.
7) Produces calibration curves, ROC/PR curves, and model comparison charts.
8) Writes churn probabilities, the best model artifact, and all reports.

Evaluation strategy (time-safe)
--------------------------------
- Train cutoff: C
- Test cutoff:  C + eval_gap_days

For each cutoff snapshot, we label churn using only activity in (cutoff, cutoff + horizon].
This prevents leakage and mimics production: train using an earlier cutoff and evaluate
using a later cutoff.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc as sklearn_auc,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LGB_AVAILABLE = False

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SHAP_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MLFLOW_AVAILABLE = False


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
    random_state: int
    cv_folds: int
    log_level: str
    mlflow_experiment: str


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


def ensure_parent_dir(path: Path) -> None:
    """Create all missing parent directories for *path*.

    Args:
        path: Target file path whose parent directory tree should be created.
            Existing directories are left untouched (idempotent).
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> ChurnConfig:
    """Parse CLI arguments and return a validated :class:`ChurnConfig`.

    All arguments have defaults so the script can be invoked without flags
    during development. Production runs should pass explicit paths and dates
    sourced from YAML configs via the pipeline orchestrator.

    Args:
        argv: Optional list of CLI tokens. Defaults to ``sys.argv[1:]`` when
            ``None`` (standard ``argparse`` behaviour).

    Returns:
        A frozen :class:`ChurnConfig` dataclass with all parameters resolved
        to their correct Python types (``Path``, ``pd.Timestamp``, ``int``).
    """
    parser = argparse.ArgumentParser(description="Train churn risk model and score customers.")
    parser.add_argument("--transactions-path", type=str,
                        default="data/interim/transactions_clean.parquet")
    parser.add_argument("--features-path", type=str,
                        default="data/processed/customer_features.parquet")
    parser.add_argument("--output-scores-path", type=str,
                        default="data/processed/customer_churn_risk_scores.parquet")
    parser.add_argument("--output-model-path", type=str,
                        default="data/processed/churn_risk_model.joblib")
    parser.add_argument("--cutoff-date", type=str, default="2011-06-01")
    parser.add_argument("--prediction-horizon-days", type=int, default=180)
    parser.add_argument("--churn-inactivity-days", type=int, default=90)
    parser.add_argument("--eval-gap-days", type=int, default=60)
    parser.add_argument("--model-type", type=str, default="auto",
                        choices=["auto", "logistic", "random_forest", "xgboost", "lightgbm"],
                        help="Model to use. 'auto' runs comparison and picks best by CV AUC.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--mlflow-experiment", type=str, default="churn_risk_model")
    parser.add_argument("--log-level", type=str, default="INFO")
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
        random_state=int(args.random_state),
        cv_folds=int(args.cv_folds),
        log_level=str(args.log_level),
        mlflow_experiment=str(args.mlflow_experiment),
    )


def _assert_columns_present(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}. Found: {list(df.columns)}")


def aggregate_to_invoice_level(txn: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line-item transactions into invoice-level purchase events."""
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

    churn=1 if the customer makes no purchase within inactivity_days after cutoff,
    or has no purchase in the observation window (cutoff, cutoff+horizon_days].
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

    labels = pd.DataFrame({"customer_id": pd.Series(customers).astype("Int64")})
    labels = labels.merge(first_post.reset_index(), on="customer_id", how="left")
    labels["days_to_next_purchase"] = (labels["first_purchase_post_cutoff"] - cutoff).dt.days
    labels["churn_label"] = (
        labels["days_to_next_purchase"].isna()
        | (labels["days_to_next_purchase"] > inactivity_days)
    ).astype(int)
    labels["days_to_next_purchase"] = labels["days_to_next_purchase"].fillna(np.inf)

    out = labels[["customer_id", "first_purchase_post_cutoff",
                  "days_to_next_purchase", "churn_label"]].copy()
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
    Join feature snapshot with churn labels.
    Customers absent from labels default to churn=1 (conservative: no observable activity = churned).
    """
    df = features.merge(labels, on="customer_id", how="left").copy()
    df["churn_label"] = df["churn_label"].fillna(1).astype(int)
    df["cutoff_date"] = cutoff
    df["prediction_horizon_days"] = horizon_days
    df["churn_inactivity_days"] = inactivity_days
    return df


# ---------------------------------------------------------------------------
# Model candidate construction
# ---------------------------------------------------------------------------

def _build_candidate_pipelines(
    feature_cols: Tuple[str, ...],
    random_state: int,
) -> Dict[str, Pipeline]:
    """
    Build one sklearn Pipeline per candidate model type.

    Design principles:
    - Logistic Regression: needs StandardScaler (distance-based loss).
    - Tree models: only need imputation (invariant to monotone transforms).
    - All pipelines share the same interface: fit(X, y) and predict_proba(X).
    """
    def _imputer_scaler_ct():
        return ColumnTransformer(
            transformers=[("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), list(feature_cols))],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _imputer_only_ct():
        return ColumnTransformer(
            transformers=[("num", SimpleImputer(strategy="median"), list(feature_cols))],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    candidates: Dict[str, Pipeline] = {
        "logistic_regression": Pipeline([
            ("preprocess", _imputer_scaler_ct()),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=random_state,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocess", _imputer_only_ct()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }

    if _XGB_AVAILABLE:
        candidates["xgboost"] = Pipeline([
            ("preprocess", _imputer_only_ct()),
            ("clf", xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                eval_metric="logloss",
                verbosity=0,
            )),
        ])

    if _LGB_AVAILABLE:
        candidates["lightgbm"] = Pipeline([
            ("preprocess", _imputer_only_ct()),
            ("clf", lgb.LGBMClassifier(
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.05,
                is_unbalance=True,
                random_state=random_state,
                verbose=-1,
            )),
        ])

    return candidates


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: Tuple[str, ...],
    cv_folds: int,
    random_state: int,
) -> Tuple[pd.DataFrame, str, Pipeline]:
    """
    Systematically compare candidate churn models via stratified k-fold CV.

    Models evaluated: Logistic Regression, Random Forest, XGBoost (if installed),
    LightGBM (if installed). Selection criterion: mean CV ROC AUC.

    Returns
    -------
    comparison_df : DataFrame with CV metrics per model, sorted best-first.
    best_model_name : name of the winning model.
    best_model : Pipeline fitted on the full training set X, y.
    """
    candidates = _build_candidate_pipelines(feature_cols, random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results = []
    for name, pipe in candidates.items():
        LOGGER.info("Running %d-fold CV for: %s", cv_folds, name)
        auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        ap_scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
        results.append({
            "model": name,
            "cv_roc_auc_mean": float(auc_scores.mean()),
            "cv_roc_auc_std": float(auc_scores.std()),
            "cv_avg_precision_mean": float(ap_scores.mean()),
            "cv_avg_precision_std": float(ap_scores.std()),
        })
        LOGGER.info("  %s: AUC=%.4f ± %.4f  AP=%.4f ± %.4f",
                    name, auc_scores.mean(), auc_scores.std(),
                    ap_scores.mean(), ap_scores.std())

    comparison_df = (
        pd.DataFrame(results)
        .sort_values("cv_roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
    best_model_name = comparison_df.iloc[0]["model"]
    LOGGER.info("Selected model: %s (CV ROC AUC=%.4f ± %.4f)",
                best_model_name,
                comparison_df.iloc[0]["cv_roc_auc_mean"],
                comparison_df.iloc[0]["cv_roc_auc_std"])

    # Fit winner on full training data
    best_model = candidates[best_model_name]
    best_model.fit(X, y)

    return comparison_df, best_model_name, best_model


# ---------------------------------------------------------------------------
# Legacy single-model builder (used by tests; kept for backwards compatibility)
# ---------------------------------------------------------------------------

def make_model(feature_cols: Tuple[str, ...], random_state: int = 42) -> Pipeline:
    """Build the logistic regression churn pipeline (baseline model)."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, list(feature_cols))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])


def cross_validate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: int,
) -> Dict[str, float]:
    """Stratified k-fold cross-validation for a single model."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    ap_scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision")
    return {
        "cv_roc_auc_mean": float(auc_scores.mean()),
        "cv_roc_auc_std": float(auc_scores.std()),
        "cv_avg_precision_mean": float(ap_scores.mean()),
        "cv_avg_precision_std": float(ap_scores.std()),
        "cv_folds": cv_folds,
    }


def extract_feature_importance(model: Pipeline, feature_cols: Tuple[str, ...]) -> pd.DataFrame:
    """Extract logistic regression coefficients as feature importance."""
    coef = model.named_steps["clf"].coef_[0]
    importance = pd.DataFrame({
        "feature": list(feature_cols),
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    importance["direction"] = importance["coefficient"].apply(
        lambda c: "increases_churn" if c > 0 else "decreases_churn"
    )
    return importance


# ---------------------------------------------------------------------------
# SHAP interpretability
# ---------------------------------------------------------------------------

def compute_shap_values(
    model: Pipeline,
    X: pd.DataFrame,
    feature_cols: Tuple[str, ...],
    model_name: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute SHAP values for the fitted pipeline.

    Uses TreeExplainer for tree-based models (fast, exact SHAP values).
    Uses LinearExplainer for Logistic Regression.

    Parameters
    ----------
    model : fitted sklearn Pipeline.
    X : raw feature DataFrame (pre-transformation).
    feature_cols : feature column names.
    model_name : one of 'logistic_regression', 'random_forest', 'xgboost', 'lightgbm'.

    Returns
    -------
    shap_vals : ndarray of shape (n_samples, n_features)
    X_transformed : DataFrame of preprocessed features (for SHAP plots)
    """
    X_transformed_arr = model.named_steps["preprocess"].transform(X)
    X_df = pd.DataFrame(X_transformed_arr, columns=list(feature_cols))
    clf = model.named_steps["clf"]

    if model_name in ("random_forest", "xgboost", "lightgbm"):
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(X_df)
        # Handle different SHAP return formats across versions:
        # - list [class0, class1] → take class1
        # - ndarray shape (n, f) → use directly
        # - ndarray shape (n, f, n_classes) → take class1 slice
        if isinstance(raw, list):
            shap_vals = np.asarray(raw[1])
        else:
            shap_vals = np.asarray(raw)
            if shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, 1]
    else:
        # Logistic Regression: LinearExplainer in standardized feature space
        explainer = shap.LinearExplainer(clf, X_df, feature_perturbation="interventional")
        shap_vals = np.asarray(explainer.shap_values(X_df))

    # Guarantee 2D output: (n_samples, n_features)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)
    elif shap_vals.ndim > 2:
        shap_vals = shap_vals.mean(axis=-1)

    return shap_vals, X_df


def _shap_feature_importance(
    shap_vals: np.ndarray,
    X_df: pd.DataFrame,
    feature_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """Build feature importance from mean |SHAP| values (model-agnostic)."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=-1)
    directions = []
    for i in range(len(feature_cols)):
        corr = float(np.corrcoef(X_df.iloc[:, i].values, shap_vals[:, i])[0, 1])
        directions.append("increases_churn" if corr > 0 else "decreases_churn")

    importance = pd.DataFrame({
        "feature": list(feature_cols),
        "mean_abs_shap": mean_abs,
        "direction": directions,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    """Horizontal bar chart comparing CV ROC AUC across candidate models."""
    fig, ax = plt.subplots(figsize=(9, max(4, len(comparison_df) * 1.1)))

    labels = comparison_df["model"].str.replace("_", " ").str.title()
    aucs = comparison_df["cv_roc_auc_mean"].values
    stds = comparison_df["cv_roc_auc_std"].values
    colors = ["#1565C0" if i == 0 else "#90CAF9" for i in range(len(labels))]

    bars = ax.barh(labels[::-1], aucs[::-1], xerr=stds[::-1],
                   color=colors[::-1], capsize=4, edgecolor="white", linewidth=0.5)

    for bar, val, std in zip(bars, aucs[::-1], stds[::-1]):
        ax.text(val + std + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f} ± {std:.3f}", va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_xlabel("CV ROC AUC (mean ± 1 std, stratified k-fold)", fontsize=11)
    ax.set_title("Churn Model Comparison — Cross-Validated Performance",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0.48, 1.05)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.6, linewidth=1, label="Random baseline (0.50)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=9)

    # Best model label
    best = comparison_df.iloc[0]["model"].replace("_", " ").title()
    ax.set_title(f"Churn Model Comparison — Best: {best}",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Model comparison chart → '%s'", str(output_path))


def plot_shap_summary(
    shap_vals: np.ndarray,
    X_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Two-panel SHAP visualization: feature importance bar + beeswarm scatter."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    if mean_abs.ndim > 1:  # safety: collapse to 1D
        mean_abs = mean_abs.mean(axis=-1)
    feature_names = [c.replace("_", " ").title() for c in X_df.columns]
    sorted_idx = np.argsort(mean_abs).tolist()  # Python int list for safe list indexing

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: mean |SHAP| bar
    ax = axes[0]
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        [mean_abs[i] for i in sorted_idx],
        color="#1565C0", edgecolor="white",
    )
    ax.set_xlabel("Mean |SHAP value|  (avg impact on model output)", fontsize=10)
    ax.set_title("Feature Importance (SHAP)", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Right panel: scatter of SHAP value vs feature value, top-5 features
    ax2 = axes[1]
    top5 = list(reversed(sorted_idx))[:5]  # Python ints
    y_positions = range(len(top5))
    for rank, feat_idx in enumerate(top5):
        col_vals = X_df.iloc[:, feat_idx].values
        sv = shap_vals[:, feat_idx]
        # Normalize color to [0,1]
        norm = (col_vals - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-9)
        ax2.scatter(sv, [rank] * len(sv), c=norm, cmap="coolwarm",
                    alpha=0.4, s=12, linewidths=0)

    ax2.set_yticks(list(y_positions))
    ax2.set_yticklabels([feature_names[i] for i in top5], fontsize=10)
    ax2.set_xlabel("SHAP value (impact on churn probability)", fontsize=10)
    ax2.set_title("SHAP Values — Top 5 Features\n(color = feature value: blue=low, red=high)",
                  fontsize=10, fontweight="bold")
    ax2.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("SHAP Model Explanation — Churn Risk Model", fontsize=13, fontweight="bold")
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("SHAP summary plot → '%s'", str(output_path))


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Reliability diagram + probability score distribution by class."""
    fop, mpv = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    ax.plot(mpv, fop, "o-", color="#1565C0", linewidth=2, markersize=7,
            label=model_name.replace("_", " ").title())
    ax.fill_between(mpv, fop, mpv, alpha=0.15, color="#F44336",
                    label="Calibration gap")
    ax.set_xlabel("Mean predicted churn probability", fontsize=11)
    ax.set_ylabel("Observed churn rate (fraction of positives)", fontsize=11)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Panel 2: score distribution by label
    ax2 = axes[1]
    ax2.hist(y_prob[y_true == 0], bins=25, alpha=0.65, color="#4CAF50",
             label="Active customers (churn=0)", density=True)
    ax2.hist(y_prob[y_true == 1], bins=25, alpha=0.65, color="#F44336",
             label="Churned customers (churn=1)", density=True)
    ax2.set_xlabel("Predicted churn probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Score Distribution by True Label", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.suptitle(f"Model Calibration Analysis — {model_name.replace('_', ' ').title()}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Calibration curve → '%s'", str(output_path))


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Two-panel ROC and Precision-Recall curves."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = sklearn_auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = sklearn_auc(recall, precision)
    base_rate = float(y_true.mean())
    display_name = model_name.replace("_", " ").title()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ROC curve
    ax = axes[0]
    ax.plot(fpr, tpr, color="#1565C0", linewidth=2.5,
            label=f"{display_name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#1565C0")
    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax.set_title("ROC Curve", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)

    # Precision-Recall curve
    ax2 = axes[1]
    ax2.plot(recall, precision, color="#C62828", linewidth=2.5,
             label=f"{display_name} (AUC = {pr_auc:.3f})")
    ax2.axhline(y=base_rate, color="gray", linestyle="--", linewidth=1.2,
                label=f"No-skill baseline ({base_rate:.2f})")
    ax2.fill_between(recall, precision, base_rate, alpha=0.12, color="#C62828",
                     where=(precision > base_rate))
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title("Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(-0.01, 1.01)
    ax2.set_ylim(-0.01, 1.01)

    fig.suptitle(f"Churn Model Evaluation — {display_name}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("ROC + PR curves → '%s'", str(output_path))


# ---------------------------------------------------------------------------
# Core train / evaluate / score
# ---------------------------------------------------------------------------

def evaluate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
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
    bins = [0.0, 0.33, 0.66, 1.0]
    labels = ["low", "medium", "high"]
    return pd.cut(prob.clip(0, 1), bins=bins, labels=labels, include_lowest=True)


def train_and_evaluate(
    inv: pd.DataFrame,
    features: pd.DataFrame,
    cfg: ChurnConfig,
) -> Tuple[Pipeline, Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full churn modeling pipeline:
      1. Generate time-safe training + test labels.
      2. Compare candidate models via stratified k-fold CV.
      3. Retrain best model on full training data.
      4. Evaluate on out-of-time holdout (different label cutoff).
      5. Generate SHAP interpretability, calibration, ROC/PR plots.

    Returns
    -------
    model : best fitted Pipeline
    metrics : dict of holdout + CV metrics
    scored : DataFrame of churn probabilities at training cutoff
    feature_importance : SHAP-based feature importance (or coefficients for LogReg)
    comparison_df : model comparison CV results table
    """
    _assert_columns_present(features, REQUIRED_FEATURE_COLUMNS, name="features")
    customers = features["customer_id"]

    # Training labels
    train_labels = compute_churn_labels(
        inv=inv, cutoff=cfg.cutoff_date,
        horizon_days=cfg.prediction_horizon_days,
        inactivity_days=cfg.churn_inactivity_days,
        customers=customers,
    )
    train_frame = build_snapshot(
        features=features, labels=train_labels,
        cutoff=cfg.cutoff_date,
        horizon_days=cfg.prediction_horizon_days,
        inactivity_days=cfg.churn_inactivity_days,
    )
    X_train = train_frame.loc[:, list(FEATURE_COLUMNS)]
    y_train = train_frame["churn_label"].astype(int)

    # Out-of-time test labels
    test_cutoff = cfg.cutoff_date + pd.Timedelta(days=cfg.eval_gap_days)
    test_labels = compute_churn_labels(
        inv=inv, cutoff=test_cutoff,
        horizon_days=cfg.prediction_horizon_days,
        inactivity_days=cfg.churn_inactivity_days,
        customers=customers,
    )
    test_frame = build_snapshot(
        features=features, labels=test_labels,
        cutoff=test_cutoff,
        horizon_days=cfg.prediction_horizon_days,
        inactivity_days=cfg.churn_inactivity_days,
    )
    X_test = test_frame.loc[:, list(FEATURE_COLUMNS)]
    y_test = test_frame["churn_label"].astype(int)

    # --- Model comparison and selection ---
    if cfg.model_type == "auto":
        comparison_df, best_model_name, model = compare_models(
            X=X_train, y=y_train,
            feature_cols=FEATURE_COLUMNS,
            cv_folds=cfg.cv_folds,
            random_state=cfg.random_state,
        )
    else:
        # Run single-model CV then fit
        candidates = _build_candidate_pipelines(FEATURE_COLUMNS, cfg.random_state)
        model_name = cfg.model_type if cfg.model_type in candidates else "logistic_regression"
        model = candidates[model_name]
        cv_single = cross_validate_model(model, X_train, y_train, cfg.cv_folds, cfg.random_state)
        model.fit(X_train, y_train)
        best_model_name = model_name
        comparison_df = pd.DataFrame([{
            "model": model_name,
            "cv_roc_auc_mean": cv_single["cv_roc_auc_mean"],
            "cv_roc_auc_std": cv_single["cv_roc_auc_std"],
            "cv_avg_precision_mean": cv_single["cv_avg_precision_mean"],
            "cv_avg_precision_std": cv_single["cv_avg_precision_std"],
        }])

    # Holdout evaluation
    y_test_prob = model.predict_proba(X_test)[:, 1]
    holdout_metrics = evaluate_probabilities(y_test.values, y_test_prob)

    best_row = comparison_df.iloc[0]
    cv_metrics = {
        "cv_roc_auc_mean": float(best_row["cv_roc_auc_mean"]),
        "cv_roc_auc_std": float(best_row["cv_roc_auc_std"]),
        "cv_avg_precision_mean": float(best_row["cv_avg_precision_mean"]),
        "cv_avg_precision_std": float(best_row["cv_avg_precision_std"]),
        "cv_folds": cfg.cv_folds,
        "best_model": best_model_name,
    }
    metrics = {**holdout_metrics, **cv_metrics}

    # --- Visualizations ---
    LOGGER.info("Generating evaluation plots...")
    plot_model_comparison(comparison_df, Path("reports/figures/churn_model_comparison.png"))
    plot_roc_pr_curves(
        y_test.values, y_test_prob, best_model_name,
        Path("reports/figures/churn_roc_pr_curves.png"),
    )
    plot_calibration_curve(
        y_test.values, y_test_prob, best_model_name,
        Path("reports/figures/churn_calibration_curve.png"),
    )

    # SHAP interpretability
    if _SHAP_AVAILABLE:
        LOGGER.info("Computing SHAP values for %s...", best_model_name)
        n_shap = min(800, len(X_train))
        rng = np.random.default_rng(cfg.random_state)
        shap_idx = rng.choice(len(X_train), size=n_shap, replace=False)
        X_shap = X_train.iloc[shap_idx].reset_index(drop=True)

        shap_vals, X_transformed = compute_shap_values(
            model, X_shap, FEATURE_COLUMNS, best_model_name,
        )
        plot_shap_summary(shap_vals, X_transformed, Path("reports/figures/churn_shap_summary.png"))
        feature_importance = _shap_feature_importance(shap_vals, X_transformed, FEATURE_COLUMNS)
        feature_importance.rename(columns={"mean_abs_shap": "importance_score"}, inplace=True)
        feature_importance["importance_type"] = "mean_abs_shap"
    elif best_model_name == "logistic_regression":
        feature_importance = extract_feature_importance(model, FEATURE_COLUMNS)
        feature_importance.rename(columns={"abs_coefficient": "importance_score"}, inplace=True)
        feature_importance["importance_type"] = "abs_logistic_coef"
    else:
        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importance_score = clf.feature_importances_
        else:
            importance_score = np.zeros(len(FEATURE_COLUMNS))
        feature_importance = pd.DataFrame({
            "feature": list(FEATURE_COLUMNS),
            "importance_score": importance_score,
            "direction": ["unknown"] * len(FEATURE_COLUMNS),
            "importance_type": "builtin",
        }).sort_values("importance_score", ascending=False).reset_index(drop=True)

    # Scores at training cutoff
    y_train_prob = model.predict_proba(X_train)[:, 1]
    scored = pd.DataFrame({
        "customer_id": train_frame["customer_id"].astype("int64"),
        "cutoff_date": cfg.cutoff_date,
        "prediction_horizon_days": cfg.prediction_horizon_days,
        "churn_inactivity_days": cfg.churn_inactivity_days,
        "churn_probability": y_train_prob,
        "model_used": best_model_name,
    })
    scored["risk_band"] = assign_risk_band(scored["churn_probability"]).astype("string")

    return model, metrics, scored, feature_importance, comparison_df


def write_outputs(
    scores: pd.DataFrame,
    model: Pipeline,
    feature_importance: pd.DataFrame,
    comparison_df: pd.DataFrame,
    cfg: ChurnConfig,
) -> None:
    """Persist scores, model artifact, feature importance, and model comparison table."""
    ensure_parent_dir(cfg.output_scores_path)
    scores.to_parquet(cfg.output_scores_path, index=False)
    LOGGER.info("Churn scores → '%s' (rows=%d)", str(cfg.output_scores_path), len(scores))

    ensure_parent_dir(cfg.output_model_path)
    dump(model, cfg.output_model_path)
    LOGGER.info("Model artifact → '%s'", str(cfg.output_model_path))

    importance_path = Path("reports/tables/churn_feature_importance.csv")
    ensure_parent_dir(importance_path)
    feature_importance.to_csv(importance_path, index=False)
    LOGGER.info("Feature importance → '%s'", str(importance_path))

    comparison_path = Path("reports/tables/churn_model_comparison.csv")
    ensure_parent_dir(comparison_path)
    comparison_df.to_csv(comparison_path, index=False)
    LOGGER.info("Model comparison → '%s'", str(comparison_path))


def _log_to_mlflow(cfg: ChurnConfig, metrics: Dict[str, float]) -> None:
    if not _MLFLOW_AVAILABLE:
        return
    try:
        mlflow.set_experiment(cfg.mlflow_experiment)
        with mlflow.start_run():
            mlflow.log_params({
                "cutoff_date": str(cfg.cutoff_date.date()),
                "prediction_horizon_days": cfg.prediction_horizon_days,
                "churn_inactivity_days": cfg.churn_inactivity_days,
                "eval_gap_days": cfg.eval_gap_days,
                "model_type": cfg.model_type,
                "best_model": metrics.get("best_model", "unknown"),
                "random_state": cfg.random_state,
                "cv_folds": cfg.cv_folds,
            })
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, float) and not np.isnan(v)})
            if cfg.output_scores_path.exists():
                mlflow.log_artifact(str(cfg.output_scores_path))
            if cfg.output_model_path.exists():
                mlflow.log_artifact(str(cfg.output_model_path))
            for path in [
                "reports/tables/churn_feature_importance.csv",
                "reports/tables/churn_model_comparison.csv",
                "reports/figures/churn_model_comparison.png",
                "reports/figures/churn_shap_summary.png",
                "reports/figures/churn_roc_pr_curves.png",
                "reports/figures/churn_calibration_curve.png",
            ]:
                if Path(path).exists():
                    mlflow.log_artifact(path)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("MLflow logging failed (non-fatal): %s", exc)


def run(cfg: ChurnConfig) -> None:
    """
    Execute the full churn modeling workflow:
    1. Load inputs.
    2. Compare models via CV → select best.
    3. Generate SHAP, calibration, ROC/PR plots.
    4. Evaluate on out-of-time holdout.
    5. Persist all outputs.
    6. Log to MLflow.
    """
    setup_logging(cfg.log_level)

    for path, step in [
        (cfg.transactions_path, "Step 2 (cleaning)"),
        (cfg.features_path, "Step 3 (features)"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}. Ensure {step} was completed."
            )

    LOGGER.info("Loading transactions from '%s'", str(cfg.transactions_path))
    txn = pd.read_parquet(cfg.transactions_path)

    LOGGER.info("Loading features from '%s'", str(cfg.features_path))
    features = pd.read_parquet(cfg.features_path)

    inv = aggregate_to_invoice_level(txn)

    model, metrics, scores, feature_importance, comparison_df = train_and_evaluate(
        inv=inv, features=features, cfg=cfg,
    )

    LOGGER.info("Holdout metrics: %s", {k: round(v, 4) for k, v in metrics.items()
                                         if isinstance(v, float)})
    LOGGER.info("Top features:\n%s", feature_importance.head(5).to_string(index=False))

    write_outputs(
        scores=scores,
        model=model,
        feature_importance=feature_importance,
        comparison_df=comparison_df,
        cfg=cfg,
    )
    _log_to_mlflow(cfg=cfg, metrics=metrics)
    LOGGER.info("Step 5 (churn modeling) completed successfully.")


def main() -> None:
    """CLI entry point: parse arguments and run the full churn-modeling workflow.

    Invoked by ``python -m src.modeling.train_churn_risk`` or via the
    ``clv-churn`` console script defined in ``pyproject.toml``.
    """
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
