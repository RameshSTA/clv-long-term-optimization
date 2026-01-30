"""

Data ingestion for: Customer Lifetime Value (CLV) with Long-Term Optimization.

This module loads the Online Retail II workbook (two sheets), standardizes schema and
types, runs basic ingestion-time validation, logs a small profiling summary, and writes
a combined dataset to Parquet for downstream pipeline stages.

Design notes
------------
- This stage is intentionally "raw-standardized", not "cleaned". We preserve cancellations,
  negative quantities, and other artifacts because business cleaning rules belong in
  `src/cleaning` (separation of concerns).
- We prioritize reproducibility and auditability. All schema assumptions are explicit.
- We standardize types after reading to avoid brittle Excel dtype inference.

"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionConfig:
    """Configuration for the ingestion pipeline."""
    input_path: Path
    output_path: Path
    sheet_names: Tuple[str, str]
    log_level: str


EXPECTED_INPUT_COLUMNS: Tuple[str, ...] = (
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
)

COLUMN_RENAME_MAP: Dict[str, str] = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_dt",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}

STANDARD_COLUMN_ORDER: Tuple[str, ...] = (
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_dt",
    "unit_price",
    "customer_id",
    "country",
    "source_sheet",
)


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


def parse_args(argv: Sequence[str] | None = None) -> IngestionConfig:
    """
    Parse CLI arguments into an IngestionConfig.

    Parameters
    ----------
    argv:
        Optional argument list. If None, argparse reads from sys.argv.

    Returns
    -------
    IngestionConfig
        Parsed configuration for ingestion.
    """
    parser = argparse.ArgumentParser(description="Ingest Online Retail II Excel data.")
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/raw/online_retail_II.xlsx",
        help="Path to Online Retail II Excel file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/interim/transactions_raw.parquet",
        help="Path to write combined standardized dataset (Parquet).",
    )
    parser.add_argument(
        "--sheet-names",
        type=str,
        nargs=2,
        default=("Year 2009-2010", "Year 2010-2011"),
        help="Two sheet names to load from the workbook.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args(args=argv)

    return IngestionConfig(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        sheet_names=(args.sheet_names[0], args.sheet_names[1]),
        log_level=args.log_level,
    )


def ensure_parent_dir(path: Path) -> None:
    """
    Ensure that the parent directory of `path` exists.

    Parameters
    ----------
    path:
        Target file path whose parent directory should exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def validate_expected_columns(df: pd.DataFrame, sheet_name: str) -> None:
    """
    Validate that all required input columns exist in the given DataFrame.

    Parameters
    ----------
    df:
        Raw DataFrame as loaded from Excel.
    sheet_name:
        Excel sheet name used for error context.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing = [c for c in EXPECTED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns in sheet '{sheet_name}': {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def read_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """
    Read a single Excel sheet into a DataFrame.

    Notes
    -----
    We avoid enforcing dtypes during read to reduce brittleness from mixed types.
    Standardization happens in `standardize_dataframe`.

    Parameters
    ----------
    path:
        Path to the Excel workbook.
    sheet_name:
        Sheet name to load.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the raw sheet content.
    """
    LOGGER.info("Reading sheet '%s' from '%s'", sheet_name, str(path))
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    LOGGER.info("Loaded sheet '%s' with shape=%s", sheet_name, df.shape)
    return df


def standardize_dataframe(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Standardize a raw sheet DataFrame into the project canonical schema.

    This function:
    - validates expected columns
    - renames columns to snake_case
    - normalizes types for downstream consistency
    - parses timestamps
    - adds a `source_sheet` column for traceability
    - orders columns deterministically

    Parameters
    ----------
    df:
        Raw DataFrame from Excel.
    sheet_name:
        Sheet name used to label the `source_sheet` column.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns in STANDARD_COLUMN_ORDER.
    """
    validate_expected_columns(df, sheet_name)

    out = df.rename(columns=COLUMN_RENAME_MAP).copy()

    # Text columns: use pandas string dtype for consistency (nullable).
    for col in ("invoice", "stock_code", "description", "country"):
        out[col] = out[col].astype("string")

    # Numeric columns: coerce invalid to NaN; keep quantity as nullable integer.
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce").astype("Int64")
    out["unit_price"] = pd.to_numeric(out["unit_price"], errors="coerce")

    # Customer ID often comes as float in Excel; normalize to nullable integer.
    out["customer_id"] = pd.to_numeric(out["customer_id"], errors="coerce").astype("Int64")

    # Datetime parsing: invalid values become NaT; keep naive timestamps (dataset local time).
    out["invoice_dt"] = pd.to_datetime(out["invoice_dt"], errors="coerce", utc=False)

    # Traceability.
    out["source_sheet"] = sheet_name

    # Deterministic ordering.
    out = out.loc[:, list(STANDARD_COLUMN_ORDER)]

    return out


def basic_profile(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute ingestion-time data quality indicators.

    These metrics are intended for visibility and logging only. They should not
    be used to filter or modify the dataset at this stage.

    Parameters
    ----------
    df:
        Standardized raw transactions dataset.

    Returns
    -------
    Dict[str, float]
        Dictionary of profiling metrics.
    """
    return {
        "rows": float(len(df)),
        "missing_customer_id_rate": float(df["customer_id"].isna().mean()),
        "missing_invoice_dt_rate": float(df["invoice_dt"].isna().mean()),
        "nonpositive_price_rate": float((df["unit_price"].fillna(0) <= 0).mean()),
        "negative_quantity_rate": float((df["quantity"].fillna(0) < 0).mean()),
        "duplicate_row_rate": float(df.duplicated().mean()),
    }


def assert_standard_schema(df: pd.DataFrame) -> None:
    """
    Assert that the standardized dataset matches the expected canonical schema.

    Parameters
    ----------
    df:
        Standardized DataFrame to validate.

    Raises
    ------
    ValueError
        If the schema does not match expectations.
    """
    expected = list(STANDARD_COLUMN_ORDER)
    actual = list(df.columns)
    if actual != expected:
        raise ValueError(
            "Standardized columns do not match expected schema/order. "
            f"Expected={expected} | Actual={actual}"
        )


def write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write a DataFrame to Parquet with safe directory creation.

    Parameters
    ----------
    df:
        DataFrame to write.
    output_path:
        Destination Parquet path.
    """
    ensure_parent_dir(output_path)
    LOGGER.info(
        "Writing Parquet to '%s' (rows=%d, cols=%d)",
        str(output_path),
        df.shape[0],
        df.shape[1],
    )
    df.to_parquet(output_path, index=False)
    LOGGER.info("Write complete")


def run(cfg: IngestionConfig) -> None:
    """
    Execute the ingestion pipeline using the provided configuration.

    Parameters
    ----------
    cfg:
        IngestionConfig for the current run.

    Raises
    ------
    FileNotFoundError
        If the input workbook does not exist.
    ValueError
        If schema validation fails.
    """
    setup_logging(cfg.log_level)

    if not cfg.input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {cfg.input_path}. "
            "Place the dataset at data/raw/online_retail_II.xlsx or pass --input-path."
        )

    sheet_a, sheet_b = cfg.sheet_names

    df_a_raw = read_excel_sheet(cfg.input_path, sheet_a)
    df_b_raw = read_excel_sheet(cfg.input_path, sheet_b)

    df_a = standardize_dataframe(df_a_raw, sheet_a)
    df_b = standardize_dataframe(df_b_raw, sheet_b)

    combined = pd.concat([df_a, df_b], ignore_index=True)

    # Validate final schema and log profiling metrics.
    assert_standard_schema(combined)
    LOGGER.info("Ingestion profiling metrics: %s", basic_profile(combined))

    # Persist standardized raw transactions for downstream stages.
    write_parquet(combined, cfg.output_path)

    LOGGER.info("Ingestion finished successfully")


def main() -> None:
    """CLI entry point."""
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()