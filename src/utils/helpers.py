"""
Shared utilities for logging configuration and filesystem safety.

This module centralizes common infrastructure concerns that are reused
across ingestion, modeling, optimization, and evaluation steps.

Design principles:
- Single responsibility
- Explicit failure on misconfiguration
- No hidden side effects
- Production-safe defaults
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(level: str) -> None:
    """
    Configure application-wide console logging.

    This function should be called once at process startup.
    It enforces consistent formatting and validates log level input.

    Parameters
    ----------
    level : str
        Logging level name (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').

    Raises
    ------
    ValueError
        If the provided log level is invalid.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(
            f"Invalid log level '{level}'. Expected one of: DEBUG, INFO, WARNING, ERROR."
        )

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_parent_dir(path: str | Path) -> None:
    """
    Ensure that the parent directory for a file path exists.

    This is used before writing artifacts such as:
    - Parquet files
    - Model binaries
    - Reports and evaluation outputs

    Parameters
    ----------
    path : Union[str, Path]
        Target file path whose parent directory should be created.

    Notes
    -----
    - This function is idempotent.
    - It does not validate write permissions.
    """
    path = Path(path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
