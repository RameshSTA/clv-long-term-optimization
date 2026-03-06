# src/utils/config_loader.py
"""
YAML configuration loader for the CLV optimization pipeline.

Provides two entry points:
- ``load_yaml``: Load and validate a single YAML file.
- ``load_configs``: Load the full standard config set (project, modeling,
  business, evaluation) as a single dict of dicts.

All functions raise on the earliest detectable error so callers receive
actionable messages rather than cryptic downstream failures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML config file from disk.

    Parameters
    ----------
    path:
        Path to a YAML file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the YAML cannot be parsed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {p}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Config file must parse to a dict: {p}")

    return data


def load_configs(config_dir: str | Path) -> dict[str, dict[str, Any]]:
    """
    Load the project's standard config set.

    Expects:
      - project.yaml
      - modeling.yaml
      - business.yaml
      - evaluation.yaml
    """
    d = Path(config_dir)
    return {
        "project": load_yaml(d / "project.yaml"),
        "modeling": load_yaml(d / "modeling.yaml"),
        "business": load_yaml(d / "business.yaml"),
        "evaluation": load_yaml(d / "evaluation.yaml"),
    }
