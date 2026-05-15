"""Shared utilities for the HCMST project.

The notebooks are the main analysis record. This module contains reusable path,
loading, saving, and validation helpers so future scripts can use the same
project conventions without duplicating boilerplate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"


def ensure_project_dirs() -> None:
    """Create standard output directories if they do not already exist."""

    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, TABLES_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


def project_path(*parts: str) -> Path:
    """Return an absolute path inside the project root."""

    return PROJECT_ROOT.joinpath(*parts)


def read_processed_csv(filename: str, **kwargs) -> pd.DataFrame:
    """Load a CSV from ``data/processed``."""

    return pd.read_csv(PROCESSED_DATA_DIR / filename, **kwargs)


def save_processed_csv(df: pd.DataFrame, filename: str, **kwargs) -> Path:
    """Save a DataFrame to ``data/processed`` and return the written path."""

    ensure_project_dirs()
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False, **kwargs)
    return path


def save_table_csv(df: pd.DataFrame, filename: str, **kwargs) -> Path:
    """Save a DataFrame to ``outputs/tables`` and return the written path."""

    ensure_project_dirs()
    path = TABLES_DIR / filename
    df.to_csv(path, index=False, **kwargs)
    return path


def read_hcmst_spss(filename: str = "HCMST 2017 to 2022 small public version 2.2.sav") -> pd.DataFrame:
    """Load the raw HCMST SPSS file from ``data/raw``."""

    return pd.read_spss(RAW_DATA_DIR / filename)


def require_columns(df: pd.DataFrame, columns: Iterable[str], dataset_name: str = "DataFrame") -> None:
    """Raise a clear error if required columns are missing."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return columns from ``columns`` that are present in ``df``."""

    return [column for column in columns if column in df.columns]


def shape_summary(name: str, df: pd.DataFrame) -> dict[str, int | str]:
    """Return a compact dataset shape summary."""

    return {
        "dataset": name,
        "n_rows": len(df),
        "n_columns": df.shape[1],
    }
