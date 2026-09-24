"""Shared test fixtures. A small, valid AI4I-shaped batch that individual tests can break on purpose."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import RAW_COLUMNS  # noqa: E402


def make_batch(n: int = 200, failures: int = 6, seed: int = 0) -> pd.DataFrame:
    """A clean batch in the raw AI4I format (same columns, same order, plausible values)."""
    rng = np.random.default_rng(seed)
    air = np.round(rng.normal(300, 1.0, n), 1)
    df = pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"L{40000 + i}" for i in range(n)],
        "Type": rng.choice(["L", "M", "H"], n, p=[0.5, 0.3, 0.2]),
        "Air temperature [K]": air,
        "Process temperature [K]": np.round(air + rng.normal(10, 0.5, n), 1),
        "Rotational speed [rpm]": rng.integers(1200, 2800, n),
        "Torque [Nm]": np.round(rng.normal(40, 8, n), 1).clip(5, 75),
        "Tool wear [min]": rng.integers(0, 240, n),
        "Machine failure": 0, "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
    })
    fail_idx = rng.choice(n, failures, replace=False)
    df.loc[fail_idx, "Machine failure"] = 1
    df.loc[fail_idx, "HDF"] = 1
    return df[RAW_COLUMNS]


@pytest.fixture
def clean_batch() -> pd.DataFrame:
    return make_batch()
