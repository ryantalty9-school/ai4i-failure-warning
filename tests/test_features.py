"""Tests for feature engineering and the leakage guard (src/features.py)."""
from __future__ import annotations

import math

import pandas as pd

from src.features import (CATEGORICAL, ENGINEERED, FAILURE_MODES, FEATURE_SETS, ID_COLUMNS, RAW_NUMERIC,
                          TARGET, add_engineered_features, model_inputs, rename_raw)
from tests.conftest import make_batch


def test_rename_gives_snake_case_columns(clean_batch):
    df = rename_raw(clean_batch)
    for col in RAW_NUMERIC + CATEGORICAL + [TARGET]:
        assert col in df.columns
    assert not any("[" in c for c in df.columns), "XGBoost rejects column names containing brackets"


def test_engineered_features_match_the_physics():
    df = pd.DataFrame([{"type": "L", "air_temp_k": 298.0, "process_temp_k": 308.5,
                        "rot_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100}])
    out = add_engineered_features(df).iloc[0]
    assert out["temp_diff_k"] == 10.5
    assert math.isclose(out["power_w"], 40.0 * 1500 * 2 * math.pi / 60, rel_tol=1e-9)
    assert out["strain_min_nm"] == 100 * 40.0


def test_feature_sets_never_include_labels_or_ids():
    """The failure-mode flags are labels: using them as inputs would leak the answer."""
    leaky = set(FAILURE_MODES + ID_COLUMNS + [TARGET])
    for name, cols in FEATURE_SETS.items():
        assert not leaky.intersection(cols), f"{name} feature set leaks {leaky.intersection(cols)}"


def test_model_inputs_returns_exactly_the_model_columns():
    df = add_engineered_features(rename_raw(make_batch(n=20)))
    X = model_inputs(df)
    assert list(X.columns) == CATEGORICAL + RAW_NUMERIC + ENGINEERED
    assert len(X) == 20
