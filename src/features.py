"""Column names and feature engineering, shared by the pipeline, training, and the app."""
from __future__ import annotations

import math

import pandas as pd

# Raw AI4I column names (as they arrive from the source file)
RAW_COLUMNS = [
    "UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF",
]

# Clean snake_case names used after processing (XGBoost does not allow "[" or "]" in names)
RENAME = {
    "UDI": "udi", "Product ID": "product_id", "Type": "type",
    "Air temperature [K]": "air_temp_k", "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm", "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min", "Machine failure": "machine_failure",
    "TWF": "twf", "HDF": "hdf", "PWF": "pwf", "OSF": "osf", "RNF": "rnf",
}

TARGET = "machine_failure"
FAILURE_MODES = ["twf", "hdf", "pwf", "osf", "rnf"]    # labels -> never used as model inputs (leakage)
ID_COLUMNS = ["udi", "product_id"]                     # identifiers -> not predictive
CATEGORICAL = ["type"]
RAW_NUMERIC = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"]
ENGINEERED = ["temp_diff_k", "power_w", "strain_min_nm"]

FEATURE_SETS = {
    "raw": RAW_NUMERIC,
    "engineered": RAW_NUMERIC + ENGINEERED,
}


def rename_raw(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=RENAME)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-based features that map to the failure modes:
    - temp_diff_k: process minus air temperature (heat dissipation)
    - power_w: torque x angular speed (power failure)
    - strain_min_nm: tool wear x torque (overstrain)
    """
    out = df.copy()
    for c in RAW_NUMERIC:
        out[c] = out[c].astype(float)
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * out["rot_speed_rpm"] * 2 * math.pi / 60
    out["strain_min_nm"] = out["tool_wear_min"] * out["torque_nm"]
    return out


def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """All candidate input columns (the model pipeline picks the ones for its feature set)."""
    return df[CATEGORICAL + RAW_NUMERIC + ENGINEERED].copy()
