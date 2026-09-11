"""Load the registered champion model and score machine runs (used by the Streamlit app)."""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import load_config
from src.features import CATEGORICAL, ENGINEERED, RAW_NUMERIC, add_engineered_features, rename_raw
from src.tracking import setup_mlflow


def load_champion(alias: str = "champion"):
    """Returns (model, info dict) for the model version behind an alias in the MLflow Model Registry."""
    setup_mlflow()
    name = load_config()["mlflow"]["registered_model"]
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    info = {"name": name, "alias": alias, "version": mv.version, "description": mv.description or "",
            "tags": dict(mv.tags or {}), "threshold": float((mv.tags or {}).get("threshold", 0.5))}
    return model, info


def registry_table() -> pd.DataFrame:
    setup_mlflow()
    name = load_config()["mlflow"]["registered_model"]
    client = MlflowClient()
    rm = client.get_registered_model(name)
    aliases = {}
    for a, v in (getattr(rm, "aliases", None) or {}).items():
        aliases.setdefault(str(v), []).append(a)
    rows = []
    for mv in client.search_model_versions(f"name='{name}'"):
        t = dict(mv.tags or {})
        rows.append({"version": int(mv.version), "aliases": ", ".join(aliases.get(str(mv.version), [])),
                     "family": t.get("model_family"), "features": t.get("feature_set"),
                     "imbalance": t.get("imbalance"), "threshold": t.get("threshold"),
                     "cv_pr_auc": t.get("cv_pr_auc"), "holdout_pr_auc": t.get("holdout_pr_auc"),
                     "holdout_recall": t.get("holdout_recall"), "holdout_precision": t.get("holdout_precision"),
                     "data_version": t.get("data_version"), "description": mv.description})
    return pd.DataFrame(rows).sort_values("version", ascending=False)


def prepare(raw_or_clean: pd.DataFrame) -> pd.DataFrame:
    """Accepts raw AI4I columns or clean snake_case columns; returns clean columns + engineered features."""
    df = raw_or_clean.copy()
    if "Type" in df.columns:
        df = rename_raw(df)
    return add_engineered_features(df)


def score(model, df_clean: pd.DataFrame, threshold: float) -> pd.DataFrame:
    X = df_clean[CATEGORICAL + RAW_NUMERIC + ENGINEERED]
    out = df_clean.copy()
    out["failure_probability"] = model.predict_proba(X)[:, 1]
    out["alert"] = out["failure_probability"] >= threshold
    return out.sort_values("failure_probability", ascending=False)


def likely_mode(row: pd.Series) -> str:
    """Plain-language hint about which documented failure mechanism the readings are closest to."""
    hints = []
    if row["temp_diff_k"] < 8.6 and row["rot_speed_rpm"] < 1380:
        hints.append("heat dissipation (small temp. gap + low speed)")
    if row["power_w"] < 3500 or row["power_w"] > 9000:
        hints.append("power outside 3,500-9,000 W")
    limit = {"L": 11000, "M": 12000, "H": 13000}.get(row["type"], 11000)
    if row["strain_min_nm"] > limit * 0.9:
        hints.append("overstrain (tool wear x torque near limit)")
    if row["tool_wear_min"] >= 200:
        hints.append("tool wear >= 200 min")
    return "; ".join(hints) if hints else "no single driver - combination of readings"


def shift_outcome(scored: pd.DataFrame) -> dict | None:
    """If the shift has true labels, count caught failures, misses, and false alarms."""
    if "machine_failure" not in scored.columns:
        return None
    y = scored["machine_failure"].astype(int).to_numpy()
    a = scored["alert"].to_numpy()
    return {"failures": int(y.sum()), "caught": int(np.sum(a & (y == 1))), "missed": int(np.sum(~a & (y == 1))),
            "false_alarms": int(np.sum(a & (y == 0))), "alerts": int(a.sum()), "runs": int(len(y))}
