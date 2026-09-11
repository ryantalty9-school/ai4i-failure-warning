"""Model building, cross-validation, cost-based thresholds, and evaluation metrics.
Pure functions (no MLflow here) so they are easy to test and reuse in the app."""
from __future__ import annotations

import ctypes
import glob
import os
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL, FEATURE_SETS, TARGET, model_inputs

FAMILY_LABELS = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost",
                 "hgb": "Gradient Boosting (sklearn)"}
SIMPLICITY = {"logreg": 0, "rf": 1, "hgb": 2, "xgb": 2}   # tie-breaker: simpler model wins


# ----------------------------------------------------------------------------- availability
def _preload_libomp() -> None:
    """XGBoost on macOS needs libomp. scikit-learn ships a copy; loading it first often lets XGBoost import."""
    try:
        import sklearn
        for p in glob.glob(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs", "libomp*.dylib")):
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


@lru_cache(maxsize=1)
def xgboost_available() -> bool:
    _preload_libomp()
    try:
        import xgboost as xgb
        xgb.XGBClassifier(n_estimators=2, verbosity=0).fit(np.random.rand(20, 3), [0, 1] * 10)
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def smote_available() -> bool:
    try:
        from imblearn.over_sampling import SMOTE  # noqa: F401
        from imblearn.pipeline import Pipeline as _P  # noqa: F401
        return True
    except Exception:
        return False


def boosting_family() -> str:
    """'xgb' when XGBoost works on this machine, otherwise sklearn's histogram gradient boosting."""
    return "xgb" if xgboost_available() else "hgb"


# ----------------------------------------------------------------------------- pipelines
def default_params(family: str) -> dict:
    return {
        "logreg": {"C": 1.0, "max_iter": 2000},
        "rf": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
        "xgb": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 5, "subsample": 0.8,
                "colsample_bytree": 0.8},
        "hgb": {"max_iter": 300, "learning_rate": 0.1, "max_depth": 5, "l2_regularization": 0.0},
    }[family]


def make_estimator(family: str, imbalance: str, params: dict, y: pd.Series, seed: int):
    weighted = imbalance == "class_weight"
    if family == "logreg":
        return LogisticRegression(class_weight="balanced" if weighted else None, random_state=seed, **params)
    if family == "rf":
        return RandomForestClassifier(class_weight="balanced_subsample" if weighted else None,
                                      n_jobs=-1, random_state=seed, **params)
    if family == "hgb":
        return HistGradientBoostingClassifier(class_weight="balanced" if weighted else None,
                                              random_state=seed, **params)
    if family == "xgb":
        import xgboost as xgb
        pos = max(int(np.sum(y)), 1)
        spw = (len(y) - pos) / pos if weighted else 1.0
        return xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", tree_method="hist",
                                 n_jobs=-1, random_state=seed, verbosity=0, **params)
    raise ValueError(f"unknown model family: {family}")


def build_pipeline(family: str, feature_set: str, imbalance: str, params: dict, y: pd.Series, seed: int):
    numeric = FEATURE_SETS[feature_set]
    prep = ColumnTransformer([
        ("num", StandardScaler() if family == "logreg" else "passthrough", numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    model = make_estimator(family, imbalance, params, y, seed)
    if imbalance == "smote":
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline([("prep", prep), ("smote", SMOTE(random_state=seed)), ("model", model)])
    return Pipeline([("prep", prep), ("model", model)])


# ----------------------------------------------------------------------------- data split
def split_train_test(df: pd.DataFrame, test_size: float, seed: int):
    """Stratified split. Same seed -> same split every time (training and registration agree)."""
    train, test = train_test_split(df, test_size=test_size, stratify=df[TARGET], random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def xy(df: pd.DataFrame):
    return model_inputs(df), df[TARGET].astype(int)


# ----------------------------------------------------------------------------- metrics
def cost_per_1k(y, pred, costs: dict) -> float:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    total = fn * costs["missed_failure"] + tp * costs["caught_failure"] + fp * costs["false_alarm"]
    return float(total / len(y) * 1000)


def no_model_cost_per_1k(y, costs: dict) -> float:
    """Status quo: every failure is a surprise."""
    return float(np.mean(y) * costs["missed_failure"] * 1000)


def metrics_at(y, proba, threshold: float, costs: dict) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "cost_per_1k_runs": cost_per_1k(y, pred, costs),
        "alerts_per_1k_runs": float(pred.mean() * 1000),
    }


def pick_threshold(y, proba, costs: dict, min_precision: float) -> tuple[float, bool]:
    """Lowest expected cost among thresholds that keep precision >= min_precision
    (so technicians are not buried in false alarms). Falls back to the lowest-cost threshold."""
    grid = np.round(np.arange(0.02, 0.981, 0.01), 2)
    rows = []
    for t in grid:
        m = metrics_at(y, proba, t, costs)
        rows.append((t, m["cost_per_1k_runs"], m["precision"], m["recall"]))
    feasible = [r for r in rows if r[2] >= min_precision and r[3] > 0]
    pool = feasible or rows
    best = min(pool, key=lambda r: (round(r[1], 2), -r[0]))
    return float(best[0]), bool(feasible)


def cross_validate(family, feature_set, imbalance, params, X, y, folds: int, seed: int) -> dict:
    """Stratified k-fold CV. Returns out-of-fold probabilities plus per-fold PR-AUC / ROC-AUC."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    pr, roc = [], []
    t0 = time.perf_counter()
    for tr, va in skf.split(X, y):
        pipe = build_pipeline(family, feature_set, imbalance, params, y.iloc[tr], seed)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        pr.append(average_precision_score(y.iloc[va], p))
        roc.append(roc_auc_score(y.iloc[va], p))
    return {"oof": oof, "fold_pr_auc": pr, "fold_roc_auc": roc,
            "cv_time_s": time.perf_counter() - t0}


def fit_final(family, feature_set, imbalance, params, X, y, seed: int):
    pipe = build_pipeline(family, feature_set, imbalance, params, y, seed)
    t0 = time.perf_counter()
    pipe.fit(X, y)
    fit_s = time.perf_counter() - t0
    one = X.iloc[[0]]
    t0 = time.perf_counter()
    for _ in range(50):
        pipe.predict_proba(one)
    latency_ms = (time.perf_counter() - t0) / 50 * 1000
    return pipe, fit_s, latency_ms


def feature_importance(pipe) -> pd.Series:
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    model = pipe.named_steps["model"]
    if hasattr(model, "coef_"):
        vals = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        return pd.Series(dtype=float)
    return pd.Series(vals, index=names).sort_values(ascending=False)


def evaluate_on(pipe, X, y, threshold: float, costs: dict) -> dict:
    p = pipe.predict_proba(X)[:, 1]
    out = metrics_at(y, p, threshold, costs)
    out["pr_auc"] = float(average_precision_score(y, p))
    out["roc_auc"] = float(roc_auc_score(y, p))
    tn, fp, fn, tp = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1]).ravel()
    out.update(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))
    return out
