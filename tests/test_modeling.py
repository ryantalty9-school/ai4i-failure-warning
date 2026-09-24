"""Tests for the cost metric, threshold choice, and the train/test split (src/modeling.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.features import TARGET, add_engineered_features, rename_raw
from src.modeling import (build_pipeline, cost_per_1k, metrics_at, no_model_cost_per_1k, pick_threshold,
                          split_train_test, xy)
from tests.conftest import make_batch

CFG = load_config()
COSTS = CFG["costs"]


def test_cost_per_1k_matches_the_cost_table():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    pred = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])   # 1 caught, 1 missed, 1 false alarm
    expected = (COSTS["caught_failure"] + COSTS["missed_failure"] + COSTS["false_alarm"]) / 10 * 1000
    assert cost_per_1k(y, pred, COSTS) == expected


def test_no_model_cost_is_every_failure_missed():
    y = np.array([1, 0, 0, 0])
    assert no_model_cost_per_1k(y, COSTS) == 0.25 * COSTS["missed_failure"] * 1000


def test_metrics_at_threshold():
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.4, 0.8, 0.1])
    m = metrics_at(y, proba, 0.5, COSTS)
    assert m["recall"] == 0.5                      # 1 of 2 failures caught
    assert m["precision"] == 0.5                   # 1 of 2 alerts is real
    assert m["alerts_per_1k_runs"] == 500


def test_threshold_respects_the_precision_floor():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], [960, 40])
    proba = np.where(y == 1, rng.uniform(0.4, 1.0, 1000), rng.uniform(0.0, 0.6, 1000))
    thr, precision_ok = pick_threshold(y, proba, COSTS, CFG["targets"]["min_precision"])
    assert precision_ok
    assert metrics_at(y, proba, thr, COSTS)["precision"] >= CFG["targets"]["min_precision"]


def test_threshold_falls_back_when_no_threshold_can_hit_the_floor():
    y = np.repeat([0, 1], [990, 10])
    proba = np.full(1000, 0.5)                     # a useless model: every run looks identical
    thr, precision_ok = pick_threshold(y, proba, COSTS, 0.9)
    assert precision_ok is False
    assert 0.0 < thr < 1.0


def test_split_is_deterministic_and_stratified():
    df = add_engineered_features(rename_raw(make_batch(n=500, failures=50)))
    a_train, a_test = split_train_test(df, 0.2, CFG["seed"])
    b_train, b_test = split_train_test(df, 0.2, CFG["seed"])
    assert a_test["udi"].tolist() == b_test["udi"].tolist(), "same seed must give the same hold-out set"
    assert len(a_test) == 100 and len(a_train) == 400
    assert set(a_train["udi"]).isdisjoint(set(a_test["udi"]))
    assert abs(a_test[TARGET].mean() - df[TARGET].mean()) < 0.01


def test_pipeline_trains_and_predicts_probabilities():
    df = add_engineered_features(rename_raw(make_batch(n=300, failures=30)))
    X, y = xy(df)
    pipe = build_pipeline("logreg", "engineered", "class_weight", {"C": 1.0, "max_iter": 500}, y, CFG["seed"])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert len(proba) == len(df)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_ignores_unseen_product_types():
    """A product type the model never saw must not crash scoring (handle_unknown='ignore')."""
    df = add_engineered_features(rename_raw(make_batch(n=200, failures=20)))
    X, y = xy(df)
    pipe = build_pipeline("logreg", "engineered", "none", {"C": 1.0, "max_iter": 500}, y, CFG["seed"])
    pipe.fit(X, y)
    odd = X.head(3).copy()
    odd["type"] = "Z"
    assert len(pipe.predict_proba(odd)) == 3
