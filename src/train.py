"""Step 4 - Model training with MLflow experiment tracking.

One MLflow experiment ("AI4I Machine Failure Prediction") with four objectives:
  1_model_comparison     LogReg vs Random Forest vs XGBoost (engineered features, class weights)
  2_feature_engineering  raw sensor readings vs + engineered features
  3_class_imbalance      no adjustment vs class weights vs SMOTE
  4_tuning               grid search on the boosting model (nested runs under one parent)

Every run logs: tags (objective, author, data version, git commit), a description, the dataset,
preprocessing + hyperparameters, 5-fold CV metrics, a cost-based alert threshold, metrics at that
threshold, timing, plots (confusion matrix, PR curve, feature importance), and the fitted model.

The 20% hold-out test set is NOT touched here. It is scored once in src/register.py.

Usage:
    python -m src.train                    # all objectives
    python -m src.train --objective 1      # just one objective
"""
from __future__ import annotations

import argparse
import itertools
import json
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src import plots
from src.config import data_version, load_config, path
from src.features import FEATURE_SETS
from src.modeling import (FAMILY_LABELS, boosting_family, cross_validate, default_params,
                          feature_importance, fit_final, metrics_at, no_model_cost_per_1k, pick_threshold,
                          smote_available, split_train_test, xy)
from src.tracking import author, log_sklearn_model, setup_mlflow

PREPROCESSING = ("drop udi/product_id; drop failure-mode flags twf/hdf/pwf/osf/rnf (label leakage); "
                 "one-hot encode type; standard-scale numeric (logreg only)")


def plan_runs(only: int | None = None) -> list[dict]:
    boost = boosting_family()
    runs = []
    if only in (None, 1):
        for fam in ["logreg", "rf", boost]:
            runs.append(dict(objective="1_model_comparison", family=fam, features="engineered",
                             imbalance="class_weight", params=default_params(fam),
                             why=f"Objective 1: compare {FAMILY_LABELS[fam]} against the other model families."))
    if only in (None, 2):
        for fam, fs in itertools.product(["logreg", boost], ["raw", "engineered"]):
            runs.append(dict(objective="2_feature_engineering", family=fam, features=fs, imbalance="class_weight",
                             params=default_params(fam),
                             why=f"Objective 2: {FAMILY_LABELS[fam]} on {fs} features to measure the value of "
                                 "temperature difference, power, and strain."))
    if only in (None, 3):
        options = ["none", "class_weight"] + (["smote"] if smote_available() else [])
        for imb in options:
            runs.append(dict(objective="3_class_imbalance", family=boost, features="engineered", imbalance=imb,
                             params=default_params(boost),
                             why=f"Objective 3: {FAMILY_LABELS[boost]} with imbalance handling = {imb}."))
    if only in (None, 4):
        if boost == "xgb":
            grid = {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1], "n_estimators": [200, 400]}
        else:
            grid = {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1], "max_iter": [200, 400]}
        keys = list(grid)
        for combo in itertools.product(*grid.values()):
            params = dict(default_params(boost), **dict(zip(keys, combo)))
            runs.append(dict(objective="4_tuning", family=boost, features="engineered", imbalance="class_weight",
                             params=params, grid_keys=keys,
                             why=f"Objective 4: tuning {FAMILY_LABELS[boost]} " +
                                 ", ".join(f"{k}={v}" for k, v in zip(keys, combo))))
    return runs


def run_base_name(r: dict) -> str:
    base = f"{r['family']}-{r['features']}-{r['imbalance'].replace('_', '')}"
    if r.get("grid_keys"):
        short = {"max_depth": "d", "learning_rate": "lr", "n_estimators": "n", "max_iter": "n"}
        base += "-" + "-".join(f"{short[k]}{r['params'][k]}" for k in r["grid_keys"])
    return base


def next_version(exp_id: str, base: str) -> int:
    prior = mlflow.search_runs([exp_id], filter_string=f"tags.run_base = '{base}'", max_results=500)
    return len(prior) + 1


def log_one(r: dict, train: pd.DataFrame, cfg: dict, exp_id: str, dv: dict, nested: bool = False) -> dict:
    seed, folds, costs = cfg["seed"], cfg["split"]["cv_folds"], cfg["costs"]
    X, y = xy(train)
    base = run_base_name(r)
    name = f"{base}-v{next_version(exp_id, base)}"
    with mlflow.start_run(run_name=name, nested=nested) as run:
        mlflow.set_tags({
            "objective": r["objective"], "author": author(), "run_base": base,
            "model_family": r["family"], "feature_set": r["features"], "imbalance": r["imbalance"],
            "data_version": dv["data_version"], "data_md5": dv["data_md5"], "git_commit": dv["git_commit"],
            "mlflow.note.content": r["why"],
        })
        # dataset lineage
        try:
            ds = mlflow.data.from_pandas(train, source=str(path(cfg["data"]["processed_path"])),
                                         name=f"ai4i-train-{dv['data_version']}", targets="machine_failure")
            mlflow.log_input(ds, context="training")
        except Exception as e:
            print(f"  (dataset logging skipped: {type(e).__name__})")
        mlflow.log_params({
            "model_family": r["family"], "feature_set": r["features"], "imbalance": r["imbalance"],
            "features": ",".join(["type"] + FEATURE_SETS[r["features"]]),
            "n_features": len(FEATURE_SETS[r["features"]]) + 1, "preprocessing": PREPROCESSING,
            "cv_folds": folds, "seed": seed, "test_size": cfg["split"]["test_size"],
            "train_rows": len(train), "train_failure_rate": round(float(y.mean()), 4),
            "threshold_rule": f"min cost with precision >= {cfg['targets']['min_precision']}",
            "cost_missed": costs["missed_failure"], "cost_caught": costs["caught_failure"],
            "cost_false_alarm": costs["false_alarm"],
            **{f"model__{k}": v for k, v in r["params"].items()},
        })

        cv = cross_validate(r["family"], r["features"], r["imbalance"], r["params"], X, y, folds, seed)
        thr, precision_ok = pick_threshold(y, cv["oof"], costs, cfg["targets"]["min_precision"])
        at = metrics_at(y, cv["oof"], thr, costs)
        base_cost = no_model_cost_per_1k(y, costs)
        pipe, fit_s, latency_ms = fit_final(r["family"], r["features"], r["imbalance"], r["params"], X, y, seed)
        metrics = {
            "cv_pr_auc_mean": float(np.mean(cv["fold_pr_auc"])), "cv_pr_auc_std": float(np.std(cv["fold_pr_auc"])),
            "cv_roc_auc_mean": float(np.mean(cv["fold_roc_auc"])), **at,
            "no_model_cost_per_1k_runs": base_cost, "savings_per_1k_runs": base_cost - at["cost_per_1k_runs"],
            "meets_recall_target": float(at["recall"] >= cfg["targets"]["min_recall"]),
            "meets_precision_target": float(at["precision"] >= cfg["targets"]["min_precision"]),
            "cv_time_s": cv["cv_time_s"], "fit_time_s": fit_s, "predict_latency_ms": latency_ms,
        }
        mlflow.log_metrics(metrics)
        mlflow.set_tag("threshold_precision_constraint_met", str(precision_ok))

        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            title = f"{name} (CV, threshold {thr:.2f})"
            plots.confusion(y, cv["oof"], thr, t / "confusion_matrix.png", title)
            plots.pr_curve(y, cv["oof"], thr, t / "pr_curve.png", title)
            imp = feature_importance(pipe)
            if len(imp):
                plots.importance(imp, t / "feature_importance.png", f"{name} feature importance")
                imp.rename("importance").to_csv(t / "feature_importance.csv")
            pd.DataFrame({"fold": range(1, folds + 1), "pr_auc": cv["fold_pr_auc"],
                          "roc_auc": cv["fold_roc_auc"]}).to_csv(t / "cv_folds.csv", index=False)
            mlflow.log_artifacts(str(t), artifact_path="evaluation")

        info = log_sklearn_model(pipe, X.head(50), y.head(50))
        mlflow.set_tag("model_uri", info.model_uri)
        print(f"  {name:<40} CV PR-AUC {metrics['cv_pr_auc_mean']:.3f}  recall {at['recall']:.2f}  "
              f"precision {at['precision']:.2f}  thr {thr:.2f}  cost/1k ${at['cost_per_1k_runs']:,.0f}")
        return {"run_id": run.info.run_id, "name": name, **metrics}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", type=int, choices=[1, 2, 3, 4])
    args = ap.parse_args()

    cfg = load_config()
    exp_id = setup_mlflow()
    dv = data_version()
    table = pd.read_parquet(path(cfg["data"]["processed_path"]))
    train, _test = split_train_test(table, cfg["split"]["test_size"], cfg["seed"])
    boost = boosting_family()
    print(f"training rows={len(train)} (hold-out {len(_test)} kept aside)  data={dv['data_version']} "
          f"commit={dv['git_commit']}  boosting model={FAMILY_LABELS[boost]}")
    if boost != "xgb":
        print("  NOTE: XGBoost could not load on this machine, so sklearn HistGradientBoosting is used instead.")

    results = []
    planned = plan_runs(args.objective)
    by_obj = {}
    for r in planned:
        by_obj.setdefault(r["objective"], []).append(r)
    for obj, runs in by_obj.items():
        print(f"\n== {obj} ({len(runs)} runs) ==")
        if obj == "4_tuning":
            with mlflow.start_run(run_name=f"{boost}-tuning-grid") as parent:
                mlflow.set_tags({"objective": obj, "is_parent": "true", "author": author(),
                                 "data_version": dv["data_version"], "git_commit": dv["git_commit"],
                                 "mlflow.note.content": "Parent run for the tuning grid; child runs hold results."})
                kids = [log_one(r, train, cfg, exp_id, dv, nested=True) for r in runs]
                best = max(kids, key=lambda k: k["cv_pr_auc_mean"])
                mlflow.log_metric("best_child_cv_pr_auc", best["cv_pr_auc_mean"])
                mlflow.set_tag("best_child_run", best["name"])
                results += kids
                _ = parent
        else:
            results += [log_one(r, train, cfg, exp_id, dv) for r in runs]

    out = path("reports/training_runs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n{len(results)} runs logged to MLflow experiment '{cfg['mlflow']['experiment_name']}'")


if __name__ == "__main__":
    main()
