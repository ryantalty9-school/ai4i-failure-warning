"""Step 5 - Compare runs, score the hold-out set once, and version models in the MLflow Model Registry.

Selection rules (from the experiment tracking plan):
  1. rank runs by cross-validated PR-AUC
  2. gates at each run's chosen threshold: recall >= 0.85 and precision >= 0.60
  3. if two runs are within 0.005 PR-AUC, the lower expected cost per 1,000 runs wins,
     then the simpler model
  4. the winner must beat the logistic regression baseline on the hold-out set

Three versions are registered under one model name so they can be compared side by side:
  alias "baseline"   -> best logistic regression
  alias "challenger" -> best random forest
  alias "champion"   -> the winner (used by the Streamlit app)

Usage:
    python -m src.register
"""
from __future__ import annotations

import json

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import data_version, load_config, path
from src.modeling import FAMILY_LABELS, SIMPLICITY, evaluate_on, split_train_test, xy
from src.tracking import author, setup_mlflow

TIE = 0.005


def leaderboard(exp_id: str, cfg: dict) -> pd.DataFrame:
    runs = mlflow.search_runs([exp_id], filter_string="attributes.status = 'FINISHED'", max_results=5000)
    if runs.empty:
        raise SystemExit("No finished runs found. Run src.train first.")
    runs = runs[runs.get("tags.model_uri").notna()].copy()           # skips parent / evaluation runs
    runs = runs[runs.get("tags.objective", "").astype(str).str.match(r"^[1-4]_")]
    t = cfg["targets"]
    runs["passes_gates"] = (runs["metrics.recall"] >= t["min_recall"]) & (runs["metrics.precision"] >= t["min_precision"])
    runs["simplicity"] = runs["tags.model_family"].map(SIMPLICITY).fillna(9)
    cols = {"tags.mlflow.runName": "run", "tags.objective": "objective", "tags.model_family": "family",
            "tags.feature_set": "features", "tags.imbalance": "imbalance",
            "metrics.cv_pr_auc_mean": "cv_pr_auc", "metrics.cv_pr_auc_std": "cv_pr_auc_std",
            "metrics.cv_roc_auc_mean": "cv_roc_auc", "metrics.recall": "recall", "metrics.precision": "precision",
            "metrics.f2": "f2", "metrics.threshold": "threshold", "metrics.cost_per_1k_runs": "cost_per_1k",
            "metrics.predict_latency_ms": "latency_ms", "run_id": "run_id", "tags.model_uri": "model_uri",
            "tags.run_base": "run_base", "passes_gates": "passes_gates", "simplicity": "simplicity",
            "start_time": "start_time"}
    lb = runs[list(cols)].rename(columns=cols)
    # re-runs create new versions of the same configuration: keep the latest of each
    lb = lb.sort_values("start_time").drop_duplicates(subset="run_base", keep="last")
    return lb.sort_values(["passes_gates", "cv_pr_auc"], ascending=[False, False]).reset_index(drop=True)


def pick_winner(lb: pd.DataFrame) -> tuple[pd.Series, bool]:
    pool = lb[lb["passes_gates"]]
    gates_met = not pool.empty
    if pool.empty:
        pool = lb
    top = pool["cv_pr_auc"].max()
    close = pool[pool["cv_pr_auc"] >= top - TIE].sort_values(["cost_per_1k", "simplicity"])
    return close.iloc[0], gates_met


def best_of(lb: pd.DataFrame, family: str):
    sub = lb[lb["family"] == family]
    return None if sub.empty else sub.sort_values("cv_pr_auc", ascending=False).iloc[0]


def main():
    cfg = load_config()
    exp_id = setup_mlflow()
    client = MlflowClient()
    name = cfg["mlflow"]["registered_model"]
    costs, dv = cfg["costs"], data_version()

    lb = leaderboard(exp_id, cfg)
    winner, gates_met = pick_winner(lb)
    candidates = {"baseline": best_of(lb, "logreg"), "challenger": best_of(lb, "rf"), "champion": winner}

    # ---- hold-out test set: scored once, for the shortlisted models only
    table = pd.read_parquet(path(cfg["data"]["processed_path"]))
    _train, test = split_train_test(table, cfg["split"]["test_size"], cfg["seed"])
    X_test, y_test = xy(test)
    holdout = {}
    with mlflow.start_run(run_name="holdout-evaluation") as ev:
        mlflow.set_tags({"objective": "5_holdout", "author": author(), "data_version": dv["data_version"],
                         "git_commit": dv["git_commit"],
                         "mlflow.note.content": "One-time scoring of the shortlisted models on the untouched 20% "
                                                "hold-out set, using each model's own cost-based threshold."})
        mlflow.log_params({"test_rows": len(test), "test_failures": int(y_test.sum())})
        for role, row in candidates.items():
            if row is None:
                continue
            model = mlflow.sklearn.load_model(row["model_uri"])
            m = evaluate_on(model, X_test, y_test, float(row["threshold"]), costs)
            holdout[role] = m
            mlflow.log_metrics({f"{role}_{k}": v for k, v in m.items()})
            mlflow.set_tag(f"{role}_run", row["run"])
            print(f"hold-out {role:<10} {row['run']:<38} PR-AUC {m['pr_auc']:.3f}  recall {m['recall']:.2f}  "
                  f"precision {m['precision']:.2f}  caught {m['tp']}/{m['tp'] + m['fn']}  false alarms {m['fp']}")
        beats = holdout["champion"]["pr_auc"] > holdout.get("baseline", {"pr_auc": 0})["pr_auc"]
        mlflow.set_tag("champion_beats_baseline", str(beats))
        evaluation_run = ev.info.run_id

    # ---- Model Registry: one registered model, three documented versions
    try:
        client.create_registered_model(name, description=(
            "Predicts the probability that a milling machine run ends in failure (AI4I 2020). "
            "Aliases: champion = model used by the app, baseline = best logistic regression, "
            "challenger = best random forest. Alert when probability >= the version's 'threshold' tag."))
    except Exception:
        pass
    existing = {v.run_id: v for v in client.search_model_versions(f"name='{name}'")}
    registered = {}
    for role in ["baseline", "challenger", "champion"]:
        row = candidates[role]
        if row is None:
            continue
        mv = existing.get(row["run_id"])
        if mv is None:
            mv = mlflow.register_model(row["model_uri"], name)
            existing[row["run_id"]] = mv
        h = holdout[role]
        desc = (f"{role.upper()}: {FAMILY_LABELS.get(row['family'], row['family'])}, {row['features']} features, "
                f"imbalance={row['imbalance']}. Source run {row['run']}. Alert threshold {row['threshold']:.2f}. "
                f"CV PR-AUC {row['cv_pr_auc']:.3f} (+/- {row['cv_pr_auc_std']:.3f}); hold-out PR-AUC {h['pr_auc']:.3f}, "
                f"recall {h['recall']:.2f}, precision {h['precision']:.2f}, cost per 1,000 runs ${h['cost_per_1k_runs']:,.0f}. "
                f"Trained on data {dv['data_version']} (commit {dv['git_commit']}).")
        client.update_model_version(name, mv.version, description=desc)
        for k, v in {"role": role, "threshold": f"{row['threshold']:.2f}", "model_family": row["family"],
                     "feature_set": row["features"], "imbalance": row["imbalance"],
                     "cv_pr_auc": f"{row['cv_pr_auc']:.4f}", "holdout_pr_auc": f"{h['pr_auc']:.4f}",
                     "holdout_recall": f"{h['recall']:.3f}", "holdout_precision": f"{h['precision']:.3f}",
                     "data_version": dv["data_version"], "source_run": row["run"],
                     "evaluation_run_id": evaluation_run}.items():
            client.set_model_version_tag(name, mv.version, k, v)
        client.set_registered_model_alias(name, role, mv.version)
        registered[role] = {"version": int(mv.version), "run": row["run"], "threshold": float(row["threshold"])}
        print(f"registered {name} v{mv.version} -> alias '{role}' ({row['run']})")

    # ---- reports for the slides / README
    rep = path("reports")
    lb.drop(columns=["model_uri", "simplicity", "start_time", "run_base"]).to_csv(rep / "leaderboard.csv", index=False)
    summary = {"winner_run": winner["run"], "winner_family": winner["family"], "gates_met": gates_met,
               "champion_beats_baseline_on_holdout": bool(beats), "holdout": holdout, "registered": registered,
               "data_version": dv, "targets": cfg["targets"], "costs": costs}
    (rep / "model_selection.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwinner: {winner['run']}  (gates met: {gates_met}; beats baseline on hold-out: {beats})")
    print("leaderboard (top 10):")
    print(lb[["run", "objective", "cv_pr_auc", "recall", "precision", "threshold", "cost_per_1k", "passes_gates"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
