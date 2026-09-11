"""MLflow setup shared by training, registration, and the app."""
from __future__ import annotations

import getpass
import inspect
import warnings

import mlflow

from src.config import ROOT, git, load_config, use_project_root

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")


def setup_mlflow() -> str:
    """Point MLflow at the local SQLite tracking store and return the experiment id."""
    use_project_root()
    cfg = load_config()["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    exp = mlflow.get_experiment_by_name(cfg["experiment_name"])
    if exp is None:
        exp_id = mlflow.create_experiment(
            cfg["experiment_name"],
            artifact_location=(ROOT / "mlruns").resolve().as_uri(),
            tags={"mlflow.note.content": cfg["experiment_description"], "project": "ai4i-failure-warning"},
        )
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def author() -> str:
    return git("config", "user.name") or getpass.getuser()


def log_sklearn_model(pipe, X_example, y_example):
    """Log a fitted pipeline with a signature and input example (works on MLflow 2.x and 3.x)."""
    from mlflow.models import infer_signature
    sig = infer_signature(X_example, pipe.predict(X_example))
    kw = {"name": "model"} if "name" in inspect.signature(mlflow.sklearn.log_model).parameters else {"artifact_path": "model"}
    return mlflow.sklearn.log_model(pipe, signature=sig, input_example=X_example.head(3), **kw)
