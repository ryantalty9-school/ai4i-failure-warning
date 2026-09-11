#!/bin/bash
# Step 3: ModelOps -> run all MLflow experiments, then pick the champion and version models in the MLflow Model Registry
cd "$(dirname "$0")" || exit 1
mkdir -p logs
exec > >(tee logs/3_run_experiments.log) 2>&1
set -o pipefail
source scripts/common.sh
date
ensure_xgboost
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

step "TRAIN (4 objectives, 5-fold CV, everything logged to MLflow)"
python -m src.train || { finish "FAILED during training"; exit 1; }

step "SELECT + HOLD-OUT + REGISTER"
python -m src.register || { finish "FAILED during registration"; exit 1; }

git add reports && git commit -q -m "experiments: training runs, leaderboard, and model selection" && echo "reports committed to git"
finish "EXPERIMENTS FINISHED - double-click 4_open_mlflow.command to see the results"
