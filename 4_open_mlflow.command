#!/bin/bash
# Step 4: open the MLflow UI (experiments, run comparison, model registry). Close this window to stop it.
cd "$(dirname "$0")" || exit 1
source scripts/common.sh
echo "Starting MLflow UI at http://127.0.0.1:5050  (close this window or press Ctrl+C to stop)"
( sleep 6; open "http://127.0.0.1:5050" ) &
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5050 2>&1 | tee logs/4_mlflow_ui.log
