#!/bin/bash
# Step 5: open the Streamlit app (serves the champion model from the MLflow registry). Close this window to stop it.
cd "$(dirname "$0")" || exit 1
source scripts/common.sh
echo "Starting the Streamlit app at http://localhost:8501  (close this window or press Ctrl+C to stop)"
( sleep 7; open "http://localhost:8501" ) &
streamlit run app/streamlit_app.py --server.headless true --browser.gatherUsageStats false 2>&1 | tee logs/5_app.log
