#!/bin/bash
# Step 6: reset the demo before recording (removes shifts 19-20 and the sensor-glitch file so you can ingest them live)
cd "$(dirname "$0")" || exit 1
source scripts/common.sh
python -m src.simulate --reset
finish "DEMO RESET - restart 5_open_app.command (or click Rerun in the app)"
