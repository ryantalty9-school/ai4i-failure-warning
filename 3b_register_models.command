#!/bin/bash
# Step 3b: re-run only the selection + hold-out + Model Registry step (no retraining)
cd "$(dirname "$0")" || exit 1
mkdir -p logs
exec > >(tee logs/3b_register_models.log) 2>&1
source scripts/common.sh
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"
date
step "SELECT + HOLD-OUT + REGISTER"
python -m src.register || { finish "FAILED during registration"; exit 1; }
git add -A && git commit -q -m "registry: baseline, champion, and challenger versions" && echo "committed to git"
finish "REGISTRATION FINISHED"
