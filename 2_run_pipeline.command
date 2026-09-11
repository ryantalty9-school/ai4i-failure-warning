#!/bin/bash
# Step 2: DataOps pipeline -> ingest shifts 1-18, validate (Great Expectations), process to Parquet, version with Git + DVC
cd "$(dirname "$0")" || exit 1
mkdir -p logs
exec > >(tee logs/2_run_pipeline.log) 2>&1
set -o pipefail
source scripts/common.sh
date
ensure_xgboost

step "1/4 INGEST (batch replay of the sensor feed, shifts 1-18)"
python -m src.ingest --through 18 || { finish "FAILED at ingest"; exit 1; }

step "2/4 VALIDATE (Great Expectations + quarantine)"
python -m src.validate || { finish "FAILED at validation"; exit 1; }

step "3/4 PROCESS (clean + engineered features -> Parquet)"
python -m src.process || { finish "FAILED at processing"; exit 1; }

step "4/4 VERSION (Git + DVC)"
[ -d .git ] || git init -b main
[ -n "$(git config user.name)" ] || git config user.name "Ryan Talty"
[ -n "$(git config user.email)" ] || git config user.email "ryantalty@users.noreply.github.com"
if [ ! -d .dvc ]; then dvc init -q && dvc config core.analytics false; fi
STORE="$HOME/dvc-storage/ai4i-failure-warning"
mkdir -p "$STORE"
dvc remote list | grep -q localstore || dvc remote add -d localstore "$STORE"
dvc add data/source/ai4i2020.csv data/raw data/processed/train_table.parquet
git add -A
git commit -q -m "data: ingest, validate, and process shifts 1-18" && echo "git commit created" || echo "(no changes to commit)"
if ! git tag --points-at HEAD | grep -q '^data-v'; then
  N=$(git tag -l 'data-v1.*' | wc -l | tr -d ' ')
  git tag -a "data-v1.$N" -m "training table from shifts 1-18" && echo "tagged data-v1.$N"
fi
dvc push
echo; git log --oneline -5; echo "tags: $(git tag | tr '\n' ' ')"
cat data/processed/train_table.parquet.dvc

finish "PIPELINE FINISHED"
