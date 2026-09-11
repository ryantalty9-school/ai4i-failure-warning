#!/bin/bash
# ------------------------------------------------------------------
# Step 1: one-time setup for the AI4I Machine Failure Early Warning prototype
# - finds a Python 3.10+ install
# - creates a virtual environment OUTSIDE OneDrive (~/.venvs/ai4i-failure-warning)
# - installs the packages in requirements.txt
# - downloads the AI4I 2020 dataset from the UCI Machine Learning Repository
# Everything printed here is also saved to logs/1_setup.log
# ------------------------------------------------------------------
cd "$(dirname "$0")" || exit 1
mkdir -p logs
exec > >(tee logs/1_setup.log) 2>&1

echo "=== AI4I prototype setup ==="
date
echo "--- system ---"
sw_vers 2>/dev/null
uname -m
echo "xcode tools: $(xcode-select -p 2>&1)"
echo "git: $(command -v git) $(git --version 2>&1)"
echo "brew: $(command -v brew || echo 'not installed')"

echo "--- python ---"
PY=""
for c in python3.12 python3.11 python3.10 python3.13 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
    echo "found $c -> $v"
    if [ -n "$v" ] && [ -z "$PY" ]; then PY="$c"; PYV="$v"; fi
  fi
done
if [ -z "$PY" ]; then
  echo "ERROR: no working Python found. Install Python 3.12 from https://www.python.org/downloads/macos/ and run this again."
  read -r -p "Press Return to close..."
  exit 1
fi
echo "using $PY ($PYV)"

VENV="$HOME/.venvs/ai4i-failure-warning"
if [ ! -d "$VENV" ]; then
  echo "--- creating virtual environment at $VENV ---"
  "$PY" -m venv "$VENV" || { echo "ERROR: venv creation failed"; read -r -p "Press Return..."; exit 1; }
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python --version

echo "--- installing packages (this can take 3-6 minutes) ---"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
PIP_STATUS=$?
echo "pip exit code: $PIP_STATUS"

echo "--- dataset ---"
mkdir -p data/source
if [ ! -f data/source/ai4i2020.csv ]; then
  curl -L --fail -o data/source/ai4i2020.zip \
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip" \
    && unzip -o data/source/ai4i2020.zip -d data/source
fi
ls -l data/source

echo "--- import check ---"
python - <<'EOF'
import importlib, platform
print("python", platform.python_version())
for m in ["pandas", "numpy", "sklearn", "pyarrow", "yaml", "matplotlib", "xgboost",
          "imblearn", "great_expectations", "dvc", "mlflow", "streamlit"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m:20s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"FAIL {m:20s} {type(e).__name__}: {str(e)[:200]}")
try:
    import numpy as np, xgboost as xgb
    xgb.XGBClassifier(n_estimators=2).fit(np.random.rand(20, 3), [0, 1] * 10)
    print("OK   xgboost can train")
except Exception as e:
    print("FAIL xgboost train:", type(e).__name__, str(e)[:300])
EOF

echo "=== SETUP FINISHED ==="
date
read -r -p "Press Return to close this window..."
