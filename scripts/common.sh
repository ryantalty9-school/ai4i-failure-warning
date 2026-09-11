# Shared setup for the numbered .command scripts (sourced, not run directly)
VENV="$HOME/.venvs/ai4i-failure-warning"
if [ ! -f "$VENV/bin/activate" ]; then
  echo "ERROR: the Python environment is missing. Double-click 1_setup.command first."
  read -r -p "Press Return to close..."; exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH="$PWD"
export GX_ANALYTICS_ENABLED=false
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
step() { echo; echo "=================== $1 ==================="; }
finish() { echo; echo "=== $1 ==="; date; read -r -p "Press Return to close this window..."; }

# XGBoost on macOS needs the OpenMP runtime (libomp). Homebrew is not installed, so point XGBoost at the
# copy of libomp that ships inside scikit-learn (changes only files inside the project's virtual environment).
xgb_ok() { python -c "import xgboost, numpy as np; xgboost.XGBClassifier(n_estimators=2).fit(np.random.rand(20,3), [0,1]*10)" >/dev/null 2>&1; }
ensure_xgboost() {
  if xgb_ok; then echo "xgboost: OK"; return; fi
  SP=$(python -c "import site; print(site.getsitepackages()[0])")
  LIB="$SP/xgboost/lib/libxgboost.dylib"
  OMP=$(ls "$SP"/sklearn/.dylibs/libomp*.dylib 2>/dev/null | head -1)
  echo "xgboost: not loading, trying to link it to scikit-learn's libomp"
  echo "  lib=$LIB"; echo "  omp=$OMP"
  if [ -f "$LIB" ] && [ -n "$OMP" ]; then
    OLD=$(otool -L "$LIB" | awk '/libomp/{print $1; exit}')
    echo "  current libomp reference: $OLD"
    [ -n "$OLD" ] && install_name_tool -change "$OLD" "$OMP" "$LIB" 2>&1
    codesign --force --sign - "$LIB" 2>&1
  fi
  if xgb_ok; then echo "xgboost: OK after fix"; else echo "xgboost: still unavailable -> sklearn gradient boosting will be used"; fi
}
