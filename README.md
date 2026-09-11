# Machine Failure Early Warning System (BANA 7075 prototype)

A predictive maintenance ML system that flags milling-machine runs at high risk of failure so maintenance can
step in before a breakdown stops the line. Built on the AI4I 2020 Predictive Maintenance dataset
(Matzka, 2020; UCI ML Repository, CC BY 4.0): 10,000 runs, 3.4% of which end in failure.

## What the prototype does

| Layer | What happens | Tools |
|---|---|---|
| **Ingestion** | The dataset is replayed in order as 20 "shifts" of 500 runs. Each shift lands untouched in an append-only raw zone (`data/raw/`), with a manifest (rows, UDI range, MD5, timestamp). | pandas |
| **Validation** | Every shift runs through a Great Expectations suite (schema, nulls, physical ranges, product type, 0/1 labels, unique IDs, process > air temperature, failure-rate and mean checks). Bad rows go to `data/quarantine/` with the reason attached; reports go to `reports/validation/`. | Great Expectations |
| **Processing** | Validated shifts 1-18 are combined, renamed, de-duplicated, and enriched with engineered features (temperature difference, mechanical power, strain), then stored as Parquet. | pandas, pyarrow |
| **Versioning** | The source file, raw zone, and training table are tracked with DVC (local remote); code, `.dvc` pointers, and reports are tracked in Git with a `data-v1.x` tag. | Git, DVC |
| **Experiment tracking** | One MLflow experiment, four objectives: model comparison, raw vs engineered features, class-imbalance handling, tuning grid (nested runs). Each run logs tags, description, dataset, parameters, 5-fold CV metrics, a cost-based threshold, plots, and the model. | MLflow, scikit-learn, XGBoost, imbalanced-learn |
| **Model registry** | The best model per the selection rules is scored once on the untouched 20% hold-out set and registered as `ai4i-failure-model`, alongside the best logistic regression (baseline) and the best model from another family (challenger). Aliases: `champion`, `baseline`, `challenger`. | MLflow Model Registry |
| **App** | Streamlit app that loads the `champion` model from the registry: ingest the next shift, validate it, and rank machines by failure risk; check a single machine; view the registry and leaderboard. | Streamlit |

## How to run it (macOS)

Double-click the scripts in order from Finder. Each one writes a log to `logs/`.

1. `1_setup.command` - creates `~/.venvs/ai4i-failure-warning`, installs `requirements.txt`, downloads the dataset.
2. `2_run_pipeline.command` - ingest shifts 1-18, validate, process, and version (Git + DVC).
3. `3_run_experiments.command` - run every MLflow experiment, then select, hold-out test, and register models.
4. `4_open_mlflow.command` - MLflow UI at http://127.0.0.1:5050 (experiments, run comparison, registry).
5. `5_open_app.command` - Streamlit app at http://localhost:8501.

Everything can also be run by hand from the project folder with the environment activated:

```bash
source ~/.venvs/ai4i-failure-warning/bin/activate
python -m src.ingest --through 18
python -m src.validate
python -m src.process
python -m src.train          # or --objective 1|2|3|4
python -m src.register
streamlit run app/streamlit_app.py
```

## Model selection rules

1. Rank runs by cross-validated PR-AUC (accuracy is misleading at a 3.4% failure rate).
2. Gates at each run's threshold: recall >= 0.85 and precision >= 0.60.
3. Runs within 0.005 PR-AUC: lower expected cost per 1,000 runs wins, then the simpler model.
4. The winner is scored once on the hold-out set and must beat the logistic regression baseline.

The alert threshold for each run is the one with the lowest expected cost (missed failure $15,000, planned
repair $3,000, false alarm $500) among thresholds that keep precision >= 0.60. The threshold is stored as a
tag on each registered model version and the app uses it.

## Project layout

```
config.yaml            all settings (paths, validation rules, costs, targets, MLflow names)
src/ingest.py          batch ingestion (append-only raw zone + manifest)
src/validate.py        Great Expectations suite + row-level quarantine
src/process.py         training table + engineered features -> Parquet
src/features.py        column names + feature engineering (shared by training and the app)
src/modeling.py        pipelines, CV, cost-based threshold, metrics
src/train.py           MLflow experiments (4 objectives)
src/register.py        selection, hold-out evaluation, Model Registry + aliases
src/scoring.py         load the champion from the registry and score runs
src/simulate.py        writes a corrupted "sensor glitch" shift to demo validation
app/streamlit_app.py   the prototype app
reports/               validation reports, processing summary, leaderboard, model selection
```

## Notes

- XGBoost needs the OpenMP runtime on macOS. The scripts link it to the copy bundled with scikit-learn; if
  that fails, the boosting model falls back to scikit-learn's `HistGradientBoostingClassifier` and says so in
  the logs and MLflow tags.
- Data files live in DVC, not Git. `dvc pull` restores them from the local DVC remote at
  `~/dvc-storage/ai4i-failure-warning`.
- `mlflow.db` and `mlruns/` are the local MLflow store and are not committed to Git.

## Push to GitHub

Create an empty repository on github.com (no README), then from this folder:

```bash
git remote add origin https://github.com/<your-account>/ai4i-failure-warning.git
git push -u origin main --tags
```
