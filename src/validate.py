"""Step 2 - Data validation with Great Expectations.

Every raw shift is checked before anything downstream can use it:
  1) schema validation      - the 14 expected columns, in order
  2) content validation     - no nulls, sensor values in physical ranges, product type in {L, M, H},
                              0/1 labels, unique UDI
  3) cross-field validation - process temperature must be above air temperature;
                              failure label should agree with the failure-mode flags (warning)
  4) statistical validation - failure rate and mean air temperature within expected bounds

The same rules drive two things: a Great Expectations suite (the formal validation record saved
in reports/validation/) and row-level masks that split each shift into validated rows
(data/validated/*.parquet) and quarantined rows (data/quarantine/*.csv) with the reason attached.
Bad rows are set aside, never silently dropped.

Usage:
    python -m src.validate            # validate every raw shift that has not been validated yet
    python -m src.validate --all      # re-validate everything
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import load_config, path
from src.features import RAW_COLUMNS

os.environ.setdefault("GX_ANALYTICS_ENABLED", "false")

NUMERIC_RAW = ["UDI", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
               "Torque [Nm]", "Tool wear [min]", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
LABELS = ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


# ----------------------------------------------------------------------------- rules
def coerce(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in NUMERIC_RAW:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def row_violations(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One boolean column per rule; True means the row breaks that rule."""
    v = pd.DataFrame(index=df.index)
    v["missing_value"] = df[RAW_COLUMNS].isna().any(axis=1)
    for col, (lo, hi) in cfg["validation"]["ranges"].items():
        v[f"out_of_range:{col}"] = df[col].notna() & ~df[col].between(lo, hi)
    v["invalid_type"] = df["Type"].notna() & ~df["Type"].isin(cfg["validation"]["allowed_types"])
    v["invalid_label"] = df[LABELS].notna().all(axis=1) & ~df[LABELS].isin([0, 1]).all(axis=1)
    v["duplicate_udi"] = df["UDI"].duplicated(keep="first") & df["UDI"].notna()
    v["process_not_above_air"] = (df["Process temperature [K]"] <= df["Air temperature [K]"]).fillna(False)
    return v


def batch_checks(df: pd.DataFrame, cfg: dict) -> list[dict]:
    val = cfg["validation"]
    checks = []
    rate = float(df["Machine failure"].mean()) if len(df) else float("nan")
    checks.append({"check": "row_count_positive", "severity": "error",
                   "value": int(len(df)), "passed": bool(len(df) > 0)})
    checks.append({"check": "failure_rate_max", "severity": "error", "value": round(rate, 4),
                   "limit": val["max_failure_rate"], "passed": bool(rate <= val["max_failure_rate"])})
    lo, hi = val["air_temp_mean_range"]
    m = float(df["Air temperature [K]"].mean())
    checks.append({"check": "air_temp_mean_range", "severity": "error", "value": round(m, 2),
                   "limit": [lo, hi], "passed": bool(lo <= m <= hi)})
    agree = (df["Machine failure"] == df[MODES].max(axis=1)).mean()
    checks.append({"check": "label_agrees_with_mode_flags", "severity": "warning",
                   "value": round(float(agree), 4), "limit": val["label_consistency_mostly"],
                   "disagreeing_rows": int((df["Machine failure"] != df[MODES].max(axis=1)).sum()),
                   "passed": bool(agree >= val["label_consistency_mostly"])})
    return checks


# ----------------------------------------------------------------------------- Great Expectations
def run_great_expectations(raw: pd.DataFrame, cfg: dict) -> dict:
    """Build and run the Great Expectations suite. Returns a JSON-friendly summary."""
    try:
        import great_expectations as gx
    except Exception as e:  # keeps the pipeline usable if GX is not installed
        return {"engine": "not available", "error": f"{type(e).__name__}: {e}", "results": []}

    val = cfg["validation"]
    ctx = gx.get_context(mode="ephemeral")
    batch_def = (ctx.data_sources.add_pandas(name="ai4i")
                 .add_dataframe_asset(name="shift")
                 .add_batch_definition_whole_dataframe("whole_shift"))
    suite = ctx.suites.add(gx.ExpectationSuite(name="ai4i_raw_shift_suite"))
    E = gx.expectations
    suite.add_expectation(E.ExpectTableColumnsToMatchOrderedList(column_list=RAW_COLUMNS))
    suite.add_expectation(E.ExpectTableRowCountToBeBetween(min_value=1))
    for c in RAW_COLUMNS:
        suite.add_expectation(E.ExpectColumnValuesToNotBeNull(column=c))
    for c, (lo, hi) in val["ranges"].items():
        suite.add_expectation(E.ExpectColumnValuesToBeBetween(column=c, min_value=lo, max_value=hi))
    suite.add_expectation(E.ExpectColumnValuesToBeInSet(column="Type", value_set=val["allowed_types"]))
    for c in LABELS:
        suite.add_expectation(E.ExpectColumnValuesToBeInSet(column=c, value_set=[0, 1]))
    suite.add_expectation(E.ExpectColumnValuesToBeUnique(column="UDI"))
    suite.add_expectation(E.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="Process temperature [K]", column_B="Air temperature [K]"))
    suite.add_expectation(E.ExpectColumnMeanToBeBetween(
        column="Machine failure", min_value=0, max_value=val["max_failure_rate"]))
    lo, hi = val["air_temp_mean_range"]
    suite.add_expectation(E.ExpectColumnMeanToBeBetween(column="Air temperature [K]", min_value=lo, max_value=hi))

    res = batch_def.get_batch(batch_parameters={"dataframe": raw}).validate(suite)
    results = []
    for r in res.results:
        conf = r.expectation_config
        etype = getattr(conf, "type", None) or getattr(conf, "expectation_type", "?")
        kw = dict(getattr(conf, "kwargs", {}) or {})
        col = kw.get("column") or (f"{kw.get('column_A')} > {kw.get('column_B')}" if "column_A" in kw else "table")
        results.append({"expectation": etype, "column": col, "success": bool(r.success),
                        "unexpected_count": (r.result or {}).get("unexpected_count")})
    return {"engine": f"great_expectations {gx.__version__}", "success": bool(res.success),
            "evaluated": len(results), "failed": sum(not r["success"] for r in results), "results": results}


# ----------------------------------------------------------------------------- batch validation
def validate_df(raw_in: pd.DataFrame, name: str, cfg: dict | None = None, use_gx: bool = True):
    """Validate one batch. Returns (report, good_rows, quarantined_rows)."""
    cfg = cfg or load_config()
    report = {"batch": name, "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "rows_in": int(len(raw_in))}

    # 1) schema - if columns are wrong nothing else can be trusted, so the whole batch is held back
    missing = [c for c in RAW_COLUMNS if c not in raw_in.columns]
    extra = [c for c in raw_in.columns if c not in RAW_COLUMNS]
    report["schema"] = {"passed": not missing, "missing_columns": missing, "extra_columns": extra}
    if missing:
        report.update(status="rejected", rows_valid=0, rows_quarantined=int(len(raw_in)),
                      reason="schema check failed")
        bad = raw_in.copy()
        bad["violations"] = "schema"
        return report, raw_in.iloc[0:0], bad

    raw = raw_in[RAW_COLUMNS]
    df = coerce(raw)

    # 2-3) row-level rules
    v = row_violations(df, cfg)
    bad_mask = v.any(axis=1)
    report["row_rules"] = {k: int(s.sum()) for k, s in v.items()}

    # 4) batch-level statistics on the rows that passed
    checks = batch_checks(df[~bad_mask], cfg)
    report["batch_checks"] = checks
    blocking = [c["check"] for c in checks if c["severity"] == "error" and not c["passed"]]

    # formal Great Expectations record (run on the raw batch as it arrived)
    report["great_expectations"] = run_great_expectations(raw, cfg) if use_gx else {"engine": "skipped"}

    if blocking:
        report.update(status="rejected", reason=f"batch check failed: {', '.join(blocking)}")
        good, bad = df.iloc[0:0], df.copy()
        bad["violations"] = "batch:" + ";".join(blocking)
    else:
        good, bad = df[~bad_mask].copy(), df[bad_mask].copy()
        bad["violations"] = [";".join(v.columns[row]) for row in v[bad_mask].to_numpy()] if len(bad) else []
        report["status"] = "passed" if not bad_mask.any() else "passed_with_quarantine"
    report["rows_valid"] = int(len(good))
    report["rows_quarantined"] = int(len(bad))
    return report, good, bad


def validate_file(csv_path: Path, cfg: dict) -> dict:
    name = csv_path.stem
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    report, good, bad = validate_df(raw, name, cfg)
    report["source_file"] = csv_path.name
    report["source_md5"] = hashlib.md5(csv_path.read_bytes()).hexdigest()

    vdir, qdir = path(cfg["data"]["validated_dir"]), path(cfg["data"]["quarantine_dir"])
    rdir = path("reports/validation")
    for d in (vdir, qdir, rdir):
        d.mkdir(parents=True, exist_ok=True)
    good.to_parquet(vdir / f"{name}.parquet", index=False)
    qfile = qdir / f"{name}.csv"
    if len(bad):
        bad.to_csv(qfile, index=False)
    elif qfile.exists():
        qfile.write_text("")  # nothing quarantined on this run
    (rdir / f"{name}.json").write_text(json.dumps(report, indent=2))
    return report


def summarize(reports: list[dict]) -> pd.DataFrame:
    rows = []
    for r in reports:
        gx_part = r.get("great_expectations", {})
        rows.append({"batch": r["batch"], "status": r["status"], "rows_in": r["rows_in"],
                     "rows_valid": r["rows_valid"], "rows_quarantined": r["rows_quarantined"],
                     "gx_expectations": gx_part.get("evaluated"), "gx_failed": gx_part.get("failed"),
                     "label_mismatch_rows": next((c.get("disagreeing_rows") for c in r.get("batch_checks", [])
                                                  if c["check"] == "label_agrees_with_mode_flags"), None)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="re-validate every raw shift")
    ap.add_argument("--file", help="validate one raw file (path)")
    args = ap.parse_args()
    cfg = load_config()
    raw_dir, rdir = path(cfg["data"]["raw_dir"]), path("reports/validation")

    files = [Path(args.file)] if args.file else sorted(raw_dir.glob("shift_*.csv"))
    reports = []
    for f in files:
        rep_file = rdir / f"{f.stem}.json"
        if rep_file.exists() and not args.all and not args.file:
            old = json.loads(rep_file.read_text())
            if old.get("source_md5") == hashlib.md5(f.read_bytes()).hexdigest():
                reports.append(old)
                continue
        r = validate_file(f, cfg)
        gxr = r.get("great_expectations", {})
        print(f"{f.name}: {r['status']:<24} valid={r['rows_valid']:<4} quarantined={r['rows_quarantined']:<4} "
              f"GX {gxr.get('evaluated', 0)} expectations, {gxr.get('failed', 0)} failed")
        reports.append(r)
    if reports:
        summary = summarize(reports)
        rdir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(rdir / "summary.csv", index=False)
        print("\nvalidation summary:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
