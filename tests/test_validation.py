"""Tests for the data validation rules (src/validate.py).

These run without Great Expectations installed: validate_df(use_gx=False) exercises the same
row-level rules that decide what gets quarantined.
"""
from __future__ import annotations

import numpy as np

from src.config import load_config
from src.validate import batch_checks, coerce, row_violations, validate_df
from tests.conftest import make_batch

CFG = load_config()


def test_clean_batch_has_no_violations(clean_batch):
    v = row_violations(coerce(clean_batch), CFG)
    assert not v.to_numpy().any(), f"clean batch flagged: {v.sum().to_dict()}"


def test_clean_batch_passes_validation(clean_batch):
    report, good, bad = validate_df(clean_batch, "clean", CFG, use_gx=False)
    assert report["status"] == "passed"
    assert len(good) == len(clean_batch)
    assert len(bad) == 0


def test_each_broken_rule_is_caught():
    df = make_batch()
    df.loc[0, "Torque [Nm]"] = np.nan                       # missing sensor value
    df.loc[1, "Air temperature [K]"] = 350.0                # impossible reading
    df.loc[2, "Type"] = "X"                                 # unknown product type
    df.loc[3, "UDI"] = df.loc[4, "UDI"]                     # duplicate machine run
    df.loc[5, "Process temperature [K]"] = df.loc[5, "Air temperature [K]"] - 2
    df.loc[6, "Machine failure"] = 2                        # label outside {0, 1}

    v = row_violations(coerce(df), CFG)
    assert v.loc[0, "missing_value"]
    assert v.loc[1, "out_of_range:Air temperature [K]"]
    assert v.loc[2, "invalid_type"]
    assert v.loc[3, "duplicate_udi"] or v.loc[4, "duplicate_udi"]
    assert v.loc[5, "process_not_above_air"]
    assert v.loc[6, "invalid_label"]


def test_bad_rows_are_quarantined_with_a_reason():
    df = make_batch()
    df.loc[0, "Torque [Nm]"] = np.nan
    df.loc[1, "Type"] = "X"
    report, good, bad = validate_df(df, "dirty", CFG, use_gx=False)
    assert report["status"] == "passed_with_quarantine"
    assert report["rows_quarantined"] == len(bad) == 2
    assert len(good) == len(df) - 2
    assert "missing_value" in bad["violations"].iloc[0]
    assert set(bad["violations"]).issuperset({"missing_value"})


def test_missing_column_rejects_the_whole_batch(clean_batch):
    df = clean_batch.drop(columns=["Torque [Nm]"])
    report, good, bad = validate_df(df, "no-torque", CFG, use_gx=False)
    assert report["status"] == "rejected"
    assert report["schema"]["passed"] is False
    assert len(good) == 0 and len(bad) == len(df)


def test_impossible_failure_rate_rejects_the_batch():
    df = make_batch(n=100, failures=60)
    report, _good, _bad = validate_df(df, "too-many-failures", CFG, use_gx=False)
    assert report["status"] == "rejected"
    assert "failure_rate_max" in report["reason"]


def test_label_mismatch_is_a_warning_not_a_rejection():
    df = make_batch()
    df.loc[10, "Machine failure"] = 1        # no failure-mode flag set on this row
    report, _good, _bad = validate_df(df, "mismatch", CFG, use_gx=False)
    check = next(c for c in report["batch_checks"] if c["check"] == "label_agrees_with_mode_flags")
    assert check["severity"] == "warning"
    assert check["disagreeing_rows"] >= 1
    assert report["status"] != "rejected"


def test_batch_checks_report_values(clean_batch):
    checks = {c["check"]: c for c in batch_checks(coerce(clean_batch), CFG)}
    assert checks["row_count_positive"]["passed"]
    assert checks["failure_rate_max"]["passed"]
    assert checks["air_temp_mean_range"]["passed"]
