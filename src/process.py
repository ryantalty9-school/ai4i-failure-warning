"""Step 3 - Processing and feature engineering.

Combines the validated rows from the training shifts, renames columns to snake_case, removes
duplicate machines runs, adds the engineered features, and stores one training table as Parquet
(columnar, compressed, keeps data types). That file is what DVC versions and what training reads.

Usage:
    python -m src.process
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from src.config import load_config, path
from src.features import ENGINEERED, FAILURE_MODES, TARGET, add_engineered_features, rename_raw


def build_training_table(cfg: dict) -> tuple[pd.DataFrame, dict]:
    vdir = path(cfg["data"]["validated_dir"])
    n = cfg["data"]["training_shifts"]
    files = [vdir / f"shift_{i:03d}.parquet" for i in range(1, n + 1)]
    present = [f for f in files if f.exists()]
    if not present:
        raise SystemExit("No validated shifts found. Run ingestion and validation first.")
    parts = []
    for f in present:
        part = pd.read_parquet(f)
        part["shift"] = int(f.stem.split("_")[1])
        parts.append(part)
    df = rename_raw(pd.concat(parts, ignore_index=True))
    before = len(df)
    df = df.drop_duplicates(subset="udi", keep="first")
    df = add_engineered_features(df)
    for c in [TARGET] + FAILURE_MODES:
        df[c] = df[c].astype(int)
    df["udi"] = df["udi"].astype(int)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "shifts_used": [int(f.stem.split("_")[1]) for f in present],
        "shifts_missing": [int(f.stem.split("_")[1]) for f in files if not f.exists()],
        "rows": int(len(df)), "duplicates_removed": int(before - len(df)),
        "failures": int(df[TARGET].sum()), "failure_rate": round(float(df[TARGET].mean()), 4),
        "engineered_features": ENGINEERED, "columns": list(df.columns),
    }
    return df, meta


def main():
    cfg = load_config()
    df, meta = build_training_table(cfg)
    out = path(cfg["data"]["processed_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta["file_md5"] = hashlib.md5(out.read_bytes()).hexdigest()
    rep = path("reports/processing.json")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(meta, indent=2))
    print(f"training table written: {out.relative_to(path('.'))}")
    print(f"  rows={meta['rows']}  failures={meta['failures']}  failure_rate={meta['failure_rate']:.2%}  "
          f"shifts={meta['shifts_used'][0]}-{meta['shifts_used'][-1]}  duplicates_removed={meta['duplicates_removed']}")


if __name__ == "__main__":
    main()
