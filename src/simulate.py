"""Create a deliberately broken shift file to show the validation step catching bad data.

Simulates a sensor-feed glitch: missing torque readings, an impossible air temperature,
an unknown product type, a duplicated machine run, and a process temperature below air temperature.

Usage:
    python -m src.simulate            # write the glitch shift
    python -m src.simulate --reset    # put the demo back to "shifts 1-18 only" before recording
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.config import load_config, path
from src.ingest import read_source

GLITCH_FILE = "shift_900_sensor_glitch.csv"


def make_glitch_shift(seed: int = 7) -> str:
    cfg = load_config()
    size = cfg["data"]["shift_size"]
    src = read_source()
    last = src.iloc[-size:].copy().reset_index(drop=True)      # based on the final shift's readings
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(last), size=12, replace=False)
    last["Torque [Nm]"] = last["Torque [Nm]"].astype(object)
    last.loc[idx[0:3], "Torque [Nm]"] = np.nan                 # missing sensor values
    last.loc[idx[3:5], "Air temperature [K]"] = 350.0          # impossible reading
    last.loc[idx[5:7], "Type"] = "X"                           # unknown product type
    last.loc[idx[7:9], "UDI"] = last.loc[idx[9:11], "UDI"].to_numpy()   # duplicate machine runs
    last.loc[idx[11], "Process temperature [K]"] = last.loc[idx[11], "Air temperature [K]"] - 2
    out = path(cfg["data"]["raw_dir"]) / GLITCH_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    last.to_csv(out, index=False)
    return out.name


def reset_demo() -> list[str]:
    """Remove every shift after the training history (and the glitch file) so the demo starts clean.
    Only touches data/raw, data/validated, data/quarantine, and reports/validation; training data is untouched."""
    cfg = load_config()
    keep = cfg["data"]["training_shifts"]
    removed = []
    folders = [(cfg["data"]["raw_dir"], ".csv"), (cfg["data"]["validated_dir"], ".parquet"),
               (cfg["data"]["quarantine_dir"], ".csv"), ("reports/validation", ".json")]
    for folder, ext in folders:
        for f in path(folder).glob(f"shift_*{ext}"):
            num = f.stem[6:9]
            if not num.isdigit() or int(num) > keep:
                f.unlink()
                removed.append(f"{folder}/{f.name}")
    manifest = path(cfg["data"]["raw_dir"]) / "_manifest.json"
    if manifest.exists():
        entries = [e for e in json.loads(manifest.read_text()) if e["shift"] <= keep]
        manifest.write_text(json.dumps(entries, indent=2))
    return removed


if __name__ == "__main__":
    if "--reset" in sys.argv:
        gone = reset_demo()
        print("demo reset; removed:", *gone, sep="\n  ") if gone else print("demo already clean")
    else:
        print("wrote", make_glitch_shift())
