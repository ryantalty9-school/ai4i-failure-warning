"""Step 1 - Batch ingestion.

A real plant would stream sensor readings into a historian. We simulate that by replaying the
AI4I file in order (by UDI) as shift-sized batches. Each shift lands untouched in data/raw as its
own CSV. The raw zone is append-only: a shift that already exists is never overwritten, so we can
always reprocess from the original records (ELT pattern from Chapter 4).

Usage:
    python -m src.ingest --through 18     # land shifts 1-18 (training history)
    python -m src.ingest --through 20     # later: land the two "new" shifts
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from src.config import load_config, path
from src.features import RAW_COLUMNS


def md5(p) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def read_source() -> pd.DataFrame:
    cfg = load_config()
    src = path(cfg["data"]["source_csv"])
    if not src.exists():
        raise SystemExit(f"Source file not found: {src}. Run 1_setup.command first.")
    # utf-8-sig strips the byte-order mark at the start of the UCI file
    df = pd.read_csv(src, encoding="utf-8-sig")
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Source file is missing expected columns: {missing}")
    return df.sort_values("UDI").reset_index(drop=True)


def ingest(through: int) -> list[dict]:
    cfg = load_config()
    size = cfg["data"]["shift_size"]
    raw_dir = path(cfg["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []

    df = read_source()
    n_shifts = -(-len(df) // size)
    through = min(through, n_shifts)
    landed = []
    for shift in range(1, through + 1):
        out = raw_dir / f"shift_{shift:03d}.csv"
        if out.exists():
            continue  # append-only: never overwrite a landed shift
        batch = df.iloc[(shift - 1) * size: shift * size]
        batch.to_csv(out, index=False)
        entry = {
            "shift": shift, "file": out.name, "rows": int(len(batch)),
            "udi_first": int(batch["UDI"].iloc[0]), "udi_last": int(batch["UDI"].iloc[-1]),
            "md5": md5(out), "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        manifest.append(entry)
        landed.append(entry)
        print(f"landed {out.name}: {len(batch)} rows (UDI {entry['udi_first']}-{entry['udi_last']})")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total = len(list(raw_dir.glob("shift_*.csv")))
    print(f"ingestion done: {len(landed)} new shift(s) landed, {total} shift file(s) in {raw_dir.name}/")
    return landed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--through", type=int, default=load_config()["data"]["training_shifts"],
                    help="land every shift up to and including this number")
    args = ap.parse_args()
    ingest(args.through)


if __name__ == "__main__":
    main()
