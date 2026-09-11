"""Shared settings and paths. Every module imports from here."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(rel: str) -> Path:
    """Resolve a project-relative path from config.yaml."""
    return ROOT / rel


def use_project_root() -> None:
    """Run from the project root so relative URIs (sqlite:///mlflow.db) always resolve the same way."""
    os.chdir(ROOT)


def git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def data_version() -> dict:
    """Describe the data the model is trained on: git tag, git commit, and the DVC hash of the training table."""
    info = {"data_version": git("describe", "--tags", "--abbrev=0") or "untagged",
            "git_commit": git("rev-parse", "--short", "HEAD") or "no-git",
            "data_md5": "unknown"}
    dvc_file = ROOT / (load_config()["data"]["processed_path"] + ".dvc")
    if dvc_file.exists():
        try:
            meta = yaml.safe_load(dvc_file.read_text())
            info["data_md5"] = meta["outs"][0].get("md5", "unknown")
        except Exception:
            pass
    return info
