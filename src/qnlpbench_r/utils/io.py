from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
    tmp.replace(path)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_yaml(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_copy_manifest(repo_root: str | Path, run_dir: str | Path) -> None:
    src = Path(repo_root) / "reproducibility_manifest.json"
    dst = Path(run_dir) / "reproducibility_manifest.json"
    if src.exists():
        shutil.copyfile(src, dst)
    else:
        write_json({"warning": "repository reproducibility_manifest.json was not found"}, dst)


def metrics_to_csv(metrics: dict[str, Any], path: str | Path) -> None:
    pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()]).to_csv(path, index=False)


def disk_free_gb(path: str | Path = ".") -> float:
    return shutil.disk_usage(Path(path)).free / (1024**3)


def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]
