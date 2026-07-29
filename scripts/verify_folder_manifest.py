from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "PACKAGE_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def publishable_paths() -> set[str]:
    try:
        top_level = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        if Path(top_level.stdout.strip()).resolve() != ROOT.resolve():
            raise subprocess.CalledProcessError(1, top_level.args)
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        candidates = sorted(ROOT.rglob("*"))
    else:
        candidates = [
            ROOT / raw.decode("utf-8")
            for raw in result.stdout.split(b"\0")
            if raw
        ]

    paths: set[str] = set()
    for path in candidates:
        relative = path.relative_to(ROOT)
        if (
            not path.is_file()
            or path == MANIFEST
            or ".git" in relative.parts
            or "SHARE_WITH_SRINJOY" in relative.parts
            or "__pycache__" in relative.parts
            or ".pytest_cache" in relative.parts
            or any(part.endswith(".egg-info") for part in relative.parts)
            or path.suffix in {".pyc", ".pyo"}
            or path.suffix == ".log"
        ):
            continue
        paths.add(relative.as_posix())
    return paths


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing: list[str] = []
    size_mismatches: list[str] = []
    hash_mismatches: list[str] = []
    for record in payload["files"]:
        path = ROOT / record["path"]
        if not path.is_file():
            missing.append(record["path"])
            continue
        if path.stat().st_size != record["size_bytes"]:
            size_mismatches.append(record["path"])
        if sha256(path) != record["sha256"]:
            hash_mismatches.append(record["path"])
    declared = {record["path"] for record in payload["files"]}
    extra = sorted(publishable_paths() - declared)
    passed = not (missing or extra or size_mismatches or hash_mismatches)
    report = {
        "status": "pass" if passed else "fail",
        "manifest_file_count": payload["file_count"],
        "verified_file_count": len(payload["files"]) - len(missing),
        "missing": missing,
        "extra": extra,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
    }
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
