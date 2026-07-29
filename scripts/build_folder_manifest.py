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

def publishable_files() -> list[Path]:
    """Return files that exist in a clean public source snapshot."""
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
            [
                "git",
                "-C",
                str(ROOT),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
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

    files: list[Path] = []
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
            or (
                path.suffix == ".log"
                and "final_public_commit_replay_1_1" not in relative.parts
            )
        ):
            continue
        files.append(path)
    return sorted(files)


def main() -> None:
    files = []
    for path in publishable_files():
        relative = path.relative_to(ROOT)
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    payload = {
        "package": "QNLPBench-R Stage 8A scientific release",
        "manifest_scope": (
            "All tracked publishable scientific-release files except this "
            "manifest and generated SHARE_WITH_SRINJOY delivery files"
        ),
        "file_count": len(files),
        "files": files,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST} with {len(files)} files.")


if __name__ == "__main__":
    main()
