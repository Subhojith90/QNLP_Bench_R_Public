from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile


RELEASE = Path(__file__).resolve().parents[1]
WORKSPACE = RELEASE.parent
MANUSCRIPT = WORKSPACE / "Manuscript"
OUTPUT_DIR = RELEASE / "SHARE_WITH_SRINJOY"
ZIP_NAME = "QNLPBench_R_Stage8A_Full_Results_Code_and_Manuscript.zip"
ZIP_PATH = OUTPUT_DIR / ZIP_NAME
SIDECAR = OUTPUT_DIR / f"{ZIP_NAME}.sha256"
REPORT = OUTPUT_DIR / "QNLPBench_R_Stage8A_MASTER_PACKAGE_VERIFICATION.json"
PREFIX = PurePosixPath("QNLPBench_R_Stage8A_Master_Release")
FIXED_TIME = (2026, 7, 23, 12, 0, 0)


START_HERE = """# QNLPBench-R Stage 8A master release

This single archive contains:

- `Release_Package_8A/`: source code, configurations, frozen inputs, raw and
  summarized outputs, exact checkpoints, RNG states, saved indices, kernel
  matrices, diagnostics, protocol records, tests, transcripts, environment
  specifications, Dockerfile, and file manifest.
- `Manuscript/`: the corrected LaTeX source, sections, bibliography, tables,
  supplementary CSV files, vector/raster figures, descriptive final PDF,
  checksum, and manuscript manifest.
- `WORKSPACE_README.md`: the local organization note.
- `MASTER_CONTENTS_MANIFEST.json`: SHA-256 and size for every payload entry in
  this archive.

The Stage 8A protocol is an internally timestamped and hash-sealed audit
protocol, not public third-party preregistration. See the protocol seal,
amendment, and clarification documents under `Release_Package_8A/docs/`.

Administrative placeholders for co-author ORCIDs, final affiliation, CRediT,
funding, repository URL, and DOI require author confirmation and are not
scientific-evidence defects.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def payload_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for source_root, archive_root in (
        (RELEASE, "Release_Package_8A"),
        (MANUSCRIPT, "Manuscript"),
    ):
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_root)
            if (
                "SHARE_WITH_SRINJOY" in relative.parts
                or ".DS_Store" in relative.parts
                or "__pycache__" in relative.parts
                or ".pytest_cache" in relative.parts
            ):
                continue
            archive_name = (PREFIX / archive_root / relative.as_posix()).as_posix()
            files.append((path, archive_name))
    workspace_readme = WORKSPACE / "README.md"
    if workspace_readme.is_file():
        files.append(
            (
                workspace_readme,
                (PREFIX / "WORKSPACE_README.md").as_posix(),
            )
        )
    return files


def zip_info(name: str, executable: bool = False) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_TIME)
    info.create_system = 3
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def build() -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = payload_files()
    records = [
        {
            "path": archive_name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path, archive_name in files
    ]

    start_name = (PREFIX / "START_HERE.md").as_posix()
    start_bytes = START_HERE.encode("utf-8")
    records.append(
        {
            "path": start_name,
            "size_bytes": len(start_bytes),
            "sha256": sha256_bytes(start_bytes),
        }
    )
    records.sort(key=lambda row: row["path"])

    manifest = {
        "package": "QNLPBench-R Stage 8A full results, code, and manuscript",
        "manifest_scope": (
            "Every payload file in this ZIP except MASTER_CONTENTS_MANIFEST.json"
        ),
        "file_count": len(records),
        "files": records,
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    manifest_name = (PREFIX / "MASTER_CONTENTS_MANIFEST.json").as_posix()

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path, archive_name in files:
            archive.write(path, archive_name)
        archive.writestr(zip_info(start_name), start_bytes)
        archive.writestr(zip_info(manifest_name), manifest_bytes)

    zip_digest = sha256_file(ZIP_PATH)
    SIDECAR.write_text(f"{zip_digest}  {ZIP_NAME}\n", encoding="utf-8")
    return {
        "zip": str(ZIP_PATH),
        "zip_size_bytes": ZIP_PATH.stat().st_size,
        "zip_sha256": zip_digest,
        "payload_files": len(records),
        "manifest_entry": manifest_name,
    }


def verify(summary: dict) -> dict:
    errors: list[str] = []
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            errors.append("duplicate archive entries")
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                errors.append(f"unsafe path: {info.filename}")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                errors.append(f"symbolic link entry: {info.filename}")

        manifest = json.loads(archive.read(summary["manifest_entry"]))
        expected = {row["path"]: row for row in manifest["files"]}
        actual = set(names) - {summary["manifest_entry"]}
        if actual != set(expected):
            errors.append("master manifest coverage mismatch")
        for name, record in expected.items():
            data = archive.read(name)
            if len(data) != record["size_bytes"]:
                errors.append(f"size mismatch: {name}")
            if sha256_bytes(data) != record["sha256"]:
                errors.append(f"hash mismatch: {name}")

    sidecar_digest = SIDECAR.read_text(encoding="utf-8").split()[0]
    if sidecar_digest != sha256_file(ZIP_PATH):
        errors.append("ZIP sidecar mismatch")

    report = {
        **summary,
        "archive_entries": len(names),
        "safe_paths": not any(error.startswith("unsafe path") for error in errors),
        "no_symlinks": not any(
            error.startswith("symbolic link") for error in errors
        ),
        "manifest_coverage_pass": "master manifest coverage mismatch" not in errors,
        "hash_and_size_pass": not any(
            error.startswith(("hash mismatch", "size mismatch"))
            for error in errors
        ),
        "sidecar_pass": "ZIP sidecar mismatch" not in errors,
        "errors": errors,
        "overall_pass": not errors,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    summary = build()
    report = verify(summary)
    print(json.dumps(report, indent=2))
    if not report["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
