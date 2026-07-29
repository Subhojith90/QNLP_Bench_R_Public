from __future__ import annotations

import hashlib
import json
from pathlib import Path


MANUSCRIPT = Path(__file__).resolve().parents[2] / "Manuscript"
MANIFEST = MANUSCRIPT / "MANUSCRIPT_MANIFEST.json"
PDF = MANUSCRIPT / "Auditing_Apparent_Quantum_Feature_Circuit_Gains.pdf"
PDF_SIDECAR = MANUSCRIPT / "Auditing_Apparent_Quantum_Feature_Circuit_Gains.pdf.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    PDF_SIDECAR.write_text(
        f"{sha256(PDF)}  Auditing_Apparent_Quantum_Feature_Circuit_Gains.pdf\n",
        encoding="utf-8",
    )

    files = []
    for path in sorted(MANUSCRIPT.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        files.append(
            {
                "path": path.relative_to(MANUSCRIPT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    payload = {
        "package": "QNLPBench-R current Stage 8A manuscript",
        "manifest_scope": "All regular files under Manuscript except this manifest",
        "file_count": len(files),
        "files": files,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST} with {len(files)} files.")


if __name__ == "__main__":
    main()
