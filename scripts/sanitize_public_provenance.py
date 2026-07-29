from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/STAGE8A_PUBLIC_PROVENANCE_REDACTION.json"
PUBLIC_HOST_LABEL = "redacted-public-host"
HISTORICAL_STAGE6D_URI = (
    "historical://QNLPBench_R_Stage6D_Replication_Output.zip"
)
PRIVATE_PATH_PATTERN = re.compile(r"/Users/[^/\s]+/[^\n\r\"]+")
HOST_LINE_PATTERN = re.compile(r"(?m)^hostname:\s*.*$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_json_value(value: Any, key: str | None = None) -> tuple[Any, int]:
    changes = 0
    if isinstance(value, dict):
        result = {}
        for child_key, child_value in value.items():
            sanitized, child_changes = sanitize_json_value(
                child_value, child_key
            )
            result[child_key] = sanitized
            changes += child_changes
        return result, changes
    if isinstance(value, list):
        result = []
        for child in value:
            sanitized, child_changes = sanitize_json_value(child, key)
            result.append(sanitized)
            changes += child_changes
        return result, changes
    if key == "hostname" and isinstance(value, str):
        return PUBLIC_HOST_LABEL, int(value != PUBLIC_HOST_LABEL)
    if key == "source_zip" and isinstance(value, str) and value.startswith("/Users/"):
        return HISTORICAL_STAGE6D_URI, 1
    if isinstance(value, str) and value.startswith("/Users/"):
        return f"redacted-private-path/{Path(value).name}", 1
    return value, 0


def sanitize_json_file(path: Path) -> tuple[int, str, str]:
    original_sha256 = sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sanitized, changes = sanitize_json_value(payload)
    if changes:
        path.write_text(
            json.dumps(sanitized, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return changes, original_sha256, sha256_file(path)


def sanitize_text_file(path: Path) -> tuple[int, str, str]:
    original_sha256 = sha256_file(path)
    text = path.read_text(encoding="utf-8", errors="strict")
    sanitized, host_changes = HOST_LINE_PATTERN.subn(
        f"host_label: {PUBLIC_HOST_LABEL}", text
    )
    sanitized, path_changes = PRIVATE_PATH_PATTERN.subn(
        "redacted-private-path", sanitized
    )
    changes = host_changes + path_changes
    if changes:
        path.write_text(sanitized, encoding="utf-8")
    return changes, original_sha256, sha256_file(path)


def candidate_files() -> list[Path]:
    roots = [
        ROOT / "inputs/stage6d_frozen",
        ROOT / "inputs/stage6e_frozen",
        ROOT / "results/end_to_end_seed11",
    ]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".txt"}
        )
    return sorted(set(paths))


def residual_sensitive_hits() -> list[str]:
    hits = []
    for path in candidate_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "/Users/" in text or "Subhojits-MacBook" in text:
            hits.append(str(path.relative_to(ROOT)))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for residual private paths or device hostnames without editing.",
    )
    args = parser.parse_args()
    if args.check:
        hits = residual_sensitive_hits()
        if hits:
            raise SystemExit(
                "Public provenance still contains sensitive values:\n"
                + "\n".join(hits)
            )
        print("Public provenance hygiene check PASS.")
        return

    redactions = []
    for path in candidate_files():
        if path.suffix.lower() == ".json":
            changes, original_sha256, sanitized_sha256 = sanitize_json_file(
                path
            )
        else:
            changes, original_sha256, sanitized_sha256 = sanitize_text_file(
                path
            )
        if changes:
            redactions.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "redaction_count": changes,
                    "pre_redaction_sha256": original_sha256,
                    "public_copy_sha256": sanitized_sha256,
                }
            )

    payload = {
        "policy": "stage8a_public_provenance_redaction_v1",
        "scope": (
            "Public extracted metadata only. Model inputs, labels, predictions, "
            "kernels, checkpoints, RNG states, and numerical results are unchanged."
        ),
        "host_replacement": PUBLIC_HOST_LABEL,
        "private_path_policy": "retain only non-sensitive basename or historical URI",
        "immutable_historical_zip_hashes": json.loads(
            (ROOT / "docs/STAGE8A_PROTOCOL_SEAL.json").read_text(
                encoding="utf-8"
            )
        )["frozen_inputs"],
        "redacted_files": redactions,
    }
    LEDGER.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    hits = residual_sensitive_hits()
    if hits:
        raise SystemExit(
            "Redaction incomplete; sensitive values remain:\n" + "\n".join(hits)
        )
    print(f"Redacted {len(redactions)} public provenance files.")


if __name__ == "__main__":
    main()
