from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from safe_zip_utils import UnsafeZipError, sha256_file, validate_zip_members

EXPECTED = {
    "QNLPBench_R_Stage5B_Pilot_Output.zip": [],
    "QNLPBench_R_Stage6AB_Calibration_Output.zip": [],
    "QNLPBench_R_Stage6_Latent_Quantum_Pilot_Output.zip": [],
    "QNLPBench_R_Stage6D_Replication_Output.zip": ["results/stage6d/artifact_sha256_manifest.json"],
    "QNLPBench_R_Stage6E_Learned_Kernel_Challenge_Output.zip": ["results/stage6e/stage6e_artifact_sha256_manifest.json", "results/stage6e/stage6e_learned_kernel_challenge_summary.csv"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the raw output ZIPs used for the comparator-escalation evidence chain.")
    parser.add_argument("--root", default="evidence")
    parser.add_argument("--output-dir", default="results/stage6g/manifests")
    parser.add_argument("--require-full-trajectory", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name, required_members in EXPECTED.items():
        path = root / name
        present = path.exists()
        valid_zip = False
        safe_zip = False
        missing_members: list[str] = []
        unsafe_error = ""
        if present:
            try:
                names = set(validate_zip_members(path))
                safe_zip = True
                valid_zip = True
                missing_members = [member for member in required_members if member not in names]
            except (OSError, UnsafeZipError, ValueError) as exc:
                unsafe_error = str(exc)
        required = args.require_full_trajectory or name in {"QNLPBench_R_Stage6D_Replication_Output.zip", "QNLPBench_R_Stage6E_Learned_Kernel_Challenge_Output.zip"}
        rows.append({"package": name, "required": required, "present": present, "valid_zip": valid_zip, "safe_zip": safe_zip, "missing_key_members": ";".join(missing_members), "unsafe_error": unsafe_error, "sha256": sha256_file(path) if present else ""})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "all_stage_package_audit.csv", index=False)
    failed = frame[(frame.required) & (~frame.present | ~frame.valid_zip | ~frame.safe_zip | (frame.missing_key_members != ""))]
    result = {"scope": "Stage 5B-6E evidence trajectory" if args.require_full_trajectory else "Stage 6D/6E verified evidence", "require_full_trajectory": args.require_full_trajectory, "pass": failed.empty, "failed_packages": failed.package.tolist(), "packages": rows}
    (output / "all_stage_package_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
