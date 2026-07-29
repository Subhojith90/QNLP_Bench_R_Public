#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="src:.:scripts"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
SOURCE_ZIP="${1:-qnlpbench_r_code_refined_v19_stage6g_public_reproducibility_hardening.zip}"
if [[ ! -f "$SOURCE_ZIP" ]]; then
  echo "Missing v19 source ZIP in project root: $SOURCE_ZIP" >&2
  echo "Copy the downloaded v19 ZIP into this extracted folder before running." >&2
  exit 2
fi
mkdir -p results/stage6g/{verification,manifests,framing,environment} figures evidence
python scripts/check_clean_source_release.py --zip "$SOURCE_ZIP"
python scripts/audit_all_stage_packages.py --root evidence --output-dir results/stage6g/manifests --require-full-trajectory
python scripts/validate_stage6d_package.py --zip evidence/QNLPBench_R_Stage6D_Replication_Output.zip
python scripts/validate_stage6e_package.py --zip evidence/QNLPBench_R_Stage6E_Learned_Kernel_Challenge_Output.zip
python scripts/verify_stage6e_from_package.py --zip evidence/QNLPBench_R_Stage6E_Learned_Kernel_Challenge_Output.zip --output-dir results/stage6g/verification
python scripts/generate_stage6g_claim_matrix.py --output-dir results/stage6g/framing
python scripts/capture_stage6g_environment.py --output-dir results/stage6g/environment
python scripts/generate_stage6g_artifact_card.py --output-dir results/stage6g/framing --verification-manifest results/stage6g/verification/stage6e_verification_manifest.json
python scripts/generate_negative_results_outline.py --output-dir results/stage6g/framing
python scripts/generate_stage6g_figures.py --figures-dir figures
python scripts/build_internal_artifact_manifest.py --evidence-dir evidence --source-zip "$SOURCE_ZIP" --output-dir results/stage6g/manifests
python scripts/package_stage6g_submission.py --manifest results/stage6g/manifests/internal_artifact_manifest.json --output-zip QNLPBench_R_Stage6G_Public_Artifact_Submission.zip
python scripts/validate_stage6g_submission.py --zip QNLPBench_R_Stage6G_Public_Artifact_Submission.zip
python scripts/build_external_delivery_manifest.py --submission-zip QNLPBench_R_Stage6G_Public_Artifact_Submission.zip --source-zip "$SOURCE_ZIP"
python scripts/verify_external_delivery_manifest.py --manifest QNLPBench_R_Stage6G_Delivery_Manifest.json --root .
python -m pytest tests/ -q
echo "Stage 6G complete: upload the public artifact ZIP, delivery manifest, SHA-256 file, and terminal output for final report preparation."
