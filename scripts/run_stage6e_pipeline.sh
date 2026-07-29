#!/usr/bin/env bash
set -euo pipefail
STAGE6D_ZIP="${1:-QNLPBench_R_Stage6D_Replication_Output.zip}"
if [ ! -f "$STAGE6D_ZIP" ]; then echo "Missing Stage 6D ZIP: $STAGE6D_ZIP" >&2; exit 1; fi
PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q
rm -rf results/stage6e
rm -f figures/stage6e_*
PYTHONPATH=src:. python scripts/import_stage6d_results.py --zip "$STAGE6D_ZIP" --dest .
PYTHONPATH=src:. python scripts/run_stage6e_learned_kernel_challenge.py --config configs/stage6e_learned_kernel.yaml
PYTHONPATH=src:. python scripts/package_stage6e_artifacts.py --output_zip QNLPBench_R_Stage6E_Learned_Kernel_Challenge_Output.zip
PYTHONPATH=src:. python scripts/validate_stage6e_package.py --zip QNLPBench_R_Stage6E_Learned_Kernel_Challenge_Output.zip
