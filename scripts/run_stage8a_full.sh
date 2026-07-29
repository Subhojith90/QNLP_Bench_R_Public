#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/qnlpbench-mpl}"
export PYTHONPATH="src:.:scripts"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/run_index_matched_comparison.py
python scripts/run_exact_initialization_controls.py
python scripts/run_tensor_train_baseline.py
python scripts/run_kernel_diagnostics.py
python scripts/run_clean_end_to_end_seed11.py
python scripts/generate_stage8a_manuscript_assets.py
python -m pytest tests -q
