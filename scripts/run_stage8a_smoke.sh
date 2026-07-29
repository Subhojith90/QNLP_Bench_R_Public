#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/qnlpbench-mpl}"
export PYTHONPATH="src:.:scripts"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/run_index_matched_comparison.py \
  --max-seeds 1 --output results/smoke/index_matched
python scripts/run_exact_initialization_controls.py \
  --max-seeds 1 \
  --indices results/smoke/index_matched/indices \
  --output results/smoke/exact_initialization
python scripts/run_tensor_train_baseline.py \
  --max-seeds 1 \
  --indices results/smoke/index_matched/indices \
  --output results/smoke/tensor_train
python scripts/run_kernel_diagnostics.py \
  --max-seeds 1 \
  --index-results results/smoke/index_matched/stage8a_index_matched_seed_level.csv \
  --index-kernels results/smoke/index_matched/kernels \
  --exact-root results/smoke/exact_initialization \
  --tensor-root results/smoke/tensor_train \
  --output results/smoke/diagnostics
python -m pytest tests/test_stage8a_scientific_strengthening.py -q
