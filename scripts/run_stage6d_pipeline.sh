#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="src:."
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
rm -rf results/stage6d
mkdir -p results/stage6d figures
rm -f figures/stage6d_*.pdf figures/stage6d_*.png figures/stage6d_*.svg
python -m pytest tests/ -q | tee results/stage6d/test_log.txt
python scripts/audit_stage6a_latent_validity.py --config configs/stage6d_validity.yaml --output_dir results/stage6d/audit
python scripts/run_feature_ablation.py --config configs/stage6d_classical.yaml
python scripts/run_feature_ablation.py --config configs/stage6d_topologies.yaml
python scripts/audit_stage6d_run_integrity.py --classical_dir results/stage6d/runs/classical --topology_dir results/stage6d/runs/topology --expected_classical 80 --expected_topology 70 --output_dir results/stage6d/manifests
python scripts/run_stage4d_kernel_replication.py --config configs/stage6d_kernel.yaml
python scripts/summarize_stage6d_replication.py --classical_dir results/stage6d/runs/classical --topology_dir results/stage6d/runs/topology --kernel_dir results/stage6d/kernel --output_dir results/stage6d/summary --figures_dir figures
python scripts/package_stage6d_artifacts.py --output_zip QNLPBench_R_Stage6D_Replication_Output.zip
python scripts/validate_stage6d_package.py --zip QNLPBench_R_Stage6D_Replication_Output.zip
