.PHONY: test stage8a-smoke stage8a-full

EVIDENCE_DIR ?= evidence
STAGE6D_ZIP ?= $(EVIDENCE_DIR)/QNLPBench_R_Stage6D_Replication_Output.zip
STAGE6E_ZIP ?= $(EVIDENCE_DIR)/QNLPBench_R_Stage6E_Learned_Kernel_Challenge_Output.zip
STAGE6G_SOURCE_ZIP ?= qnlpbench_r_code_refined_v19_stage6g_public_reproducibility_hardening.zip

# No target below executes a new quantum scale-up. Stage 6G verifies and freezes prior evidence only.
test:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=src:.:scripts PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q

stage8a-smoke:
	bash scripts/run_stage8a_smoke.sh

stage8a-full:
	bash scripts/run_stage8a_full.sh

validate-stage6d:
	PYTHONPATH=src:.:scripts python scripts/validate_stage6d_package.py --zip $(STAGE6D_ZIP)

validate-stage6e:
	PYTHONPATH=src:.:scripts python scripts/validate_stage6e_package.py --zip $(STAGE6E_ZIP)

audit-stage6e-summary:
	PYTHONPATH=src:.:scripts python scripts/verify_stage6e_from_package.py --zip $(STAGE6E_ZIP) --output-dir results/stage6g/verification

rerun-stage6e:
	@echo "Stage 6E is scientifically frozen. Re-run only if explicitly approved for verification."; exit 2

freeze-stage6g:
	bash scripts/run_stage6g_pipeline.sh $(STAGE6G_SOURCE_ZIP)

verify-delivery:
	PYTHONPATH=src:.:scripts python scripts/verify_external_delivery_manifest.py --manifest QNLPBench_R_Stage6G_Delivery_Manifest.json --root .
