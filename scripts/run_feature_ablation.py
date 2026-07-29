from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnlpbench_r.config import deep_update, load_config, save_config
from qnlpbench_r.experiments.run_experiment import run_experiment_from_config
from qnlpbench_r.utils.io import ensure_dir, write_json


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def build_feature_config(base: dict[str, Any], feature_case: dict[str, Any], results_dir: str | Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    name = str(feature_case["name"])
    cfg["dataset"]["feature_set"] = str(feature_case["feature_set"])
    if "feature_columns" in feature_case:
        cfg["dataset"]["feature_columns"] = list(feature_case["feature_columns"])
    has_custom_stage = "stage_prefix" in cfg["experiment"]
    stage_prefix = str(cfg["experiment"].get("stage_prefix", "qnlpbench_r_stage3d_feature"))
    desired_group = str(cfg["experiment"].get("run_group", "stage3d_feature_ablation")) if has_custom_stage else "stage3d_feature_ablation"
    cfg["experiment"]["name"] = f"{stage_prefix}_{name}"
    cfg["experiment"]["run_group"] = desired_group
    cfg["experiment"]["feature_ablation"] = name
    cfg["experiment"]["feature_set"] = str(feature_case["feature_set"])
    cfg["experiment"]["description"] = str(feature_case.get("description", "Stage 3D feature-set ablation."))
    cfg["output"]["base_dir"] = str(results_dir)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage-3D feature-set ablations.")
    parser.add_argument("--config", default="configs/stage3d_feature_ablation.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--feature-set", action="append", default=None, help="Feature ablation name to run; repeatable.")
    args = parser.parse_args()
    spec = _read_yaml(args.config)
    base_path = Path(spec.get("base_config", "configs/stage3d.yaml"))
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    base = load_config(base_path)
    if isinstance(spec.get("models"), list) and spec["models"]:
        base["models"] = copy.deepcopy(spec["models"])
    if isinstance(spec.get("experiment"), dict):
        base["experiment"].update(copy.deepcopy(spec["experiment"]))
    out_cfg = spec.get("output", {}) or {}
    results_dir = out_cfg.get("results_dir", "results_stage3d_feature_ablation")
    cfg_dir = ensure_dir(out_cfg.get("generated_config_dir", "paper_assets/diagnostics/stage3d_feature_ablation/generated_configs"))
    manifest_path = Path(out_cfg.get("manifest_path", "paper_assets/diagnostics/stage3d_feature_ablation/feature_ablation_manifest.json"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cases = spec.get("feature_sets", [])
    selected = set(args.feature_set or [])
    if selected:
        cases = [c for c in cases if str(c.get("name")) in selected]
    manifest = {"stage": str(spec.get("stage_label", "Feature/Representation Ablation")), "created_at": datetime.now().isoformat(), "results_dir": str(results_dir), "dry_run": bool(args.dry_run), "feature_sets": []}
    for case in cases:
        cfg = build_feature_config(base, case, results_dir)
        cfg_path = cfg_dir / f"{case['name']}.yaml"
        cfg["_config_path"] = str(cfg_path)
        save_config(cfg, cfg_path)
        expected = len(cfg["seed_list"]) * len(cfg["models"])
        manifest["feature_sets"].append({"name": case["name"], "feature_set": case["feature_set"], "feature_columns": case.get("feature_columns"), "intended_strict": bool(case.get("intended_strict", False)), "generated_config": str(cfg_path), "expected_runs": expected})
        print(f"Prepared feature ablation {case['name']} ({expected} runs)")
        if not args.dry_run:
            run_experiment_from_config(cfg)
    write_json(manifest, manifest_path)
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
