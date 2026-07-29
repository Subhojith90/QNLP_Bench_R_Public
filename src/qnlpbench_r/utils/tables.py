from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from qnlpbench_r.utils.io import read_json

CORE_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_precision",
    "macro_recall",
    "roc_auc",
    "brier",
    "ece_10",
    "nll",
]
CORE_SPLITS = ["train", "val", "test", "ood", "ood_composition", "ood_depth"]
PAPER_MAIN_METRICS = [
    "test_accuracy",
    "test_balanced_accuracy",
    "test_macro_f1",
    "test_roc_auc",
    "ood_accuracy",
    "ood_balanced_accuracy",
    "ood_macro_f1",
    "ood_roc_auc",
    "ood_composition_accuracy",
    "ood_composition_balanced_accuracy",
    "ood_composition_macro_f1",
    "ood_composition_roc_auc",
    "ood_depth_accuracy",
    "ood_depth_balanced_accuracy",
    "ood_depth_macro_f1",
    "ood_depth_roc_auc",
    "elapsed_seconds",
    "model_parameter_count",
    "model_n_qubits",
    "model_n_layers",
    "model_depth_proxy",
    "model_two_qubit_gate_count_proxy",
]
KERNEL_METRICS = [
    "alignment",
    "effective_rank",
    "mean_offdiag",
    "trace",
    "sample_n",
    "sample_class0_n",
    "sample_class1_n",
]
ROBUSTNESS_METRICS_FOR_SUMMARY = [
    "accuracy",
    "accuracy_gap_from_clean",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "brier",
    "ece_10",
]

SEV_RE = re.compile(r"^robust_(?P<split>.+?)_(?P<variant>.+?)_sev(?P<severity>[-+]?\d*\.?\d+)_(?P<metric>.+)$")
SHOT_RE = re.compile(r"^robust_(?P<split>.+?)_(?P<variant>shot_noise)_shots(?P<shots>\d+)_(?P<metric>.+)$")
SPLIT_PATTERN = r"(?:ood_composition|ood_depth|train|val|test|ood)"
KERNEL_RE = re.compile(rf"^(?P<split>{SPLIT_PATTERN})_quantum_kernel_(?P<kernel_metric>.+)$")
SPLIT_METRIC_RE = re.compile(rf"^(?P<split>{SPLIT_PATTERN})_(?P<metric>.+)$")


def collect_metric_files(results_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(results_dir).glob("**/metrics.json") if "__MACOSX" not in p.parts)


def _base_row(run_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "experiment_name": meta.get("experiment_name"),
        "run_group": meta.get("run_group"),
        "model_name": meta.get("model_name"),
        "model_type": meta.get("model_type"),
        "seed": meta.get("seed"),
        "status": meta.get("status"),
        "elapsed_seconds": meta.get("elapsed_seconds"),
    }


def _load_metric_records(results_dir: str | Path) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for metrics_path in collect_metric_files(results_dir):
        run_dir = metrics_path.parent
        metrics = read_json(metrics_path)
        meta_path = run_dir / "run_metadata.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        records.append((run_dir, metrics, meta))
    if not records:
        raise FileNotFoundError(f"No valid metrics.json files found under {results_dir}")
    return records


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)


def _summarize(df: pd.DataFrame, group_cols: list[str], value_col: str = "value") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + ["count", "mean", "std", "sem", "ci95_low", "ci95_high"])
    summary = df.groupby(group_cols, dropna=False)[value_col].agg(["count", "mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    summary["sem"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    summary["ci95_low"] = summary["mean"] - 1.96 * summary["sem"]
    summary["ci95_high"] = summary["mean"] + 1.96 * summary["sem"]
    return summary


def _write_table_bundle(df: pd.DataFrame, output_dir: Path, stem: str, latex: bool = True) -> None:
    df.to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / f"{stem}.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    if latex:
        (output_dir / f"{stem}.tex").write_text(df.to_latex(index=False, float_format="%.4f"), encoding="utf-8")


def load_results_table(results_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir, metrics, meta in _load_metric_records(results_dir):
        base = _base_row(run_dir, meta)
        for k, v in metrics.items():
            if _is_finite_number(v):
                rows.append({**base, "metric": k, "value": float(v)})
    if not rows:
        raise FileNotFoundError(f"No finite numeric metrics found under {results_dir}")
    return pd.DataFrame(rows)


def build_main_results_long(results_dir: str | Path, metrics: Iterable[str] | None = None) -> pd.DataFrame:
    metric_set = set(metrics or PAPER_MAIN_METRICS)
    rows: list[dict[str, Any]] = []
    for run_dir, run_metrics, meta in _load_metric_records(results_dir):
        base = _base_row(run_dir, meta)
        for key in metric_set:
            value = run_metrics.get(key)
            if _is_finite_number(value):
                split = None
                metric_name = key
                m = SPLIT_METRIC_RE.match(key)
                if m:
                    split = m.group("split")
                    metric_name = m.group("metric")
                rows.append({**base, "split": split, "metric": metric_name, "source_metric": key, "value": float(value)})
    return pd.DataFrame(rows)


def build_main_results_summary(results_dir: str | Path) -> pd.DataFrame:
    long_df = build_main_results_long(results_dir)
    return _summarize(long_df, ["run_group", "experiment_name", "model_name", "model_type", "split", "metric", "source_metric"])


def build_paper_main_wide(results_dir: str | Path) -> pd.DataFrame:
    summary = build_main_results_summary(results_dir)
    if summary.empty:
        return summary
    display = summary.copy()
    display["mean_ci95"] = display.apply(lambda r: f"{r['mean']:.4f} [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]", axis=1)
    keep_metrics = [
        "test_accuracy",
        "test_balanced_accuracy",
        "ood_accuracy",
        "ood_balanced_accuracy",
        "ood_composition_accuracy",
        "ood_composition_balanced_accuracy",
        "ood_depth_accuracy",
        "ood_depth_balanced_accuracy",
        "elapsed_seconds",
        "model_parameter_count",
    ]
    display = display[display["source_metric"].isin(keep_metrics)]
    if display.empty:
        return display
    wide = display.pivot_table(
        index=["run_group", "experiment_name", "model_name", "model_type"],
        columns="source_metric",
        values="mean_ci95",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def build_robustness_long(results_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir, metrics, meta in _load_metric_records(results_dir):
        base = _base_row(run_dir, meta)
        for key, value in metrics.items():
            if not _is_finite_number(value) or not key.startswith("robust_"):
                continue
            m = SEV_RE.match(key)
            if m:
                rows.append({
                    **base,
                    "split": m.group("split"),
                    "perturbation": m.group("variant"),
                    "severity": float(m.group("severity")),
                    "shots": np.nan,
                    "metric": m.group("metric"),
                    "source_metric": key,
                    "value": float(value),
                })
                continue
            m = SHOT_RE.match(key)
            if m:
                rows.append({
                    **base,
                    "split": m.group("split"),
                    "perturbation": m.group("variant"),
                    "severity": np.nan,
                    "shots": int(m.group("shots")),
                    "metric": m.group("metric"),
                    "source_metric": key,
                    "value": float(value),
                })
    return pd.DataFrame(rows)


def build_robustness_summary(results_dir: str | Path) -> pd.DataFrame:
    long_df = build_robustness_long(results_dir)
    if long_df.empty:
        return pd.DataFrame()
    long_df = long_df[long_df["metric"].isin(ROBUSTNESS_METRICS_FOR_SUMMARY)]
    return _summarize(long_df, ["run_group", "experiment_name", "model_name", "model_type", "split", "perturbation", "severity", "shots", "metric"])


def build_paper_robustness_wide(results_dir: str | Path) -> pd.DataFrame:
    summary = build_robustness_summary(results_dir)
    if summary.empty:
        return summary
    sub = summary[summary["metric"].isin(["accuracy", "accuracy_gap_from_clean"])]
    if sub.empty:
        return sub
    sub = sub.copy()
    sub["condition"] = sub.apply(
        lambda r: f"{r['split']}|{r['perturbation']}|sev={r['severity']:.3f}" if pd.notna(r["severity"]) else f"{r['split']}|{r['perturbation']}|shots={int(r['shots'])}",
        axis=1,
    )
    sub["mean_ci95"] = sub.apply(lambda r: f"{r['mean']:.4f} [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]", axis=1)
    return sub.pivot_table(
        index=["run_group", "experiment_name", "model_name", "model_type", "condition"],
        columns="metric",
        values="mean_ci95",
        aggfunc="first",
    ).reset_index()


def build_kernel_diagnostics_long(results_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir, metrics, meta in _load_metric_records(results_dir):
        base = _base_row(run_dir, meta)
        for key, value in metrics.items():
            m = KERNEL_RE.match(key)
            if not m or not _is_finite_number(value):
                continue
            rows.append({
                **base,
                "split": m.group("split"),
                "kernel_metric": m.group("kernel_metric"),
                "source_metric": key,
                "value": float(value),
            })
    return pd.DataFrame(rows)


def build_kernel_diagnostics_summary(results_dir: str | Path) -> pd.DataFrame:
    long_df = build_kernel_diagnostics_long(results_dir)
    if long_df.empty:
        return pd.DataFrame()
    long_df = long_df[long_df["kernel_metric"].isin(KERNEL_METRICS)]
    return _summarize(long_df, ["run_group", "experiment_name", "model_name", "model_type", "split", "kernel_metric"])


def build_ablation_summary(results_dir: str | Path) -> pd.DataFrame:
    main = build_main_results_long(results_dir)
    if main.empty:
        return pd.DataFrame()
    mask = main["run_group"].fillna("").str.contains("ablation", case=False, regex=False) | main["experiment_name"].fillna("").str.contains("ablation", case=False, regex=False)
    sub = main[mask & main["source_metric"].isin(["test_accuracy", "ood_accuracy", "ood_composition_accuracy", "ood_depth_accuracy", "test_balanced_accuracy", "ood_balanced_accuracy", "ood_composition_balanced_accuracy", "ood_depth_balanced_accuracy", "elapsed_seconds", "model_parameter_count", "model_n_qubits", "model_n_layers", "model_depth_proxy", "model_two_qubit_gate_count_proxy"])]
    if sub.empty:
        return pd.DataFrame()
    return _summarize(sub, ["run_group", "experiment_name", "model_name", "model_type", "split", "metric", "source_metric"])


def aggregate_results(results_dir: str | Path, output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results_table(results_dir)
    full_summary = _summarize(df, ["run_group", "experiment_name", "model_name", "model_type", "metric"])
    _write_table_bundle(full_summary, output_dir, "aggregated_metrics")

    main_long = build_main_results_long(results_dir)
    _write_table_bundle(main_long, output_dir, "main_results_long", latex=False)
    main_summary = build_main_results_summary(results_dir)
    _write_table_bundle(main_summary, output_dir, "main_results_summary")
    paper_main = build_paper_main_wide(results_dir)
    _write_table_bundle(paper_main, output_dir, "paper_main_results")

    robustness_long = build_robustness_long(results_dir)
    if not robustness_long.empty:
        _write_table_bundle(robustness_long, output_dir, "robustness_long", latex=False)
        robustness_summary = build_robustness_summary(results_dir)
        _write_table_bundle(robustness_summary, output_dir, "robustness_summary")
        paper_robustness = build_paper_robustness_wide(results_dir)
        _write_table_bundle(paper_robustness, output_dir, "paper_robustness_results")

    kernel_long = build_kernel_diagnostics_long(results_dir)
    if not kernel_long.empty:
        _write_table_bundle(kernel_long, output_dir, "kernel_diagnostics_long", latex=False)
        kernel_summary = build_kernel_diagnostics_summary(results_dir)
        _write_table_bundle(kernel_summary, output_dir, "kernel_diagnostics_summary")

    ablation_summary = build_ablation_summary(results_dir)
    if not ablation_summary.empty:
        _write_table_bundle(ablation_summary, output_dir, "ablation_summary")

    manifest = {
        "input_results_dir": str(results_dir),
        "n_metric_files": len(collect_metric_files(results_dir)),
        "n_raw_metric_rows": int(len(df)),
        "n_main_rows": int(len(main_long)),
        "n_robustness_rows": int(len(robustness_long)),
        "n_kernel_rows": int(len(kernel_long)),
        "written_files": sorted(p.name for p in output_dir.glob("*")),
    }
    import json
    (output_dir / "table_export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return full_summary
