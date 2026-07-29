from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qnlpbench_r.evaluation.evaluate import select_kernel_sample_indices
from qnlpbench_r.models import build_model
from qnlpbench_r.utils.io import ensure_dir, read_json, write_json


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _squared_cross_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)


def _l1_cross_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)


def _median_gamma(X: np.ndarray) -> float:
    d2 = _squared_cross_distances(X, X)
    nz = d2[d2 > 1e-12]
    return 1.0 / max(float(np.median(nz)), 1e-6) if nz.size else 1.0


def _normalize_cross_kernel(K: np.ndarray, diag_a: np.ndarray | None = None, diag_b: np.ndarray | None = None) -> np.ndarray:
    K = np.asarray(K, dtype=np.float64)
    if diag_a is None:
        diag_a = np.ones(K.shape[0], dtype=np.float64)
    if diag_b is None:
        diag_b = np.ones(K.shape[1], dtype=np.float64)
    denom = np.outer(np.sqrt(np.clip(diag_a, 1e-12, None)), np.sqrt(np.clip(diag_b, 1e-12, None)))
    return K / denom


def _classical_kernel_cross(A: np.ndarray, B: np.ndarray, kind: str, params: dict[str, float]) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if kind == "linear":
        K = A @ B.T
        diag_a = np.sum(A * A, axis=1)
        diag_b = np.sum(B * B, axis=1)
        return _normalize_cross_kernel(K, diag_a, diag_b)
    if kind == "cosine":
        An = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-12)
        Bn = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-12)
        return An @ Bn.T
    if kind == "poly":
        gamma = float(params.get("gamma", 1.0 / max(1, A.shape[1])))
        coef0 = float(params.get("coef0", 1.0))
        degree = int(params.get("degree", 2))
        K = (gamma * (A @ B.T) + coef0) ** degree
        diag_a = (gamma * np.sum(A * A, axis=1) + coef0) ** degree
        diag_b = (gamma * np.sum(B * B, axis=1) + coef0) ** degree
        return _normalize_cross_kernel(K, diag_a, diag_b)
    if kind == "rbf":
        gamma = float(params.get("gamma", _median_gamma(A)))
        return np.exp(-gamma * _squared_cross_distances(A, B))
    if kind == "laplacian":
        gamma = float(params.get("gamma", _median_gamma(A)))
        return np.exp(-gamma * _l1_cross_distances(A, B))
    raise ValueError(f"Unknown classical kernel kind: {kind}")


def _candidate_kernels(X_train: np.ndarray, spec: dict[str, Any]) -> list[dict[str, Any]]:
    kinds = spec.get("classical_kernels", ["linear", "cosine", "rbf", "laplacian", "poly"])
    median_gamma = _median_gamma(X_train)
    gamma_multipliers = [float(x) for x in spec.get("gamma_multipliers", [0.25, 0.5, 1.0, 2.0, 4.0])]
    poly_degrees = [int(x) for x in spec.get("poly_degrees", [2, 3])]
    out: list[dict[str, Any]] = []
    if "linear" in kinds:
        out.append({"kernel_source": "classical", "kernel_family": "linear", "kernel_name": "linear", "params": {}})
    if "cosine" in kinds:
        out.append({"kernel_source": "classical", "kernel_family": "cosine", "kernel_name": "cosine", "params": {}})
    if "rbf" in kinds:
        for mult in gamma_multipliers:
            out.append({"kernel_source": "classical", "kernel_family": "rbf", "kernel_name": f"rbf_gamma_x{mult:g}", "params": {"gamma": median_gamma * mult, "gamma_multiplier": mult}})
    if "laplacian" in kinds:
        for mult in gamma_multipliers:
            out.append({"kernel_source": "classical", "kernel_family": "laplacian", "kernel_name": f"laplacian_gamma_x{mult:g}", "params": {"gamma": median_gamma * mult, "gamma_multiplier": mult}})
    if "poly" in kinds:
        for degree in poly_degrees:
            out.append({"kernel_source": "classical", "kernel_family": "poly", "kernel_name": f"poly_degree_{degree}", "params": {"degree": degree, "gamma": 1.0 / max(1, X_train.shape[1]), "coef0": 1.0}})
    return out


def _load_preprocessed(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    pre = read_json(run_dir / "data" / "preprocessor.json")
    df = pd.read_csv(run_dir / "data" / "dataset.csv")
    cols = pre["feature_columns"]
    mean = np.asarray(pre["mean"], dtype=np.float32)
    std = np.asarray(pre["std"], dtype=np.float32)
    sub = df[df["split"].astype(str) == split]
    if split == "ood" and sub.empty:
        sub = df[df["split"].astype(str).str.startswith("ood")]
    X = sub[cols].to_numpy(dtype=np.float32)
    y = sub[pre["label_column"]].to_numpy(dtype=np.int64)
    return ((X - mean) / std).astype(np.float32), y


def _load_qfc(run_dir: Path, trained: bool):
    cfg = _read_yaml(run_dir / "config.yaml")
    meta = read_json(run_dir / "run_metadata.json")
    pre = read_json(run_dir / "data" / "preprocessor.json")
    model_cfg = cfg["models"][0]
    model = build_model(model_cfg, input_dim=len(pre["feature_columns"]), n_classes=2, feature_columns=pre["feature_columns"])
    if trained:
        ckpt = run_dir / "model_best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    return model, cfg, meta, pre


def _quantum_states(model: Any, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.quantum_states(torch.from_numpy(X.astype(np.float32))).detach().cpu().numpy()


def _quantum_kernel_cross(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    K = np.abs(states_a @ states_b.conj().T) ** 2
    return np.asarray(K, dtype=np.float64)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, score))
        except Exception:
            out["roc_auc"] = float("nan")
    return out


def _fit_eval_precomputed(K_train: np.ndarray, y_train: np.ndarray, K_eval: np.ndarray, y_eval: np.ndarray, C: float) -> tuple[dict[str, float], SVC]:
    clf = SVC(kernel="precomputed", C=float(C), probability=False)
    clf.fit(K_train, y_train)
    pred = clf.predict(K_eval)
    score = None
    if hasattr(clf, "decision_function"):
        try:
            score = clf.decision_function(K_eval)
        except Exception:
            score = None
    return _metrics(y_eval, pred, score), clf


def _infer_feature_set(meta: dict[str, Any], cfg: dict[str, Any], run_dir: Path) -> str:
    exp = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    if exp.get("feature_ablation"):
        return str(exp.get("feature_ablation"))
    ds = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
    if ds.get("feature_set"):
        return str(ds.get("feature_set"))
    name = run_dir.name.lower()
    for key in ["no_rule_features", "rule_features_only", "raw_ids_only", "full_features"]:
        if key in name:
            return key
    return "full"


def _find_qfc_runs(results_dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for root in results_dirs:
        if not root.exists():
            continue
        for meta_path in sorted(root.glob("**/run_metadata.json")):
            run_dir = meta_path.parent
            try:
                meta = read_json(meta_path)
            except Exception:
                continue
            if meta.get("model_type") in {"quantum_feature", "grammar_structured_quantum_feature"} and (run_dir / "data" / "dataset.csv").exists():
                key = str(run_dir.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(run_dir)
    return found


def _sample(X: np.ndarray, y: np.ndarray, cap: int, seed: int, strategy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = select_kernel_sample_indices(y, sample_cap=cap, seed=seed, strategy=strategy)
    return X[idx], y[idx], idx


def _analyze_run(run_dir: Path, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trained_model, cfg, meta, _ = _load_qfc(run_dir, trained=True)
    untrained_model, _, _, _ = _load_qfc(run_dir, trained=False)
    seed = int(meta.get("seed", 0))
    run_group = str(meta.get("run_group", ""))
    exp_name = str(meta.get("experiment_name", ""))
    model_name = str(meta.get("model_name", "qfc"))
    feature_set = _infer_feature_set(meta, cfg, run_dir)
    splits = list(spec.get("splits", ["test", "ood_composition", "ood_depth"]))
    train_cap = int(spec.get("train_sample_cap", 384))
    eval_cap = int(spec.get("eval_sample_cap", 256))
    strategy = str(spec.get("kernel_sample_strategy", "stratified"))
    c_values = [float(c) for c in spec.get("c_values", [0.1, 1.0, 10.0])]

    X_train, y_train = _load_preprocessed(run_dir, "train")
    X_val, y_val = _load_preprocessed(run_dir, "val")
    Xtr, ytr, train_idx = _sample(X_train, y_train, train_cap, seed + 7001, strategy)
    Xva, yva, val_idx = _sample(X_val, y_val, min(eval_cap, len(y_val)), seed + 7003, strategy)

    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    def base_row(kernel_source: str, kernel_family: str, kernel_name: str, C: float) -> dict[str, Any]:
        return {
            "run_dir": run_dir.name,
            "run_group": run_group,
            "experiment_name": exp_name,
            "feature_set": feature_set,
            "model_name": model_name,
            "seed": seed,
            "train_sample_cap": train_cap,
            "eval_sample_cap": eval_cap,
            "kernel_source": kernel_source,
            "kernel_family": kernel_family,
            "kernel_name": kernel_name,
            "C": C,
            "train_sample_n": int(len(ytr)),
            "val_sample_n": int(len(yva)),
        }

    # Quantum candidates: trained and untrained QFC state kernels, C selected on validation split.
    for q_source, model in [("quantum_trained", trained_model), ("quantum_untrained", untrained_model)]:
        states_train = _quantum_states(model, Xtr)
        states_val = _quantum_states(model, Xva)
        K_train = _quantum_kernel_cross(states_train, states_train)
        K_val = _quantum_kernel_cross(states_val, states_train)
        best = None
        best_score = -np.inf
        for C in c_values:
            val_metrics, _ = _fit_eval_precomputed(K_train, ytr, K_val, yva, C)
            rec = {**base_row(q_source, "qfc_state_overlap", q_source, C), "selection_split": "val", "selected": False, **{f"val_{k}": v for k, v in val_metrics.items()}}
            selection_rows.append(rec)
            score = val_metrics["balanced_accuracy"]
            if score > best_score:
                best_score = score
                best = (C, K_train, states_train, model)
        assert best is not None
        best_C, K_train_best, states_train_best, model_best = best
        # Evaluate selected C on all requested splits.
        for split in splits:
            X_eval, y_eval = _load_preprocessed(run_dir, split)
            if len(y_eval) == 0:
                continue
            Xev, yev, _ = _sample(X_eval, y_eval, min(eval_cap, len(y_eval)), seed + 7100 + len(split), strategy)
            states_eval = _quantum_states(model_best, Xev)
            K_eval = _quantum_kernel_cross(states_eval, states_train_best)
            m, _ = _fit_eval_precomputed(K_train_best, ytr, K_eval, yev, best_C)
            rows.append({**base_row(q_source, "qfc_state_overlap", q_source, best_C), "split": split, "eval_sample_n": int(len(yev)), "selection_metric": "val_balanced_accuracy", "selection_score": float(best_score), **m})
        for rec in selection_rows:
            if rec["kernel_source"] == q_source and rec["C"] == best_C:
                rec["selected"] = True

    # Classical kernels: select kernel and C on validation split, evaluate selected model on all splits.
    best_classical = None
    best_score = -np.inf
    for cand in _candidate_kernels(Xtr, spec):
        K_train = _classical_kernel_cross(Xtr, Xtr, cand["kernel_family"], cand["params"])
        K_val = _classical_kernel_cross(Xva, Xtr, cand["kernel_family"], cand["params"])
        for C in c_values:
            val_metrics, _ = _fit_eval_precomputed(K_train, ytr, K_val, yva, C)
            rec = {**base_row("classical", cand["kernel_family"], cand["kernel_name"], C), "selection_split": "val", "selected": False, **{f"val_{k}": v for k, v in val_metrics.items()}}
            selection_rows.append(rec)
            score = val_metrics["balanced_accuracy"]
            if score > best_score:
                best_score = score
                best_classical = (cand, C, K_train)
    assert best_classical is not None
    best_cand, best_C, K_train_best = best_classical
    for split in splits:
        X_eval, y_eval = _load_preprocessed(run_dir, split)
        if len(y_eval) == 0:
            continue
        Xev, yev, _ = _sample(X_eval, y_eval, min(eval_cap, len(y_eval)), seed + 7100 + len(split), strategy)
        K_eval = _classical_kernel_cross(Xev, Xtr, best_cand["kernel_family"], best_cand["params"])
        m, _ = _fit_eval_precomputed(K_train_best, ytr, K_eval, yev, best_C)
        rows.append({**base_row("classical", best_cand["kernel_family"], best_cand["kernel_name"], best_C), "split": split, "eval_sample_n": int(len(yev)), "selection_metric": "val_balanced_accuracy", "selection_score": float(best_score), **m})
    for rec in selection_rows:
        if rec["kernel_source"] == "classical" and rec["kernel_name"] == best_cand["kernel_name"] and rec["C"] == best_C:
            rec["selected"] = True
    return rows, selection_rows


def _summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    summary = df.groupby(["feature_set", "model_name", "train_sample_cap", "eval_sample_cap", "split", "kernel_source", "kernel_family", "kernel_name"], as_index=False).agg(
        n=("seed", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        roc_auc_mean=("roc_auc", "mean"),
    )
    pivots: list[dict[str, Any]] = []
    for (feature_set, model_name, train_cap, eval_cap, split), g in df.groupby(["feature_set", "model_name", "train_sample_cap", "eval_sample_cap", "split"]):
        qt = g[g["kernel_source"] == "quantum_trained"]
        qu = g[g["kernel_source"] == "quantum_untrained"]
        cl = g[g["kernel_source"] == "classical"]
        if qt.empty or cl.empty:
            continue
        pivots.append({
            "feature_set": feature_set,
            "model_name": model_name,
            "train_sample_cap": train_cap,
            "eval_sample_cap": eval_cap,
            "split": split,
            "quantum_trained_balanced_accuracy_mean": qt["balanced_accuracy"].mean(),
            "quantum_untrained_balanced_accuracy_mean": qu["balanced_accuracy"].mean() if not qu.empty else np.nan,
            "best_classical_balanced_accuracy_mean": cl["balanced_accuracy"].mean(),
            "delta_quantum_minus_best_classical": qt["balanced_accuracy"].mean() - cl["balanced_accuracy"].mean(),
            "delta_trained_minus_untrained_quantum": qt["balanced_accuracy"].mean() - (qu["balanced_accuracy"].mean() if not qu.empty else np.nan),
            "best_classical_kernel_names": ";".join(sorted(cl["kernel_name"].unique().tolist())),
        })
    return summary, pd.DataFrame(pivots)


def _make_figures(comparison: pd.DataFrame, figures_dir: Path) -> None:
    if comparison.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    comp = comparison[comparison["split"].isin(["test", "ood_composition", "ood_depth"])].copy()
    labels = comp["feature_set"] + "\n" + comp["split"]
    ax.bar(np.arange(len(comp)), comp["delta_quantum_minus_best_classical"])
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(comp)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("QFC kernel-SVM balanced acc - selected classical kernel-SVM")
    ax.set_title("Stage 4B predictive kernel-transfer diagnostic")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(figures_dir / f"stage4b_quantum_minus_classical_kernel_svm_delta.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B: precomputed-kernel SVM comparison of QFC state kernels versus tuned classical kernels.")
    parser.add_argument("--config", default="configs/stage4b_kernel_classifier.yaml")
    parser.add_argument("--results-dir", action="append", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    cfg = _read_yaml(args.config)
    results_dirs = [Path(x) for x in (args.results_dir or cfg.get("results_dirs", ["results_stage3d_dedup", "results_stage3d_feature_ablation"]))]
    out = ensure_dir(args.output_dir or cfg.get("output_dir", "paper_assets/diagnostics/stage4b_kernel_classifier"))
    fig_dir = ensure_dir(args.figures_dir or cfg.get("figures_dir", "figures_stage4b"))
    max_runs = int(args.max_runs if args.max_runs is not None else cfg.get("max_runs", 500))

    qfc_runs = _find_qfc_runs(results_dirs)[:max_runs]
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run_dir in qfc_runs:
        try:
            r, s = _analyze_run(run_dir, cfg)
            rows.extend(r)
            selections.extend(s)
        except Exception as exc:
            failures.append({"run_dir": str(run_dir), "error": repr(exc)})

    df = pd.DataFrame(rows)
    sel = pd.DataFrame(selections)
    df.to_csv(out / "stage4b_kernel_classifier_long.csv", index=False)
    sel.to_csv(out / "stage4b_kernel_classifier_selection.csv", index=False)
    summary, comparison = _summaries(df)
    summary.to_csv(out / "stage4b_kernel_classifier_summary.csv", index=False)
    comparison.to_csv(out / "stage4b_quantum_vs_best_classical_kernel_svm.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out / "stage4b_failures.csv", index=False)
    _make_figures(comparison, fig_dir)
    write_json({
        "stage": "Stage 4B - Predictive Kernel-Transfer Diagnostic",
        "results_dirs": [str(p) for p in results_dirs],
        "n_qfc_runs_found": len(qfc_runs),
        "n_failures": len(failures),
        "outputs": [
            "stage4b_kernel_classifier_long.csv",
            "stage4b_kernel_classifier_selection.csv",
            "stage4b_kernel_classifier_summary.csv",
            "stage4b_quantum_vs_best_classical_kernel_svm.csv",
        ],
        "claim_boundary": "Diagnostic only. This tests whether QFC state geometry transfers to a precomputed-kernel classifier relative to validation-selected classical kernels. It is not quantum advantage.",
    }, out / "stage4b_kernel_classifier_manifest.json")
    print(f"Analyzed {len(qfc_runs)} QFC runs with {len(failures)} failures. Wrote Stage 4B outputs to {out}")


if __name__ == "__main__":
    main()
