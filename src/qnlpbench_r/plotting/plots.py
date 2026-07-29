from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from qnlpbench_r.utils.tables import build_robustness_long, load_results_table


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def plot_metric_bar(df: pd.DataFrame, metric: str, output_dir: str | Path, stem: str | None = None) -> None:
    sub = df[df["metric"] == metric]
    if sub.empty:
        raise ValueError(f"Metric not found in results: {metric}")
    agg = sub.groupby("model_name")["value"].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"].fillna(0.0) / agg["count"].clip(lower=1).pow(0.5)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(agg["model_name"], agg["mean"], yerr=1.96 * agg["sem"], capsize=4)
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    ax.set_title(f"{metric} by model")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, Path(output_dir), stem or f"metric_{metric}")


def plot_robustness_gaps(df: pd.DataFrame, output_dir: str | Path) -> None:
    sub = df[df["metric"].str.contains("accuracy_gap_from_clean", regex=False)]
    if sub.empty:
        raise ValueError("No robustness gap metrics found. Run robustness config first.")
    agg = sub.groupby(["model_name", "metric"])["value"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{m}\n{metric.replace('robust_', '')[:32]}" for m, metric in zip(agg["model_name"], agg["metric"])]
    ax.bar(range(len(agg)), agg["value"])
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy gap from clean")
    ax.set_title("Robustness stress-test gaps")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, Path(output_dir), "robustness_gaps")


def plot_structured_robustness(results_dir: str | Path, output_dir: str | Path) -> list[str]:
    robust = build_robustness_long(results_dir)
    if robust.empty:
        return []
    made: list[str] = []

    gap = robust[robust["metric"] == "accuracy_gap_from_clean"].copy()
    if not gap.empty:
        sev_gap = gap[gap["severity"].notna()]
        for split in sorted(sev_gap["split"].dropna().unique()):
            split_df = sev_gap[sev_gap["split"] == split]
            for perturbation in sorted(split_df["perturbation"].dropna().unique()):
                sub = split_df[split_df["perturbation"] == perturbation]
                if sub.empty:
                    continue
                agg = sub.groupby(["model_name", "severity"])["value"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(8, 4.5))
                for model_name, model_df in agg.groupby("model_name"):
                    model_df = model_df.sort_values("severity")
                    ax.plot(model_df["severity"], model_df["value"], marker="o", label=model_name)
                ax.set_xlabel("Severity")
                ax.set_ylabel("Accuracy gap from clean")
                ax.set_title(f"Robustness gap: {split} / {perturbation}")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
                stem = f"robustness_curve_{split}_{perturbation}"
                _save(fig, Path(output_dir), stem)
                made.append(stem)

        shot_gap = gap[gap["shots"].notna()]
        if not shot_gap.empty:
            for split in sorted(shot_gap["split"].dropna().unique()):
                sub = shot_gap[shot_gap["split"] == split]
                agg = sub.groupby(["model_name", "shots"])["value"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(8, 4.5))
                for model_name, model_df in agg.groupby("model_name"):
                    model_df = model_df.sort_values("shots")
                    ax.plot(model_df["shots"], model_df["value"], marker="o", label=model_name)
                ax.set_xlabel("Shots")
                ax.set_ylabel("Accuracy gap from clean")
                ax.set_title(f"Shot-noise approximation gap: {split}")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
                stem = f"robustness_curve_{split}_shot_noise"
                _save(fig, Path(output_dir), stem)
                made.append(stem)

    acc = robust[robust["metric"] == "accuracy"].copy()
    if not acc.empty:
        sev_acc = acc[acc["severity"].notna()]
        for split in sorted(sev_acc["split"].dropna().unique()):
            split_df = sev_acc[sev_acc["split"] == split]
            for perturbation in sorted(split_df["perturbation"].dropna().unique()):
                sub = split_df[split_df["perturbation"] == perturbation]
                if sub.empty:
                    continue
                agg = sub.groupby(["model_name", "severity"])["value"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(8, 4.5))
                for model_name, model_df in agg.groupby("model_name"):
                    model_df = model_df.sort_values("severity")
                    ax.plot(model_df["severity"], model_df["value"], marker="o", label=model_name)
                ax.set_xlabel("Severity")
                ax.set_ylabel("Accuracy")
                ax.set_title(f"Robustness accuracy: {split} / {perturbation}")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
                stem = f"robustness_accuracy_{split}_{perturbation}"
                _save(fig, Path(output_dir), stem)
                made.append(stem)
    return made


def plot_training_curves(results_dir: str | Path, output_dir: str | Path) -> None:
    histories = sorted(Path(results_dir).glob("**/history.csv"))
    histories = [p for p in histories if "__MACOSX" not in p.parts]
    if not histories:
        raise FileNotFoundError(f"No history.csv files found under {results_dir}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for path in histories[:30]:
        hist = pd.read_csv(path)
        if "epoch" in hist.columns and "val_accuracy" in hist.columns:
            ax.plot(hist["epoch"], hist["val_accuracy"], alpha=0.7, label=path.parent.name[-45:])
            plotted = True
    if not plotted:
        raise ValueError("No histories contained val_accuracy.")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Training curves")
    ax.grid(alpha=0.3)
    if len(histories) <= 8:
        ax.legend(fontsize=7)
    _save(fig, Path(output_dir), "training_curves")


def make_all_plots(results_dir: str | Path, output_dir: str | Path) -> list[str]:
    output_dir = Path(output_dir)
    df = load_results_table(results_dir)
    made = []
    for metric in ["test_accuracy", "ood_accuracy", "ood_composition_accuracy", "ood_depth_accuracy", "test_balanced_accuracy", "ood_composition_balanced_accuracy", "ood_depth_balanced_accuracy", "elapsed_seconds"]:
        if metric in set(df["metric"]):
            plot_metric_bar(df, metric, output_dir, stem=f"{metric}_by_model")
            made.append(f"{metric}_by_model")
    try:
        plot_robustness_gaps(df, output_dir)
        made.append("robustness_gaps")
    except ValueError:
        pass
    try:
        made.extend(plot_structured_robustness(results_dir, output_dir))
    except Exception:
        pass
    try:
        plot_training_curves(results_dir, output_dir)
        made.append("training_curves")
    except Exception:
        pass
    if not made:
        raise ValueError("No plots could be generated from available metrics.")
    return made
