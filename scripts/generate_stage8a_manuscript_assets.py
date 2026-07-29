from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_DIR = ROOT.parent / "Manuscript"
FIGURE_DIR = MANUSCRIPT_DIR / "figures"
TABLE_DIR = MANUSCRIPT_DIR / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "generic": "#2C7FB8",
    "learned": "#D95F0E",
    "all": "#3B6FB6",
    "random": "#B34A6F",
    "tensor": "#548C2F",
    "neutral": "#555555",
}
TOPOLOGY_LABEL = {
    "qfc_all_to_all": "All-to-all",
    "qfc_random_01": "Random-01",
}
SPLIT_ORDER = ["test", "ood_composition", "ood_depth"]
SPLIT_LABEL = {
    "test": "ID test",
    "ood_composition": "Held-out pair",
    "ood_depth": "OOD depth",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def bootstrap(values: pd.Series, seed: int, resamples: int = 10000):
    array = values.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
        int((array > 0).sum()),
    )


def save_figure(figure: plt.Figure, name: str) -> None:
    figure.savefig(FIGURE_DIR / f"{name}.pdf")
    figure.savefig(FIGURE_DIR / f"{name}.svg")
    figure.savefig(FIGURE_DIR / f"{name}.png", dpi=600)
    figure.savefig(
        FIGURE_DIR / f"{name}.jpg",
        dpi=600,
        pil_kwargs={"quality": 95},
    )
    plt.close(figure)


def audit_protocol_figure() -> None:
    figure, axis = plt.subplots(figsize=(7.2, 2.35))
    axis.set_xlim(0, 5)
    axis.set_ylim(0, 1)
    axis.axis("off")
    titles = [
        "Shortcut audit",
        "Latent task",
        "Replicated\nscreen",
        "Matched\ncontrols",
        "Mechanism\nand rerun",
    ]
    details = [
        "Rule-aware controls\nretire deterministic task",
        "Fixed hidden programme\nreserved pairs + depth",
        "10 seeds; two retained\nQFC topologies",
        "Same cap-128 indices\nlearned, initial, TT",
        "Kernel diagnostics\nseed-11 end-to-end",
    ]
    box_colors = ["#E8EEF7", "#E8EEF7", "#E8EEF7", "#FBE9DF", "#E8F2E3"]
    for index, (title, detail, color) in enumerate(zip(titles, details, box_colors)):
        x = index + 0.08
        rectangle = mpl.patches.FancyBboxPatch(
            (x, 0.27),
            0.84,
            0.48,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            linewidth=0.9,
            edgecolor="#555555",
            facecolor=color,
        )
        axis.add_patch(rectangle)
        axis.text(
            x + 0.42,
            0.62,
            title,
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=7.5,
        )
        axis.text(x + 0.42, 0.41, detail, ha="center", va="center", fontsize=6.5)
        if index < 4:
            axis.annotate(
                "",
                xy=(index + 1.04, 0.51),
                xytext=(index + 0.94, 0.51),
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.0},
            )
    axis.text(
        2.5,
        0.10,
        "Each control changes the permissible interpretation; later evidence supersedes earlier screens.",
        ha="center",
        va="center",
        fontsize=8,
    )
    figure.tight_layout()
    save_figure(figure, "fig1_audit_protocol")


def comparator_assets() -> pd.DataFrame:
    data = pd.read_csv(
        ROOT / "results/index_matched/stage8a_index_matched_seed_level.csv"
    )
    rows = []
    for topology in TOPOLOGY_LABEL:
        for split in SPLIT_ORDER:
            group = data[(data.topology == topology) & (data.split == split)]
            base = {
                "topology": topology,
                "split": split,
                "qfc_mean": group.qfc_balanced_accuracy.mean(),
                "generic_mean": group.generic_balanced_accuracy.mean(),
                "learned_mean": group.learned_balanced_accuracy.mean(),
            }
            for position, column in enumerate(
                ["qfc_minus_generic", "qfc_minus_learned", "learned_minus_generic"]
            ):
                mean, low, high, positive = bootstrap(
                    group[column], 8100 + position * 100 + len(topology) + len(split)
                )
                base[f"{column}_mean"] = mean
                base[f"{column}_low"] = low
                base[f"{column}_high"] = high
                base[f"{column}_positive"] = positive
            rows.append(base)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        ROOT / "results/index_matched/stage8a_index_matched_summary_with_intervals.csv",
        index=False,
    )

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    for axis, split in zip(axes, SPLIT_ORDER):
        axis.axhline(0, color="#777777", linewidth=0.8)
        subset = summary[summary.split == split]
        for t_index, topology in enumerate(TOPOLOGY_LABEL):
            row = subset[subset.topology == topology].iloc[0]
            for offset, column, color, label in [
                (-0.11, "qfc_minus_generic", COLORS["generic"], "QFC − generic"),
                (0.11, "qfc_minus_learned", COLORS["learned"], "QFC − learned"),
            ]:
                mean = row[f"{column}_mean"]
                low = row[f"{column}_low"]
                high = row[f"{column}_high"]
                axis.errorbar(
                    t_index + offset,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="o",
                    color=color,
                    capsize=3,
                    markersize=4.5,
                    linewidth=1.2,
                    label=label if t_index == 0 else None,
                )
        axis.set_title(SPLIT_LABEL[split])
        axis.set_xticks(range(2), ["All-to-all", "Random-01"])
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Paired balanced-accuracy difference")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93))
    figure.suptitle("Index-matched comparator escalation (ten paired seeds)", y=1.03)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(figure, "fig2_index_matched_comparator_escalation")
    return summary


def exact_control_assets() -> pd.DataFrame:
    data = pd.read_csv(
        ROOT
        / "results/exact_initialization/stage8a_exact_initialization_seed_level.csv"
    )
    rows = []
    for topology in TOPOLOGY_LABEL:
        for split in SPLIT_ORDER:
            group = data[(data.topology == topology) & (data.split == split)]
            row = {
                "topology": topology,
                "split": split,
                "initial_mean": group.exact_initial_balanced_accuracy.mean(),
                "trained_mean": group.trained_balanced_accuracy.mean(),
                "shuffled_mean": group.shuffled_label_balanced_accuracy.mean(),
            }
            for position, column in enumerate(
                ["trained_minus_exact_initial", "trained_minus_shuffled_label"]
            ):
                mean, low, high, positive = bootstrap(
                    group[column], 9100 + position * 100 + len(topology) + len(split)
                )
                row[f"{column}_mean"] = mean
                row[f"{column}_low"] = low
                row[f"{column}_high"] = high
                row[f"{column}_positive"] = positive
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(
        ROOT
        / "results/exact_initialization/stage8a_exact_initialization_summary_with_intervals.csv",
        index=False,
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for axis, topology in zip(axes, TOPOLOGY_LABEL):
        subset = summary[summary.topology == topology]
        axis.axhline(0, color="#777777", linewidth=0.8)
        x = np.arange(3)
        for offset, column, color, label in [
            (-0.11, "trained_minus_exact_initial", COLORS["all"], "Trained − exact initial"),
            (0.11, "trained_minus_shuffled_label", COLORS["neutral"], "Trained − shuffled-label"),
        ]:
            means = subset.set_index("split").loc[SPLIT_ORDER, f"{column}_mean"].to_numpy()
            lows = subset.set_index("split").loc[SPLIT_ORDER, f"{column}_low"].to_numpy()
            highs = subset.set_index("split").loc[SPLIT_ORDER, f"{column}_high"].to_numpy()
            axis.errorbar(
                x + offset,
                means,
                yerr=[means - lows, highs - means],
                fmt="o",
                color=color,
                capsize=3,
                linewidth=1.2,
                markersize=4.5,
                label=label,
            )
        axis.set_title(TOPOLOGY_LABEL[topology])
        axis.set_xticks(x, [SPLIT_LABEL[value] for value in SPLIT_ORDER])
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Paired balanced-accuracy difference")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93))
    figure.suptitle("Initialization-matched training controls (ten paired seeds)", y=1.03)
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(figure, "fig3_exact_initialization_controls")
    return summary


def depth_assets() -> pd.DataFrame:
    data = pd.read_csv(
        ROOT / "results/index_matched/stage8a_depth_stratified_seed_level.csv"
    )
    rows = []
    for topology in TOPOLOGY_LABEL:
        for stratum in ["modifier_count_lt_3", "modifier_count_ge_3"]:
            group = data[(data.topology == topology) & (data.stratum == stratum)]
            mean, low, high, positive = bootstrap(
                group.qfc_minus_learned, 10100 + len(topology) + len(stratum)
            )
            rows.append(
                {
                    "topology": topology,
                    "stratum": stratum,
                    "sample_count_mean": group.sample_count.mean(),
                    "sample_count_range": f"{group.sample_count.min()}--{group.sample_count.max()}",
                    "class_0_range": f"{group.class_0.min()}--{group.class_0.max()}",
                    "class_1_range": f"{group.class_1.min()}--{group.class_1.max()}",
                    "all_seeds_have_both_classes": bool(
                        ((group.class_0 > 0) & (group.class_1 > 0)).all()
                    ),
                    "qfc_mean": group.qfc_balanced_accuracy.mean(),
                    "learned_mean": group.learned_balanced_accuracy.mean(),
                    "difference_mean": mean,
                    "difference_low": low,
                    "difference_high": high,
                    "positive_seeds": positive,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        ROOT / "results/index_matched/stage8a_depth_stratified_summary_with_intervals.csv",
        index=False,
    )
    figure, axis = plt.subplots(figsize=(5.8, 2.9))
    axis.axhline(0, color="#777777", linewidth=0.8)
    labels = ["<3 modifiers", "≥3 modifiers"]
    for offset, topology, color in [
        (-0.10, "qfc_all_to_all", COLORS["all"]),
        (0.10, "qfc_random_01", COLORS["random"]),
    ]:
        subset = summary[summary.topology == topology].set_index("stratum").loc[
            ["modifier_count_lt_3", "modifier_count_ge_3"]
        ]
        means = subset.difference_mean.to_numpy()
        axis.errorbar(
            np.arange(2) + offset,
            means,
            yerr=[
                means - subset.difference_low.to_numpy(),
                subset.difference_high.to_numpy() - means,
            ],
            fmt="o",
            color=color,
            capsize=3,
            linewidth=1.2,
            label=TOPOLOGY_LABEL[topology],
        )
    axis.set_xticks(range(2), labels)
    axis.set_ylabel("QFC − learned balanced accuracy")
    axis.set_title("Held-out-pair results stratified by modifier depth")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, "fig4_depth_stratified_heldout_pairs")
    return summary


def tensor_summary() -> pd.DataFrame:
    primary = pd.read_csv(
        ROOT / "results/index_matched/stage8a_index_matched_seed_level.csv"
    )
    tensor = pd.read_csv(
        ROOT / "results/tensor_train/stage8a_tensor_train_seed_level.csv"
    )
    paired = primary.merge(
        tensor[["seed", "split", "kernel_balanced_accuracy"]],
        on=["seed", "split"],
    )
    paired["qfc_minus_tensor_train"] = (
        paired.qfc_balanced_accuracy - paired.kernel_balanced_accuracy
    )
    paired.to_csv(
        ROOT / "results/tensor_train/stage8a_qfc_tensor_train_paired.csv", index=False
    )
    rows = []
    for topology in TOPOLOGY_LABEL:
        for split in SPLIT_ORDER:
            group = paired[(paired.topology == topology) & (paired.split == split)]
            mean, low, high, positive = bootstrap(
                group.qfc_minus_tensor_train, 11100 + len(topology) + len(split)
            )
            rows.append(
                {
                    "topology": topology,
                    "split": split,
                    "qfc_mean": group.qfc_balanced_accuracy.mean(),
                    "tensor_train_mean": group.kernel_balanced_accuracy.mean(),
                    "difference_mean": mean,
                    "difference_low": low,
                    "difference_high": high,
                    "positive_seeds": positive,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        ROOT / "results/tensor_train/stage8a_qfc_tensor_train_summary.csv", index=False
    )
    return summary


def diagnostic_assets() -> None:
    data = pd.read_csv(
        ROOT / "results/diagnostics/stage8a_kernel_diagnostics_seed_level.csv"
    )
    display = data[
        data.kernel.isin(
            [
                "frozen_trained_qfc",
                "selected_learned_classical",
                "selected_tensor_train",
            ]
        )
    ].copy()
    display["series"] = display.apply(
        lambda row: (
            TOPOLOGY_LABEL[row.topology]
            if row.kernel == "frozen_trained_qfc"
            else (
                "Learned classical"
                if row.kernel == "selected_learned_classical"
                else "Tensor train"
            )
        ),
        axis=1,
    )
    # Classical kernels are identical across topology records; retain one copy per seed.
    display = pd.concat(
        [
            display[display.kernel == "frozen_trained_qfc"],
            display[
                (display.kernel != "frozen_trained_qfc")
                & (display.topology == "qfc_all_to_all")
            ],
        ],
        ignore_index=True,
    )
    order = ["All-to-all", "Random-01", "Learned classical", "Tensor train"]
    metrics = [
        ("centered_kernel_target_alignment", "Kernel–target alignment"),
        ("effective_rank_entropy", "Entropy effective rank"),
        ("support_vector_fraction", "Support-vector fraction"),
    ]
    colors = [COLORS["all"], COLORS["random"], COLORS["learned"], COLORS["tensor"]]
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    for axis, (metric, title) in zip(axes, metrics):
        grouped = display.groupby("series")[metric]
        means = grouped.mean().reindex(order)
        standard_errors = grouped.sem().reindex(order)
        for position, (series, color) in enumerate(zip(order, colors)):
            values = display.loc[display.series == series, metric].to_numpy()
            jitter = np.linspace(-0.075, 0.075, len(values))
            axis.scatter(
                position + jitter,
                values,
                s=13,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                alpha=0.78,
                zorder=2,
            )
            axis.errorbar(
                position,
                means.loc[series],
                yerr=standard_errors.loc[series],
                fmt="_",
                markersize=13,
                markeredgewidth=2,
                color="#222222",
                capsize=2.5,
                linewidth=1.0,
                zorder=3,
            )
        axis.set_title(title)
        axis.set_xticks(np.arange(len(order)), ["All", "R-01", "Learned", "TT"])
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    figure.suptitle("Training-kernel geometry on matched cap-128 samples", y=1.02)
    figure.tight_layout()
    save_figure(figure, "fig5_kernel_geometry_diagnostics")


def supplementary_assets() -> None:
    selection = pd.read_csv(
        ROOT / "results/index_matched/stage8a_validation_selection_candidates.csv"
    )
    learned = selection[selection.candidate_family == "learned"].copy()
    learned.to_csv(
        MANUSCRIPT_DIR / "supplementary_stage8a_learned_selection_grid.csv",
        index=False,
    )
    selected = learned[learned.selected == True].copy()  # noqa: E712
    selected.to_csv(
        MANUSCRIPT_DIR / "supplementary_stage8a_learned_selection_winners.csv",
        index=False,
    )
    diagnostics = pd.read_csv(
        ROOT / "results/diagnostics/stage8a_kernel_diagnostics_seed_level.csv"
    )
    diagnostics.to_csv(
        MANUSCRIPT_DIR / "supplementary_stage8a_kernel_diagnostics_seed_level.csv",
        index=False,
    )
    comparisons = pd.read_csv(
        ROOT / "results/diagnostics/stage8a_cross_kernel_comparisons_seed_level.csv"
    )
    comparisons.to_csv(
        MANUSCRIPT_DIR / "supplementary_stage8a_cross_kernel_diagnostics_seed_level.csv",
        index=False,
    )
    eigenspectra = pd.read_csv(
        ROOT / "results/diagnostics/stage8a_kernel_eigenspectra.csv"
    )
    eigenspectra.to_csv(
        MANUSCRIPT_DIR / "supplementary_stage8a_kernel_eigenspectra.csv",
        index=False,
    )

    winner_lines = []
    for _, row in selected.sort_values("seed").iterrows():
        winner_lines.append(
            f"{int(row.seed)} & {row.source.replace('_', r'\_')} & "
            f"{row.kernel_name.replace('_', r'\_')} & {row.C:g} & "
            f"{row.validation_balanced_accuracy:.3f} \\\\"
        )
    (TABLE_DIR / "table_stage8a_learned_winners.tex").write_text(
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Validation-selected learned-representation kernel in each seed.}\n"
        "\\label{tab:stage8a-learned-winners}\n\\small\n"
        "\\begin{tabular}{rllrr}\n\\toprule\n"
        "Seed & Representation & Kernel & $C$ & Validation BA \\\\\n\\midrule\n"
        + "\n".join(winner_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )

    depth = pd.read_csv(
        ROOT / "results/index_matched/stage8a_depth_stratified_seed_level.csv"
    )[
        ["seed", "stratum", "sample_count", "class_0", "class_1"]
    ].drop_duplicates()
    depth_lines = []
    for seed in sorted(depth.seed.unique()):
        shallow = depth[(depth.seed == seed) & (depth.stratum == "modifier_count_lt_3")].iloc[0]
        deep = depth[(depth.seed == seed) & (depth.stratum == "modifier_count_ge_3")].iloc[0]
        depth_lines.append(
            f"{seed} & {int(shallow.sample_count)} & {int(shallow.class_0)} & "
            f"{int(shallow.class_1)} & {int(deep.sample_count)} & "
            f"{int(deep.class_0)} & {int(deep.class_1)} \\\\"
        )
    (TABLE_DIR / "table_stage8a_depth_class_counts.tex").write_text(
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Class counts in the capped held-out-pair depth strata. Every stratum contains both classes.}\n"
        "\\label{tab:stage8a-depth-counts}\n\\small\n"
        "\\begin{tabular}{rrrrrrr}\n\\toprule\n"
        "Seed & \\multicolumn{3}{c}{Fewer than 3 modifiers} & "
        "\\multicolumn{3}{c}{At least 3 modifiers} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        " & $n$ & Class 0 & Class 1 & $n$ & Class 0 & Class 1 \\\\\n\\midrule\n"
        + "\n".join(depth_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )


def write_latex_tables(
    comparator: pd.DataFrame,
    exact: pd.DataFrame,
    depth: pd.DataFrame,
    tensor: pd.DataFrame,
) -> None:
    comparator_lines = []
    for _, row in comparator.iterrows():
        comparator_lines.append(
            f"{TOPOLOGY_LABEL[row.topology]} & {SPLIT_LABEL[row.split]} & "
            f"{row.qfc_mean:.3f} & {row.generic_mean:.3f} & {row.learned_mean:.3f} & "
            f"{row.qfc_minus_generic_mean:+.3f} [{row.qfc_minus_generic_low:+.3f}, {row.qfc_minus_generic_high:+.3f}] & "
            f"{row.qfc_minus_learned_mean:+.3f} [{row.qfc_minus_learned_low:+.3f}, {row.qfc_minus_learned_high:+.3f}] \\\\"
        )
    (TABLE_DIR / "table_stage8a_index_matched.tex").write_text(
        "\\begin{table*}[t]\n\\centering\n\\caption{Index-matched comparison on cap-128 samples. Intervals are descriptive 95\\% paired bootstrap intervals over ten seeds.}\n"
        "\\label{tab:stage8a-index-matched}\n\\small\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "Topology & Split & QFC BA & Generic BA & Learned BA & QFC--generic & QFC--learned \\\\\n\\midrule\n"
        + "\n".join(comparator_lines)
        + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n",
        encoding="utf-8",
    )
    exact_lines = []
    for _, row in exact.iterrows():
        exact_lines.append(
            f"{TOPOLOGY_LABEL[row.topology]} & {SPLIT_LABEL[row.split]} & "
            f"{row.initial_mean:.3f} & {row.trained_mean:.3f} & {row.shuffled_mean:.3f} & "
            f"{row.trained_minus_exact_initial_mean:+.3f} [{row.trained_minus_exact_initial_low:+.3f}, {row.trained_minus_exact_initial_high:+.3f}] & "
            f"{int(row.trained_minus_exact_initial_positive)}/10 & "
            f"{row.trained_minus_shuffled_label_mean:+.3f} [{row.trained_minus_shuffled_label_low:+.3f}, {row.trained_minus_shuffled_label_high:+.3f}] & "
            f"{int(row.trained_minus_shuffled_label_positive)}/10 \\\\"
        )
    (TABLE_DIR / "table_stage8a_exact_controls.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Initialization-matched QFC controls. Intervals are descriptive 95\\% paired bootstrap intervals.}\n"
        "\\label{tab:stage8a-exact-controls}\n\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{llrrrrrrr}\n\\toprule\n"
        "Topology & Split & Initial BA & Trained BA & Shuffled BA & Trained--initial & Positive & Trained--shuffled & Positive \\\\\n\\midrule\n"
        + "\n".join(exact_lines)
        + "\n\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n",
        encoding="utf-8",
    )
    depth_lines = []
    for _, row in depth.iterrows():
        stratum = "$<3$ modifiers" if row.stratum.endswith("lt_3") else "$\\geq3$ modifiers"
        depth_lines.append(
            f"{TOPOLOGY_LABEL[row.topology]} & {stratum} & {row.sample_count_mean:.1f} ({row.sample_count_range}) & "
            f"{row.qfc_mean:.3f} & {row.learned_mean:.3f} & "
            f"{row.difference_mean:+.3f} [{row.difference_low:+.3f}, {row.difference_high:+.3f}] \\\\"
        )
    (TABLE_DIR / "table_stage8a_depth_strata.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Depth-stratified held-out-pair comparison. Counts are mean and range across seeds.}\n"
        "\\label{tab:stage8a-depth}\n\\small\n\\begin{tabular}{llrrrr}\n\\toprule\n"
        "Topology & Stratum & Count & QFC BA & Learned BA & Difference \\\\\n\\midrule\n"
        + "\n".join(depth_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )
    tensor_lines = []
    for _, row in tensor.iterrows():
        tensor_lines.append(
            f"{TOPOLOGY_LABEL[row.topology]} & {SPLIT_LABEL[row.split]} & {row.qfc_mean:.3f} & "
            f"{row.tensor_train_mean:.3f} & {row.difference_mean:+.3f} "
            f"[{row.difference_low:+.3f}, {row.difference_high:+.3f}] \\\\"
        )
    (TABLE_DIR / "table_stage8a_tensor_train.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{Paired comparison with the validation-selected tensor-train representation kernel.}\n"
        "\\label{tab:stage8a-tensor-train}\n\\small\n\\begin{tabular}{llrrr}\n\\toprule\n"
        "Topology & Split & QFC BA & Tensor-train BA & Difference \\\\\n\\midrule\n"
        + "\n".join(tensor_lines)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n",
        encoding="utf-8",
    )


def main() -> None:
    audit_protocol_figure()
    comparator = comparator_assets()
    exact = exact_control_assets()
    depth = depth_assets()
    tensor = tensor_summary()
    diagnostic_assets()
    write_latex_tables(comparator, exact, depth, tensor)
    supplementary_assets()
    print("Generated Stage 8A manuscript tables and vector figures.")


if __name__ == "__main__":
    main()
