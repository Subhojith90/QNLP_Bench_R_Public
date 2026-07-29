from pathlib import Path
from qnlpbench_r.plotting.plots import make_all_plots


def create_paper_figures(results_dir: str | Path, output_dir: str | Path) -> list[str]:
    return make_all_plots(results_dir, output_dir)
