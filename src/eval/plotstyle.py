"""Shared matplotlib style for figures (static PNG + PDF).
Categorical colours: fixed palette, assigned in fixed order, never cycled."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5,
        "legend.frameon": False, "legend.fontsize": 8, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save(fig, name, outdir="results/figures"):
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
