"""IEEE two-column format matplotlib styling and shared plot helpers."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IEEE_WIDTH_IN = 3.5
IEEE_FONT_PT = 8

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def set_ieee_style():
    plt.rcParams.update({
        "font.size": IEEE_FONT_PT,
        "axes.titlesize": IEEE_FONT_PT,
        "axes.labelsize": IEEE_FONT_PT,
        "xtick.labelsize": IEEE_FONT_PT - 1,
        "ytick.labelsize": IEEE_FONT_PT - 1,
        "legend.fontsize": IEEE_FONT_PT - 1,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.6,
        "legend.frameon": False,
    })


def new_figure(height_in: float = 2.4):
    set_ieee_style()
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH_IN, height_in))
    return fig, ax


def save_figure(fig, name: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    fig.tight_layout(pad=0.4)
    fig.savefig(path, format="pdf")
    png_path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(png_path, format="png", dpi=200)
    plt.close(fig)
    return path
