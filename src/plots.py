"""Shared Matplotlib style for all figures."""

from pathlib import Path

import matplotlib.pyplot as plt

IMAGES = Path(__file__).resolve().parents[1] / "images"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
ACTUAL = "#52514e"   # observed values: neutral ink
MODEL = "#2a78d6"    # our model: blue
SNAM = "#eb6834"     # Snam benchmark: orange
ACCENT = "#1baf7a"


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "legend.frameon": False,
        "legend.labelcolor": TEXT,
        "lines.linewidth": 2,
        "font.size": 10.5,
        "figure.dpi": 110,
    })


def save(fig, name: str):
    IMAGES.mkdir(exist_ok=True)
    fig.savefig(IMAGES / name, dpi=150, bbox_inches="tight")
