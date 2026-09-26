# -*- coding: utf-8 -*-
"""figstyle_v05: shared Nature-style rc for the v05 figure suite.
No CJK font injection (figures are English-only). Use Arial if present, else DejaVu Sans.
SVG carries editable text (fonttype none)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
    })

def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)

def save(fig, path_base):
    fig.savefig(path_base + ".png", bbox_inches="tight")
    fig.savefig(path_base + ".svg", bbox_inches="tight")
    fig.savefig(path_base + ".pdf", bbox_inches="tight")
    print("saved:", path_base + ".{png,svg,pdf}")
