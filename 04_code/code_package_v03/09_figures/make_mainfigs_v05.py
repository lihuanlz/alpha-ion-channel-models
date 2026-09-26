# -*- coding: utf-8 -*-
# Re-render Fig2/Fig4/Fig5 from make_figures_v01 with unified style + svg.
# Does NOT touch fig1 (superseded by Fig1_pipeline_v06), fig3 (make_fig3_v05)
# or EDFig2 (superseded by make_EDFig2_v05.py, data-driven 3-panel version).
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_figures_v01 as M

plt.rcParams["svg.fonttype"] = "none"

_orig_save = M.save


def save3(fig, name):
    fig.savefig(os.path.join(M.OUT, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(M.OUT, name + ".svg"), bbox_inches="tight")
    fig.savefig(os.path.join(M.OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved:", name)


M.save = save3
M.fig2()
M.fig4()
M.fig5()
print("done")
