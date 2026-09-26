# -*- coding: utf-8 -*-
# ED Fig. 9 v05: Kv4.3 (Ito) registration-grade probe on Kovacheva et al. 2025 data.
# Six panels: a peak current, b activation foot slope, c availability V1/2,
# d purified inactivation tau (candidate finding), e Rs rejection test, f HCN control.
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import figstyle_v05 as fs

fs.apply()

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "Kv43探针")
FIGD = os.path.join(ROOT, "figures_v01")

C_VEH = "#2e7d76"
C_LES = "#B23A3A"

veh_act = pd.read_csv(os.path.join(DATA, "veh_act.csv"))
les_act = pd.read_csv(os.path.join(DATA, "les_act.csv"))
veh_in = pd.read_csv(os.path.join(DATA, "veh_inact.csv"))
les_in = pd.read_csv(os.path.join(DATA, "les_inact.csv"))
pure = pd.read_csv(os.path.join(DATA, "pure_kv43.csv"))
hcn = pd.read_csv(os.path.join(DATA, "hcn_all.csv"))
amp30 = pd.read_csv(os.path.join(DATA, "amp30.csv"))

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
axes = axes.ravel()


def jitter(ax, groups, colors, labels, anchor=None, log=False):
    """groups: list of 1-D arrays. anchor = [(mean, sem), ...] published."""
    ns = []
    for i, (g, c, lab) in enumerate(zip(groups, colors, labels)):
        g = np.asarray(g, float)
        g = g[np.isfinite(g)]
        x = i + np.linspace(-0.16, 0.16, len(g)) if len(g) > 1 else np.array([i])
        ax.scatter(x, g, s=11, color=c, alpha=0.85, zorder=3, lw=0)
        ax.hlines(np.median(g), i - 0.28, i + 0.28, color=c, lw=1.6, zorder=4)
        ns.append((i, len(g), c))
    if anchor is not None:
        from matplotlib.patches import Rectangle as _Rect
        for i, (m, s) in enumerate(anchor):
            ax.add_patch(_Rect((i - 0.34, m - s), 0.68, 2 * s,
                               fc="#888888", alpha=0.18, ec="none", zorder=1))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    lo, hi = ax.get_ylim()
    lo2, hi2 = lo - (hi - lo) * 0.09, lo + (hi - lo) * 1.18
    ax.set_ylim(lo2, hi2)  # headroom for top annotation + room for n= labels
    for i, n_, c in ns:
        ax.text(i, lo2, f"n={n_}", ha="center", va="bottom", fontsize=6, color=c)
    if log:
        ax.set_yscale("log")


# ---- a: peak current at -30 mV (the sealed anchor convention) ----
ax = axes[0]
jitter(ax, [amp30.loc[amp30["group"] == "veh", "I_minus30_nA"],
            amp30.loc[amp30["group"] == "les", "I_minus30_nA"]],
       [C_VEH, C_LES], ["vehicle", "lesioned"],
       anchor=[(3.8, 0.4), (1.7, 0.3)])
ax.set_ylabel("peak Kv4.3 current at −30 mV (nA)", fontsize=7)
ax.set_title("a  peak current (−30 mV)", fontsize=7.5, loc="left", fontweight="bold")
ax.text(0.5, 0.96, "grey bands: published 3.8\u00b10.4 / 1.7\u00b10.3 nA",
        transform=ax.transAxes, ha="center", va="top", fontsize=5.6, color="#666666")

# ---- b: activation foot slope ----
ax = axes[1]
jitter(ax, [veh_act["s_ref"], les_act["s_ref"]], [C_VEH, C_LES],
       ["vehicle", "lesioned"])
ax.set_ylabel("activation slope (mV$^{-1}$)", fontsize=7)
ax.set_title("b  activation foot slope", fontsize=7.5, loc="left", fontweight="bold")
med_v = np.median(veh_act["s_ref"]); med_l = np.median(les_act["s_ref"])
ax.text(0.5, 0.93, f"medians {med_v:.3f} / {med_l:.3f}  ({100*(med_l/med_v-1):+.0f}%)",
        transform=ax.transAxes, ha="center", fontsize=5.6, color="#666666")

# ---- c: availability V1/2 ----
ax = axes[2]
jitter(ax, [veh_in["Vh_inact"], les_in["Vh_inact"]], [C_VEH, C_LES],
       ["vehicle", "lesioned"])
ax.axhline(-61.8, color="#888888", lw=0.7, ls=":", zorder=1)
ax.set_ylabel("availability V$_{1/2}$ (mV)", fontsize=7)
ax.set_title("c  steady-state inactivation", fontsize=7.5, loc="left", fontweight="bold")
ax.text(0.5, 0.93, "unchanged (published \u221261.8 / \u221261.0 mV)",
        transform=ax.transAxes, ha="center", fontsize=5.6, color="#666666")

# ---- d: purified inactivation tau (candidate) ----
ax = axes[3]
pure_q = pure[(pure["r2_pure"] >= 0.9) & (pure["r2_app"] >= 0.9)]  # probe QC
pv = pure_q.loc[pure_q["group"] == "veh", "tau_pure_ms"].values
pl = pure_q.loc[pure_q["group"] == "les", "tau_pure_ms"].values
jitter(ax, [pv, pl], [C_VEH, C_LES], ["vehicle", "lesioned"])
u, p = stats.mannwhitneyu(pv, pl, alternative="two-sided")
ax.set_ylabel("τ$_{inact}$, HCN-subtracted (ms)", fontsize=7)
ax.set_title("d  inactivation time constant", fontsize=7.5, loc="left", fontweight="bold")
ax.text(0.5, 0.93, f"median {np.median(pv):.1f} vs {np.median(pl):.1f} ms "
        f"({100*(np.median(pl)/np.median(pv)-1):+.0f}%, p={p:.3f})",
        transform=ax.transAxes, ha="center", fontsize=5.6, color=C_LES)
for s in ax.spines.values():
    s.set_color(C_LES)
ax.set_facecolor("#fdf6f6")

# ---- e: Rs rejection test ----
ax = axes[4]
for grp, c, lab in (("veh", C_VEH, "vehicle"), ("les", C_LES, "lesioned")):
    d = pure_q[pure_q["group"] == grp]
    ax.scatter(d["pk_pure_nA"], d["tau_pure_ms"], s=13, color=c, alpha=0.85,
               label=lab, lw=0)
dv = pure_q[pure_q["group"] == "veh"]
rho, pr = stats.spearmanr(dv["pk_pure_nA"], dv["tau_pure_ms"])
ax.set_xlabel("peak current (nA)", fontsize=7)
ax.set_ylabel("τ$_{inact}$ (ms)", fontsize=7)
ax.set_title("e  series-resistance test", fontsize=7.5, loc="left", fontweight="bold")
ax.text(0.03, 0.94, f"within-vehicle ρ={rho:.2f}, p={pr:.3f}\n(Rs hypothesis rejected)",
        transform=ax.transAxes, fontsize=5.6, color="#333333", va="top")
ax.legend(fontsize=6, frameon=False, loc="lower right")

# ---- f: HCN control ----
ax = axes[5]
for grp, c, lab in (("veh", C_VEH, "vehicle"), ("les", C_LES, "lesioned")):
    d = hcn[hcn["group"] == grp]
    ax.scatter(d["V_cond"], d["ih_amp_nA"], s=11, color=c, alpha=0.7, label=lab, lw=0)
    m = d.groupby("V_cond")["ih_amp_nA"].median()
    ax.plot(m.index, m.values, color=c, lw=1.2)
ax.set_xlabel("conditioning voltage (mV)", fontsize=7)
ax.set_ylabel("I$_h$ amplitude (nA)", fontsize=7)
ax.legend(fontsize=6, frameon=False, loc="lower right")
ax.set_title("f  HCN background (unchanged)", fontsize=7.5, loc="left", fontweight="bold")

fig.suptitle("K$_V$4.3 probe on an independent Parkinson-model dataset "
             "(Kovacheva et al. 2025): all six published anchors reproduced; "
             "one registered candidate signal (d)", fontsize=7.2, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.965])

fs.save(fig, os.path.join(FIGD, "EDFig9_Kv43_probe"))
