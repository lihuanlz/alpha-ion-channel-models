# -*- coding: utf-8 -*-
# EDFig2 v05: identifiability + history-dependence, all three panels data-driven.
# a: Fisher eigenspectrum (sealed chi = 7.4e12, 6 null directions)
# b: signed eigenvector loading heatmap (13 parameters x 13 directions)
# c: 192 P2 model-free probe, per-cell current at -40 mV after two histories
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import figstyle_v05 as fs

fs.apply()

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(ROOT, "figures_v01")
W182 = os.path.join(ROOT, "..", "w182", "代码182_model62可识别性谱卡_16713003_2026-09-11", "model62_identifiability")
W192 = os.path.join(ROOT, "..", "w192_第四尺度指纹卡", "de192_result.json")

eig = json.load(open(os.path.join(W182, "fisher_eigen.json"), encoding="utf-8"))
mat = json.load(open(os.path.join(W182, "fisher_matrix_13x13.json"), encoding="utf-8"))
lam = np.array(eig["特征值_升序"])
V = np.array(eig["特征向量_列对应特征值"])  # columns = directions, ascending lambda
slots = mat["槽序"]

def pretty(s):
    s = s.replace("log10", "")
    return {"GKr": "G$_{Kr}$", "α_amp": "α$_{amp}$", "α_τ": "α$_{τ}$"}.get(s, s)
names = [pretty(s) for s in slots]

txt = open(W192, encoding="utf-8").read().replace("NaN", "null")
d192 = json.loads(txt)
cells, i_inact, i_closed, neg_flag = [], [], [], []
for cell, cc in d192["per_cell"].items():
    p = cc["P2"]["-40"]
    cells.append(cell)
    i_inact.append(p["hook末"])
    i_closed.append(p["SA_Iss"])
    neg_flag.append(p["SA_Iss"] < 0)
ratio = [a / b if b > 0 else a / abs(b) for a, b in zip(i_inact, i_closed)]
order = np.argsort(ratio)
cells = [cells[i] for i in order]
i_inact = [i_inact[i] for i in order]
i_closed = [i_closed[i] for i in order]
neg_flag = [neg_flag[i] for i in order]

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7),
                         gridspec_kw={"width_ratios": [1.0, 1.45, 1.05], "wspace": 0.42})

# ---------- a: eigenspectrum ----------
axa = axes[0]
idx = np.arange(1, 14)
thr = lam.max() * 1e-3
axa.axhspan(lam.min() * 0.3, thr, color="#E8913A", alpha=0.10, zorder=0)
axa.semilogy(idx, lam, "o-", color="#4C9BD6", mec="#2f6ea3", ms=4, lw=1.0, zorder=3)
axa.axhline(thr, color="#a3621f", ls="--", lw=0.8, zorder=2)
axa.text(13.2, 3.2e-3, "6 null directions", fontsize=5.6, color="#a3621f", ha="right")
axa.text(13.2, 6.5e-4, "loop-inversion: 0 / 451", fontsize=5.4, color="#333333", ha="right")
axa.set_xlabel("eigenvalue index (ascending)", fontsize=6.5)
axa.set_ylabel("Fisher eigenvalue $\\lambda$", fontsize=6.5)
axa.set_title("13-parameter Markov formulation, 4 protocol families\ncondition number $\\chi$ = 7.4e+12", fontsize=6.5)
axa.set_xlim(0.3, 13.7); axa.set_ylim(lam.min() * 0.2, lam.max() * 4)
axa.text(0.02, 1.01, "a", transform=axa.transAxes, fontsize=9, fontweight="bold", va="bottom")

# ---------- b: eigenvector loading heatmap ----------
axb = axes[1]
norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
im = axb.imshow(V, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
axb.set_xticks(range(13)); axb.set_xticklabels(range(1, 14), fontsize=5.5)
axb.set_yticks(range(13)); axb.set_yticklabels(names, fontsize=5.5)
axb.axvline(5.5, color="#a3621f", lw=1.2)
axb.set_ylim(12.5, -1.4)
axb.text(2.5, -0.95, "6 null directions", fontsize=5.8, color="#a3621f", ha="center")
axb.text(9.5, -0.95, "identifiable", fontsize=5.8, color="#2f6ea3", ha="center")
axb.set_xlabel("eigen-direction (ascending $\\lambda$)", fontsize=6.5)
axb.set_title("eigenvector loadings (signed)\nall coordinates log$_{10}$ except Pf2 (linear)", fontsize=6.5)
cb = fig.colorbar(im, ax=axb, fraction=0.035, pad=0.02)
cb.ax.tick_params(labelsize=5)
axb.text(-0.28, 1.01, "b", transform=axb.transAxes, fontsize=9, fontweight="bold", va="bottom")

# ---------- c: history dependence at -40 mV ----------
axc = axes[2]
xs = np.arange(9)
for i in range(9):
    axc.plot([i, i], [abs(i_closed[i]), i_inact[i]], color="#bbbbbb", lw=0.8, zorder=1)
axc.plot(xs, i_inact, "o", color="#B23A3A", ms=4.5, zorder=3, label="from +50 mV (inactivated)")
axc.plot(xs, [abs(v) for v in i_closed], "o", mfc="white", mec="#2f6ea3", ms=4.5, zorder=3, label="from −80 mV (closed)")
for i, neg in enumerate(neg_flag):
    if neg:
        axc.plot([i], [abs(i_closed[i])], "x", color="#2f6ea3", ms=5, zorder=4)
axc.set_yscale("log")
axc.set_ylim(2e-3, 6)
axc.set_xticks(xs)
axc.set_xticklabels([c[4:] for c in cells], fontsize=5.2, rotation=45)
axc.set_xlabel("cell (167…)", fontsize=6.5)
axc.set_ylabel("current at −40 mV (nA)", fontsize=6.5)
axc.set_title("same voltage, different history (model-free probe)\nfactors 17–112$\\times$, 9/9 cells", fontsize=6.5)
axc.legend(fontsize=5.4, loc="upper left", frameon=False, handlelength=1.2)
axc.text(0.0, -0.42, "×: closed-history current crosses zero (sign unstable), |value| plotted",
         transform=axc.transAxes, fontsize=5.0, color="#555555")
axc.text(0.02, 1.01, "c", transform=axc.transAxes, fontsize=9, fontweight="bold", va="bottom")

for a in axes:
    a.tick_params(labelsize=6)

fig.suptitle("Identifiability and history-dependence audits of the 13-parameter Markov formulation",
             fontsize=8.5, fontweight="bold", y=1.05)
fs.save(fig, os.path.join(FIGD, "EDFig2_identifiability"))
print("saved EDFig2_identifiability")
print("ratios sorted:", [round(r, 1) for r in sorted(ratio)])
