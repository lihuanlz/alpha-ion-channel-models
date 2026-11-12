# -*- coding: utf-8 -*-
"""
SI figure for S8: dataset boundaries and defect registry (6 panels).
All numbers recomputed from frozen analysis CSVs and printed for reconciliation.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()
from matplotlib import rcParams

ANA = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia\分析_全药池_2026-11-10"
OUT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\latex\figs"

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.2, "axes.titlesize": 7.6, "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300,
})
LABC = {"lab_1": "#D81B60", "lab_2": "#1E88E5", "lab_3": "#004D40",
        "lab_4": "#F58231", "lab_5": "#7B1FA2"}
LABN = {k: f"Lab {k[-1]}" for k in LABC}

def plabel(ax, s):
    ax.text(-0.24, 1.05, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")

shift = pd.read_csv(os.path.join(ANA, "stage2a_lab_shift.csv"))
leak = pd.read_csv(os.path.join(ANA, "stage5f_leak_vs_block.csv"))
w5e = pd.read_csv(os.path.join(ANA, "stage5e_per_cell_waveform.csv"))
seg = pd.read_csv(os.path.join(ANA, "stage5a_per_segment.csv"))
pairs = pd.read_csv(os.path.join(ANA, "stage5a_two_conc_pairs.csv"))

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
fig.subplots_adjust(hspace=0.70, wspace=0.42, left=0.085, right=0.99,
                    top=0.88, bottom=0.115)

# (a) lab shift nominal vs corrected
ax = axes[0, 0]
x = np.arange(len(shift)); wd = 0.38
ax.bar(x - wd/2, shift["geo_shift_nominal"], wd, color="#90A4AE",
       edgecolor="k", lw=0.4, label="nominal concentration")
ax.bar(x + wd/2, shift["geo_shift_corrected"], wd, color="#F58231",
       edgecolor="k", lw=0.4, label="measured concentration")
ax.axhline(1.0, color="k", ls="--", lw=0.9)
for i, r in shift.iterrows():
    ax.text(i, max(r["geo_shift_nominal"], r["geo_shift_corrected"]) + 0.12,
            f"{r['geo_shift_corrected']:.2f}", ha="center", fontsize=6.4,
            fontweight="bold", color="#B34700")
ax.set_xticks(x); ax.set_xticklabels(shift["Lab"])
ax.set_ylim(0, 5.4)
ax.set_ylabel(r"geometric IC$_{50}$ shift (fold)")
ax.set_title("Laboratory 2 shift localises to direct\npipetting (Lab 2*: real samples, shift gone)",
             fontsize=7.6)
ax.legend(frameon=False, fontsize=5.8, loc="upper left")
plabel(ax, "a")

# (b) leak vs block (lab_2, the shifted lab)
ax = axes[0, 1]
g = leak[leak["lab"] == "lab_2"].dropna(subset=["leak_hold", "block_last"])
rho = g["leak_hold"].corr(g["block_last"], method="spearman")
ax.scatter(g["leak_hold"].abs(), g["block_last"], s=4, alpha=0.35,
           color="#1E88E5", lw=0)
ax.set_xlabel("holding leak current (pA)")
ax.set_ylabel("block fraction")
ax.set_ylim(-0.05, 1.05)
ax.set_title(f"Leak current does not explain the shift\n"
             f"Lab 2 Spearman $\\rho$={rho:+.3f} (n={len(g)})", fontsize=7.6)
print(f"[b] leak lab_2 rho={rho:+.4f} n={len(g)}")
plabel(ax, "b")

# (c) template-voltage reconstruction failure
ax = axes[0, 2]
rows = []
for lab, g2 in w5e.groupby("lab"):
    r = (g2["tau_sim_ctrl"] / g2["tau_meas_ctrl"]).median()
    rows.append((lab, r, len(g2)))
    print(f"[c] {lab} sim/meas ctrl tau median {r:.2f} n={len(g2)}")
labs = [r[0] for r in rows]; vals = [r[1] for r in rows]
bars = ax.bar([LABN[l] for l in labs], vals, width=0.55,
              color=[LABC[l] for l in labs], edgecolor="k", lw=0.4)
ax.axhline(1.0, color="k", ls="--", lw=0.9)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v * 1.12, f"{v:.2f}", ha="center",
            fontsize=7.2, fontweight="bold")
ax.set_yscale("log"); ax.set_ylim(0.05, 60)
ax.set_ylabel(r"template $\tau$ / measured $\tau$")
ax.set_title("Intended protocol templates cannot\nreplace measured voltage", fontsize=7.6)
plabel(ax, "c")

# (d) E-4031 residual
ax = axes[1, 0]
e4 = seg[seg["e4031"].astype(str).str.lower().isin(["true", "1"])]
res = (1 - e4["block_seg"]).dropna() * 100
ax.hist(res.clip(0, 20), bins=np.linspace(0, 20, 60), color="#004D40",
        alpha=0.8, edgecolor="k", lw=0.3)
ax.axvline(res.median(), color="#D81B60", lw=1.6,
           label=f"median {res.median():.1f}% (IQR {res.quantile(.25):.1f}-{res.quantile(.75):.1f}%)")
ax.set_xlabel("residual current at E-4031 (%)")
ax.set_ylabel("segments")
ax.set_title(f"E-4031 positive control confirms hERG\nidentity (n={len(res)} segments)",
             fontsize=7.6)
ax.legend(frameon=False, fontsize=6.0)
print(f"[d] e4031 n={len(res)} med={res.median():.2f} IQR {res.quantile(.25):.2f}-{res.quantile(.75):.2f}")
plabel(ax, "d")

# (e) monotone pairs
ax = axes[1, 1]
mono = (pairs["b_hi"] > pairs["b_lo"]).sum(); nmono = len(pairs) - mono
ax.bar(["monotone", "non-monotone"], [mono / len(pairs) * 100, nmono / len(pairs) * 100],
       color=["#2E7D32", "#C62828"], width=0.5, edgecolor="k", lw=0.4)
for i, v in enumerate([mono, nmono]):
    ax.text(i, (v if i == 0 else v) / len(pairs) * 100 + 2,
            f"{v}/{len(pairs)}\n({v/len(pairs)*100:.1f}%)", ha="center",
            fontsize=7.0, fontweight="bold")
ax.set_ylim(0, 112); ax.set_ylabel("pairs (%)")
ax.set_title("Paired two-concentration cells are\nmonotone in block", fontsize=7.6)
print(f"[e] monotone {mono}/{len(pairs)} = {mono/len(pairs)*100:.1f}%")
plabel(ax, "e")

# (f) cross-laboratory IC50 variability
ax = axes[1, 2]
fold = pd.read_csv(os.path.join(ANA, "stage1_crosslab_fold.csv"))
v = fold["fold_ours"].dropna()
v = v[v > 0]
ax.hist(np.log10(v), bins=12, color="#1E88E5", alpha=0.75, edgecolor="k", lw=0.4)
ax.axvline(np.log10(v.median()), color="#D81B60", lw=1.8,
           label=f"median {v.median():.2f}-fold")
ax.axvline(np.log10(5), color="k", ls="--", lw=1.1,
           label="source study: about 5-fold")
ax.set_xlabel(r"cross-laboratory IC$_{50}$ ratio (log$_{10}$)")
ax.set_ylabel("drugs")
ax.set_title(f"Potency varies across laboratories\n({len(v)} drugs tested in >=2 labs)",
             fontsize=7.6)
ax.legend(frameon=False, fontsize=6.0)
print(f"[f] fold n={len(v)} median={v.median():.2f}")
plabel(ax, "f")

fig.suptitle("Dataset character and registered boundaries of the five-laboratory campaign",
             fontsize=8.6, fontweight="bold", y=0.985)
fig.savefig(os.path.join(OUT, "figS_fivelab_defects.png"), bbox_inches="tight")
plt.close(fig)
print("saved figS_fivelab_defects.png")
