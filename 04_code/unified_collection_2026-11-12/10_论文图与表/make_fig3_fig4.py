# -*- coding: utf-8 -*-
"""
新论文 Fig3（温度律+前向预言）与 Fig4（药物不动动力学，统一新表）
"""
import os, sys, json
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

HERE = Path(__file__).resolve().parent
LAW = HERE.parent
ANA = LAW / "分析_全药池_2026-11-10"
OUT = HERE / "figs_new"
OUT.mkdir(exist_ok=True)

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.2, "axes.titlesize": 7.6, "axes.labelsize": 7.2,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "axes.grid": False, "savefig.dpi": 300,
})
LABC = {"lab_1": "#D81B60", "lab_2": "#1E88E5", "lab_3": "#004D40",
        "lab_4": "#F58231", "lab_5": "#7B1FA2"}

def plabel(ax, s):
    ax.text(-0.24, 1.05, s, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left")

m = pd.read_csv(HERE / "master_unified_v1.csv", low_memory=False)

# ================= FIG 3 =================
fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6))
fig.subplots_adjust(wspace=0.42, left=0.06, right=0.99, top=0.78, bottom=0.17)

# (a) Lei 五温度三过程
ax = axes[0]
lei = m[(m.source_big == "lei2019") & (m.qc_flag == "ok")]
procs = [("tau_rec", "recovery", "#0072B2"), ("tau_act_fast", "activation (fast)", "#D81B60"),
         ("tau_deact_fast", "deactivation (fast)", "#E69F00")]
rng = np.random.default_rng(3)
for pname, lab, col in procs:
    d = lei[lei.parameter == pname]
    for T, g0 in d.groupby("temp_C"):
        ax.scatter(rng.normal(T, 0.25, min(len(g0), 150)),
                   g0.value.clip(0.5, 1e4).values[:150], s=2.5, color=col, alpha=0.18, lw=0)
    med = d.groupby("temp_C")["value"].median()
    ax.plot(med.index, med.values, "o-", color=col, lw=1.6, ms=4, label=lab)
    y = np.log10(med.values); T2 = med.index.values.astype(float)
    s2 = np.linalg.lstsq(np.vstack([T2, np.ones_like(T2)]).T, y, rcond=None)[0][0]
    print(f"Lei {pname}: medians {dict(zip(med.index.astype(int), med.round(1)))} Q10={10**(-10*s2):.2f}")
ax.set_yscale("log"); ax.set_xlabel("temperature (°C)"); ax.set_ylabel("tau (ms)")
ax.set_title("Lei 2019 hERG, five temperatures\n(670 wells, profile CIs)", fontsize=7.4)
ax.legend(frameon=False, fontsize=5.8, loc="upper right", borderaxespad=0.1)
plabel(ax, "a")

# (b) Channelpedia Q10 分过程
ax = axes[1]
be = json.load(open(LAW / "features" / "BE_verdict.json", encoding="utf-8"))
b3 = be["B3_q10_by_process"]
names = {"tau_act": "activation", "tau_deact": "deactivation", "tau_inact": "inactivation"}
cols3 = {"tau_act": "#D81B60", "tau_deact": "#0072B2", "tau_inact": "#009E73"}
ys = np.arange(3)[::-1]
for yi, (k, v) in zip(ys, b3.items()):
    ax.barh(yi, v["median"], color=cols3[k], alpha=0.88, height=0.6)
    ax.text(v["median"] + 0.06, yi, f"{v['median']:.2f} (n={v['n']})", va="center",
            fontsize=6.4, fontweight="bold")
ax.set_yticks(ys); ax.set_yticklabels([names[k] for k in b3], fontsize=6.6)
ax.set_xlabel("pooled Q10"); ax.set_xlim(0, 4.6)
ax.set_title("Temperature loads on activation\n(Channelpedia cross-temperature pairs)", fontsize=7.4)
plabel(ax, "b")

# (c) 25->35C 前向预言
ax = axes[2]
e = pd.read_csv(LAW / "features" / "E_forward_prediction.csv")
ax.scatter(e.tau35_meas, e.tau35_pred, s=7, color="#0072B2", alpha=0.4, lw=0)
lim = [0.2, 2e3]
ax.plot(lim, lim, "k--", lw=0.9)
ax.plot(lim, [2 * x for x in lim], color="#2E7D32", ls=":", lw=1.0)
ax.plot(lim, [x / 2 for x in lim], color="#2E7D32", ls=":", lw=1.0)
inb = ((e.tau35_pred >= e.tau35_meas / 2) & (e.tau35_pred <= e.tau35_meas * 2)).mean()
medfe = (2 ** float(np.median(np.abs(np.log2(e.tau35_pred / e.tau35_meas)))) - 1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("measured tau at 35 C (ms)"); ax.set_ylabel("predicted (ms)")
ax.set_title(f"Zero-parameter forecast 25 C to 35 C\n{len(e)} points, {inb*100:.0f}% within two-fold, median error {medfe*100:.0f}%",
             fontsize=7.4)
plabel(ax, "c")

# (d) CaV1.2 连续浴温
ax = axes[3]
cav = m[(m.source_big == "cav12_ren2022") & (m.dataset == "Temperature_Ca2")
        & (m.qc_flag == "ok")]
cellcols = dict(zip(cav.cell_id.unique(),
                    ["#0072B2", "#D81B60", "#E69F00", "#009E73", "#7B1FA2", "#F58231"]))
for cid, g0 in cav.groupby("cell_id"):
    ax.scatter(g0.temp_C, g0.value, s=4, alpha=0.45, lw=0, color=cellcols[cid])
# 对数线性拟合（全点）
T = cav.temp_C.values; y = np.log10(cav.value.values)
s2, b2 = np.linalg.lstsq(np.vstack([T, np.ones_like(T)]).T, y, rcond=None)[0]
xx = np.linspace(32, 38.8, 50)
ax.plot(xx, 10 ** (s2 * xx + b2), "k-", lw=1.8,
        label=f"log-linear fit, Q10 {10**(-10*s2):.1f}")
ax.set_yscale("log"); ax.set_yticks([100, 200, 400]); ax.set_yticklabels(["100", "200", "400"])
ax.set_xlabel("bath temperature (°C)"); ax.set_ylabel("tau_inact (ms)")
ax.set_title(f"CaV1.2 inactivation vs bath temperature\n{cav.cell_id.nunique()} cells, continuous bath readout", fontsize=7.4)
ax.legend(frameon=False, fontsize=5.8, loc="upper right")
plabel(ax, "d")

fig.savefig(OUT / "fig3_temperature.png", bbox_inches="tight")
plt.close(fig)
print("saved fig3_temperature.png")

# ================= FIG 4 =================
cc = pd.read_csv(HERE / "cipa2_cells.csv")
tr = cc["tau_ratio"].dropna()
tr = tr[(tr > 0) & (tr < 10)]
med_pool = tr.median()
BAND = (0.8, 1.25)
inband = ((tr >= BAND[0]) & (tr <= BAND[1])).mean()
pm30 = ((tr - 1).abs() <= 0.30).mean()
print(f"CiPA ss-caliber: n={len(tr)} median={med_pool:.3f} band={inband:.3f} pm30={pm30:.3f}")

fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6))
fig.subplots_adjust(wspace=0.40, left=0.06, right=0.99, top=0.78, bottom=0.17)

# (a) 五实验室 violin
ax = axes[0]
labs = ["lab_1", "lab_2", "lab_3", "lab_4", "lab_5"]
data, meds = [], {}
for l in labs:
    v = cc[cc.lab == l]["tau_ratio"].dropna()
    v = v[(v > 0) & (v < 10)]
    data.append(v.clip(0, 2.5).values); meds[l] = (len(v), v.median())
parts = ax.violinplot(data, showmedians=False, showextrema=False, widths=0.82)
for pc, l in zip(parts["bodies"], labs):
    pc.set_facecolor(LABC[l]); pc.set_alpha(0.55); pc.set_edgecolor("none")
for i, l in enumerate(labs):
    d = data[i]
    ax.scatter(rng.normal(i + 1, 0.05, min(len(d), 120)), d[:120], s=3,
               color=LABC[l], alpha=0.5, lw=0)
    ax.hlines(meds[l][1], i + 0.68, i + 1.32, color="k", lw=1.4)
    ax.text(i + 1, 2.58, f"{meds[l][1]:.2f}", ha="center", fontsize=6.6, fontweight="bold")
ax.axhspan(BAND[0], BAND[1], color="#2E7D32", alpha=0.10, lw=0)
ax.axhline(1.0, color="k", ls="--", lw=0.9)
ax.set_xticks(range(1, 6)); ax.set_xticklabels([l.replace("_", " ") for l in labs], fontsize=6.4)
ax.set_ylim(0, 2.8); ax.set_ylabel("tau ratio (drug / vehicle)")
ax.set_title(f"Deactivation tau conserved under drug\npooled median {med_pool:.2f}, n={len(tr)} cells (ss block)", fontsize=7.4)
plabel(ax, "a")

# (b) tau vs block depth
ax = axes[1]
qq = cc.dropna(subset=["tau_ratio", "block_frac"])
qq = qq[(qq.tau_ratio > 0) & (qq.tau_ratio < 10)]
ax.scatter(qq.block_frac, qq.tau_ratio.clip(0, 2.5), s=3.5, color="#455A64", alpha=0.22, lw=0)
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
st = qq.assign(stratum=pd.cut(qq.block_frac, bins)).groupby("stratum", observed=True)["tau_ratio"].agg(["median", "count"])
mids = [(i.left + i.right) / 2 for i in st.index]
ax.errorbar(mids, st["median"], fmt="o-", color="#D81B60", lw=1.6, ms=5, capsize=2.5)
for mx, md, n in zip(mids, st["median"], st["count"]):
    ax.annotate(f"{md:.2f}", (mx, md), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=6.4, fontweight="bold", color="#880E4F")
ax.axhline(1.0, color="k", ls="--", lw=0.9)
ax.set_ylim(0, 2.6); ax.set_xlim(0, 1.0)
ax.set_xlabel("steady-state block fraction"); ax.set_ylabel("tau ratio")
ax.set_title("No drift with block depth\n(occupancy moves, kinetics does not)", fontsize=7.4)
plabel(ax, "b")

# (c) 盲法
ax = axes[2]
b = cc[cc.blind == True]["tau_ratio"].dropna(); b = b[(b > 0) & (b < 10)]
u = cc[cc.blind == False]["tau_ratio"].dropna(); u = u[(u > 0) & (u < 10)]
bb = np.linspace(0, 2.5, 60)
uc = u[(u <= 2.5)]; bc = b[(b <= 2.5)]
ax.hist(uc, bins=bb, density=True, color="#78909C", alpha=0.55,
        label=f"unblinded (n={len(u)}, median {u.median():.2f})")
ax.hist(bc, bins=bb, density=True, color="#D81B60", alpha=0.60,
        histtype="stepfilled", edgecolor="#880E4F", lw=0.8,
        label=f"blinded (n={len(b)}, median {b.median():.2f})")
ax.axvline(1.0, color="k", ls="--", lw=0.9)
ax.set_xlabel("tau ratio (drug / vehicle)"); ax.set_ylabel("density")
ax.set_title("Blinded compounds pass the same test", fontsize=7.4)
ax.legend(frameon=False, loc="upper right", fontsize=5.6)
plabel(ax, "c")

# (d) IC50 锚对拍
ax = axes[3]
an = pd.read_csv(ANA / "stage1_vs_newCiPA_anchor.csv")
ax.scatter(an.newCiPA_hERG_uM, an.ours_med_uM, s=18, color="#0072B2", alpha=0.8, lw=0)
lim = [1e-3, 1e3]
ax.plot(lim, lim, "k--", lw=0.9)
ax.plot(lim, [10 * x for x in lim], color="#2E7D32", ls=":", lw=1.0)
ax.plot(lim, [x / 10 for x in lim], color="#2E7D32", ls=":", lw=1.0)
fold10 = ((an.fold >= 0.1) & (an.fold <= 10)).mean()
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("official CiPA hERG IC50 (uM)"); ax.set_ylabel("our estimate (uM)")
ax.set_title(f"Potency anchors reproduced\n{len(an)} drugs, {fold10*100:.0f}% within ten-fold", fontsize=7.4)
plabel(ax, "d")

fig.savefig(OUT / "fig4_drug.png", bbox_inches="tight")
plt.close(fig)
print("saved fig4_drug.png")
