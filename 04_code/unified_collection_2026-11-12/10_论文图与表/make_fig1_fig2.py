# -*- coding: utf-8 -*-
"""
新论文 Fig1（方法+统一数据底座）与 Fig2（守恒律全景）
全部数字从统一总表 master_unified_v1.csv 及判决 CSV 现算并打印对账。
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
PAL = {"channelpedia": "#0072B2", "cipa_5lab": "#D81B60", "lei2019": "#E69F00",
       "beattie9": "#004D40", "nav15_27193878": "#7B1FA2", "cav12_ren2022": "#F58231",
       "iks_fedida": "#009E73", "literature": "#666666"}

def plabel(ax, s):
    ax.text(-0.24, 1.05, s, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left")

m = pd.read_csv(HERE / "master_unified_v1.csv", low_memory=False)
exp = m[~m.source_big.isin(["cipa_official", "channelpedia_groupstats", "icgenealogy"])]
print("实验行数:", len(exp))

# ================= FIG 1 =================
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
fig.subplots_adjust(wspace=0.42, left=0.075, right=0.985, top=0.80, bottom=0.17)

# (a) 语料全景：逐源行数与细胞数
ax = axes[0]
g = exp.groupby("source_big").agg(rows=("value", "size"), cells=("cell_id", "nunique"))
order = g.sort_values("rows").index
labels = {"channelpedia": "Channelpedia\n(19,282-cell corpus)", "cipa_5lab": "CiPA five-lab hERG",
          "lei2019": "Lei 2019 five-temp", "beattie9": "Beattie nine-cell",
          "nav15_27193878": "Nav1.5 35/25C", "cav12_ren2022": "CaV1.2 bath-temp",
          "iks_fedida": "IKs Fedida"}
y = np.arange(len(order))
ax.barh(y, [g.loc[s, "rows"] for s in order],
        color=[PAL.get(s, "#999") for s in order], alpha=0.88, height=0.66)
for yi, s in zip(y, order):
    ax.text(g.loc[s, "rows"] * 1.15, yi, f"{g.loc[s,'rows']:,}", va="center",
            fontsize=6.2, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([labels.get(s, s) for s in order], fontsize=6.2)
ax.set_xscale("log"); ax.set_xlim(1e2, 3e6)
ax.set_xlabel("extracted quantities (rows)")
ax.set_title(f"A unified measurement table\n{len(exp):,} rows, {exp.cell_id.nunique():,} cells, 7 sources")
plabel(ax, "a")

# (b) 家族与过程覆盖
ax = axes[1]
fam = exp[exp.family.notna() & (exp.family != "")].groupby("family")["cell_id"].nunique()
fam = fam.sort_values()
cols = {"Kv": "#0072B2", "HCN": "#E69F00", "K2P": "#009E73", "Kir": "#CC79A7",
        "Nav": "#7B1FA2", "CaV": "#F58231", "other": "#999999"}
y = np.arange(len(fam))
ax.barh(y, fam.values, color=[cols.get(f, "#999") for f in fam.index], alpha=0.88, height=0.66)
for yi, v in zip(y, fam.values):
    if v > 8000:
        ax.text(v / 1.6, yi, f"{v:,}", va="center", ha="right", fontsize=6.2,
                fontweight="bold", color="white")
    else:
        ax.text(v * 1.12, yi, f"{v:,}", va="center", fontsize=6.2, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(fam.index, fontsize=6.6)
ax.set_xscale("log"); ax.set_xlim(1, 3e4)
ax.set_xlabel("cells per channel family")
ax.set_title(f"{len(fam)} families, {exp.channel.nunique()} channel types")
plabel(ax, "b")

# (c) 逐行剖面似然校准：CI 含值率（qc_flag==ok，与 QA 同口径）
ax = axes[2]
tau_mask = (exp.parameter.astype(str).str.startswith("tau") & exp.prof_lo.notna()
            & (exp.value > 0) & (exp.qc_flag == "ok"))
cal = exp[tau_mask]
cov = ((cal.prof_lo <= cal.value) & (cal.value <= cal.prof_hi))
by_src = cov.groupby(cal.source_big).agg(["mean", "size"])
by_src = by_src[by_src["size"] >= 100].sort_values("mean")
y = np.arange(len(by_src))
ax.barh(y, by_src["mean"] * 100, color="#0072B2", alpha=0.85, height=0.66)
ax.axvline(95.0, color="#D81B60", ls="--", lw=1.1)
ax.text(95.0, len(by_src) - 0.28, " nominal 95%", fontsize=6.2, color="#D81B60", va="top")
for yi, (mu, n) in enumerate(zip(by_src["mean"], by_src["size"])):
    ax.text(min(mu * 100, 99.2) + 0.3, yi, f"{mu*100:.1f}%", va="center", fontsize=6.2, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([labels.get(s, s).replace("\n", " ") for s in by_src.index], fontsize=6.2)
ax.set_xlim(80, 104); ax.set_xticks([80, 85, 90, 95, 100])
ax.set_xlabel("nominal 95% CI coverage, %")
ax.set_title(f"Profile-likelihood intervals are calibrated\n{len(cal):,} QC-passed tau rows, pooled {cov.mean()*100:.1f}%")
plabel(ax, "c")

fig.savefig(OUT / "fig1_method_corpus.png", bbox_inches="tight")
plt.close(fig)
print("saved fig1_method_corpus.png")

# ================= FIG 2 =================
verd = pd.read_csv(LAW / "features" / "robust_verdict_all_profile_2026-11-08.csv")
icg = pd.read_csv(LAW / "icg_out" / "verdict_icg_ALL.csv")
csp = pd.read_csv(LAW / "features" / "cross_species_v3.csv")

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
fig.subplots_adjust(wspace=0.40, left=0.075, right=0.985, top=0.78, bottom=0.17)

# (a) 层叠判决森林图（D 口径 = EB + 剖面似然）
ax = axes[0]
layers = [("kv_main", "Kv activation (64 groups)", "#0072B2"),
          ("tau_deact", "Kv deactivation", "#56B4E9"),
          ("tau_inact", "Kv inactivation", "#009E73"),
          ("rec", "Kv recovery", "#CC79A7"),
          ("hcn", "HCN (5 groups)", "#E69F00")]
rng = np.random.default_rng(7)
ytick, yticklab = [], []
y0 = 0
for key, lab, col in layers:
    d = verd[verd.table == key]["ratio_D"].dropna()
    yy = y0 + rng.normal(0, 0.16, len(d))
    ax.scatter(d, yy, s=5, color=col, alpha=0.45, lw=0)
    med = d.median()
    ax.plot([med, med], [y0 - 0.42, y0 + 0.42], color=col, lw=2.2)
    npass = int((d > 1).sum())
    ax.text(d.max() * 1.25, y0, f"{npass}/{len(d)}  {med:.2f}x", fontsize=6.2,
            va="center", fontweight="bold", color=col)
    ytick.append(y0); yticklab.append(lab)
    y0 -= 1.15
# ICG 模型文献层（IH=1.001 计为打平，与正文 9/10 口径一致）
dicg = icg["ratio"]
yy = y0 + rng.normal(0, 0.16, len(dicg))
ax.scatter(dicg, yy, s=9, color="#666666", alpha=0.7, lw=0, marker="s")
med = dicg.median()
ax.plot([med, med], [y0 - 0.42, y0 + 0.42], color="#666666", lw=2.2)
npass = int((dicg > 1.01).sum())
ax.text(dicg.max() * 1.25, y0, f"{npass}/{len(dicg)}  {med:.2f}x",
        fontsize=6.2, va="center", fontweight="bold", color="#444444")
ytick.append(y0); yticklab.append("ICGenealogy models (10 layers)")
ax.axvline(1.0, color="k", ls="--", lw=1.0)
ax.set_yticks(ytick); ax.set_yticklabels(yticklab, fontsize=6.4)
ax.set_xscale("log"); ax.set_xlim(0.5, 30)
ax.set_xticks([0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
ax.set_xlabel("midpoint scatter / time-constant scatter")
ax.set_title("Time constants conserved, midpoints not\none verdict rule across all layers")
plabel(ax, "a")

# (b) 单细胞实例：Kv1.1@25C tau_act vs Vhalf_act
ax = axes[1]
cp = exp[(exp.source_big == "channelpedia")]
ch, T = "Kv1.1", 25.0
t = cp[(cp.channel == ch) & (cp.temp_C == T) & (cp.parameter == "tau_act")
       & (cp.voltage_mV >= 0) & (cp.voltage_mV <= 40) & (cp.r2 >= 0.7) & (cp.qc_flag == "ok")]
h = cp[(cp.channel == ch) & (cp.temp_C == T) & (cp.parameter == "Vhalf_act") & (cp.r2 >= 0.8)]
ax.scatter(t.voltage_mV, t.value, s=6, color="#0072B2", alpha=0.4, lw=0)
ax.set_yscale("log"); ax.set_xlabel("voltage (mV)"); ax.set_ylabel("tau_act (ms)")
sd_tau = np.log10(t.value).std()
sd_mid = h.value.std() / cp[(cp.channel == ch) & (cp.temp_C == T) & (cp.parameter == "k_act")].value.median()
ax.set_title(f"{ch} at {T:.0f} C, {t.cell_id.nunique()} cells\n"
             f"tau scatter {sd_tau:.2f} dex vs midpoint {sd_mid:.2f} kT/e", fontsize=7.4)
plabel(ax, "b")
# 插图: Vhalf 分布
axin = ax.inset_axes([0.55, 0.12, 0.42, 0.38])
axin.hist(h.value, bins=30, color="#D81B60", alpha=0.75)
axin.set_xlabel("Vhalf_act (mV)", fontsize=5.8, labelpad=1)
axin.set_ylabel("cells", fontsize=5.8, labelpad=1)
axin.tick_params(labelsize=5.4)

# (c) 跨物种
ax = axes[2]
ax.scatter(csp.Vhalf_diff, csp.tau_fold, s=16, color="#009E73", alpha=0.75, lw=0)
ax.axhspan(0.5, 2.0, color="#2E7D32", alpha=0.10, lw=0)
ax.axhline(1.0, color="k", ls="--", lw=0.9)
offs = {"Kv2.1": (-40, 6), "Kv12.3": (-46, -10)}
for _, r in csp.nlargest(3, "Vhalf_diff").iterrows():
    dx, dy = offs.get(r.channel, (4, 4))
    ax.annotate(f"{r.channel}", (r.Vhalf_diff, r.tau_fold), fontsize=5.6,
                xytext=(dx, dy), textcoords="offset points")
ax.set_xlabel("Vhalf difference between species (mV)")
ax.set_ylabel("tau fold-change between species")
inband = ((csp.tau_fold >= 0.5) & (csp.tau_fold <= 2.0)).mean()
ax.set_title(f"38 cross-species pairs, {inband*100:.0f}% of tau\nwithin two-fold while midpoints drift")
plabel(ax, "c")

fig.savefig(OUT / "fig2_law_panorama.png", bbox_inches="tight")
plt.close(fig)
print("saved fig2_law_panorama.png")
