# -*- coding: utf-8 -*-
"""
新论文 Fig5（通道卡：Nav1.5 / CaV1.2 / IKs / Beattie hERG）与 Fig6（ICG 模型文献层 + 边界登记）
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

def plabel(ax, s):
    ax.text(-0.24, 1.05, s, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", va="top", ha="left")

m = pd.read_csv(HERE / "master_unified_v1.csv", low_memory=False)

# ================= FIG 5 =================
fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6))
fig.subplots_adjust(wspace=0.44, left=0.06, right=0.99, top=0.78, bottom=0.18)

# (a) Nav1.5 tau_inact(V)，两温度；插图：CP 浓度下 tau 不动
ax = axes[0]
nav = m[(m.source_big == "nav15_27193878") & (m.qc_flag == "ok")]
for T, col in [(25.0, "#0072B2"), (35.0, "#D81B60")]:
    d = nav[nav.temp_C == T]
    for cid, g0 in d.groupby("cell_id"):
        g0 = g0.sort_values("voltage_mV")
        ax.plot(g0.voltage_mV, g0.value, "-", color=col, lw=0.5, alpha=0.35)
    med = d.groupby("voltage_mV")["value"].median()
    ax.plot(med.index, med.values, "o-", color=col, lw=1.6, ms=3, label=f"{T:.0f} C")
ax.set_yscale("log"); ax.set_xlabel("voltage (mV)"); ax.set_ylabel("tau_inact (ms)")
ax.set_title("Nav1.5 inactivation kinetics\n4 cells, two temperatures", fontsize=7.4)
ax.legend(frameon=False, fontsize=6.0, loc="upper left")
plabel(ax, "a")
axin = ax.inset_axes([0.58, 0.56, 0.40, 0.40])
d35 = nav[nav.temp_C == 35.0].copy()
d35["drug"] = d35["drug"].replace("", "CP0").fillna("CP0")
order = ["CP0", "CP20", "CP40", "CP60", "CP80"]
meds = [d35[d35.drug == c]["value"].median() for c in order]
ns = [len(d35[d35.drug == c]) for c in order]
axin.bar(range(5), meds, color="#7B1FA2", alpha=0.75, width=0.62)
for i, (md, n) in enumerate(zip(meds, ns)):
    axin.text(i, md + 0.008, f"{md:.2f}", ha="center", fontsize=5.2, fontweight="bold")
axin.set_xticks(range(5)); axin.set_xticklabels(["0", "20", "40", "60", "80"], fontsize=5.0)
axin.set_ylim(0, max(meds) * 1.3)
axin.set_xlabel("CP level", fontsize=5.4, labelpad=1)
axin.set_ylabel("tau_inact (ms)", fontsize=5.4, labelpad=1)
axin.set_title("tau flat across CP", fontsize=5.6)
axin.tick_params(labelsize=5.0)

# (b) CaV1.2 逐细胞 tau_inact 分布
ax = axes[1]
cav = m[(m.source_big == "cav12_ren2022") & (m.dataset == "Temperature_Ca2")]
ok = cav[cav.qc_flag == "ok"]
ed = cav[cav.qc_flag == "tau_edge"]
cellcols = dict(zip(cav.cell_id.unique(),
                    ["#0072B2", "#D81B60", "#E69F00", "#009E73", "#7B1FA2", "#F58231"]))
for cid, g0 in cav.groupby("cell_id"):
    col = cellcols[cid]
    ax.scatter(g0.temp_C, g0.value, s=4, color=col,
               alpha=0.8 if (g0.qc_flag == "ok").all() else 0.35, lw=0)
ax.axhline(500, color="#999", ls=":", lw=1.0)
ax.text(32.2, 520, "fit ceiling (500 ms)", fontsize=5.8, color="#666666")
med_ok = ok.value.median()
ax.axhline(med_ok, color="#F58231", ls="--", lw=1.4)
ax.text(32.2, med_ok * 0.78, f"median {med_ok:.0f} ms", fontsize=6.2, color="#F58231",
        fontweight="bold")
ax.set_yscale("log"); ax.set_yticks([100, 200, 500]); ax.set_yticklabels(["100", "200", "500"])
ax.set_xlabel("bath temperature (°C)"); ax.set_ylabel("tau_inact (ms)")
ax.set_title("CaV1.2 CDI time constant at +47 mV\n6 cells; faint points hit the fit ceiling", fontsize=7.4)
plabel(ax, "b")

# (c) IKs tau_act / tau_deact 中位带
ax = axes[2]
iks = m[(m.source_big == "iks_fedida") & (m.qc_flag == "ok")]
for pname, col, lab in [("tau_act", "#009E73", "activation"),
                        ("tau_deact", "#0072B2", "deactivation (tail)")]:
    d = iks[(iks.parameter == pname) & (iks.voltage_mV >= -40)]
    g = d.groupby("voltage_mV")["value"]
    med, q25, q75 = g.median(), g.quantile(0.25), g.quantile(0.75)
    ax.plot(med.index, med.values, "o-", color=col, lw=1.6, ms=3, label=lab)
    ax.fill_between(med.index, q25.values, q75.values, color=col, alpha=0.18, lw=0)
ax.set_yscale("log"); ax.set_xlabel("voltage (mV)"); ax.set_ylabel("tau (ms)")
ax.set_title("IKs activation and deactivation\n283 recordings, median with IQR", fontsize=7.4)
ax.legend(frameon=False, fontsize=6.0, loc="upper left")
plabel(ax, "c")

# (d) Beattie 九细胞（hERG 去激活双指数：慢分量为标志性慢门控）
ax = axes[3]
b9 = m[(m.source_big == "beattie9") & (m.qc_flag == "ok")]
h = b9[b9.parameter == "Vhalf_act"]
for pname, col, lab in [("tau_deact_slow", "#004D40", "slow component"),
                        ("tau_deact_fast", "#56B4E9", "fast component")]:
    t = b9[b9.parameter == pname]
    for cid, g0 in t.groupby("cell_id"):
        g0 = g0.sort_values("voltage_mV")
        ax.plot(g0.voltage_mV, g0.value, "-", color=col, lw=0.5, alpha=0.4)
    med = t.groupby("voltage_mV")["value"].median()
    ax.plot(med.index, med.values, "o-", color=col, lw=1.7, ms=3, label=lab)
ax.set_yscale("log"); ax.set_xlabel("voltage (mV)"); ax.set_ylabel("tau_deact (ms)")
ax.set_title(f"hERG deactivation, nine Beattie cells\nVhalf_act median {h.value.median():.1f} mV "
             f"(IQR {h.value.quantile(0.25):.1f} to {h.value.quantile(0.75):.1f})", fontsize=7.4)
ax.legend(frameon=False, fontsize=6.0, loc="upper left")
plabel(ax, "d")

fig.savefig(OUT / "fig5_channel_cards.png", bbox_inches="tight")
plt.close(fig)
print("saved fig5_channel_cards.png")

# ================= FIG 6 =================
icg = pd.read_csv(LAW / "icg_out" / "verdict_icg_ALL.csv")
verd = pd.read_csv(LAW / "features" / "robust_verdict_all_profile_2026-11-08.csv")

fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.7))
fig.subplots_adjust(wspace=0.40, left=0.06, right=0.99, top=0.78, bottom=0.17)

# (a) ICG 十层判决
ax = axes[0]
icg2 = icg.copy()
GATE = {"act": "activation", "inact": "inactivation"}
icg2["lab"] = icg2["cls"].astype(str) + " " + icg2["gate"].map(GATE)
FAMC = {"K": "#0072B2", "Na": "#7B1FA2", "KCa": "#009E73", "IH": "#E69F00", "Ca": "#F58231"}
y = np.arange(len(icg2))[::-1]
ax.barh(y, icg2.ratio, color=[FAMC[f] for f in icg2.family], alpha=0.85, height=0.66)
for yi, (_, r) in zip(y, icg2.iterrows()):
    tag = f"{r.ratio:.2f}x (n={r.n_models})"
    if abs(r.ratio - 1) <= 0.01:
        tag += "  tie"
    ax.text(r.ratio + 0.07, yi, tag, va="center", fontsize=6.0, fontweight="bold")
ax.axvline(1.0, color="k", ls="--", lw=1.0)
ax.set_yticks(y); ax.set_yticklabels(icg2["lab"], fontsize=5.8)
ax.set_xlim(0, 6.4); ax.set_xlabel("midpoint scatter / time-constant scatter")
ax.set_title("The same verdict in 933 published models\n(ICGenealogy, curves refit in-house)", fontsize=7.4)
plabel(ax, "a")

# (b) tau_obs vs mid_obs 散布（编号+图内名录，防遮挡）
ax = axes[1]
names = []
for i, (_, r) in enumerate(icg.iterrows(), 1):
    ax.scatter(r.tau_obs, r.mid_obs, s=58, color=FAMC[r.family], alpha=0.85, lw=0)
    ax.text(r.tau_obs, r.mid_obs, str(i), fontsize=5.6, color="white",
            ha="center", va="center", fontweight="bold")
    names.append(f"{i} {r.cls} {r.gate}")
ax.text(0.03, 0.97, "\n".join(names[:5]), fontsize=5.2, va="top", transform=ax.transAxes)
ax.text(0.35, 0.97, "\n".join(names[5:]), fontsize=5.2, va="top", transform=ax.transAxes)
lim = [0, 4.6]
ax.plot(lim, lim, "k--", lw=0.9)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("time-constant scatter (dex)")
ax.set_ylabel("midpoint scatter (kT/e units)")
ax.set_title("All layers above the diagonal\nexcept the inbred IH model set", fontsize=7.4)
plabel(ax, "b")

# (c) 诚实登记
ax = axes[2]
ax.axis("off")
entries = [
    ("Kv3.2 at 15 C", "sole failure of the main verdict (0.83x);\nkinetics, not midpoint, carries its diversity", "#D81B60"),
    ("IH model layer", "ties at 1.00x; small and inbred model set\n(same family in cells passes at 8.25x)", "#E69F00"),
    ("Window-current hypothesis", "negative: activation and inactivation midpoints\nare independent set points (median r = 0.30, 35 pairs)", "#0072B2"),
    ("hERG midpoint T-drift", "+1.14 / -1.47 mV per C is hERG-specific,\nnot a family-wide rule (family median -0.05)", "#009E73"),
    ("kon not extractable", "apparent onset rate flattened by solution exchange;\nonly equilibrium quantities are pinned by this dataset", "#7B1FA2"),
]
y0 = 0.98
for t, d, c in entries:
    ax.text(0.02, y0, t, fontsize=7.0, fontweight="bold", color=c, va="top", transform=ax.transAxes)
    ax.text(0.02, y0 - 0.075, d, fontsize=6.2, va="top", transform=ax.transAxes, color="#333333")
    y0 -= 0.21
ax.set_title("Boundary registry (pre-registered exceptions)", fontsize=7.6, loc="left")
plabel(ax, "c")

fig.savefig(OUT / "fig6_icg_boundary.png", bbox_inches="tight")
plt.close(fig)
print("saved fig6_icg_boundary.png")
