# make_EDFig345_channel_cards_2026-09-21.py
# Manuscript ED3/ED4/ED5 channel cards: Nav1.5 / CaV1.2 / IKs, each 2x2, box+strip style same as ED8 v04.
# Data: per-cell master table + drug-arm table + verdict JSONs/CSVs (all from sealed/registered sources; the figure only reads).
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
LINE = os.path.dirname(ROOT)
RES = os.path.join(LINE, "结果")
AM = os.path.join(LINE, "α模型")
FIGD = os.path.join(ROOT, "figures_v01")

import figstyle_v05 as fs
fs.apply()

df = pd.read_csv(os.path.join(RES, "逐细胞参数总表_四通道_2026-09-21.csv"))
df = df[df["ok"] == 1]
arm = pd.read_csv(os.path.join(RES, "逐细胞参数总表_加药臂_2026-09-21.csv"))

def box_strip(ax, d, x, color, w=0.52):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return
    xj = x + np.random.default_rng(7).uniform(-0.15, 0.15, n)
    ax.scatter(xj, d, s=3.0, c="k", alpha=0.4, linewidths=0, zorder=3)
    med = np.median(d)
    if n >= 5:
        q1, q3 = np.percentile(d, [25, 75])
        p10, p90 = np.percentile(d, [10, 90])
        ax.add_patch(Rectangle((x - w / 2, q1), w, q3 - q1, facecolor=color,
                               alpha=0.20, edgecolor=color, lw=0.9, zorder=2))
        ax.plot([x, x], [p10, q1], color=color, lw=0.9, zorder=2)
        ax.plot([x, x], [q3, p90], color=color, lw=0.9, zorder=2)
        ax.plot([x - 0.12, x + 0.12], [p10, p10], color=color, lw=0.9, zorder=2)
        ax.plot([x - 0.12, x + 0.12], [p90, p90], color=color, lw=0.9, zorder=2)
        ax.plot([x - w / 2, x + w / 2], [med, med], color=color, lw=2.0, zorder=4)
    else:
        ax.plot([x - 0.26, x + 0.26], [med, med], color=color, lw=1.8, zorder=4)

def letters(fig, axes):
    for i, ax in enumerate(axes):
        ax.text(-0.24, 1.05, "abcd"[i], transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")

NAVC, CAVC, IKSC = "#DD8452", "#8172B3", "#55A868"

# ================= ED3 · Nav1.5 =================
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.13, hspace=0.52, wspace=0.38)

# a: tau_h(V) per-cell curves, all cells (Lei-Nav-HEK35, 4 cells x -20..+40)
ax = axes[0, 0]
th = df[(df.channel == "Nav1.5") & (df.parameter == "tau_h") & (df.dataset == "Lei-Nav-HEK35")]
for cid, sub in th.groupby("cell_id"):
    sub = sub.sort_values("voltage_mV")
    ax.plot(sub.voltage_mV, sub.value, "-o", ms=2.5, lw=0.8, color=NAVC, alpha=0.55)
med = th.groupby("voltage_mV")["value"].median()
ax.plot(med.index, med.values, "k-", lw=1.8, zorder=5)
ax.set_xlabel("voltage (mV)")
ax.set_ylabel(r"per-cell $\tau_h$(V) (ms)")
ax.set_title("Nav1.5 · $\\tau_h$(V) mid-band constant, per-cell (n=4)", fontsize=6.8)

# b: V1/2,h per construct (G1_WT/G2_BC2/G3_dKPQ)
ax = axes[0, 1]
gs_ = ["G1_WT", "G2_BC2", "G3_dKPQ"]
data = [df[(df.channel == "Nav1.5") & (df.parameter == "V_half_inact") & (df.group == g)].value.to_numpy(float)
        for g in gs_]
for i, d in enumerate(data):
    box_strip(ax, d, i + 1, NAVC)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f"{g.replace('G1_','').replace('G2_','').replace('G3_','')}\n({len(d)})"
                    for g, d in zip(gs_, data)], fontsize=6)
ax.set_ylabel(r"per-cell $V_{1/2}$ inact (mV)")
ax.set_title("Nav1.5 · availability $V_{1/2}$: per-cell, not constant (Nα-2 C2)", fontsize=6.8)

# c: s_ref criterion-1 pool (WT+BC2, n=12, sealed CV=0.184) + not-portable controls (dKPQ/myo/Lei)
# read the Nα-3 verdict JSON directly, numbers verbatim identical to the verdict card; P10-P90 band = criterion-3 sealed interval
ax = axes[1, 0]
n3 = json.load(open(os.path.join(AM, "2026-09-16_Nα3_Nav15激活脚部可携带性判决_结果.json"),
                    encoding="utf-8"))
v1 = n3["verdict"]["判1"]
p10, p90 = v1["P10"], v1["P90"]
gs2 = ["G1_WT", "G2_BC2", "G3_dKPQ", "G4_myo", "Lei_0CP", "Lei_80CP"]
lab2 = ["WT*", "BC2*", "dKPQ", "myo", "Lei\n0CP", "Lei\n80CP"]
data = []
for g in gs2:
    vals = [c["s_ref"] for c in n3["cells"] if c["group"] == g and c.get("ok")]
    data.append(np.asarray(vals, float))
ax.axhspan(p10, p90, color=NAVC, alpha=0.10, zorder=1)
ax.axhline(v1["median"], color=NAVC, lw=1.0, ls=":", zorder=2)
for i, d in enumerate(data):
    col = NAVC if i < 2 else "0.55"
    box_strip(ax, d, i + 1, col)
ax.set_xticks(range(1, 7))
ax.set_xticklabels([f"{l}\n({len(d)})" for l, d in zip(lab2, data)], fontsize=5.0)
ax.set_ylabel(r"per-cell $s_{ref}$ (mV$^{-1}$)")
ax.set_title(f"Nav1.5 · activation foot: pool* n=12, CV=0.184 real dispersion "
             f"(Q=77.9 >> chi2=19.7);\nnot portable: dKPQ −34.9%, Lei 0/3 in band (Nα-3)",
             fontsize=6.2)

# d: tau(-40) vs late_pct scatter (61 patches, negative-result panel)
ax = axes[1, 1]
td = df[(df.channel == "Nav1.5") & (df.parameter == "tau_decay") & (df.group == "多通道膜片")]
lp = df[(df.channel == "Nav1.5") & (df.parameter == "late_pct") & (df.group == "多通道膜片")]
j = td.set_index("cell_id")["value"].to_frame("tau").join(
    lp.set_index("cell_id")["value"].to_frame("late"), how="inner").dropna()
rho, p = spearmanr(j.tau, j.late)
ax.scatter(j.late, j.tau, s=6, c=NAVC, alpha=0.6, linewidths=0)
ax.set_yscale("log")
ax.set_xlabel("late-current fraction")
ax.set_ylabel(r"per-patch $\tau_h$(−40 mV) (ms)")
ax.set_title(f"Nav1.5 · negative result: $\\tau_h$(−40) is a distribution "
             f"(n={len(j)}, CV=0.77, ρ={rho:.2f})", fontsize=6.6)
letters(fig, axes.flat)
fig.text(0.09, 0.008, "All values from the deposited per-cell master table (SI S7); "
                      "verdicts sealed/registered per Nα-1…Nα-4 cards (SI S1.2).", fontsize=6, color="0.35")
fig.suptitle("Nav1.5 judgement card: inactivation constants, portable and non-portable coordinates",
             fontsize=8.5, fontweight="bold", y=0.995)
fs.save(fig, os.path.join(FIGD, "EDFig3_Nav15_channel_card"))
print("ED3 saved; s_ref pool CV =", round(v1["CV"], 3), "n =", v1["n"], "; rho =", round(rho, 3))

# ================= ED4 · CaV1.2 =================
c1b = json.load(open(os.path.join(AM, "2026-09-15_Cα1b_CaV12失活电荷载子判决_17mV段_结果.json"), encoding="utf-8"))
c3 = json.load(open(os.path.join(AM, "2026-09-15_Cα3v12_CDI结构形式判决_结果.json", ), encoding="utf-8"))
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.11, hspace=0.52, wspace=0.38)

# a: f_inact characterization batch Ca2+ vs Ba2+
ax = axes[0, 0]
fca = list(c1b["verdict"]["f_inact_cells"]["Ca2"].values())
fba = list(c1b["verdict"]["f_inact_cells"]["Ba2"].values())
box_strip(ax, fca, 1, CAVC); box_strip(ax, fba, 2, CAVC)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"Ca$^{{2+}}$\n({len(fca)})", f"Ba$^{{2+}}$\n({len(fba)})"], fontsize=6.5)
ax.set_ylabel(r"pre-drug $f_{inact}$(+17 mV, 40 ms)")
ax.set_ylim(0, 1.0)
ax.set_title("CaV1.2 · inactivation fraction by carrier (Cα-1b)", fontsize=6.8)

# b: w_fast per separable cell + population line + delta-f dashed line
ax = axes[0, 1]
ws, taus_f = [], []
for cell, v in c3["cells"]["Ca2"].items():
    if v["fit"].get("pick") == "H1":
        ws.append(v["fit"]["H1"]["w"])
        taus_f.append(v["fit"]["H1"]["tauf"])
gv = c3["verdict"]["group_Ca2"]["H1"]
box_strip(ax, ws, 1, CAVC)
ax.axhline(gv["w"], color=CAVC, lw=1.4, zorder=5)
ax.text(1.32, gv["w"] + 0.014, f"group $w_{{fast}}$ = {gv['w']:.3f}", fontsize=5.8, color=CAVC)
ax.axhline(0.293, color="#B22222", lw=1.0, ls="--", zorder=5)
ax.text(1.32, 0.293 - 0.024, "independent Δf = 0.293", fontsize=5.8, color="#B22222")
ax.set_xticks([1])
ax.set_xticklabels([f"separable\ncells\n({len(ws)})"], fontsize=6)
ax.set_xlim(0.4, 2.2)
ax.set_ylim(0.15, 0.45)
ax.set_ylabel(r"$w_{fast}$")
ax.set_title("CaV1.2 · CDI additive fast component (Cα-3)", fontsize=6.8)

# c: tau_f per cell + tau_slow population two points
ax = axes[1, 0]
box_strip(ax, taus_f, 1, CAVC)
gb = c3["verdict"]["group_Ba2"]["H1"]
ax.scatter([2, 3], [gv["tau"], gb["tau"]], s=30, c=CAVC, zorder=5)
for x, val in [(2, gv["tau"]), (3, gb["tau"])]:
    ax.text(x, val + 3, f"{val:.1f}", ha="center", fontsize=6, color=CAVC)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f"$\\tau_f$\n({len(taus_f)})", "$\\tau_{slow}$\nCa$^{2+}$", "$\\tau_{slow}$\nBa$^{2+}$"],
                   fontsize=6)
ax.set_ylabel("ms")
ax.set_title("CaV1.2 · CDI kinetics: fast 6–12 ms, slow 50.0/41.6 ms", fontsize=6.8)

# d: drug effect delta-f per cell by carrier (VeraRT highlighted in red)
ax = axes[1, 1]
sub = arm[arm.parameter.isin(["f_inact", "f_inact_drug"]) & (arm.group != "表征对照_Ca2") & (arm.group != "表征对照_Ba2")]
fc = sub[sub.parameter == "f_inact"].set_index(["group", "cell_id"])["value"]
fd = sub[sub.parameter == "f_inact_drug"].set_index(["group", "cell_id"])["value"]
dff = (fd - fc).dropna().rename("df").reset_index()
dff["carrier"] = dff.group.str.contains("Ca2").map({True: "Ca", False: "Ba"})
dff["rt"] = dff.group.str.contains("RT")
for i, car in enumerate(["Ca", "Ba"]):
    d_pt = dff[(dff.carrier == car) & (~dff.rt)]["df"].to_numpy(float)
    box_strip(ax, d_pt, i * 2 + 1, CAVC)
    d_rt = dff[(dff.carrier == car) & (dff.rt)]["df"].to_numpy(float)
    if len(d_rt):
        xj = i * 2 + 2 + np.random.default_rng(7).uniform(-0.15, 0.15, len(d_rt))
        ax.scatter(xj, d_rt, s=5, c="#B22222", alpha=0.7, linewidths=0, zorder=3)
        ax.plot([i * 2 + 2 - 0.26, i * 2 + 2 + 0.26], [np.median(d_rt)] * 2,
                color="#B22222", lw=1.4, zorder=4)
ax.axhline(0, lw=0.7, color="0.4", zorder=1)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Ca$^{2+}$ pool", "Ca$^{2+}$ VeraRT\n(registered)", "Ba$^{2+}$ pool"], fontsize=5.8)
ax.set_ylabel(r"$\Delta f_{inact}$ (drug − pre)")
ax.set_title("CaV1.2 · drug action by carrier pool (Cα-5)", fontsize=6.8)
letters(fig, axes.flat)
fig.text(0.09, 0.008, "Panels a–c: characterization batch (Cα-1b/Cα-3); panel d: drug-screening batch (Cα-5, "
                      "pre-registered). Absolute f_inact batch-anchored (SI S1.3a).", fontsize=6, color="0.35")
fig.suptitle("CaV1.2 judgement card: carrier-dependent inactivation and one additive CDI term",
             fontsize=8.5, fontweight="bold", y=0.995)
fs.save(fig, os.path.join(FIGD, "EDFig4_CaV12_channel_card"))
print("ED4 saved; separable cells:", len(ws), "w =", [round(w, 3) for w in ws],
      "tauf =", taus_f, "tau_slow =", round(gv["tau"], 1), "/", round(gb["tau"], 1))

# ================= ED5 · IKs =================
k1 = pd.read_csv(os.path.join(AM, "2026-09-15_Kα1_IKs激活去激活恒定性判决_逐文件表.csv"))
k3 = json.load(open(os.path.join(AM, "2026-09-15_Kα3_IKs前向验证_结果.json"), encoding="utf-8"))
pool = k1[k1.in_pool == 1]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.11, hspace=0.52, wspace=0.38)

# a: V1/2 / k WT pool + Chan anchor star
ax = axes[0, 0]
box_strip(ax, pool.Vh.to_numpy(float), 1, IKSC)
box_strip(ax, pool.k.to_numpy(float), 2, IKSC)
ax.scatter([0.62], [25.4], marker="*", s=50, c="#B22222", zorder=6)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"$V_{{1/2}}$ act\n({pool.Vh.notna().sum()})", f"$k$ act\n({pool.k.notna().sum()})"],
                   fontsize=6.5)
ax.set_ylabel("mV")
ax.text(0.03, 0.95, "red star: Chan 2023 anchor 25.4 (Kα-1 J0 PASS)", transform=ax.transAxes,
        fontsize=5.5, color="#B22222", va="top")
ax.set_title("IKs · WT pool G–V parameters (n=8)", fontsize=6.8)

# b: instantaneous component inst_med per cell + beta=0.30 line
ax = axes[0, 1]
box_strip(ax, pool.inst_med.to_numpy(float), 1, IKSC)
ax.axhline(0.30, color=IKSC, lw=1.2, ls="--", zorder=5)
ax.text(1.35, 0.305, "pool constant β = 0.30", fontsize=5.8, color="#2d5a3a")
ax.set_xticks([1])
ax.set_xticklabels([f"WT pool\n({pool.inst_med.notna().sum()})"], fontsize=6.5)
ax.set_xlim(0.4, 2.0)
ax.set_ylabel("instantaneous fraction")
ax.set_title("IKs · instantaneous component, per cell (Kα-1)", fontsize=6.8)

# c: Kα3 arm A activation forward R2 per cell + 0.80 criterion line
ax = axes[1, 0]
cellsA = k3["arms"]["A"]["cells"]
r2 = [c["r2_act"] for c in cellsA]
box_strip(ax, r2, 1, IKSC)
ax.axhline(0.80, color="#B22222", lw=1.0, ls="--", zorder=5)
ax.text(1.35, 0.815, "pre-registered line 0.80", fontsize=5.8, color="#B22222")
npass = k3["arms"]["A"]["判A_激活R2≥0.80细胞数"]
ax.set_xticks([1])
ax.set_xticklabels([f"held-out\ncells\n({len(r2)})"], fontsize=6.5)
ax.set_xlim(0.4, 2.0)
ax.set_ylim(0.5, 1.02)
ax.set_ylabel(r"activation forward $R^2$")
ax.set_title(f"IKs · activation forward: {npass}/{len(r2)} pass (Kα-3 arm A)", fontsize=6.8)

# d: tau_app apparent rejection: archived per-file table n=8 colored by protocol (red); sealed J2 numbers in text, not plotted; anchor band 2-3 s
ax = axes[1, 1]
ta_pool = pool.dropna(subset=["tau_app_s"])
for proto, mk, lab in [("KCNQ act 4sec - low to high", "o", "4-s protocol"),
                       ("IKs activation -80", "s", "step protocol")]:
    subp = ta_pool[ta_pool.protocol == proto]
    xj = 1 + np.random.default_rng(7).uniform(-0.15, 0.15, len(subp))
    ax.scatter(xj, subp.tau_app_s, s=9, marker=mk, c="#B22222", alpha=0.8,
               linewidths=0, zorder=3, label=lab)
ax.plot([1 - 0.26, 1 + 0.26], [np.median(ta_pool.tau_app_s)] * 2,
        color="#B22222", lw=1.8, zorder=4)
ax.axhspan(2.0, 3.0, color=IKSC, alpha=0.15, zorder=1)
ax.text(1.42, 2.5, "anchored magnitude 2–3 s\n(J3: $\\tau_{60}$ med 1.5 s, registered)",
        fontsize=5.6, color="#2d5a3a", va="center")
ax.text(1.42, 0.62, "sealed J2: n=5, med 0.445 s, CV 0.313\n→ apparent $\\tau$ REJECTED (window artefact)",
        fontsize=5.6, color="#B22222", va="center")
ax.legend(fontsize=5.2, loc="upper left", frameon=False, handletextpad=0.2,
          borderaxespad=0.1, labelspacing=0.25)
ax.set_xticks([1])
ax.set_xticklabels([f"deposited per-file\n$\\tau_{{app}}$ (n={len(ta_pool)})"], fontsize=6.5)
ax.set_xlim(0.4, 2.6)
ax.set_ylabel(r"apparent $\tau_{deact}$(−40 mV) (s)")
ax.set_title("IKs · deactivation: apparent rejected, magnitude anchored", fontsize=6.8)
letters(fig, axes.flat)
fig.text(0.09, 0.008, "Values from Kα-1 per-file table and Kα-3 forward card (SI S1.4). Panel d: deposited "
                      "re-extracted set (n=8); sealed J2 used the pre-registration extraction (n=5); both reject "
                      "a constant apparent τ.", fontsize=5.8, color="0.35")
fig.suptitle("IKs judgement card: pool constant beta, bell-shaped activation timescale, window artefacts",
             fontsize=8.5, fontweight="bold", y=0.995)
fs.save(fig, os.path.join(FIGD, "EDFig5_IKs_channel_card"))
print("ED5 saved; pool n =", len(pool), "; forward pass =", npass, "/", len(r2),
      "; tau_app deposited med =", round(float(np.median(ta_pool.tau_app_s)), 3))
