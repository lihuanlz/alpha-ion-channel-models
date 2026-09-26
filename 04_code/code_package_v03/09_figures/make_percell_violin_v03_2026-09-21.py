# make_percell_violin_v03_2026-09-21.py
# ED8 v03：在 v02 基础上把药理臂细胞补全进图（加药/不加药分列）。
#   e: Nav 药物臂剂量-效应（Nα药，63 条件）
#   f: IKs V½/k —— WT(8) | Mef(15) | DIDS(26) | HMR-sub(6) 分列
#   h: CaV f_inact 对照 —— 表征批(6/4) | 筛选批对照段(77/20) 分列
#   i: CaV 药物效应 Δf 逐组（加药段单独成面板）
# 同时输出加药臂逐细胞长表 CSV（主表封卷 18,710 行不动，臂表独立登记）。
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
LINE = os.path.dirname(ROOT)
RES = os.path.join(LINE, "结果")
AM = os.path.join(LINE, "α模型")
FIGD = os.path.join(ROOT, "figures_v01")
os.makedirs(FIGD, exist_ok=True)

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    pass
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

df = pd.read_csv(os.path.join(RES, "逐细胞参数总表_四通道_2026-09-21.csv"))
df = df[df["ok"] == 1]

def L(channel, param, voltage=None, group=None, temp=None, dataset=None):
    d = df[(df.channel == channel) & (df.parameter == param)]
    if voltage is not None:
        d = d[d.voltage_mV == voltage]
    if group is not None:
        d = d[d.group == group]
    if temp is not None:
        d = d[d.temp_C.astype(str) == str(temp)]
    if dataset is not None:
        d = d[d.dataset == dataset]
    return d["value"].to_numpy(float)

# ============ 1) 组装加药臂长表 ============
arm_rows = []

ik = pd.read_csv(os.path.join(AM, "2026-09-15_Kα药_IKs药物表分解判决_逐文件表_结果.csv"))
for _, r in ik.iterrows():
    for param, val, unit in [("V_half_act_drug", r.vh, "mV"), ("k_act_drug", r.k, "mV")]:
        arm_rows.append(["IKs", "Chan2023-Zenodo8226585", f"{r.arm}药臂", 22, r.file,
                         param, np.nan, val, unit, 1, f"drug-treated ({r.arm}); G-V R2={r.gv_r2:.3f}"])

kj = json.load(open(os.path.join(AM, "2026-09-15_Kα药_IKs药物表分解判决_结果.json"), encoding="utf-8"))
for f, v in kj["hmr"].items():
    for param, val, unit in [("V_half_act_drug", v["gv"][0], "mV"), ("k_act_drug", v["gv"][1], "mV")]:
        arm_rows.append(["IKs", "Chan2023-Zenodo8226585", "HMR减除臂", 22, f,
                         param, np.nan, val, unit, 1, f"HMR-1556 subtraction; G-V R2={v['gv'][2]:.3f}"])

c1 = json.load(open(os.path.join(AM, "2026-09-15_Cα1b_CaV12失活电荷载子判决_17mV段_结果.json"), encoding="utf-8"))
for carrier, cells in c1["verdict"]["f_inact_cells"].items():
    for f, val in cells.items():
        arm_rows.append(["CaV1.2", "Ren2022-g3msb", f"表征对照_{carrier}", "31-37", f,
                         "f_inact", 17.0, val, "frac", 1, "characterization-batch control (Cα1b)"])

c5 = json.load(open(os.path.join(AM, "2026-09-15_Cα5_CaV12药物形状调制判决_结果.json"), encoding="utf-8"))
for c in c5["cells"]:
    t = round(c["T_med"]) if c.get("T_med") else ""
    if c.get("f_ctrl") is not None:
        arm_rows.append(["CaV1.2", "Ren2022-g3msb", c["group"], t, c["file"],
                         "f_inact", 17.0, c["f_ctrl"], "frac", 1, "pre-drug control segment (Cα5)"])
    if c.get("f_drug") is not None:
        arm_rows.append(["CaV1.2", "Ren2022-g3msb", c["group"], t, c["file"],
                         "f_inact_drug", 17.0, c["f_drug"], "frac", 1, "drug segment (Cα5)"])

nv = pd.read_csv(os.path.join(AM, "2026-09-15_Nα药_Nav15药物形状调制判决_逐条件表_结果.csv"))
for _, r in nv.iterrows():
    arm_rows.append(["Nav1.5", "Tarasov2026-Dryad", f"{r.drug}药臂", "", r.file_id,
                     "f_block_step", np.nan, r.f_step, "frac", 1,
                     f"drug-treated; conc {r.conc_uM} uM; group {r.group}"])

arm = pd.DataFrame(arm_rows, columns=["channel", "dataset", "group", "temp_C", "cell_id",
                                      "parameter", "voltage_mV", "value", "unit", "ok", "note"])
csv_path = os.path.join(RES, "逐细胞参数总表_加药臂_2026-09-21.csv")
arm.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("arm table:", arm.shape, "->", csv_path)

def A(channel, param, group):
    d = arm[(arm.channel == channel) & (arm.parameter == param) & (arm.group == group)]
    return d["value"].to_numpy(float)

# ============ 2) 画图 ============
VIOLIN_MIN_N = 20

def draw_groups(ax, data_list, pos, color):
    for i, dd in enumerate(data_list):
        dd = np.asarray(dd, float)
        dd = dd[np.isfinite(dd)]
        if len(dd) == 0:
            continue
        if len(dd) >= VIOLIN_MIN_N:
            vp = ax.violinplot([dd], positions=[pos[i]], widths=0.72,
                               showmeans=False, showmedians=False, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(color); b.set_alpha(0.25)
                b.set_edgecolor(color); b.set_linewidth(0.8)
        xj = pos[i] + np.random.default_rng(7).uniform(-0.16, 0.16, len(dd))
        ax.scatter(xj, dd, s=3.5, c="k", alpha=0.45, linewidths=0, zorder=3)
        ax.hlines(np.median(dd), pos[i]-0.28, pos[i]+0.28, colors=color, lw=1.6, zorder=4)
        if 2 <= len(dd) < VIOLIN_MIN_N:
            q1, q3 = np.percentile(dd, [25, 75])
            ax.vlines(pos[i]+0.28, q1, q3, colors=color, lw=1.0, zorder=4)
            ax.hlines([q1, q3], pos[i]+0.22, pos[i]+0.34, colors=color, lw=1.0, zorder=4)

def panel(ax, data_list, labels, color, tick_fs=6.0, rot=0, star=None, star_label=""):
    pos = np.arange(1, len(data_list) + 1)
    draw_groups(ax, data_list, pos, color)
    ax.set_xticks(pos)
    labs = ["{}\n({})".format(l, np.isfinite(np.asarray(d, float)).sum())
            for l, d in zip(labels, data_list)]
    if rot:
        ax.set_xticklabels(labs, fontsize=tick_fs, rotation=rot, ha="right")
    else:
        ax.set_xticklabels(labs, fontsize=tick_fs)
    if star is not None:
        ax.scatter([star[1]], [star[0]], marker="*", s=80, c="#B22222", zorder=6)
        ax.annotate(star_label, (star[1], star[0]), textcoords="offset points",
                    xytext=(7, 3), fontsize=6, color="#B22222")
    ax.spines[["top", "right"]].set_visible(False)

fig, axes = plt.subplots(3, 3, figsize=(7.6, 7.0))
TEMPS = [25, 27, 30, 33, 37]

# a: hERG E_rev × T
ax = axes[0, 0]
panel(ax, [L("hERG", "E_rev", temp=t) for t in TEMPS], [str(t) for t in TEMPS],
      "#4C72B0", tick_fs=5.5, star=(-92.9, 5.0), star_label="model62-chain fit −92.9")
ax.set_ylabel("per-cell $E_{rev}$ (mV)")
ax.set_xlabel("temperature (°C)")
ax.set_title("hERG · Lei CHO 670 wells", fontsize=7)

# b: hERG h_ss limb 25 °C
ax = axes[0, 1]
panel(ax, [L("hERG", "h_ss", voltage=v, temp=25) for v in [-140, -120, -100]],
      ["−140", "−120", "−100"], "#4C72B0")
ax.set_ylabel("per-cell $h_{ss}$")
ax.set_xlabel("voltage (mV)")
ax.set_title("hERG · $h_{ss}$ negative limb, 25 °C", fontsize=7)

# c: hERG τ_rec(−120) × T
ax = axes[0, 2]
panel(ax, [L("hERG", "tau_rec", voltage=-120, temp=t) for t in TEMPS],
      [str(t) for t in TEMPS], "#4C72B0", tick_fs=5.5,
      star=(3.04, 5.0), star_label="Beattie HEK 3.04 ms")
ax.set_ylabel(r"per-cell $\tau_{rec}$(−120 mV) (ms)")
ax.set_xlabel("temperature (°C)")
ax.set_ylim(0, 18)
ax.set_yticks([0, 3, 6, 9, 12, 15, 18])
ax.set_title("hERG · $\tau_{rec}$ travels: $Q_{10}$=2.84, $R^2$=0.991", fontsize=7)

# d: Nav τ(−40) by modality
ax = axes[1, 0]
d1 = L("Nav1.5", "tau_decay", group="多通道膜片")
d2 = L("Nav1.5", "tau_decay", group="单通道膜片")
d3 = L("Nav1.5", "tau_h", voltage=-40.0, dataset="Nα2-Lei-Nav35")
draw_groups(ax, [d1, d2, []], [1, 2, 3], "#DD8452")
xj = 3 + np.random.default_rng(7).uniform(-0.16, 0.16, len(d3))
ax.scatter(xj, d3, marker="D", s=14, c="#B22222", alpha=0.9, zorder=3)
ax.hlines(np.median(d3), 3-0.28, 3+0.28, colors="#DD8452", lw=1.6, zorder=4)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f"multi\nch\n({len(d1)})", f"single\nch\n({len(d2)})",
                    f"whole\ncell\n({len(d3)})"], fontsize=5.8)
cv1 = np.std(d1) / np.mean(d1)
ax.set_ylim(0, 7.5)
ax.set_title(f"Nav1.5 · τ(−40 mV): a distribution (CV={cv1:.2f})", fontsize=6.5)
ax.set_ylabel("per-patch / per-cell τ (ms)")
ax.set_xlabel("recording modality")
ax.spines[["top", "right"]].set_visible(False)

# e: Nav 药物臂 —— 剂量-效应散点
ax = axes[1, 1]
drugs = ["Naltrexone", "Methadone", "Naloxone", "Buprenorphine", "Norbuprenorphine"]
cmap = plt.get_cmap("tab10")
for j, dr in enumerate(drugs):
    sub = nv[nv.drug == dr]
    ax.scatter(sub.conc_uM, sub.f_step, s=10, color=cmap(j), alpha=0.75,
               linewidths=0, label=f"{dr} ({len(sub)})")
ax.axhline(0, lw=0.7, color="0.4", zorder=1)
ax.set_xscale("log")
ax.set_xlabel("drug concentration (µM)")
ax.set_ylabel("fractional block of step $I_{Na}$")
ax.set_title("Nav1.5 · drug arm: dose-dependent block", fontsize=6.5)
ax.legend(fontsize=4.6, frameon=False, loc="upper left", handletextpad=0.2,
          borderaxespad=0.1, labelspacing=0.25)
ax.spines[["top", "right"]].set_visible(False)

# f: IKs V½ / k —— WT | Mef | DIDS | HMR 分列
ax = axes[1, 2]
wt_v = L("IKs", "V_half_act"); wt_k = L("IKs", "k_act")
mef_v = A("IKs", "V_half_act_drug", "Mef药臂"); mef_k = A("IKs", "k_act_drug", "Mef药臂")
dids_v = A("IKs", "V_half_act_drug", "DIDS药臂"); dids_k = A("IKs", "k_act_drug", "DIDS药臂")
hmr_v = A("IKs", "V_half_act_drug", "HMR减除臂"); hmr_k = A("IKs", "k_act_drug", "HMR减除臂")
all_d = [wt_v, mef_v, dids_v, hmr_v, wt_k, mef_k, dids_k, hmr_k]
labels = ["WT", "Mef", "DIDS", "HMR"] * 2
pos = [1, 2, 3, 4, 6, 7, 8, 9]
draw_groups(ax, all_d, pos, "#55A868")
ax.axvline(5, lw=0.6, color="0.7", ls=":")
ax.set_ylim(-100, 58)
ax.text(2.5, 50, "$V_{1/2}$ act", ha="center", fontsize=6.5)
ax.text(7.5, 50, "$k$ act", ha="center", fontsize=6.5)
ax.set_xticks(pos)
stagger = ["WT", "\nMef", "DIDS", "\nHMR"] * 2
ax.set_xticklabels(stagger, fontsize=5.5)
ax.scatter([0.62], [25.4], marker="*", s=45, c="#B22222", zorder=6)
ax.set_ylabel("mV")
ax.set_title("IKs · WT pool vs drug arms ($V_{1/2}$ & $k$; n = 8/15/26/6)", fontsize=6.2)
ax.spines[["top", "right"]].set_visible(False)

# g: CaV E_chord / Q10
ax = axes[2, 0]
f1 = L("CaV1.2", "E_chord")
panel(ax, [f1], ["$E_{chord}$"], "#8172B3", tick_fs=6.5)
ax.axhline(46, ls="--", lw=0.9, color="#B22222", zorder=2)
ax.text(0.62, 41.5, "literature anchor +46 mV", fontsize=6, color="#B22222")
ax.set_ylabel("mV")
ax.set_ylim(28, 66)
ax2 = ax.twinx()
f2 = L("CaV1.2", "Q10_tau_slow")
xj = 2 + np.random.default_rng(7).uniform(-0.10, 0.10, len(f2))
ax2.scatter(xj, f2, s=12, c="#4a3d6b", alpha=0.85, linewidths=0)
ax2.hlines(np.median(f2), 2-0.28, 2+0.28, colors="#8172B3", lw=1.6)
ax2.set_ylabel(r"$Q_{10}(\tau_{slow})$", color="#4a3d6b")
ax2.set_ylim(0.5, 2.5)
ax2.spines["top"].set_visible(False)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"$E_{{chord}}$\n({len(f1)})", f"$Q_{{10}}$($\\tau_{{slow}}$)\n({len(f2)})"],
                   fontsize=6.5)
ax.set_title("CaV1.2 · Ren 2022 (Ca$^{2+}$), complete record", fontsize=6.5)

# h: CaV f_inact 对照 —— 表征批 vs 筛选批对照段，载子分列
ax = axes[2, 1]
h1 = A("CaV1.2", "f_inact", "表征对照_Ca2")
h2 = np.concatenate([A("CaV1.2", "f_inact", g) for g in
                     ["Buprenorphine_Ca2+_PT", "Methadone_Ca2+_PT", "Norbuprenorphine_Ca2+_PT",
                      "Verapamil_Ca2+_PT", "Verapamil_Ca2+_RT"]])
h3 = A("CaV1.2", "f_inact", "表征对照_Ba2")
h4 = np.concatenate([A("CaV1.2", "f_inact", g) for g in
                     ["Diltiazem_Ba2+_PT", "Norbuprenorphine_Ba2+_PT", "Naloxone_Ba2+_PT"]])
panel(ax, [h1, h2, h3, h4], ["char.\nCa$^{2+}$", "screen\nCa$^{2+}$",
                             "char.\nBa$^{2+}$", "screen\nBa$^{2+}$"],
      "#8172B3", tick_fs=5.5)
ax.set_ylabel(r"pre-drug $f_{inact}$(+17 mV, 40 ms)")
ax.set_ylim(0, 1.05)
ax.set_title("CaV1.2 · pre-drug $f_{inact}$: batch-anchored", fontsize=6.5)

# i: CaV 药物效应 Δf 逐组（45° 标签）
ax = axes[2, 2]
groups = ["Buprenorphine_Ca2+_PT", "Methadone_Ca2+_PT", "Norbuprenorphine_Ca2+_PT",
          "Verapamil_Ca2+_PT", "Verapamil_Ca2+_RT",
          "Diltiazem_Ba2+_PT", "Norbuprenorphine_Ba2+_PT", "Naloxone_Ba2+_PT"]
short = ["Bup·Ca", "Meth·Ca", "Norb·Ca", "VeraPT·Ca", "VeraRT·Ca",
         "Dilt·Ba", "Norb·Ba", "Nalo·Ba"]
df_groups = []
for g in groups:
    sub = arm[(arm.group == g)]
    fc = sub[sub.parameter == "f_inact"].set_index("cell_id")["value"]
    fd = sub[sub.parameter == "f_inact_drug"].set_index("cell_id")["value"]
    common = fc.index.intersection(fd.index)
    df_groups.append((fd[common] - fc[common]).to_numpy(float))
panel(ax, df_groups, short, "#8172B3", tick_fs=5.0, rot=45)
ax.axhline(0, lw=0.7, color="0.4", zorder=1)
ax.set_ylabel(r"$\Delta f_{inact}$ (drug − pre-drug)")
ax.set_title("CaV1.2 · drug effect per cell by group", fontsize=6.5)

for i, ax in enumerate(axes.flat):
    ax.text(-0.24, 1.04, "abcdefghi"[i], transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")
fig.tight_layout(w_pad=2.0, h_pad=1.8)

fig.text(0.01, 0.005,
         "Violin density shown only for groups with n ≥ 20; smaller groups show every cell/patch. "
         "a–d, g: complete QC-passed public record (SI S9). e, f, h, i: pre-registered pharmacology arms; "
         "drug-naïve and drug-treated conditions shown separately (SI S7 arm table).",
         fontsize=6, color="0.35")

png = os.path.join(FIGD, "EDFig_percell_violin_v03.png")
pdf = os.path.join(FIGD, "EDFig_percell_violin_v03.pdf")
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
print("PNG pixels:", Image.open(png).size)
print("saved:", png)
