# -*- coding: utf-8 -*-
"""图4 全库定律主图（六面板）+ 图5 独立印证与边界（四面板），出版级。
数字全部来自 features/ 与 icg_out/ 判决表，画前打印校验。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np, pandas as pd, pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from daimon_runtime import setup_plot
setup_plot()

R = Path(__file__).parent
F = R / "features"

# ---------- 鲜艳色板 ----------
C = dict(red="#E63946", orange="#F4842B", teal="#12B5A5", blue="#3A6FF7",
         purple="#9B5DE5", pink="#EF476F", yellow="#FFC53D", green="#52B788",
         grey="#8D99AE", dark="#22223B")
plt.rcParams.update({"axes.edgecolor": "#333333", "axes.linewidth": 1.0,
                     "axes.titlesize": 13, "axes.labelsize": 11.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 9.5, "figure.facecolor": "white"})

# ============ 数据载入与校验 ============
pp = pd.read_csv(F / "per_process_v5.csv")
kv_act = pp[["channel", "temp", "mid_obs", "tau_act"]].dropna()
kv_act["ratio"] = kv_act.mid_obs / kv_act.tau_act
kv_de = pp[["mid_obs", "tau_deact"]].dropna(); kv_de["ratio"] = kv_de.mid_obs / kv_de.tau_deact
kv_in = pp[["mid_obs", "tau_inact"]].dropna(); kv_in["ratio"] = kv_in.mid_obs / kv_in.tau_inact
rec = pd.read_csv(F / "recovery_verdict_v5_rebuilt.csv")
hcn = pd.read_csv(F / "hcn_verdict_v5.csv")
print("校验  Kv act:", len(kv_act), "组, 通过率 %.1f%%" % (100 * (kv_act.ratio > 1).mean()),
      "中位比 %.2f" % kv_act.ratio.median())
print("校验  Kv deact:", len(kv_de), "%.1f%%" % (100 * (kv_de.ratio > 1).mean()))
print("校验  Kv inact:", len(kv_in), "%.1f%%" % (100 * (kv_in.ratio > 1).mean()))
print("校验  Rec:", len(rec), "%.1f%%" % (100 * (rec.ratio > 1).mean()), "中位 %.2f" % rec.ratio.median())
print("校验  HCN:", len(hcn), "全过" if (hcn.ratio > 1).all() else "有败", "中位 %.2f" % hcn.ratio.median())

cs = pd.read_csv(F / "cross_species_v3.csv")
print("校验  跨物种:", len(cs), "对, τ倍数中位 %.3f" % cs.tau_fold.median(),
      "2倍内 %.0f%%" % (100 * ((cs.tau_fold >= 0.5) & (cs.tau_fold <= 2)).mean()),
      "|ΔV½|max %.1f" % cs.Vhalf_diff.abs().max())
qc = pd.read_csv(F / "qc_split.csv")
ep = pd.read_csv(F / "E_forward_prediction.csv")
b3 = pd.read_csv(F / "B3_q10_by_process.csv")

# ============ 图 4 ============
fig = plt.figure(figsize=(17.5, 10.5))
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30,
              left=0.055, right=0.985, top=0.90, bottom=0.085)
fig.suptitle("门控定律：时间常数与门控电荷是硬件，电压调定点是软件（Channelpedia 全库 17,965 细胞）",
             fontsize=16, fontweight="bold", color=C["dark"])

# --- a 判决总览 ---
ax = fig.add_subplot(gs[0, 0])
groups = [("Kv 激活", kv_act.ratio, C["blue"], "98.5% (65/66)"),
          ("Kv 去激活", kv_de.ratio, C["teal"], "93.8% (45/48)"),
          ("Kv 失活", kv_in.ratio, C["purple"], "85.7% (30/35)"),
          ("恢复 τ_rec", rec.ratio, C["orange"], "95.1% (39/41)"),
          ("HCN 激活", hcn.ratio, C["red"], "100% (5/5)")]
for i, (lab, vals, col, note) in enumerate(groups):
    x = np.random.default_rng(7 + i).normal(i, 0.055, len(vals))
    ax.scatter(x, np.log10(vals), s=26, color=col, alpha=0.75, edgecolor="white", lw=0.4, zorder=3)
    ax.hlines(np.log10(vals.median()), i - 0.22, i + 0.22, color=C["dark"], lw=2.4, zorder=4)
    ax.text(i, -0.62, note, ha="center", va="top", fontsize=8.6, color=col, fontweight="bold")
ax.axhline(0, color=C["red"], ls="--", lw=1.6, zorder=2)
ax.text(4.42, 0.06, "判决线 ratio=1", color=C["red"], fontsize=9, ha="right")
ax.set_xticks(range(5)); ax.set_xticklabels([g[0] for g in groups])
ax.set_ylabel("log$_{10}$(中点散布 / 时间常数散布)")
ax.set_ylim(-0.85, 1.75)
ax.set_title("a  五大战场判决：全部位于判决线上方", loc="left", fontweight="bold")

# --- b HCN 反向极性 ---
ax = fig.add_subplot(gs[0, 1])
hcn_s = hcn.sort_values("ratio")
labels = [f"{c}@{t}°C" for c, t in zip(hcn_s.channel, hcn_s.temp)]
bars = ax.bar(range(len(hcn_s)), hcn_s.ratio, width=0.62,
              color=[C["red"], C["orange"], C["yellow"], C["teal"], C["blue"]], edgecolor="white")
for i, v in enumerate(hcn_s.ratio):
    ax.text(i, v + 0.18, f"{v:.1f}×", ha="center", fontsize=10, fontweight="bold", color=C["dark"])
ax.axhline(1, color=C["red"], ls="--", lw=1.6)
ax.axhline(kv_act.ratio.median(), color=C["grey"], ls=":", lw=1.4)
ax.text(4.4, kv_act.ratio.median() + 0.12, "Kv 中位 3.2×", fontsize=8.6, color=C["grey"], ha="right")
ax.set_xticks(range(len(hcn_s))); ax.set_xticklabels(labels, rotation=18)
ax.set_ylabel("散布比（倍）"); ax.set_ylim(0, 15.2)
ax.set_title("b  反向极性 HCN 族：5/5 通过，中位 8.25×", loc="left", fontweight="bold")

# --- c 跨物种主图 ---
ax = fig.add_subplot(gs[0, 2])
ax.axhspan(0.5, 2.0, color=C["green"], alpha=0.13, zorder=1)
ax.axhline(1, color=C["grey"], lw=1.0, ls="-", zorder=2)
sc = ax.scatter(cs.Vhalf_diff.abs(), cs.tau_fold, s=64, c=cs.temp,
                cmap=matplotlib.colors.ListedColormap([C["blue"], C["orange"], C["red"]]),
                edgecolor="white", lw=0.8, zorder=3)
ax.set_xlabel("同源通道电压中点差 |ΔV½|（mV）")
ax.set_ylabel("时间常数倍数 τ$_a$ / τ$_b$")
ax.set_xlim(-1.5, 36); ax.set_ylim(0.35, 2.15)
ax.annotate("τ：38/38 对落在两倍内\n（中位 0.956）", xy=(1.1, 1.02), xytext=(8.5, 0.62),
            fontsize=10.5, fontweight="bold", color=C["green"],
            arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.4))
ax.annotate("V½：最大漂移 33 mV", xy=(32.9, 1.35), xytext=(14.5, 1.86),
            fontsize=10.5, fontweight="bold", color=C["red"],
            arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.4))
hs = [plt.Line2D([], [], marker="o", ls="", color=c, markeredgecolor="white", label=f"{t} °C")
      for t, c in zip([15, 25, 35], [C["blue"], C["orange"], C["red"]])]
ax.legend(handles=hs, loc="lower right", framealpha=0.95, title="温度", title_fontsize=9.5)
ax.set_title("c  跨物种（人/鼠/大鼠 38 对同源通道）", loc="left", fontweight="bold")

# --- d QC 分层反杀 ---
ax = fig.add_subplot(gs[1, 0])
order = ["active", "bad", "silent", "excl"]
cols = [C["green"], C["orange"], C["purple"], C["red"]]
data_q = [np.log10(qc.ratio[qc.flag == f].values) for f in order]
bp = ax.boxplot(data_q, positions=range(4), widths=0.55, patch_artist=True, showfliers=False)
for patch, col in zip(bp["boxes"], cols):
    patch.set_facecolor(col); patch.set_alpha(0.75)
for med in bp["medians"]:
    med.set_color(C["dark"]); med.set_lw(2.2)
for i, d in enumerate(data_q):
    x = np.random.default_rng(3 + i).normal(i, 0.06, len(d))
    ax.scatter(x, d, s=12, color=cols[i], alpha=0.45, zorder=3, edgecolor="none")
ax.axhline(0, color=C["red"], ls="--", lw=1.6)
ns = [int((qc.flag == f).sum()) for f in order]
ax.set_xticks(range(4))
ax.set_xticklabels([f"{f}\n({n} 组)" for f, n in zip(order, ns)])
ax.set_ylabel("log$_{10}$(散布比)")
ax.set_title("d  质控假象被反杀：差细胞里定律反而放大", loc="left", fontweight="bold")

# --- e 温度前向预言 ---
ax = fig.add_subplot(gs[1, 1])
cmap_p = {"tau_act": C["blue"], "tau_deact": C["teal"], "tau_inact": C["purple"]}
for pname, grp in ep.groupby("parameter"):
    ax.scatter(grp.tau35_meas, grp.tau35_pred, s=30, color=cmap_p.get(pname, C["grey"]),
               alpha=0.7, edgecolor="white", lw=0.3,
               label={"tau_act": "激活", "tau_deact": "去激活", "tau_inact": "失活"}.get(pname, pname))
lim = [ep.tau35_meas.min() * 0.6, ep.tau35_meas.max() * 1.8]
ax.plot(lim, lim, color=C["dark"], lw=1.5, zorder=4)
ax.plot(lim, [2 * v for v in lim], color=C["red"], lw=1.2, ls="--", zorder=4)
ax.plot(lim, [0.5 * v for v in lim], color=C["red"], lw=1.2, ls="--", zorder=4)
ax.text(lim[0] * 1.25, lim[0] * 2.5, "2 倍窗", color=C["red"], fontsize=9.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("实测 τ（35 °C，ms）"); ax.set_ylabel("预言 τ（25 °C 表 ÷ Q$_{10}$）")
in2 = 100 * (ep.log2err.abs() <= 1).mean()
ax.text(0.04, 0.96, f"325 点 / 20 通道\n{in2:.1f}% 落在两倍窗内\n中位倍数误差 28%",
        transform=ax.transAxes, va="top", fontsize=10, fontweight="bold", color=C["dark"],
        bbox=dict(fc="white", ec=C["grey"], alpha=0.9, boxstyle="round,pad=0.35"))
ax.legend(loc="lower right", framealpha=0.95)
ax.set_title("e  温度前向预言：25 °C 表 → 35 °C（零额外参数）", loc="left", fontweight="bold")

# --- f Q10 过程结构 ---
ax = fig.add_subplot(gs[1, 2])
proc = [("tau_act", "激活", C["blue"]), ("tau_deact", "去激活", C["teal"]), ("tau_inact", "失活", C["purple"])]
for i, (pname, lab, col) in enumerate(proc):
    v = b3.Q10[(b3.parameter == pname) & (b3.R2 >= 0.8)].dropna()
    x = np.random.default_rng(11 + i).normal(i, 0.09, len(v))
    ax.scatter(x, v, s=22, color=col, alpha=0.55, edgecolor="white", lw=0.3)
    ax.hlines(v.median(), i - 0.28, i + 0.28, color=C["dark"], lw=2.6)
    ax.text(i, 10.35, f"中位 {v.median():.2f}\nn={len(v)}", ha="center", va="top",
            fontsize=9.5, fontweight="bold", color=col,
            bbox=dict(fc="white", ec=col, alpha=0.92, boxstyle="round,pad=0.25"))
ax.set_xticks(range(3)); ax.set_xticklabels([p[1] for p in proc])
ax.set_ylabel("Q$_{10}$（15–35 °C）"); ax.set_ylim(0, 11.2)
ax.set_title("f  温度主要由激活过程承载", loc="left", fontweight="bold")

out = F / "fig4_law_verdict_2026-11-08.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
