# -*- coding: utf-8 -*-
"""图5 独立印证与边界：ICG 五族 / Nav 三源闭环 / K2P+Kir 形状守恒 / 阴性登记。"""
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

C = dict(red="#E63946", orange="#F4842B", teal="#12B5A5", blue="#3A6FF7",
         purple="#9B5DE5", pink="#EF476F", yellow="#FFC53D", green="#52B788",
         grey="#8D99AE", dark="#22223B")
plt.rcParams.update({"axes.edgecolor": "#333333", "axes.linewidth": 1.0,
                     "axes.titlesize": 13, "axes.labelsize": 11.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 9.5, "figure.facecolor": "white"})

icg = pd.read_csv(R / "icg_out" / "verdict_icg_ALL.csv")
navc = pd.read_csv(F / "verdict_nav_closure_2026-11-08.csv")
k2p_sum = pd.read_csv(F / "k2p_recon_2026-11-08.csv")
with open(F / "k2p_recon_cells_2026-11-08.pkl", "rb") as f:
    k2p_cells = pickle.load(f)
with open(F / "kir_recon_cells_2026-11-08.pkl", "rb") as f:
    kir_cells = pickle.load(f)
wc = pd.read_csv(F / "window_covar_v5.csv")
print("校验 ICG:", (icg.ratio > 1).sum(), "/", len(icg), "通过")
print("校验 K2P 亚型:", k2p_sum.channel.tolist())
print("校验 窗口共变: n=%d, r 中位 %.2f" % (len(wc), wc.r.median()))

fig = plt.figure(figsize=(17.5, 10.0))
gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.26,
              left=0.065, right=0.985, top=0.90, bottom=0.09)
fig.suptitle("独立印证与边界：模型库复证 · 钠族闭环 · 无 VSD 家族的形状守恒 · 阴性登记",
             fontsize=16, fontweight="bold", color=C["dark"])

# --- a ICG 五族十条 ---
ax = fig.add_subplot(gs[0, 0])
fam_col = {"K": C["blue"], "Na": C["red"], "KCa": C["purple"], "IH": C["grey"], "Ca": C["orange"]}
icg_s = icg.sort_values("ratio").reset_index(drop=True)
labels = [f"{r.family} {r.gate} · {r.cls}" if r.family == "Ca" else f"{r.family} {r.gate}"
          for _, r in icg_s.iterrows()]
labels = [l.replace("act", "激活").replace("inact", "失活") for l in labels]
ypos = np.arange(len(icg_s))
cols = [fam_col[f] for f in icg_s.family]
ax.barh(ypos, icg_s.ratio, color=cols, edgecolor="white", height=0.66)
for y, r, n in zip(ypos, icg_s.ratio, icg_s.n_models):
    ax.text(r + 0.07, y, f"{r:.2f}×  (n={n})", va="center", fontsize=9.3,
            color=C["dark"], fontweight="bold")
ax.axvline(1, color=C["red"], ls="--", lw=1.8)
ax.text(1.02, -0.62, "判决线 1×", color=C["red"], fontsize=9.5)
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=9.6)
ax.set_xlim(0, 5.6); ax.set_xlabel("散布比（中点散布 / 时间常数散布，倍）")
ax.set_title("a  已发表模型库复证：933 模型 10 层判决，9 层通过", loc="left", fontweight="bold")
ax.text(0.97, 0.05, "唯一打平：IH 模型集（38 个，近交）\n同家族原始细胞判决为 8.25×",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        color=C["grey"], bbox=dict(fc="white", ec=C["grey"], alpha=0.9, boxstyle="round,pad=0.3"))

# --- b Nav 三源闭环 ---
ax = fig.add_subplot(gs[0, 1])
bars = [("灵长类皮层\nDANDI 全 QC 膜片", 3.05, "n=12", C["red"]),
        ("灵长类皮层\nDANDI 快 τ 主群", 4.52, "n=6", C["orange"]),
        ("模型库\nICG Na 激活", 2.27, "n=231", C["blue"]),
        ("模型库\nICG Na 失活", 2.18, "n=206", C["teal"])]
x = np.arange(len(bars))
ax.bar(x, [b[1] for b in bars], width=0.58, color=[b[3] for b in bars], edgecolor="white")
for i, (lab, v, n, col) in enumerate(bars):
    ax.text(i, v + 0.09, f"{v:.2f}×", ha="center", fontsize=11.5, fontweight="bold", color=C["dark"])
    ax.text(i, v / 2, n, ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")
ax.axhline(1, color=C["red"], ls="--", lw=1.8)
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=9.6)
ax.set_ylabel("散布比（倍）"); ax.set_ylim(0, 5.6)
ax.set_title("b  钠族三源闭环：原生组织 + 细胞系 + 模型库", loc="left", fontweight="bold")
ax.text(0.03, 0.95, "另有 Na_v1.5 细胞系判决：中点逐细胞 ±4 mV、\n失活斜率 k_h 共享（预印本 Nα-1）",
        transform=ax.transAxes, va="top", fontsize=9, color=C["grey"],
        bbox=dict(fc="white", ec=C["grey"], alpha=0.9, boxstyle="round,pad=0.3"))

# --- c K2P 六亚型形状面条 + 形状/幅度对比 ---
ax = fig.add_subplot(gs[1, 0])
V_K2P = np.arange(-160, 81, 15)
sub_cols = {"TASK1": C["blue"], "TASK2": C["teal"], "TASK3": C["red"],
            "TRAAK": C["orange"], "TREK1": C["purple"], "TWIK1": C["pink"]}
for ch, col in sub_cols.items():
    sub = k2p_cells[k2p_cells.channel == ch]
    S = np.array([np.asarray(s, float) for s in sub["shape"]])
    for row in S[:: max(1, len(S) // 25)]:
        ax.plot(V_K2P := V_K2P, row, color=col, alpha=0.10, lw=0.7)
    ax.plot(V_K2P, np.nanmedian(S, axis=0), color=col, lw=2.6, label=f"{ch} (n={len(sub)})")
ax.axhline(0, color=C["grey"], lw=0.8)
ax.set_xlabel("膜电位（mV）"); ax.set_ylabel("归一化电流（形状）")
ax.legend(ncol=3, loc="lower right", framealpha=0.95, columnspacing=0.9, handlelength=1.4)
ax.set_title("c  K2P 六亚型：归一化 I–V 形状跨细胞重合", loc="left", fontweight="bold")
# 右上内嵌: 形状SD vs 幅度CV
axin = ax.inset_axes([0.035, 0.52, 0.33, 0.44])
xx = np.arange(len(k2p_sum))
axin.bar(xx - 0.19, k2p_sum.amp_CV, width=0.38, color=C["red"], alpha=0.85, label="幅度 CV")
axin.bar(xx + 0.19, k2p_sum.shape_SD_med, width=0.38, color=C["teal"], alpha=0.9, label="形状 SD")
axin.set_xticks(xx); axin.set_xticklabels(k2p_sum.channel, fontsize=7.2, rotation=30)
axin.legend(fontsize=7.5, loc="upper right")
axin.set_title("幅度漂移 ≈ 形状散布的 10 倍", fontsize=8.6, fontweight="bold", color=C["dark"])
axin.tick_params(labelsize=7.5)

# --- d 阴性/反例登记 ---
ax = fig.add_subplot(gs[1, 1])
r = wc.r.values
x = np.random.default_rng(5).normal(0, 0.05, len(r))
ax.scatter(x, r, s=42, color=C["purple"], alpha=0.65, edgecolor="white", lw=0.4, zorder=3)
ax.hlines(np.median(r), -0.3, 0.3, color=C["dark"], lw=2.6)
ax.axhline(1, color=C["green"], ls="--", lw=1.6)
ax.axhline(0, color=C["grey"], ls=":", lw=1.2)
ax.text(0.34, 1.0, "完全锁定 r=1", color=C["green"], fontsize=9.5, va="center")
ax.text(0.34, np.median(r), f"实测中位 r={np.median(r):.2f}", color=C["dark"], fontsize=10,
        va="center", fontweight="bold")
ax.set_xlim(-0.55, 0.75); ax.set_ylim(-0.35, 1.25)
ax.set_xticks([0]); ax.set_xticklabels(["同细胞 V½_激活 × V½_失活\n35 对"])
ax.set_ylabel("Pearson r")
ax.set_title("d  阴性登记①：窗口共变假说不成立（两调定点各自独立）", loc="left", fontweight="bold")
ax.text(0.03, 0.03,
        "阴性登记②：Kv3.2@15°C 为 66 组中唯一反例（比值 0.825）——\n该通道低温下把异质性放在动力学而非中点，如实登记不豁免。\n"
        "阴性登记③：hERG 中点双向温度漂移（+1.14/−1.47 mV/°C）\n在全族不普适（全族中位 −0.05 mV/°C），登记为 hERG 特性。",
        transform=ax.transAxes, va="bottom", fontsize=9.2, color=C["dark"],
        bbox=dict(fc="#FFF8E7", ec=C["orange"], boxstyle="round,pad=0.4"))

out = F / "fig5_validation_boundary_2026-11-08.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
