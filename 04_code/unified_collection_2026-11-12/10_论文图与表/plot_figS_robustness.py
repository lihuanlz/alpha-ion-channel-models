# -*- coding: utf-8 -*-
"""SI 稳健性图: 三口径散布比对照(66组) + tau 侧 EB-vs-冻结 一致性。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from daimon_runtime import setup_plot
setup_plot()

R = Path(__file__).parent
df = pd.read_csv(R / "features" / "robust_verdict_66groups_2026-11-08.csv")
C = dict(red="#E63946", orange="#F4842B", teal="#12B5A5", blue="#3A6FF7",
         purple="#9B5DE5", dark="#22223B", grey="#8D99AE", green="#52B788")

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

# a: A vs C 散点
ax = axes[0]
kv = df[df.channel.str.startswith("Kv")]
task = df[~df.channel.str.startswith("Kv")]
ax.axhline(1, color=C["red"], ls="--", lw=1.4); ax.axvline(1, color=C["red"], ls="--", lw=1.4)
lim = [0.3, 30]
ax.plot(lim, lim, color=C["grey"], lw=1.0, ls=":")
ax.scatter(kv.ratio_A, kv.ratio_C, s=55, color=C["blue"], alpha=0.8, edgecolor="white", lw=0.5,
           label=f"Kv 组 (n={len(kv)})", zorder=3)
ax.scatter(task.ratio_A, task.ratio_C, s=80, color=C["orange"], marker="^", edgecolor="white",
           lw=0.6, label="TASK3（误入的漏通道组）", zorder=4)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("现行口径散布比（含撞界行）"); ax.set_ylabel("分层贝叶斯口径散布比")
ax.legend(loc="upper left", fontsize=9.5)
ax.set_title("a  逐组对照：判决方向全程一致", loc="left", fontweight="bold")
ax.text(0.97, 0.05, "三组口径翻转集合完全相同\n（唯一反例均为 Kv3.2@15）", transform=ax.transAxes,
        ha="right", fontsize=9.5, color=C["dark"],
        bbox=dict(fc="white", ec=C["grey"], boxstyle="round,pad=0.3"))

# b: 三口径中位比与通过率
ax = axes[1]
lab = ["A 现行\n(含撞界)", "B 过滤撞界\n(V½∈[−80,60])", "C 分层贝叶斯\n(双侧 EB)"]
kvA, kvB, kvC = kv.ratio_A.dropna(), kv.ratio_B.dropna(), kv.ratio_C.dropna()
meds = [kvA.median(), kvB.median(), kvC.median()]
pas = [100 * (v > 1).mean() for v in (kvA, kvB, kvC)]
bars = ax.bar(range(3), meds, width=0.56, color=[C["grey"], C["teal"], C["blue"]], edgecolor="white")
for i, (m, p) in enumerate(zip(meds, pas)):
    ax.text(i, m + 0.07, f"{m:.2f}×", ha="center", fontsize=12, fontweight="bold", color=C["dark"])
    ax.text(i, 0.18, f"通过率\n{p:.1f}%", ha="center", fontsize=9.5, color="white", fontweight="bold")
ax.axhline(1, color=C["red"], ls="--", lw=1.6)
ax.set_xticks(range(3)); ax.set_xticklabels(lab, fontsize=10)
ax.set_ylabel("散布比中位（倍）"); ax.set_ylim(0, 3.9)
ax.set_title("b  纯 Kv 64 组：口径越严数值越保守，判决不动", loc="left", fontweight="bold")

# c: tau 侧 EB vs 冻结
ax = axes[2]
g = kv.dropna(subset=["tau_eb", "tau_obs"])
ax.plot([0.05, 1.0], [0.05, 1.0], color=C["grey"], lw=1.0, ls=":")
ax.scatter(g.tau_obs, g.tau_eb, s=55, color=C["teal"], alpha=0.8, edgecolor="white", lw=0.5, zorder=3)
rel = 100 * (g.tau_eb - g.tau_obs) / g.tau_obs
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("冻结口径 τ 散布（log10-SD）"); ax.set_ylabel("分层贝叶斯 τ 散布")
ax.set_title("c  τ 侧：两条独立路线互证", loc="left", fontweight="bold")
ax.text(0.05, 0.95, f"中位相对差 {rel.median():+.1f}%\nIQR [{rel.quantile(.25):+.1f}%, {rel.quantile(.75):+.1f}%]",
        transform=ax.transAxes, va="top", fontsize=10.5, fontweight="bold", color=C["dark"],
        bbox=dict(fc="white", ec=C["teal"], boxstyle="round,pad=0.35"))

fig.suptitle("稳健性附录：撞界过滤 + 分层贝叶斯双侧收缩，主判决不变（纯 Kv 63/64 = 98.4%）",
             fontsize=14, fontweight="bold", color=C["dark"])
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = R / "features" / "figS_robustness_2026-11-08.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
