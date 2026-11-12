# -*- coding: utf-8 -*-
"""Stage2e 最终汇总出图 v2：列污染修复后的干净版 fig6
数据：stage2_final_cell_metrics.csv（lab_1/2 来自 stage2c，lab_3/4/5 来自列修复版 stage2d）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()

OUT = os.path.dirname(os.path.abspath(__file__))
q = pd.read_csv(os.path.join(OUT, "stage2_final_cell_metrics.csv"))

LAB_COLORS = {"lab_1": "#E6194B", "lab_2": "#3CB44B", "lab_3": "#4363D8", "lab_4": "#F58231", "lab_5": "#911EB4"}

fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

# (a) 形状比变化分布（lab_1 主力；lab_3 峰太早平台窗不满足，n=12 不画）
ax = axes[0, 0]
bins = np.linspace(0, 2.5, 76)
v = q[q["lab"] == "lab_1"]["shape_chg"].dropna()
ax.hist(v.clip(0, 2.5), bins=bins, alpha=0.65, color=LAB_COLORS["lab_1"],
        label=f"lab_1 (n={len(v)}, 中位={v.median():.2f})")
ax.axvline(1.0, color="k", ls="--", lw=1.5)
ax.set_xlabel("形状比变化（药物稳态 ÷ 对照）"); ax.set_ylabel("细胞数")
ax.set_title("(a) 复极化峰/平台形状比：1=纯占据；<1=峰削更多（状态依赖迹象）", fontweight="bold", fontsize=11)
ax.legend()

# (b) 形状比变化 vs 阻断深度（lab_1/2）
ax = axes[0, 1]
for lab in ["lab_1", "lab_2"]:
    g = q[q["lab"] == lab]
    ax.scatter(g["block_frac"], g["shape_chg"].clip(0, 2.5), s=10, alpha=0.4,
               color=LAB_COLORS[lab], label=lab)
ax.axhline(1.0, color="k", ls="--", lw=1.2)
ax.set_xlabel("阻断深度"); ax.set_ylabel("形状比变化")
ax.set_title("(b) 形状扰动随阻断深度（深阻断处更明显=状态依赖）", fontweight="bold", fontsize=11)
ax.legend(markerscale=3)

# (c) τ 比值分布（四实验室分协议标注）
ax = axes[1, 0]
bins2 = np.linspace(0, 3, 76)
notes = {"lab_1": "尾窗130ms", "lab_3": "去激活·长尾窗金标准", "lab_4": "阶跃失活τ", "lab_5": "尾窗~2.1s"}
for lab in ["lab_3", "lab_1", "lab_4", "lab_5"]:
    v = q[q["lab"] == lab]["tau_ratio"].dropna()
    v = v[(v > 0) & (v < 10)]
    ax.hist(v.clip(0, 3), bins=bins2, alpha=0.45, color=LAB_COLORS[lab],
            label=f"{lab} (n={len(v)}, 中位={v.median():.2f}) {notes[lab]}")
ax.axvline(1.0, color="k", ls="--", lw=1.5)
ax.set_xlabel("动力学时间常数比值（药物 ÷ 对照）"); ax.set_ylabel("细胞数")
ax.set_title("(c) τ 守恒检验：跨 4 实验室中位 0.81–1.19，无翻倍/减半", fontweight="bold", fontsize=11)
ax.legend(fontsize=8.5)

# (d) τ比 vs 阻断深度 分箱中位（守恒不随深度漂移=占据说的核心证据）
ax = axes[1, 1]
depth_bins = [(0.2, 0.5), (0.5, 0.7), (0.7, 1.01)]
xlabels = ["20-50%", "50-70%", "70-100%"]
width = 0.19
for i, lab in enumerate(["lab_1", "lab_3", "lab_4", "lab_5"]):
    g = q[q["lab"] == lab]
    g = g[g["tau_ratio"].notna() & (g["tau_ratio"] > 0) & (g["tau_ratio"] < 10)]
    meds, los, his, ns = [], [], [], []
    for lo, hi in depth_bins:
        s = g[(g["block_frac"] >= lo) & (g["block_frac"] < hi)]["tau_ratio"]
        meds.append(s.median()); los.append(s.quantile(.25)); his.append(s.quantile(.75)); ns.append(len(s))
    x = np.arange(3) + (i - 1.5) * width
    ax.errorbar(x, meds, yerr=[np.array(meds) - np.array(los), np.array(his) - np.array(meds)],
                fmt="o-", color=LAB_COLORS[lab], lw=1.8, ms=6, capsize=3,
                label=f"{lab} (n={ns[0]}/{ns[1]}/{ns[2]})")
ax.axhline(1.0, color="k", ls="--", lw=1.5)
ax.set_xticks(np.arange(3)); ax.set_xticklabels(xlabels)
ax.set_xlabel("阻断深度分箱"); ax.set_ylabel("τ比 中位 ± IQR")
ax.set_ylim(0, 2.2)
ax.set_title("(d) τ 守恒不随阻断深度漂移（lab_5 深阻断+30%已标注存疑）", fontweight="bold", fontsize=11)
ax.legend(fontsize=8.5)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_波形tau守恒检验.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print("fig6 v2 已重画")

print("\n=== 最终核心数字（干净版）===")
for lab in ["lab_1", "lab_2", "lab_3", "lab_4", "lab_5"]:
    g = q[q["lab"] == lab]
    tr = g["tau_ratio"].dropna(); tr = tr[(tr > 0) & (tr < 10)]
    bf = g["block_frac"].dropna()
    print(f"{lab}: τ比 {tr.median():.2f}[{tr.quantile(.25):.2f},{tr.quantile(.75):.2f}] n={len(tr)} | "
          f"block {bf.median():.2f} n={len(bf)}")
gg = q[q["lab"].isin(["lab_1", "lab_3", "lab_4", "lab_5"])]
tr = gg["tau_ratio"].dropna(); tr = tr[(tr > 0) & (tr < 10)]
within20 = ((tr >= 0.8) & (tr <= 1.25)).mean()
within30 = ((tr >= 0.7) & (tr <= 1.43)).mean()
print(f"\n四实验室合并 τ比: 中位 {tr.median():.2f} (n={len(tr)})")
print(f"τ比在±20%内占比 {within20*100:.0f}%，±30%内占比 {within30*100:.0f}%")
deep = q[(q["lab"] == "lab_1") & (q["block_frac"] > 0.6)]["shape_chg"].dropna()
print(f"lab_1 深阻断(>60%)形状比: 中位 {deep.median():.2f}，偏离>20%占比 {((deep<0.8)|(deep>1.25)).mean()*100:.0f}% (n={len(deep)})")
print("DONE")
