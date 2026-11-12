# -*- coding: utf-8 -*-
"""正式稿图8合成：全药池五实验室规模验证（Stage1 表级 + Stage2 波形级）
(a) 跨实验室IC50倍数分布 (b) 实验室偏移(nominal/corrected) (c) Hill全景
(d) τ比分布四实验室 (e) τ比vs阻断深度分箱 (f) lab_1形状比
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
fold = pd.read_csv(os.path.join(OUT, "stage1_crosslab_fold.csv"))
shift = pd.read_csv(os.path.join(OUT, "stage2a_lab_shift.csv"))
s1 = pd.read_csv(os.path.join(OUT, "stage1_drug_lab_summary.csv"))
q = pd.read_csv(os.path.join(OUT, "stage2_final_cell_metrics.csv"))

LAB_COLORS = {"lab_1": "#E6194B", "lab_2": "#3CB44B", "lab_3": "#4363D8", "lab_4": "#F58231", "lab_5": "#911EB4"}

fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

# (a) 跨实验室 fold 分布
ax = axes[0, 0]
v = fold["fold_ours"].dropna()
ax.hist(np.log10(v), bins=14, color="#4363D8", alpha=0.75, edgecolor="k", lw=0.4)
ax.axvline(np.log10(v.median()), color="#E6194B", lw=2.2, label=f"中位 {v.median():.2f}×")
ax.axvline(np.log10(5), color="k", ls="--", lw=1.5, label="源研究 ~5×")
ax.set_xlabel("跨实验室 IC50 倍数（log10 刻度）"); ax.set_ylabel("药物数")
ax.set_title("(a) 逐药跨实验室 potency 变异性（28 药）", fontweight="bold", fontsize=11)
ax.legend()

# (b) 实验室偏移：nominal vs corrected
ax = axes[0, 1]
x = np.arange(len(shift)); w = 0.38
ax.bar(x - w/2, shift["geo_shift_nominal"], w, color="#9E9E9E", edgecolor="k", lw=0.4, label="标称浓度")
ax.bar(x + w/2, shift["geo_shift_corrected"], w, color="#F58231", edgecolor="k", lw=0.4, label="实测修正浓度")
ax.axhline(1.0, color="k", ls="--", lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(shift["Lab"])
ax.set_ylim(0, 5.3)
ax.set_ylabel("几何偏移倍数（>1=IC50 系统性偏高）")
ax.set_title("(b) 实验室偏移不能被浓度修正解释（lab_2 偏 ~4×）", fontweight="bold", fontsize=11)
ax.legend()

# (c) Hill 系数全景
ax = axes[0, 2]
s1ok = s1[s1["fit_ok"]].copy()
for lab, g in s1ok.groupby("lab"):
    ax.scatter([lab] * len(g), g["hill"].clip(0, 4), s=22, alpha=0.75,
               color=LAB_COLORS[lab], edgecolor="k", lw=0.3)
ax.axhline(1.0, color="k", ls="--", lw=1.2)
ib = s1ok[s1ok["drug"] == "ibutilide"]
for _, r in ib.iterrows():
    if r["hill"] > 2:
        ax.annotate(f"ibutilide {r['hill']:.2f}", (r["lab"], min(r["hill"], 4)),
                    xytext=(10, -12), textcoords="offset points", fontsize=9, color="#B71C1C",
                    arrowprops=dict(arrowstyle="->", color="#B71C1C"))
ax.set_ylabel("Hill 系数（截断显示至 4）")
ax.set_title("(c) Hill≠1 目录 42/134：阻断非 1:1 简单占据", fontweight="bold", fontsize=11)

# (d) τ 比分布（四实验室）
ax = axes[1, 0]
bins = np.linspace(0, 3, 76)
notes = {"lab_1": "尾窗130ms", "lab_3": "去激活主口径", "lab_4": "阶跃失活τ", "lab_5": "尾窗~2.1s"}
for lab in ["lab_3", "lab_1", "lab_4", "lab_5"]:
    v = q[q["lab"] == lab]["tau_ratio"].dropna()
    v = v[(v > 0) & (v < 10)]
    ax.hist(v.clip(0, 3), bins=bins, alpha=0.45, color=LAB_COLORS[lab],
            label=f"{lab} (n={len(v)}, {v.median():.2f}) {notes[lab]}")
ax.axvline(1.0, color="k", ls="--", lw=1.5)
ax.set_xlabel("τ 比值（药物稳态 ÷ 载体对照）"); ax.set_ylabel("细胞数")
ax.set_title("(d) τ 守恒：合并中位 0.95，无翻倍/减半", fontweight="bold", fontsize=11)
ax.legend(fontsize=8)

# (e) τ比 vs 阻断深度分箱
ax = axes[1, 1]
depth_bins = [(0.2, 0.5), (0.5, 0.7), (0.7, 1.01)]
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
ax.set_xticks(np.arange(3)); ax.set_xticklabels(["20–50%", "50–70%", "70–100%"])
ax.set_xlabel("阻断深度分箱"); ax.set_ylabel("τ 比 中位 ± IQR"); ax.set_ylim(0, 2.2)
ax.set_title("(e) τ 守恒不随阻断深度漂移（lab_5 深阻断存疑）", fontweight="bold", fontsize=11)
ax.legend(fontsize=8)

# (f) lab_1 形状比
ax = axes[1, 2]
v1 = q[q["lab"] == "lab_1"]["shape_chg"].dropna()
bins3 = np.linspace(0, 2.5, 76)
ax.hist(v1.clip(0, 2.5), bins=bins3, color="#E6194B", alpha=0.65, edgecolor="k", lw=0.3,
        label=f"全部 (n={len(v1)})")
deep = q[(q["lab"] == "lab_1") & (q["block_frac"] > 0.6)]["shape_chg"].dropna()
ax.axvline(1.0, color="k", ls="--", lw=1.5)
ax.axvline(deep.median(), color="#7B1FA2", lw=2.2,
           label=f"深阻断(>60%)中位 {deep.median():.2f}\n70% 偏离>20% (n={len(deep)})")
ax.set_xlabel("峰/平台形状比（药物 ÷ 对照）"); ax.set_ylabel("细胞数")
ax.set_title("(f) 状态依赖结合证据：峰削更多（lab_1）", fontweight="bold", fontsize=11)
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_全药池五实验室_正式稿.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("图8 已生成：fig8_全药池五实验室_正式稿.png")
print("DONE")
