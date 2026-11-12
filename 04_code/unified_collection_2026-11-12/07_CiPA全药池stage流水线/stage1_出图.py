# -*- coding: utf-8 -*-
"""Stage1 出图：5 张彩色图（剂量反应面板 / IC50 全景 / 对拍验证 / hill 全景 / lab 偏移热图）"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(__file__ and os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()

OUT = os.path.dirname(os.path.abspath(__file__))
dfl = pd.read_csv(os.path.join(OUT, "stage1_drug_lab_summary.csv"))
cs = pd.read_csv(os.path.join(OUT, "stage1_conc_level_stats.csv"))
anchor = pd.read_csv(os.path.join(OUT, "stage1_vs_newCiPA_anchor.csv"))

LAB_COLORS = {"lab_1": "#E6194B", "lab_2": "#3CB44B", "lab_3": "#4363D8", "lab_4": "#F58231", "lab_5": "#911EB4"}
LAB_MARK = {"lab_1": "o", "lab_2": "s", "lab_3": "^", "lab_4": "D", "lab_5": "v"}

def hill_curve(logC, logIC50, hill):
    return 1.0 / (1.0 + 10 ** ((logIC50 - logC) * hill))

# ============ fig1: 28 药剂量反应面板 ============
drugs = sorted(dfl[~dfl["blind"]]["drug"].unique())
n = len(drugs)
ncol = 5
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(22, 3.4 * nrow))
axes = axes.ravel()
for i, drug in enumerate(drugs):
    ax = axes[i]
    g = dfl[(dfl["drug"] == drug) & (~dfl["blind"])]
    for _, r in g.iterrows():
        c = LAB_COLORS.get(r["lab"], "gray")
        sub = cs[(cs["drug"] == drug) & (cs["lab"] == r["lab"])].sort_values("conc_uM")
        if len(sub):
            ax.errorbar(sub["conc_uM"], sub["med"],
                        yerr=[sub["med"] - sub["q25"], sub["q75"] - sub["med"]],
                        fmt=LAB_MARK.get(r["lab"], "o"), color=c, ms=5, capsize=2, lw=1,
                        label=f'{r["lab"]} IC50={r["IC50_uM"]:.3g} µM', alpha=0.9)
        if pd.notna(r["IC50_uM"]) and pd.notna(r["hill"]):
            xs = np.logspace(np.log10(max(sub["conc_uM"].min() * 0.5, 1e-4)),
                             np.log10(sub["conc_uM"].max() * 2), 100)
            ax.plot(xs, hill_curve(np.log10(xs), np.log10(r["IC50_uM"]), r["hill"]),
                    color=c, ls="--", lw=1.2, alpha=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-0.15, 1.15)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_title(drug, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    if i % ncol == 0:
        ax.set_ylabel("hERG 抑制率", fontsize=10)
    if i >= n - ncol:
        ax.set_xlabel("浓度 (µM)", fontsize=10)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
for j in range(n, len(axes)):
    axes[j].axis("off")
fig.suptitle("CiPA 全药池 28 药 hERG 剂量-反应（分实验室，点=浓度中位数±IQR，虚线=pooled Hill 拟合）",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(os.path.join(OUT, "fig1_剂量反应面板_28药.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ============ fig2: IC50 全景（各实验室点 + newCiPA 锚） ============
amap = dict(zip(anchor["drug"], anchor["newCiPA_hERG_uM"]))
order = dfl[~dfl["blind"]].groupby("drug")["IC50_uM"].median().sort_values().index.tolist()
fig, ax = plt.subplots(figsize=(13, 11))
for y, drug in enumerate(order):
    g = dfl[(dfl["drug"] == drug) & (~dfl["blind"])].dropna(subset=["IC50_uM"])
    med = g["IC50_uM"].median()
    for _, r in g.iterrows():
        ax.scatter(r["IC50_uM"], y, color=LAB_COLORS.get(r["lab"], "gray"),
                   marker=LAB_MARK.get(r["lab"], "o"), s=64, zorder=3, edgecolor="k", lw=0.4)
    ax.plot([g["IC50_uM"].min(), g["IC50_uM"].max()], [y, y], color="lightgray", lw=4, zorder=2)
    ax.scatter(med, y, marker="|", s=400, color="black", zorder=4)
    if drug in amap:
        ax.scatter(amap[drug], y, marker="x", s=90, color="#FF0000", lw=2.5, zorder=5)
ax.set_xscale("log")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order, fontsize=11)
ax.set_xlabel("hERG IC50 (µM，log 轴)", fontsize=13)
ax.set_title("28 药 IC50 全景：彩点=五实验室各自 pooled 值，黑竖=跨实验室中位，红×=CiPA 官方锚(newCiPA)",
             fontsize=13, fontweight="bold")
handles = [plt.Line2D([], [], marker=LAB_MARK[l], color=LAB_COLORS[l], ls="", ms=9, markeredgecolor="k", label=l)
           for l in LAB_COLORS]
handles.append(plt.Line2D([], [], marker="x", color="#FF0000", ls="", ms=10, mew=2.5, label="newCiPA 官方锚"))
handles.append(plt.Line2D([], [], marker="|", color="black", ls="", ms=16, label="跨实验室中位"))
ax.legend(handles=handles, fontsize=10, loc="lower right")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_IC50全景.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ============ fig3: 对拍验证双联 ============
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.5))
g = dfl[~dfl["blind"]].dropna(subset=["IC50_uM", "off_IC50_uM"])
a1.scatter(g["off_IC50_uM"], g["IC50_uM"], c=[LAB_COLORS.get(l, "gray") for l in g["lab"]], s=40, alpha=0.85)
lim = [min(g["off_IC50_uM"].min(), g["IC50_uM"].min()) / 3, max(g["off_IC50_uM"].max(), g["IC50_uM"].max()) * 3]
a1.plot(lim, lim, "k--", lw=1)
a1.plot(lim, [x * 3 for x in lim], "r:", lw=1); a1.plot(lim, [x / 3 for x in lim], "r:", lw=1)
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("官方 pooled IC50 (µM)", fontsize=12); a1.set_ylabel("我们 pooled IC50 (µM)", fontsize=12)
a1.set_title("复现验证：我们 vs 官方 ic50nh（红虚线=±3×）", fontsize=12, fontweight="bold")
a2.scatter(anchor["newCiPA_hERG_uM"], anchor["ours_med_uM"], s=60, color="#FF8C00", edgecolor="k", lw=0.5, zorder=3)
_flagged = anchor[anchor["fold"] > 3].sort_values("ours_med_uM")
for k, (_, r) in enumerate(_flagged.iterrows()):
    dx, dy = (6, 6) if k % 2 == 0 else (6, -13)
    a2.annotate(r["drug"], (r["newCiPA_hERG_uM"], r["ours_med_uM"]), fontsize=9,
                xytext=(dx, dy), textcoords="offset points", color="#B00000", fontweight="bold")
lim2 = [min(anchor["newCiPA_hERG_uM"].min(), anchor["ours_med_uM"].min()) / 5,
        max(anchor["newCiPA_hERG_uM"].max(), anchor["ours_med_uM"].max()) * 5]
a2.plot(lim2, lim2, "k--", lw=1)
a2.plot(lim2, [x * 3 for x in lim2], "r:", lw=1); a2.plot(lim2, [x / 3 for x in lim2], "r:", lw=1)
a2.set_xscale("log"); a2.set_yscale("log")
a2.set_xlabel("CiPA 官方锚 newCiPA hERG IC50 (µM)", fontsize=12)
a2.set_ylabel("本数据集跨实验室中位 (µM)", fontsize=12)
a2.set_title("锚对拍：本数据集(HESI手动膜片钳) vs CiPA一期官方（红虚线=±3×）", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_对拍验证.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ============ fig4: hill 全景 ============
fig, ax = plt.subplots(figsize=(13, 9))
order2 = dfl[~dfl["blind"]].groupby("drug")["hill"].median().sort_values(ascending=False).index.tolist()
for y, drug in enumerate(order2):
    g = dfl[(dfl["drug"] == drug) & (~dfl["blind"])].dropna(subset=["hill"])
    for _, r in g.iterrows():
        ax.scatter(r["hill"], y, color=LAB_COLORS.get(r["lab"], "gray"), marker=LAB_MARK.get(r["lab"], "o"),
                   s=64, edgecolor="k", lw=0.4, zorder=3)
    ax.plot([g["hill"].min(), g["hill"].max()], [y, y], color="lightgray", lw=4, zorder=2)
ax.axvline(1.0, color="red", ls="--", lw=2)
ax.text(1.01, len(order2) - 0.5, "hill=1（单一位点）", color="red", fontsize=11, fontweight="bold")
ax.set_yticks(range(len(order2)))
ax.set_yticklabels(order2, fontsize=11)
ax.set_xlabel("Hill 系数（pooled 拟合）", fontsize=13)
ax.set_title("28 药 Hill 系数全景：>1=协同/多态结合迹象，<1=负协同或群体异质", fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
handles = [plt.Line2D([], [], marker=LAB_MARK[l], color=LAB_COLORS[l], ls="", ms=9, markeredgecolor="k", label=l) for l in LAB_COLORS]
ax.legend(handles=handles, fontsize=10, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_hill全景.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ============ fig5: 实验室偏移热图（log10 IC50/跨实验室中位） ============
piv = []
for drug in order:
    g = dfl[(dfl["drug"] == drug) & (~dfl["blind"])].dropna(subset=["IC50_uM"])
    med = g["IC50_uM"].median()
    row = {"drug": drug}
    for lab in ["lab_1", "lab_2", "lab_3", "lab_4", "lab_5"]:
        sub = g[g["lab"] == lab]
        row[lab] = np.log10(sub["IC50_uM"].iloc[0] / med) if len(sub) else np.nan
    piv.append(row)
pv = pd.DataFrame(piv).set_index("drug")
fig, ax = plt.subplots(figsize=(8.5, 11))
im = ax.imshow(pv.to_numpy(), cmap="RdYlGn_r", vmin=-0.7, vmax=0.7, aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels(pv.columns, fontsize=12)
ax.set_yticks(range(len(pv))); ax.set_yticklabels(pv.index, fontsize=10)
for i in range(len(pv)):
    for j in range(5):
        v = pv.to_numpy()[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{10**v:.2f}×", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.35 else "black",
                    fontweight="bold" if abs(v) > 0.48 else "normal")
cb = fig.colorbar(im, ax=ax, shrink=0.6)
cb.set_label("log10(实验室IC50 / 跨实验室中位)", fontsize=11)
ax.set_title("实验室偏移热图：值=该实验室 IC50 ÷ 该药跨实验室中位\n（整列偏红=该实验室系统性偏高）", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_实验室偏移热图.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

print("5 张图已出：", [f for f in os.listdir(OUT) if f.startswith("fig")])
print("DONE")
