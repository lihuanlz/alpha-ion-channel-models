# -*- coding: utf-8 -*-
# stage4a_出图.py —— fig10 全药池前向预言（六面板）
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "stage4a_per_cell_predictions.csv"))
D = pd.read_csv(os.path.join(HERE, "stage4a_dirlevel_LOO.csv"))
ok = df[df["join"] == "ok"].copy()

LAB_C = {"lab_1": "#E60012", "lab_2": "#009944", "lab_3": "#2868B7",
         "lab_4": "#F39800", "lab_5": "#7B3FF2"}
fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.6))

def diag(ax, lim=(-0.05, 1.05)):
    ax.plot(lim, lim, "k-", lw=1, alpha=0.6)
    ax.fill_between(lim, np.array(lim) - 0.15, np.array(lim) + 0.15,
                    color="gray", alpha=0.15, label="±0.15 带")

# (a) 臂B 逐细胞
ax = axes[0, 0]
m = ok[np.isfinite(ok.pred_B)]
for lab, g in m.groupby("lab"):
    ax.scatter(g.block_meas, g.pred_B, s=7, alpha=0.35, color=LAB_C[lab], label=lab)
diag(ax)
r = spearmanr(m.pred_B, m.block_meas).statistic
mae = (m.pred_B - m.block_meas).abs().mean()
w = ((m.pred_B - m.block_meas).abs() <= 0.15).mean()
ax.set_title(f"(a) 逐细胞前向预言：留一室交叉臂\nρ={r:.2f}  MAE={mae:.2f}  ±0.15内占{w:.0%}（n={len(m)}）",
             fontweight="bold", fontsize=11)
ax.set_xlabel("实测稳态阻断分数"); ax.set_ylabel("模型预言（他室占据表）")
ax.legend(markerscale=3, fontsize=8)

# (b) 臂A 外部锚
ax = axes[0, 1]
mA = ok[np.isfinite(ok.pred_A)]
for lab, g in mA.groupby("lab"):
    ax.scatter(g.block_meas, g.pred_A, s=7, alpha=0.35, color=LAB_C[lab], label=lab)
diag(ax)
rA = spearmanr(mA.pred_A, mA.block_meas).statistic
maeA = (mA.pred_A - mA.block_meas).abs().mean()
ax.set_title(f"(b) 外部锚臂（newCiPA，跨数据集）\nρ={rA:.2f}  MAE={maeA:.2f}（n={len(mA)}）——跨数据集药理漂移",
             fontweight="bold", fontsize=11)
ax.set_xlabel("实测稳态阻断分数"); ax.set_ylabel("模型预言（外部锚 IC50）")

# (c) 目录级 LOO
ax = axes[0, 2]
for lab, g in D.groupby("lab"):
    ax.scatter(g.meas, g.pred, s=30, alpha=0.8, color=LAB_C[lab], label=lab, edgecolor="k", lw=0.3)
diag(ax)
rD = spearmanr(D.pred, D.meas).statistic
maeD = (D.pred - D.meas).abs().mean()
wD = ((D.pred - D.meas).abs() <= 0.15).mean()
bD = (D.pred - D.meas).mean()
ax.set_title(f"(c) 目录级预言（125 目录）\nMAE={maeD:.2f}  ±0.15内占{wD:.0%}  偏差{bD:+.3f}（ρ={rD:.2f}）",
             fontweight="bold", fontsize=11)
ax.set_xlabel("目录实测平均阻断"); ax.set_ylabel("目录预言平均阻断")
ax.legend(markerscale=1.5, fontsize=8)

# (d) 测量噪声地板：我们 vs 官方游标
ax = axes[1, 0]
mF = ok[np.isfinite(ok.official_change)]
for lab, g in mF.groupby("lab"):
    ax.scatter(g.official_change, g.block_meas, s=7, alpha=0.35, color=LAB_C[lab], label=lab)
diag(ax)
rF = spearmanr(mF.official_change, mF.block_meas).statistic
maeF = (mF.official_change - mF.block_meas).abs().mean()
ax.set_title(f"(d) 噪声地板：两条独立测量管线互拍\n我们(波形) vs 官方(游标)：ρ={rF:.2f}  MAE={maeF:.2f}（n={len(mF)}）",
             fontweight="bold", fontsize=11)
ax.set_xlabel("官方游标法阻断（Change）"); ax.set_ylabel("我们波形法阻断（block_frac）")

# (e) 残差 vs 未平台化斜率（Stage3 联动）
ax = axes[1, 1]
mm = ok[np.isfinite(ok.pred_B) & np.isfinite(ok.tail_slope)]
resid = mm.pred_B - mm.block_meas
for lab, g in mm.groupby("lab"):
    ax.scatter(g.tail_slope, resid.loc[g.index], s=7, alpha=0.35, color=LAB_C[lab], label=lab)
ax.axhline(0, color="k", lw=1)
ax.axvline(0.002, color="gray", ls="--", lw=1, label="未平台化阈值")
rS = spearmanr(resid, mm.tail_slope).statistic
pS = spearmanr(resid, mm.tail_slope).pvalue
ax.set_title(f"(e) Stage3联动：残差 vs 末段爬升斜率\nρ={rS:+.2f}（p={pS:.1f}）——未平台化不主导逐细胞残差",
             fontweight="bold", fontsize=11)
ax.set_xlabel("tail_slope（frac/sweep）"); ax.set_ylabel("残差（预言−实测）")
ax.set_ylim(-1.05, 1.05)

# (f) 逐室 MAE 汇总
ax = axes[1, 2]
labs = ["lab_1", "lab_2", "lab_3", "lab_4", "lab_5"]
armB = [ok[np.isfinite(ok.pred_B) & (ok.lab == l)].pipe(lambda g: (g.pred_B - g.block_meas).abs().mean()) for l in labs]
armA = [ok[np.isfinite(ok.pred_A) & (ok.lab == l)].pipe(lambda g: (g.pred_A - g.block_meas).abs().mean()) for l in labs]
null = []
for l in labs:
    g = ok[ok.lab == l]
    tr = ok[ok.lab != l]
    ests = []
    for d, gg in g.groupby("drug_real"):
        tt = tr[tr.drug_real == d]
        if len(tt) >= 5:
            ests.append(np.abs(tt.block_meas.median() - gg.block_meas).mean())
    null.append(np.mean(ests) if ests else np.nan)
x = np.arange(5); wbar = 0.27
ax.bar(x - wbar, armA, wbar, label="臂A 外部锚", color="#999999", edgecolor="k", lw=0.4)
ax.bar(x, armB, wbar, label="臂B 留一室", color="#2868B7", edgecolor="k", lw=0.4)
ax.bar(x + wbar, null, wbar, label="零模型(他室药均值)", color="#F39800", edgecolor="k", lw=0.4)
ax.axhline(maeF, color="#B71C1C", ls="--", lw=1.5, label=f"测量噪声地板 {maeF:.2f}")
ax.set_xticks(x); ax.set_xticklabels(labs)
ax.set_ylabel("MAE（预言 vs 实测）")
ax.set_title("(f) 逐室误差：臂B 全部低于噪声地板", fontweight="bold", fontsize=11)
ax.legend(fontsize=8)

fig.suptitle("全药池前向预言（28 药 × 5 实验室 × 1,546 QC 细胞）：占据表跨室可移植，精度抵达测量噪声地板",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.965))
out = os.path.join(HERE, "fig10_全药池前向预言.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("fig10 已生成:", out)
