# -*- coding: utf-8 -*-
"""Stage3 汇总出图：阻断时间进程挖掘（未平台化 + E-4031 时钟 + 例证曲线）"""
import os, json, glob, zipfile
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

LAW = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia"
CIPA = os.path.join(LAW, "数据总库", "01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t")
OUT = os.path.join(LAW, "分析_全药池_2026-11-10")
q = pd.read_csv(os.path.join(OUT, "stage3_onset_metrics.csv"))
LAB_COLORS = {"lab_1": "#E6194B", "lab_2": "#3CB44B", "lab_3": "#4363D8", "lab_4": "#F58231", "lab_5": "#911EB4"}

fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))

# (a) 例证洗入曲线：lab_1/lab_3/lab_4 各一条 dofetilide
ax = axes[0, 0]
shown = 0
for lab, col in [("lab_1", LAB_COLORS["lab_1"]), ("lab_3", LAB_COLORS["lab_3"]), ("lab_4", LAB_COLORS["lab_4"])]:
    cands = q[(q["lab"] == lab) & (q["drug"] == "dofetilide") & (q["e4_found"])]
    if len(cands) == 0:
        continue
    row = cands.iloc[len(cands)//2]
    zp = os.path.join(CIPA, lab, "hERG", "dofetilide", "subtracted", "ted",
                      row["cell"].split("/")[-1] + ".csv.zip")
    if not os.path.exists(zp):
        # blinded 目录回退
        cand2 = glob.glob(os.path.join(CIPA, lab, "*", "dofetilide*", "subtracted", "ted", "*.csv.zip"))
        zp = None
        for c2 in cand2:
            if os.path.basename(c2).replace(".csv.zip","") == row["cell"].split("/")[-1]:
                zp = c2; break
        if zp is None: continue
    z = zipfile.ZipFile(zp)
    df0 = pd.read_csv(z.open(z.namelist()[0]))
    cols = [c for c in df0.columns] if False else [c for c in df0.columns if c.startswith("trace_#") and "_current_" in c]
    uf = {"pa":1.0,"a":1e12}.get(cols[0].rsplit("_current_",1)[-1].lower(), 1.0)
    amps = []
    for c in cols:
        y = df0[c].to_numpy(float)*uf
        if np.std(y) > 5 and np.ptp(y) <= 4000 and abs(np.max(y)) <= 3000:
            i0 = int(0.05*len(y)); amps.append(float(y[i0:].max()))
    amps = np.array(amps); a_ref = np.median(amps[:max(3, len(amps)//6)])
    b = 1 - amps/a_ref
    ax.plot(np.arange(len(b)), b, ".", ms=4, alpha=0.75, color=col,
            label=f"{lab}（span={len(b)} sweeps）")
    shown += 1
ax.axhline(1.0, color="k", ls=":", lw=1)
ax.set_xlabel("sweep 序号（时间进程）"); ax.set_ylabel("阻断分数 1−A/A_ref")
ax.set_title("(a) 例证：dofetilide 阻断发展曲线（三实验室）", fontweight="bold", fontsize=11)
ax.legend(fontsize=9); ax.set_ylim(-0.05, 1.15)

# (b) 各实验室未平台化占比 + E-4031 时钟
ax = axes[0, 1]
labs = ["lab_1","lab_2","lab_3","lab_4","lab_5"]
unset = [ (q[q['lab']==l]['tail_slope']>0.002).mean()*100 for l in labs ]
clock = [ q[q['lab']==l]['tau_e4031_sweeps'].median() for l in labs ]
x = np.arange(len(labs))
bars = ax.bar(x, unset, 0.55, color=[LAB_COLORS[l] for l in labs], edgecolor="k", lw=0.4)
for xi, u, c in zip(x, unset, clock):
    ax.text(xi, u+1.5, f"{u:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.text(xi, 5, f"E-4031\n时钟\n{c:.1f}", ha="center", fontsize=8, color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labs)
ax.set_ylabel("末段仍爬升细胞占比（%）")
ax.set_title("(b) 43–70% 细胞记录结束时阻断未达平台（柱内=E-4031 交换时钟）", fontweight="bold", fontsize=11)
ax.set_ylim(0, 88)

# (c) 逐药未平台化排序
ax = axes[1, 0]
g = q.groupby("drug").agg(u=("tail_slope", lambda s: (s>0.002).mean()*100), n=("cell","count")).query("n>=20").sort_values("u")
cols3 = []
anchors = {"pimozide","tamoxifen","dofetilide","metoprolol"}
for d in g.index:
    cols3.append("#B71C1C" if d in anchors else "#4363D8")
ax.barh(range(len(g)), g["u"], color=cols3, edgecolor="k", lw=0.3)
ax.set_yticks(range(len(g)))
ax.set_yticklabels([f"{d} (n={r['n']})" for d, (_, r) in zip(g.index, g.iterrows())], fontsize=7.5)
ax.set_xlabel("未平台化细胞占比（%）")
ax.set_title("(c) 逐药未平台化排序（红=锚表离群四药，全部 ≥58%）", fontweight="bold", fontsize=11)
ax.axvline(60, color="gray", ls="--", lw=1)

# (d) 末段爬升 ≠ rundown：斜率与对照漂移脱钩
ax = axes[1, 1]
F = pd.read_csv(os.path.join(OUT, "stage2_final_cell_metrics.csv"))
m = q.merge(F[["cell","control_drift"]], on="cell", how="left")
for lab in labs:
    gg = m[m["lab"]==lab]
    ax.scatter(gg["control_drift"], gg["tail_slope"], s=9, alpha=0.35, color=LAB_COLORS[lab], label=lab)
ax.axhline(0.002, color="k", ls="--", lw=1, label="未平台化阈值")
rho = m[["control_drift","tail_slope"]].dropna().corr(method="spearman").iloc[0,1]
ax.set_xlabel("对照段漂移（rundown）"); ax.set_ylabel("加药段末 25% 斜率（frac/sweep）")
ax.set_title(f"(d) 末段爬升与 rundown 脱钩（Spearman ρ={rho:.2f}）", fontweight="bold", fontsize=11)
ax.legend(markerscale=3, fontsize=8.5); ax.set_ylim(-0.01, 0.03)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_阻断时间进程_未平台化.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print("fig9 已生成")
print("DONE")
