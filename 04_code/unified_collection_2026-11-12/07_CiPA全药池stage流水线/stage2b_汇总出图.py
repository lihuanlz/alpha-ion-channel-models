# -*- coding: utf-8 -*-
"""Stage2b 汇总与出图：τ/形状比守恒检验 + 与 Stage1 hill 异常交叉"""
import os, json
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
rows = []
with open(os.path.join(OUT, "stage2b_ckpt.jsonl"), encoding="utf-8") as fh:
    for line in fh:
        try:
            r = json.loads(line)
            if r.get("ok"):
                rows.append(r)
        except Exception:
            pass
df = pd.DataFrame(rows).drop_duplicates(subset=["cell"], keep="last")
print(f"细胞总数 {len(df)}；QC 过 {int(df['qc_pass'].sum())}")
q = df[df["qc_pass"]].copy()
q.to_csv(os.path.join(OUT, "stage2b_cell_metrics.csv"), index=False, encoding="utf-8-sig")

# ---- 药×实验室汇总 ----
summ = []
for (lab, drug), g in q.groupby(["lab", "drug"]):
    deep = g[g["block_frac"] > 0.8]
    tr = g["tau_ratio"].dropna()
    sr = g["shape_ratio_chg"].dropna()
    srd = deep["shape_ratio_chg"].dropna()
    trd = deep["tau_ratio"].dropna()
    summ.append({"lab": lab, "drug": drug, "n_cells": len(g),
                 "block_med": float(g["block_frac"].median()),
                 "tau_ratio_med": float(tr.median()) if len(tr) else np.nan,
                 "tau_ratio_iqr": f"[{tr.quantile(0.25):.2f},{tr.quantile(0.75):.2f}]" if len(tr) else "",
                 "shape_chg_med": float(sr.median()) if len(sr) else np.nan,
                 "shape_chg_iqr": f"[{sr.quantile(0.25):.2f},{sr.quantile(0.75):.2f}]" if len(sr) else "",
                 "n_deep": len(deep),
                 "shape_chg_deep_med": float(srd.median()) if len(srd) else np.nan,
                 "tau_ratio_deep_med": float(trd.median()) if len(trd) else np.nan})
dfs = pd.DataFrame(summ)
dfs.to_csv(os.path.join(OUT, "stage2b_drug_lab_summary.csv"), index=False, encoding="utf-8-sig")

# ---- 全池统计 ----
print("\n=== 全池（QC 过细胞）===")
sr_all = q["shape_ratio_chg"].dropna()
tr_all = q["tau_ratio"].dropna()
print(f"形状比变化（药物稳态/对照）: n={len(sr_all)} 中位={sr_all.median():.3f} IQR=[{sr_all.quantile(.25):.3f},{sr_all.quantile(.75):.3f}]")
print(f"τ_tail 比值: n={len(tr_all)} 中位={tr_all.median():.3f} IQR=[{tr_all.quantile(.25):.3f},{tr_all.quantile(.75):.3f}]")
deep = q[q["block_frac"] > 0.8]
srd = deep["shape_ratio_chg"].dropna(); trd = deep["tau_ratio"].dropna()
print(f"\n=== 深阻断(>80%)亚组 n={len(deep)} ===")
print(f"形状比变化: 中位={srd.median():.3f} IQR=[{srd.quantile(.25):.3f},{srd.quantile(.75):.3f}]")
print(f"τ_tail 比值: n={len(trd)} 中位={trd.median():.3f} IQR=[{trd.quantile(.25):.3f},{trd.quantile(.75):.3f}]")
# 形状比 |变化|>20% 的比例
big = (srd < 0.8) | (srd > 1.25)
print(f"深阻断下形状比偏离>20%: {int(big.sum())}/{len(srd)} = {big.mean()*100:.0f}%")

# ---- 与 Stage1 hill 交叉 ----
s1 = pd.read_csv(os.path.join(OUT, "stage1_drug_lab_summary.csv"))
m = dfs.merge(s1[["lab", "drug", "off_nh"]], on=["lab", "drug"], how="left")
hi = m[m["off_nh"] > 1.15]["shape_chg_deep_med"].dropna()
lo = m[m["off_nh"] < 0.85]["shape_chg_deep_med"].dropna()
mid = m[(m["off_nh"] >= 0.85) & (m["off_nh"] <= 1.15)]["shape_chg_deep_med"].dropna()
print(f"\n=== 交叉：hill 方向 × 深阻断形状比变化 ===")
print(f"hill>1.15 (n={len(hi)}): 形状比中位 {hi.median():.3f}" if len(hi) else "hill>1.15 无数据")
print(f"hill~1   (n={len(mid)}): 形状比中位 {mid.median():.3f}" if len(mid) else "")
print(f"hill<0.85(n={len(lo)}): 形状比中位 {lo.median():.3f}" if len(lo) else "")
m.to_csv(os.path.join(OUT, "stage2b_with_hill.csv"), index=False, encoding="utf-8-sig")

# ================= 出图 =================
LAB_COLORS = {"lab_1": "#E6194B", "lab_2": "#3CB44B", "lab_3": "#4363D8", "lab_4": "#F58231", "lab_5": "#911EB4"}
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# (a) 形状比变化分布：全体 vs 深阻断
ax = axes[0, 0]
bins = np.linspace(0, 2, 81)
ax.hist(sr_all.clip(0, 2), bins=bins, alpha=0.55, color="#4363D8", label=f"全部 QC 细胞 (n={len(sr_all)})")
ax.hist(srd.clip(0, 2), bins=bins, alpha=0.65, color="#E6194B", label=f"深阻断>80% (n={len(srd)})")
ax.axvline(1.0, color="k", ls="--", lw=1.5)
ax.set_xlabel("形状比变化（药物稳态 ÷ 对照）"); ax.set_ylabel("细胞数")
ax.set_title("(a) 动力学形状守恒检验：1=纯占据效应", fontweight="bold")
ax.legend()

# (b) 形状比变化 vs 阻断深度
ax = axes[0, 1]
for lab, g in q.groupby("lab"):
    ax.scatter(g["block_frac"], g["shape_ratio_chg"].clip(0, 2.5), s=8, alpha=0.35,
               color=LAB_COLORS.get(lab, "gray"), label=lab)
ax.axhline(1.0, color="k", ls="--", lw=1.2)
ax.set_xlabel("阻断深度 (1 − A_drug/A_ctrl)"); ax.set_ylabel("形状比变化")
ax.set_title("(b) 形状变化 vs 阻断深度（分实验室）", fontweight="bold")
ax.legend(markerscale=3)

# (c) τ_tail 比值分布（深阻断）
ax = axes[1, 0]
bins2 = np.linspace(0, 4, 81)
ax.hist(tr_all.clip(0, 4), bins=bins2, alpha=0.55, color="#3CB44B", label=f"全部 (n={len(tr_all)})")
ax.hist(trd.clip(0, 4), bins=bins2, alpha=0.65, color="#F58231", label=f"深阻断 (n={len(trd)})")
ax.axvline(1.0, color="k", ls="--", lw=1.5)
ax.set_xlabel("τ_tail 比值（药物 ÷ 对照）"); ax.set_ylabel("细胞数")
ax.set_title("(c) 去激活 τ 守恒检验（注：1s sweep 尾窗短，分散大）", fontweight="bold")
ax.legend()

# (d) 药级形状比变化（深阻断中位，按药排序）
ax = axes[1, 1]
dd = dfs.dropna(subset=["shape_chg_deep_med"])
dd = dd[dd["n_deep"] >= 2].sort_values("shape_chg_deep_med")
colors = [LAB_COLORS.get(l, "gray") for l in dd["lab"]]
ax.barh(range(len(dd)), dd["shape_chg_deep_med"], color=colors, edgecolor="k", lw=0.3)
ax.axvline(1.0, color="k", ls="--", lw=1.5)
ax.set_yticks(range(len(dd)))
ax.set_yticklabels([f"{r['drug']}({r['lab'][-1]})" for _, r in dd.iterrows()], fontsize=7)
ax.set_xlabel("深阻断形状比变化（药×实验室中位）")
ax.set_title("(d) 各药动力学扰动排序（n_deep≥2）", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_波形tau守恒检验.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print("\nfig6 已出")
print("DONE")
