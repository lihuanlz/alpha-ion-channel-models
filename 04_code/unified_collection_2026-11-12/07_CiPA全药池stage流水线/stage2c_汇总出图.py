# -*- coding: utf-8 -*-
"""Stage2c 汇总出图：分实验室 τ/形状比守恒（lab_1/2/3 主体，lab_4/5 标注协议限制）"""
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
ok = {}
with open(os.path.join(OUT, "stage2c_ckpt.jsonl"), encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        if r.get("ok"):
            ok[r["cell"]] = r
df = pd.DataFrame(list(ok.values()))
q = df[df["qc_pass"]].copy()
q.to_csv(os.path.join(OUT, "stage2c_cell_metrics.csv"), index=False, encoding="utf-8-sig")
print(f"ok={len(df)} QC过={len(q)}")

LAB_COLORS = {"lab_1": "#E6194B", "lab_2": "#3CB44B", "lab_3": "#4363D8", "lab_4": "#F58231", "lab_5": "#911EB4"}
KIN_LABS = ["lab_1", "lab_2", "lab_3"]  # 有动力学信息的协议

fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

# (a) 形状比变化分布（lab_1/3 主力）
ax = axes[0, 0]
bins = np.linspace(0, 2.5, 76)
for lab in ["lab_1", "lab_3"]:
    v = q[q["lab"] == lab]["shape_chg"].dropna()
    ax.hist(v.clip(0, 2.5), bins=bins, alpha=0.55, color=LAB_COLORS[lab],
            label=f"{lab} (n={len(v)}, 中位={v.median():.2f})")
ax.axvline(1.0, color="k", ls="--", lw=1.5)
ax.set_xlabel("形状比变化（药物稳态 ÷ 对照）"); ax.set_ylabel("细胞数")
ax.set_title("(a) 复极化峰/平台形状比：1=纯占据；<1=峰削更多（状态依赖迹象）", fontweight="bold", fontsize=11)
ax.legend()

# (b) 形状比变化 vs 阻断深度（lab_1/2/3）
ax = axes[0, 1]
for lab in KIN_LABS:
    g = q[q["lab"] == lab]
    ax.scatter(g["block_frac"], g["shape_chg"].clip(0, 2.5), s=10, alpha=0.4,
               color=LAB_COLORS[lab], label=lab)
ax.axhline(1.0, color="k", ls="--", lw=1.2)
ax.set_xlabel("阻断深度"); ax.set_ylabel("形状比变化")
ax.set_title("(b) 形状扰动随阻断深度（深阻断处更明显=状态依赖）", fontweight="bold", fontsize=11)
ax.legend(markerscale=3)

# (c) τ_tail 比值（lab_3 长尾窗主力 + lab_1）
ax = axes[1, 0]
bins2 = np.linspace(0, 3, 76)
for lab, note in [("lab_3", "长尾窗~1.1s 可靠"), ("lab_1", "短尾窗 分散大")]:
    v = q[q["lab"] == lab]["tau_ratio"].dropna()
    v = v[(v > 0) & (v < 10)]
    ax.hist(v.clip(0, 3), bins=bins2, alpha=0.55, color=LAB_COLORS[lab],
            label=f"{lab} (n={len(v)}, 中位={v.median():.2f}) {note}")
ax.axvline(1.0, color="k", ls="--", lw=1.5)
ax.set_xlabel("去激活 τ 比值（药物 ÷ 对照）"); ax.set_ylabel("细胞数")
ax.set_title("(c) 去激活时间常数守恒检验：τ 基本不动", fontweight="bold", fontsize=11)
ax.legend()

# (d) 药级汇总：形状比变化中位（lab_1/3，n≥3）
ax = axes[1, 1]
qq = q[q["lab"].isin(["lab_1", "lab_3"])]
rows = []
for (lab, drug), g in qq.groupby(["lab", "drug"]):
    v = g["shape_chg"].dropna()
    if len(v) >= 3:
        rows.append({"drug": drug, "lab": lab, "med": v.median(), "n": len(v),
                     "block": g["block_frac"].median()})
dd = pd.DataFrame(rows).sort_values("med")
colors = [LAB_COLORS[l] for l in dd["lab"]]
ax.barh(range(len(dd)), dd["med"], color=colors, edgecolor="k", lw=0.3)
ax.axvline(1.0, color="k", ls="--", lw=1.5)
ax.set_yticks(range(len(dd)))
ax.set_yticklabels([f"{r['drug']}({r['lab'][-1]}) n={r['n']}" for _, r in dd.iterrows()], fontsize=6.5)
ax.set_xlabel("形状比变化中位（lab_1/3 合并视图）")
ax.set_title("(d) 各药动力学扰动排序", fontweight="bold", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_波形tau守恒检验.png"), dpi=140, bbox_inches="tight")
plt.close(fig)

# ---- 文本汇总 ----
print("\n=== 核心数字 ===")
for lab in KIN_LABS:
    g = q[q["lab"] == lab]
    sr = g["shape_chg"].dropna()
    tr = g["tau_ratio"].dropna()
    tr = tr[(tr > 0) & (tr < 10)]
    deep = g[g["block_frac"] > 0.6]["shape_chg"].dropna()
    print(f"{lab}: 形状比 {sr.median():.2f}[{sr.quantile(.25):.2f},{sr.quantile(.75):.2f}] (n={len(sr)}) | "
          f"τ比 {tr.median():.2f}[{tr.quantile(.25):.2f},{tr.quantile(.75):.2f}] (n={len(tr)}) | "
          f"深阻断形状比 {deep.median():.2f} (n={len(deep)})")
# lab_1+3 合并深阻断统计
gg = q[q["lab"].isin(["lab_1", "lab_3"])]
deep = gg[gg["block_frac"] > 0.6]["shape_chg"].dropna()
sig = ((deep < 0.8) | (deep > 1.25)).mean()
print(f"\nlab_1+3 深阻断(>60%): 形状比中位 {deep.median():.2f}，偏离>20%占比 {sig*100:.0f}% (n={len(deep)})")
tr13 = gg["tau_ratio"].dropna(); tr13 = tr13[(tr13 > 0) & (tr13 < 10)]
tsig = ((tr13 < 0.8) | (tr13 > 1.25)).mean()
print(f"lab_1+3 τ比: 中位 {tr13.median():.2f}，偏离>20%占比 {tsig*100:.0f}% (n={len(tr13)})")
print("DONE")
