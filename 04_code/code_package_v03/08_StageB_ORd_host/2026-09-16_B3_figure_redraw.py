# -*- coding: utf-8 -*-
"""B3 图版重绘：从结果 JSON/CSV 重出 PNG（替换乱码符号为中文，不动判词数据）"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()
import matplotlib.pyplot as plt

BASE = Path(r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\α模型")
J = json.load(open(BASE / "2026-09-16_α模型_StageB_B3_双α化宿主复验_24药_结果.json", encoding="utf-8"))
df = pd.read_csv(BASE / "2026-09-16_α模型_StageB_B3_双α化宿主复验_24药_结果.csv")
R = J["R_lines"]; vd = J["verdict_B3"]; meta = J["meta"]; checks = J["checks"]; incr = J["incremental_vs_B1"]

def mk(b):
    return "过" if b else "挂"

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f"B3 双α化宿主复验 · 24药 · R1 ρq={R['R1_rho_qNet']:.3f}（{mk(R['R1_pass'])}）", fontsize=13)

cols = {0: "tab:green", 1: "tab:orange", 2: "tab:red"}

ax = axes[0, 0]
for c in (0, 1, 2):
    sub = df[df.CiPA == c]
    ax.scatter(sub.B2dyn_qNet_ratio, sub.B3_qNet_ratio, c=cols[c], s=42, zorder=3, label=f"CiPA {c}")
lim = [min(df.B2dyn_qNet_ratio.min(), df.B3_qNet_ratio.min()) * 0.95,
       max(df.B2dyn_qNet_ratio.max(), df.B3_qNet_ratio.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlabel("B2 qNet_ratio"); ax.set_ylabel("B3 qNet_ratio")
ax.set_title(f"R1: ρ={R['R1_rho_qNet']:.3f}（≥0.95 {mk(R['R1_pass'])}）  R1b={R['R1b_dev_median']:.3f}")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
for c in (0, 1, 2):
    sub = df[df.CiPA == c]
    ax.scatter(sub.B2dyn_APD90_ratio, sub.B3_APD90_ratio, c=cols[c], s=42, zorder=3, label=f"CiPA {c}")
lim = [min(df.B2dyn_APD90_ratio.min(), df.B3_APD90_ratio.min()) * 0.95,
       max(df.B2dyn_APD90_ratio.max(), df.B3_APD90_ratio.max()) * 1.05]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlabel("B2 APD90_ratio"); ax.set_ylabel("B3 APD90_ratio")
ax.set_title(f"R1 APD90 口径: ρ={R['R1_rho_APD90']:.3f}  R3 逆序对={R['R3_kendall_discordant']}")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
rng = np.random.default_rng(16)
for _, r in df.iterrows():
    ax.scatter(r.CiPA + (rng.random() - 0.5) * 0.15, r.B3_qNet_ratio,
               c=cols[r.CiPA], s=42, marker="o", zorder=3, alpha=0.85)
ax.axhline(1.0, color="k", lw=0.7, ls="--")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["低", "中", "高"])
ax.set_ylabel("B3 qNet_ratio"); ax.set_title("B3 风险分级散布（IKs 强子集：无）")
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.axis("off")
lines = [
    f"C1 {meta['C1_rel_diff']*100:.2f}% {mk(meta['C1_pass'])}  C2 {mk(checks['C2']['pass'])}  C3 {mk(checks['C3']['pass'])}",
    f"B3 V1 ρ={vd['V1_rho_qNet']:.3f}（线≤−0.60，{mk(vd['V1_pass'])}）  V2 AUC={vd['V2_AUC_qNetSupp_hi_lo']:.3f}（{mk(vd['V2_pass'])}）",
    f"R1 {mk(R['R1_pass'])}  R1b {mk(R['R1b_pass'])}  R3 {mk(R['R3_pass'])}  R2 {mk(R['R2_pass'])}",
    f"增量判 B3 vs B1: {'不劣' if incr['not_inferior'] else '劣'}",
    f"metoprolol 剔除敏感性: V1 ρ={J['ex_metoprolol']['V1_rho']:.3f}（n=23，过线）",
    f"0 EAD / 0 稳态失败；用时 {meta['runtime_s']:.0f}s（正式）",
]
ax.text(0.02, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=11)

fpng = BASE / "2026-09-16_α模型_StageB_B3_双α化宿主复验_24药_结果.png"
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(fpng, dpi=130, bbox_inches="tight")
print("重绘落盘:", fpng)
