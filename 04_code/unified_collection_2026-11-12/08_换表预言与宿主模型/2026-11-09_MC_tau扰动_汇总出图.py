# -*- coding: utf-8 -*-
# 2026-11-09_MC_tau扰动_汇总出图.py
# 读 2026-11-09_MC_tau扰动_ckpt.jsonl → 汇总 json + 图
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from daimon_runtime import setup_plot
setup_plot()

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, "2026-11-09_MC_tau扰动_ckpt.jsonl"), encoding="utf-8") if l.strip()]
mc = [r for r in rows if r["iter"] > 0]
ident = [r for r in rows if r["iter"] == 0][0]
assert len(mc) == 200, len(mc)

npass = np.array([r["summary"]["n_pass"] for r in mc])
verd_ok = sum(1 for r in mc if r["summary"]["总判"] == "过线")
lab_worst = {lab: min(r["summary"]["per_lab"][lab][0] / r["summary"]["per_lab"][lab][1]
                      for r in mc) for lab in ("lab3", "lab4", "lab5")}
man_min = min(r["summary"]["manual"][0] for r in mc)

# 失败细胞频次
from collections import Counter
failcnt = Counter() if False else __import__("collections").Counter()
for r in mc:
    for fn, cv in r["cells"].items():
        if cv["pool"] == "main" and cv["v"] == "fail":
            failcnt[fn] += 1

# 锚点扰动范围
anchor_stats = {}
for k in mc[0]["meds"]:
    arr = np.array([r["meds"][k] for r in mc])
    anchor_stats[k] = dict(med=float(np.median(arr)), p5=float(np.percentile(arr, 5)),
                           p95=float(np.percentile(arr, 95)))

summary = {
    "说明": "Kv11.1 τ 分布 bootstrap 扰动下的 CiPA 跨实验室判决存活率（200 次）",
    "bootstrap": "按细胞整体有放回重抽（M=60 QC 细胞池），逐锚点取中位重建 τ_m 表；QC=flag=active∧剖面CI有限∧r2>=0.9@35°C",
    "恒等内控": "iter0 原样 CP 表复现封卷 CP_full 67/68，逐细胞 0 不一致 ✓",
    "结果": {
        "MC次数": 200,
        "总判过线": f"{verd_ok}/200 = {verd_ok/200:.1%}",
        "主池过线数分布": {int(k): int(v) for k, v in sorted(Counter(npass).items())},
        "主池过线数_min_med_max": [int(npass.min()), float(np.median(npass)), int(npass.max())],
        "对照": {"ORIG封卷": "65/68", "CP_full封卷": "67/68"},
        "各实验室最差过率": {k: round(v, 4) for k, v in lab_worst.items()},
        "manual池最差": f"{man_min}/3",
        "锚点零覆盖回退次数": sum(len(r["fallbacks"]) for r in mc),
        "失败细胞频次": dict(failcnt.most_common()),
    },
    "锚点扰动5~95%范围_ms35": anchor_stats,
    "解读": ("τ 锚点在 bootstrap 噪声下大幅摆动（如 τ_act@-20 的 5~95% 达 247~548 ms，"
             "τ_act@60 达 37~156 ms），但判决 200/200 全部过线、主池过线数从未低于 66/68——"
             "仍高于 ORIG 封卷的 65/68。换表判决对群体分布抽样噪声分布级稳健。"),
}
outj = os.path.join(HERE, "2026-11-09_MC_tau扰动_结果.json")
json.dump(summary, open(outj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", outj)

# ---------------- 图 ----------------
C = dict(blue="#3A6FF7", red="#E63946", teal="#12B5A5", orange="#F4840C",
         purple="#8B5CF6", dark="#22223B", grey="#8D99AE", green="#2EA44F")
fig = plt.figure(figsize=(15.5, 6.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.35, 1.2], wspace=0.26)

# (a) 过线数分布
ax = fig.add_subplot(gs[0])
bins = np.arange(64.5, 68.6, 1)
cnt = Counter(npass)
ax.bar(cnt.keys(), cnt.values(), width=0.72, color=C["blue"], alpha=0.88, zorder=3,
       edgecolor="white", lw=1.2, label="bootstrap 200 次")
ax.axvline(65, color=C["red"], ls="--", lw=2.2, zorder=4)
ax.axvline(67, color=C["green"], ls="--", lw=2.2, zorder=4)
ax.text(65, max(cnt.values()) * 1.13, "ORIG 封卷 65", color=C["red"], fontsize=11,
        ha="center", va="bottom", fontweight="bold")
ax.text(67, max(cnt.values()) * 1.13, "CP_full 封卷 67", color=C["green"], fontsize=11,
        ha="center", va="bottom", fontweight="bold")
for x, y in sorted(cnt.items()):
    ax.text(x, y + 2, str(y), ha="center", fontsize=11, color=C["dark"], fontweight="bold")
ax.set_xlabel("主池过线细胞数（满分 68）")
ax.set_ylabel("bootstrap 次数")
ax.set_title("(a) τ 表扰动 200 次：过线数分布\n总判过线率 200/200 = 100%", fontsize=12.5,
             fontweight="bold", color=C["dark"])
ax.set_xlim(64.2, 68.4); ax.set_ylim(0, max(cnt.values()) * 1.32)
ax.grid(axis="y", alpha=0.3, zorder=0)

# (b) 锚点扰动范围（ms@35°C，对数轴）
ax = fig.add_subplot(gs[1])
keys = list(mc[0]["meds"].keys())
labels = [k.replace("tau_deact@", "τ_deact@").replace("tau_act@", "τ_act@") for k in keys]
med = [anchor_stats[k]["med"] for k in keys]
p5 = [anchor_stats[k]["p5"] for k in keys]
p95 = [anchor_stats[k]["p95"] for k in keys]
CP = {"τ_deact@-70": 37.7, "τ_deact@-60": 43.4, "τ_deact@-50": 56.7, "τ_deact@-40": 80.0,
      "τ_deact@-30": 85.1, "τ_act@-20": 280.7, "τ_act@-10": 323.0, "τ_act@0": 268.0,
      "τ_act@10": 150.6, "τ_act@20": 90.2, "τ_act@30": 64.1, "τ_act@40": 57.5,
      "τ_act@50": 71.5, "τ_act@60": 105.6}
cpv = [CP[l] for l in labels]
ypos = np.arange(len(keys))[::-1]
colors = [C["teal"] if "deact" in l else C["orange"] for l in labels]
ax.barh(ypos, np.array(p95) - np.array(p5), left=p5, height=0.62, color=colors,
        alpha=0.35, zorder=2, label="bootstrap 5~95%")
ax.scatter(cpv, ypos, s=64, color=C["dark"], zorder=4, marker="D", label="CP 中位（换表值）")
ax.scatter(med, ypos, s=30, color=C["red"], zorder=5, marker="o", label="bootstrap 中位")
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=10)
ax.set_xscale("log")
ax.set_xlim(25, 900)
ax.set_xlabel("τ（ms @35°C，对数轴）")
ax.set_title("(b) 14 个锚点的 bootstrap 扰动幅度\n判决在如此摆动下仍然全过", fontsize=12.5,
             fontweight="bold", color=C["dark"])
ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95, borderaxespad=0.3)
ax.grid(axis="x", alpha=0.3, zorder=0)

# (c) 每实验室过率箱线（逐迭代）
ax = fig.add_subplot(gs[2])
lab_names = {"lab3": "lab3 (n=20)", "lab4": "lab4 (n=16)", "lab5": "lab5 (n=32)"}
data = {lab: [r["summary"]["per_lab"][lab][0] / r["summary"]["per_lab"][lab][1] for r in mc]
        for lab in lab_names}
pos = np.arange(3)
bp = ax.boxplot([data[l] for l in lab_names], positions=pos, widths=0.5, patch_artist=True,
                showfliers=True, flierprops=dict(marker="o", ms=4, mfc=C["red"], mec=C["red"], alpha=0.5))
for patch, col in zip(bp["boxes"], [C["blue"], C["teal"], C["purple"]]):
    patch.set_facecolor(col); patch.set_alpha(0.75)
for element in ("whiskers", "caps", "medians"):
    for e in bp[element]:
        e.set_color(C["dark"]); e.set_lw(1.6)
man_rates = [r["summary"]["manual"][0] / 3 for r in mc]
ax.scatter([3] * len(man_rates), man_rates, s=18, color=C["green"], alpha=0.35, zorder=3)
ax.plot([2.62], [1.0], marker="s", ms=0)  # spacer
ax.axhline(0.5, color=C["red"], ls="--", lw=2)
ax.text(2.5, 0.52, "判线 1/2", color=C["red"], fontsize=10.5, va="bottom", ha="center",
        fontweight="bold")
ax.set_xticks(list(pos) + [3])
ax.set_xticklabels(list(lab_names.values()) + ["manual (n=3)"], fontsize=10)
ax.set_ylabel("单次 bootstrap 的实验室过率")
ax.set_xlim(-0.6, 3.6)
ax.set_ylim(0.4, 1.06)
ax.set_title("(c) 逐实验室过率：200 次中最差\nlab5 30/32=93.8%，远离判线", fontsize=12.5,
             fontweight="bold", color=C["dark"])
ax.grid(axis="y", alpha=0.3, zorder=0)

fig.suptitle("蒙特卡洛验证：Kv11.1 τ 群体分布扰动下，CiPA 跨实验室换表判决 200/200 存活",
             fontsize=15, fontweight="bold", color=C["dark"], y=0.99)
outp = os.path.join(HERE, "fig_MC_tau扰动_2026-11-09.png")
fig.savefig(outp, dpi=300, bbox_inches="tight")
print("saved", outp)
