# -*- coding: utf-8 -*-
# stage4b_出图.py —— fig11 波形级前向预言（边界实证）
import os, sys, json, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import importlib.util
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("s4b", os.path.join(HERE, "stage4b_波形级前向预言.py"))
s4b = importlib.util.module_from_spec(spec); spec.loader.exec_module(s4b)

rows = [json.loads(l) for l in open(os.path.join(HERE, "stage4b_ckpt.jsonl"), encoding="utf-8")]
ok = pd.DataFrame(rows).query("ok==True").copy()
ok["tau_ratio_ctrl"] = ok.tau_sim_ctrl / ok.tau_meas_ctrl
s3 = pd.read_csv(os.path.join(HERE, "stage3_onset_metrics.csv"))
ok = ok.merge(s3[["cell", "tail_slope"]], on="cell", how="left")

LAB_C = {"lab_3": "#2868B7", "lab_4": "#F39800", "lab_5": "#7B3FF2"}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) 例证细胞：lab_3 cisapride 对照迹线（模型抓住斜坡复现峰，漏慢去活分量）
rec = ok[ok.lab == "lab_3"].iloc[0]
zp = s4b.find_zip(rec.cell)
z = zipfile.ZipFile(zp)
with z.open(z.namelist()[0]) as f:
    d = pd.read_csv(f)
t = d["t_ms"].to_numpy(float)
cur = {int(c.split("#")[1].split("_")[0]): c for c in d.columns if c.startswith("trace_#") and "_current_" in c}
vol = {int(c.split("#")[1].split("_")[0]): c for c in d.columns if c.startswith("trace_#") and c.endswith("_voltage_mV")}
tr = sorted(set(cur) & set(vol))[2]
V = d[vol[tr]].to_numpy(float); y = d[cur[tr]].to_numpy(float)
tabs = s4b.build_tabs(36.5)
m, h = s4b.forward_dt(V, tabs, float(np.median(np.diff(t))) / 1000)
G = np.sum((m * h * (V - s4b.E_REV)) * y) / np.sum((m * h * (V - s4b.E_REV)) ** 2)
ax = axes[0]
ax.plot(t, y, "r-", lw=0.9, label="实测（lab_3 对照段）")
ax.plot(t, G * m * h * (V - s4b.E_REV), "b-", lw=0.9, alpha=0.8, label="α 模型前向仿真")
ax2 = ax.twinx(); ax2.plot(t, V, "k-", lw=0.5, alpha=0.35); ax2.set_ylabel("V (mV)", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")
ax.set_xlabel("ms"); ax.set_ylabel("pA")
ax.set_title("(a) 斜坡复现峰被抓住，去活慢尾巴缺分量\n（ICH S7B 阶跃+斜坡协议，lab_3 例证细胞）", fontweight="bold", fontsize=10.5)
ax.legend(fontsize=9, loc="upper right")

# (b) 对照段全迹线 RMS 残差（稳健形状保真度，n 充足）
ax = axes[1]
data = [ok[ok.lab == l]["rms_ctrl"].replace([np.inf, -np.inf], np.nan).dropna() for l in ("lab_3", "lab_4", "lab_5")]
pos = [1, 2, 3]
bp = ax.boxplot([d.values for d in data], positions=pos, widths=0.55,
                patch_artist=True, showfliers=False)
for patch, lab in zip(bp["boxes"], ("lab_3", "lab_4", "lab_5")):
    patch.set_facecolor(LAB_C[lab]); patch.set_alpha(0.75)
for i, (dd, lab) in enumerate(zip(data, ("lab_3", "lab_4", "lab_5"))):
    ax.scatter(np.full(len(dd), pos[i]) + np.random.uniform(-0.12, 0.12, len(dd)), dd,
               s=6, color=LAB_C[lab], alpha=0.25, edgecolor="none")
ax.set_xticks(pos); ax.set_xticklabels([f"lab_3\n(n={len(data[0])})", f"lab_4\n(n={len(data[1])})", f"lab_5\n(n={len(data[2])})"])
ax.set_ylabel("对照段归一化 RMS 残差（仿真 vs 实测）")
ax.set_title("(b) 形状保真度：RMS 残差 0.57–0.75\n缺口 = 起始尖峰（伪迹）+ 慢去活分量（已登记边界）", fontweight="bold", fontsize=10.5)

# (c) 药物段幅度比 vs tail_slope（Stage3 联动，阴性）
ax = axes[2]
mm = ok[np.isfinite(ok.A_ratio_drug) & np.isfinite(ok.tail_slope)]
for lab, g in mm.groupby("lab"):
    ax.scatter(g.tail_slope, g.A_ratio_drug, s=9, alpha=0.4, color=LAB_C.get(lab, "gray"), label=lab)
r = spearmanr(mm.A_ratio_drug, mm.tail_slope)
ax.axhline(1.0, color="k", lw=1, ls="--")
ax.set_xlabel("tail_slope（frac/sweep）"); ax.set_ylabel("药物段幅度比（仿真 / 实测）")
p_txt = f"{r.pvalue:.2f}" if r.pvalue >= 0.01 else f"{r.pvalue:.0e}"
ax.set_title(f"(c) Stage3 联动：ρ={r.statistic:+.2f}（p={p_txt}）\n幅度比被未建模的起始尖峰压低——如实登记", fontweight="bold", fontsize=10.5)
ax.legend(fontsize=8.5)

fig.suptitle("Stage4b 波形级前向预言（971 细胞，实测 V(t)）：快分量结构复现协议特征，慢分量缺席边界显形",
             fontsize=12.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = os.path.join(HERE, "fig11_波形级前向预言_边界实证.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("fig11 已生成")
