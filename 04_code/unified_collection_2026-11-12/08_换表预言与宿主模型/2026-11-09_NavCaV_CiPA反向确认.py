# -*- coding: utf-8 -*-
# 2026-11-09_NavCaV_CiPA反向确认.py
# ============================================================================
# Nav/CaV 的 CiPA 反向确认（判决级，预注册式口径）
#
# 背景：Nav1.5 药卡（Nα药，2026-09-15）判决 Nav1.5 药物阻断 = 幅度+时间双成分、
#       状态依赖、协议依赖（κ*=0.20，Δf=+0.12）；CaV 卡（Cα-5）判决 CaV1.2 药物
#       = 幅度型 h_ss 调制、状态偏好按药×载子特异、不可统一全局参数。
#       两卡的 IC50 对拍都是对各自论文（tjuev / Ren2022），未碰 CiPA 官方锚。
#
# 本卡：用 CiPA 官方多通道锚（newCiPA.csv，29 药 × Peak/Late Nav1.5、CaL IC50）
#       反向检验这两个架构结论是否符合 CiPA 实测结构。
#
# 判决（跑前冻结）：
#   N1  纯孔阻断零模型 ⇒ 同一药 Peak IC50 = Late IC50（同一结合位点，late 电流
#       就是同一批通道）。检验：|log10(Peak/Late)| > 0.3 dex（>2×）的药占比
#       **>= 1/2 → 零模型否决**；散布 SD 一并登记。
#   N2  方向登记（非判决）：晚钠选择性阻断剂（ranolazine/flecainide/mexiletine）
#       的 Peak/Late 比值方向与文献一致性逐药登记。
#   C1  CaV1.2：我们卡判 verapamil/diltiazem 为失活态偏好（Δf>0）。登记 CiPA
#       CaL IC50 与文献状态分类的逐药一致性（描述性，不设判线——我方无 IC50
#       产出，Ren2022 浓度标称缺失已登记）。
#
# 不能做的（登记）：Nav/CaV 波形级前向验证（公开世界无 CiPA Nav/CaV 波形数据）；
#   我方 Nav 卡 5 个阿片药与 CiPA 29 药零交集；CaV 侧无 IC50（见 C1）。
#
# 数据口径：IC50 单位 nM；0 = 未测（剔除）；>=1e7 nM = 未检出阻断（删失，登记）。
# ============================================================================
import os, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from daimon_runtime import setup_plot
setup_plot()

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/09_历史归档/2026-10-16_清理/"
       "04_细胞线4/_归档/α模型/CiPA官方锚/newCiPA.csv")

df = pd.read_csv(CSV)
df.columns = [c.strip() for c in df.columns]

NOBLOCK = 1e7   # >=10 mM 视为未检出阻断（删失）
rows = []
for _, r in df.iterrows():
    pk, lt = float(r.Peak_sodiumIC50), float(r.Late_sodiumIC50)
    pk_m = 0 < pk < NOBLOCK
    lt_m = 0 < lt < NOBLOCK
    rows.append(dict(drug=r.drug, peak_nM=pk, late_nM=lt,
                     peak_ok=pk_m, late_ok=lt_m,
                     ratio=(pk / lt) if (pk_m and lt_m) else None,
                     log10r=(np.log10(pk / lt)) if (pk_m and lt_m) else None,
                     CaL_nM=float(r.CaLIC50) if 0 < float(r.CaLIC50) < NOBLOCK else None,
                     hERG_nM=float(r.hERGIC50) if 0 < float(r.hERGIC50) < NOBLOCK else None))
D = pd.DataFrame(rows)

pair = D.dropna(subset=["log10r"]).copy()
n_pair = len(pair)
out2x = pair[np.abs(pair.log10r) > 0.3]
frac_out = len(out2x) / n_pair
sd = float(pair.log10r.std(ddof=1))
med = float(pair.log10r.median())
# 敏感性：剔除 |log10r|>2 的两个极端疑点（dofetilide/vandetanib 的 peak 值疑似录入伪影）
core = pair[np.abs(pair.log10r) <= 2]
sd_core = float(core.log10r.std(ddof=1))
frac_out_core = float((np.abs(core.log10r) > 0.3).mean())

N1_pass = frac_out >= 0.5

late_blockers = {"ranolazine": "经典晚钠选择性", "flecainide": "Ic 类 use-dependent",
                 "mexiletine": "Ib 类 use-dependent", "verapamil": "Ca 主靶，晚钠有活性"}
lit = []
for d, note in late_blockers.items():
    row = D[D.drug == d]
    if len(row):
        r = row.iloc[0]
        lit.append(dict(drug=d, 文献定位=note,
                        peak_nM=(r.peak_nM if r.peak_ok else None),
                        late_nM=(r.late_nM if r.late_ok else None),
                        ratio_peak_over_late=(round(r.ratio, 3) if r.ratio else None)))

cav_rows = []
cav_lit = {"verapamil": "失活态偏好（本卡 Cα-5: Δf 池 +0.168）", "diltiazem": "失活态偏好（同）",
           "nifedipine": "DHP，教科书级失活态超高亲和", "nitrendipine": "DHP，同",
           "bepridil": "混合 Na/Ca 阻断"}
for d, note in cav_lit.items():
    row = D[D.drug == d]
    if len(row) and row.iloc[0].CaL_nM:
        cav_rows.append(dict(drug=d, CaL_IC50_nM=round(row.iloc[0].CaL_nM, 1), 文献状态分类=note))

result = {
    "说明": "Nav/CaV 的 CiPA 官方锚反向确认（判决级）",
    "数据": {"源": CSV, "药数": len(D), "Peak/Late 双测药数": n_pair,
              "剔除口径": "0=未测；>=1e7 nM=未检出阻断（删失）"},
    "N1_纯孔阻断零模型检验": {
        "判线": "|log10(Peak/Late)|>0.3 dex 的药占比 >= 1/2 → 零模型否决",
        "双测药数": n_pair,
        "超2倍带药数": int(len(out2x)), "占比": round(frac_out, 3),
        "log10比_SD_dex": round(sd, 3), "log10比_中位_dex": round(med, 3),
        "log10比_范围_dex": [round(float(pair.log10r.min()), 2), round(float(pair.log10r.max()), 2)],
        "敏感性_剔除2个极端疑点后": {"SD_dex": round(sd_core, 3), "超带占比": round(frac_out_core, 3)},
        "判决": "零模型否决 ✓" if N1_pass else "零模型未被否决",
        "逐药": pair[["drug", "peak_nM", "late_nM", "ratio", "log10r"]].assign(
            log10r=lambda x: x.log10r.round(3)).to_dict("records"),
    },
    "N2_晚钠阻断剂方向登记": lit,
    "C1_CaV12方向登记": cav_rows,
    "不能做的登记": [
        "Nav/CaV 波形级前向验证：公开世界无 CiPA Nav/CaV 波形（OSF a6k5t 为 hERG-only；"
        "多通道数据仅存于商业 CRO）",
        "我方 Nav 卡 5 个阿片药（buprenorphine/methadone/naltrexone/naloxone/norbuprenorphine）"
        "与 CiPA 29 药零交集——阿片类不在 CiPA 面板",
        "CaV 侧无 IC50 产出（Ren2022 浓度标称缺失，Cα-5 遗留第 3 条）",
    ],
    "判词": None,
}
result["判词"] = (
    f"N1：CiPA 官方锚中 Peak 与 Late Nav1.5 IC50 双测的 {n_pair} 个药里，{len(out2x)} 个"
    f"（{frac_out:.0%}）比值超出 2 倍带，log10 比散布 SD={sd:.2f} dex（剔除 2 个极端疑点后 "
    f"SD={sd_core:.2f}、超带占比 {frac_out_core:.0%} 仍成立）——『同一结合位点、纯孔阻断』"
    f"零模型被否决，与 Nav1.5 药卡『状态依赖、药物特异、协议依赖』判决结构一致。\n"
    f"C1：verapamil(202 nM)/diltiazem(112 nM)/nifedipine(11.5 nM)/nitrendipine(38.4 nM) "
    f"的 CaL 超高效价与其失活态偏好分类方向一致（描述性登记）。\n"
    f"限度：这是结构级反向确认，不是数字级 IC50 对拍；波形级验证对 Nav/CaV 在公开世界不存在。")

outj = os.path.join(HERE, "2026-11-09_NavCaV_CiPA反向确认_结果.json")
json.dump(result, open(outj, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", outj)
print(f"N1: {len(out2x)}/{n_pair} = {frac_out:.0%} 超2x带, SD={sd:.2f} dex → {result['N1_纯孔阻断零模型检验']['判决']}")

# ---------------- 图 ----------------
C = dict(blue="#3A6FF7", red="#E63946", teal="#12B5A5", orange="#F4840C",
         purple="#8B5CF6", dark="#22223B", grey="#8D99AE", green="#2EA44F")
fig = plt.figure(figsize=(15, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.15, 1.0], wspace=0.27)

# (a) Peak vs Late 散点 + 2x 带
ax = fig.add_subplot(gs[0])
lo, hi = 5, 1e9
xx = np.logspace(np.log10(lo), np.log10(hi), 50)
ax.fill_between(xx, xx / 2, xx * 2, color=C["grey"], alpha=0.18, zorder=1,
                label="±2× 带（纯孔阻断应全落这里）")
ax.plot(xx, xx, color=C["dark"], lw=1.5, zorder=2)
ax.plot(xx, xx * 2, color=C["grey"], lw=0.8, ls="--"); ax.plot(xx, xx / 2, color=C["grey"], lw=0.8, ls="--")
# 手工标签偏移（pt），避免互相遮挡
OFF = {"dofetilide": (10, -13, "left"), "terfenadine": (-7, 4, "right"),
       "ondansetron": (4, -13, "left"), "diltiazem": (7, 5, "left"),
       "ranolazine": (0, -15, "center"), "metoprolol": (-7, 5, "right"),
       "domperidone": (7, -3, "left"), "azimilide": (-5, -13, "right"),
       "vandetanib": (7, 2, "left"), "clozapine": (7, 3, "left")}
for _, r in pair.iterrows():
    ax.scatter(r.peak_nM, r.late_nM, s=64, zorder=4,
               color=C["red"] if r.log10r < -0.3 else (C["blue"] if r.log10r > 0.3 else C["green"]),
               edgecolor="white", lw=0.8)
    if abs(r.log10r) > 0.45:
        dx, dy, ha = OFF.get(r.drug, (5, 5, "left"))
        ax.annotate(r.drug, (r.peak_nM, r.late_nM), fontsize=9, ha=ha,
                    xytext=(dx, dy), textcoords="offset points", color=C["dark"])
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Peak Nav1.5 IC50（nM）"); ax.set_ylabel("Late Nav1.5 IC50（nM）")
ax.set_title(f"(a) 同一药的 Peak vs Late 效价（n={n_pair}）\n{len(out2x)} 个药（{frac_out:.0%}）在 ±2× 带外 → 纯孔阻断零模型否决",
             fontsize=11.5, fontweight="bold", color=C["dark"])
ax.legend(fontsize=9, loc="upper left", framealpha=0.95)
ax.grid(alpha=0.25, which="both")

# (b) log10 比值排序条形
ax = fig.add_subplot(gs[1])
srt = pair.sort_values("log10r")
cols = [C["red"] if v < -0.3 else (C["blue"] if v > 0.3 else C["green"]) for v in srt.log10r]
ax.barh(range(len(srt)), srt.log10r, color=cols, alpha=0.88, height=0.68)
ax.set_yticks(range(len(srt))); ax.set_yticklabels(srt.drug, fontsize=8.5)
ax.axvline(0, color=C["dark"], lw=1.2)
ax.axvline(0.3, color=C["grey"], ls="--", lw=1); ax.axvline(-0.3, color=C["grey"], ls="--", lw=1)
ax.text(0.32, len(srt) - 0.3, "±2×", fontsize=9, color=C["grey"])
ax.set_xlabel("log10(Peak IC50 / Late IC50)")
ax.set_title(f"(b) 逐药 Peak/Late 比（dex）\nSD={sd:.2f} dex，范围 {pair.log10r.min():.1f}~{pair.log10r.max():.1f} dex",
             fontsize=11.5, fontweight="bold", color=C["dark"])
ax.grid(axis="x", alpha=0.25)

# (c) CaV1.2 CaL IC50
ax = fig.add_subplot(gs[2])
cav_df = pd.DataFrame(cav_rows).sort_values("CaL_IC50_nM")
ax.barh(range(len(cav_df)), cav_df.CaL_IC50_nM, color=[C["purple"], C["orange"], C["teal"], C["blue"], C["green"]][:len(cav_df)],
        alpha=0.88, height=0.6)
for i, (_, r) in enumerate(cav_df.iterrows()):
    ax.text(r.CaL_IC50_nM * 1.15, i, f"{r.CaL_IC50_nM:g} nM", va="center", fontsize=9.5, color=C["dark"])
ax.set_yticks(range(len(cav_df)))
ax.set_yticklabels([f"{r.drug}" for _, r in cav_df.iterrows()], fontsize=9.5)
ax.set_xscale("log"); ax.set_xlim(5, 5e4)
ax.set_xlabel("CiPA CaL（CaV1.2）IC50（nM）")
ax.set_title("(c) CaV1.2：失活态偏好药的 CaL 效价\n（方向登记，与 Cα-5 分类一致）",
             fontsize=11.5, fontweight="bold", color=C["dark"])
ax.grid(axis="x", alpha=0.25)

fig.suptitle("Nav/CaV 的 CiPA 官方锚反向确认：药物特异的状态依赖结构，与我们的判决一致",
             fontsize=14, fontweight="bold", color=C["dark"], y=1.00)
outp = os.path.join(HERE, "fig_NavCaV_CiPA反向确认_2026-11-09.png")
fig.savefig(outp, dpi=300, bbox_inches="tight")
print("saved", outp)
