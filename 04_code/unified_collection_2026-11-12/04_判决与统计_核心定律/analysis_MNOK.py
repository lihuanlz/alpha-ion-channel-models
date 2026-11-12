# -*- coding: utf-8 -*-
# M/N/O/K 收官补弹（全本地）
# M: Boltzmann 斜率 k 的散布——"陡峭度是不是也紧"（若紧：三量分解完整：τ紧、k紧、只有V½松）
# N: 全过程 τ 散布（tau_act/deact/inact 分开算，不只激活）
# O: hERG 跨实验室对拍（Beattie 37°C 手工 vs Lei 25°C 自动，仓库内已有）
# K: 全族有效门控电荷地图 z = RT/(k·F)
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "features"
df = pd.read_csv(OUT / "features_all.csv")

# ---------- M+N: 三量分解（每 通道×名义温度） ----------
MINC = 15
rows = []
for (ch, T), g in df.groupby(["channel", "temp_nominal"]):
    if g.cell_id.nunique() < MINC: continue
    r = dict(channel=ch, temp=T, n_cells=g.cell_id.nunique())
    # k 散布
    k = g[(g.parameter == "k_act") & (g.r2 >= 0.8)]["value"]
    if len(k) >= MINC:
        r["k_CV"] = float(k.std() / k.mean())
        r["z_eff"] = float(8.314 * 298.15 / (k.median() * 1e-3 * 96485))  # k in mV
    # V½ 散布
    vh = g[(g.parameter == "Vhalf_act") & (g.r2 >= 0.8)]["value"]
    if len(vh) >= MINC: r["Vhalf_SD"] = float(vh.std())
    # 三过程 τ 散布
    for par in ["tau_act", "tau_deact", "tau_inact"]:
        t = g[(g.parameter == par) & (g.value > 0) & (g.r2 >= 0.7)]
        lsd = t.groupby("voltage_mV")["value"].apply(
            lambda s: np.log10(s).std() if len(s) >= MINC else np.nan).dropna()
        if len(lsd): r[f"logSD_{par}"] = float(lsd.median())
    rows.append(r)
mn = pd.DataFrame(rows)
mn.to_csv(OUT / "MN_decomposition.csv", index=False)

# ---------- O: hERG 跨实验室对拍（Beattie vs Lei） ----------
MDIR = Path(r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\_github\alpha-ion-channel-models\03_data_csv")
m = pd.read_csv(MDIR / "per_cell_master_table_four_channels_2026-09-21.csv")
h = m[(m.channel == "hERG") & (m.ok == 1)].copy()
print("hERG dataset 清单:", h.dataset.unique())
# 按 dataset 分实验室: Beattie* vs Lei*
o_rows = []
for par in ["tau_deact_slow", "tau_rec", "tau_inact", "tau_act"]:
    for V, gv in h[h.parameter == par].groupby("voltage_mV"):
        med = g2 = None
        labs = {}
        for ds, gg in h[h.parameter == par].groupby("dataset"):
            pass
    # 简化: 直接按 dataset 分组统计各参数
for ds, gg in h.groupby("dataset"):
    for par in ["tau_deact_slow", "tau_rec", "tau_inact"]:
        t = gg[(gg.parameter == par) & (gg.value > 0)]
        if t.cell_id.nunique() >= 8:
            o_rows.append(dict(dataset=ds, parameter=par, n_cells=t.cell_id.nunique(),
                               tau_median=float(t.value.median()),
                               logSD=float(np.log10(t.value).std())))
o = pd.DataFrame(o_rows)
o.to_csv(OUT / "O_herg_datasets.csv", index=False)
print(o.to_string(index=False))

# ---------- 汇总判读 ----------
ok = mn.dropna(subset=["k_CV", "Vhalf_SD", "logSD_tau_act"])
summary = dict(
    n_groups=len(mn),
    k_CV_median=float(mn.k_CV.median()) if "k_CV" in mn else None,
    Vhalf_SD_median_mV=float(mn.Vhalf_SD.median()) if "Vhalf_SD" in mn else None,
    tau_logSD=dict(
        act=float(mn.logSD_tau_act.median()) if mn.logSD_tau_act.notna().any() else None,
        deact=float(mn.logSD_tau_deact.median()) if "logSD_tau_deact" in mn and mn.logSD_tau_deact.notna().any() else None,
        inact=float(mn.logSD_tau_inact.median()) if "logSD_tau_inact" in mn and mn.logSD_tau_inact.notna().any() else None),
    z_eff_range=[float(mn.z_eff.min()), float(mn.z_eff.max())] if "z_eff" in mn else None,
)
# 判读：k 是否紧（k_CV 与 Vhalf 散布的相对大小——把 Vhalf_SD 换成以 k 为单位的相对散布已在主分析）
mn["rel_mid"] = mn.Vhalf_SD / (mn.Vhalf_SD / 1)  # placeholder
with open(OUT / "MNOK_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
