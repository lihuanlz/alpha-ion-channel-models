# -*- coding: utf-8 -*-
"""统一总表 QA：关键判决从统一表重算，与冻结存档值对拍"""
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
m = pd.read_csv(OUT / "master_unified_v1.csv", low_memory=False)
print("=" * 64)
print("统一总表 QA 对拍（2026-11-11 晚）")
print("=" * 64)
print(f"\n总行数 {len(m):,}；细胞数 {m['cell_id'].nunique():,}")
print(m.groupby("source_big").agg(rows=("value", "size"), cells=("cell_id", "nunique"),
                                  channels=("channel", "nunique")))
print("\n家族覆盖:", m.groupby("family")["channel"].nunique().to_dict())
print("参数覆盖:", m["parameter"].value_counts().to_dict())

print("\n--- QA1 Channelpedia τ守恒（Kv 66组判线） ---")
cp = m[(m["source_big"] == "channelpedia")]
# 组级判决：tau_act 的组内 CV vs Vhalf_act 的组内 CV（判线：τ_CV 显著小于 Vhalf_CV）
kv = cp[(cp["family"] == "Kv")]
g_tau = kv[kv["parameter"] == "tau_act"].groupby(["channel", "temp_C"])["value"].agg(["median", "std", "count"])
g_vh = kv[kv["parameter"] == "Vhalf_act"].groupby(["channel", "temp_C"])["value"].agg(["median", "std", "count"])
# 守恒比 = Vhalf 散布 / tau 散布（log10 域 std）
tau_g = kv[kv["parameter"] == "tau_act"].copy()
tau_g["lv"] = np.log10(tau_g["value"])
vh_g = kv[kv["parameter"] == "Vhalf_act"].copy()
sd_tau = tau_g.groupby(["channel", "temp_C"])["lv"].std()
sd_vh = vh_g.groupby(["channel", "temp_C"])["value"].std() / 20.0  # 粗归一：Vhalf 按 20mV 特征尺度折 dex
both = pd.concat([sd_tau.rename("sd_tau"), sd_vh.rename("sd_vh")], axis=1).dropna()
ratio = both["sd_vh"] / both["sd_tau"]
print(f"组数 {len(both)}；中位散布比 {ratio.median():.2f}×（存档：3.2×，稳健区间 2.5-2.9×）")

print("\n--- QA2 CiPA τ守恒（全药池合并中位） ---")
cells = pd.read_csv(OUT / "cipa2_cells.csv")
q = cells[cells["qc_pass"]]
tr = q["tau_ratio"].dropna()
tr = tr[(tr > 0.3) & (tr < 3)]
print(f"τ_ratio 中位 {tr.median():.3f}（冻结 0.96）；±30% 带内 {(abs(tr-1)<=0.3).mean()*100:.0f}%（存档 56%）")
print(f"QC 通过 {len(q)}/{len(cells)}")

print("\n--- QA3 Lei 温度律（tau_deact_slow -40mV 中位 vs 温度） ---")
lei = m[(m["source_big"] == "lei2019") & (m["parameter"] == "tau_deact_slow") & (m["voltage_mV"] == -40)]
g = lei.groupby("temp_C")["value"].median()
print(g.to_string())
if len(g) >= 3:
    T = g.index.to_numpy(float)
    lt = np.log10(g.to_numpy())
    sl, ic = np.polyfit(1 / (T + 273.15), lt, 1) if False else np.polyfit(T, lt, 1)
    # Q10 from slope over 10C
    q10 = 10 ** (-sl * 10)
    print(f"log10(τ) 对 T 斜率 {sl:.4f}/°C → 表观 Q10 ≈ {q10:.2f}（存档 Q10=2.84）")

print("\n--- QA4 Beattie 九细胞（Vhalf_act / tau_act / Vhalf_inact） ---")
b9 = m[m["source_big"] == "beattie9"]
for p in ["Vhalf_act", "k_act", "Vhalf_inact", "k_inact", "tau_act"]:
    s = b9[b9["parameter"] == p]["value"]
    if len(s):
        print(f"{p}: n={len(s)} 中位 {s.median():.2f} IQR [{s.quantile(.25):.2f}, {s.quantile(.75):.2f}]")

print("\n--- QA5 剖面CI覆盖率 ---")
print("prof_lo 非空率: %.3f" % m["prof_lo"].notna().mean())
print("CI 含值率: %.4f" % ((m["prof_lo"] <= m["value"]) & (m["value"] <= m["prof_hi"])).mean())
