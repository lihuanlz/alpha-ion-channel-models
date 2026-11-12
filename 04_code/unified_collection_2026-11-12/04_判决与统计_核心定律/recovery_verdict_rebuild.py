# -*- coding: utf-8 -*-
"""
recovery_verdict_rebuild.py — τ_rec 判决正式化重建（零记忆审计产物 2026-11-08）
口径（由存档表反推确证）：
  tau_rec_sd: tau_rec 行, value>0, r2>=0.9, log10 跨细胞 SD (ddof=1)
  mid_obs:    全体该组细胞 Vhalf_act r2>=0.8 且 V½∈[-80,+60] mV / median(k_act r2>=0.8)
  ratio = mid_obs / tau_rec_sd;  MIN_CELLS=15
产出: features/recovery_verdict_v5_rebuilt.csv
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
rv = pd.read_csv(ROOT / "features" / "features_recovery_v5.csv")
ft = pd.read_csv(ROOT / "features" / "features_corpus_v5.csv", low_memory=False)
MIN_CELLS = 15
VWIN = (-80, 60)

rows = []
for (ch, T), g in rv.groupby(["channel", "temp_nominal"]):
    tr = g[(g.parameter == "tau_rec") & (g.value > 0) & (g.r2 >= 0.9)]
    n = tr.cell_id.nunique()
    if n < MIN_CELLS:
        continue
    tau_rec_sd = float(np.log10(tr.value.astype(float)).std(ddof=1))
    gg = ft[(ft.channel == ch) & (ft.temp_nominal == T)]
    vh = gg[(gg.parameter == "Vhalf_act") & (gg.r2 >= 0.8) &
            (gg.value >= VWIN[0]) & (gg.value <= VWIN[1])].value
    kk = gg[(gg.parameter == "k_act") & (gg.r2 >= 0.8)].value
    if len(vh) < MIN_CELLS or len(kk) < MIN_CELLS:
        continue
    mid_obs = float(vh.std() / kk.median())
    rows.append(dict(channel=ch, temp=T, n=n, tau_rec_sd=tau_rec_sd,
                     mid_obs=mid_obs, ratio=mid_obs / tau_rec_sd))

vd = pd.DataFrame(rows)
vd.to_csv(ROOT / "features" / "recovery_verdict_v5_rebuilt.csv", index=False)
print(f"groups: {len(vd)}, ratio>1: {(vd.ratio > 1).mean():.3f}, median ratio: {vd.ratio.median():.2f}")

# 与存档表对拍
old = pd.read_csv(ROOT / "features" / "recovery_verdict_v5.csv")
m = old.merge(vd, on=["channel", "temp"], suffixes=("_old", "_new"), how="outer", indicator=True)
print("\n行对齐:", m._merge.value_counts().to_dict())
m["dratio"] = (m.ratio_new - m.ratio_old).abs()
print("ratio 最大偏差:", m.dratio.max())
bad = m[(m.ratio_old > 1) != (m.ratio_new > 1)]
print("判决翻转行数:", len(bad.dropna(subset=['ratio_old','ratio_new'])))
print(m[["channel","temp","ratio_old","ratio_new","dratio"]].round(3).sort_values("dratio", ascending=False).head(8).to_string(index=False))
