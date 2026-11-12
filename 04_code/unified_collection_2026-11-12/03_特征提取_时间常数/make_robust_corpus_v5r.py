# -*- coding: utf-8 -*-
"""v5r 稳健层: 在冻结提取器之上补齐提取语义级旗标, 官方重出判决表。
不加改动 extract_features_v2.py(冻结), 所有旗标由已存拟合值精确导出:
  bound_hit: V½ 拟合值贴 ±250 mV 界 (|value|>=249)
  in_window: V½∈[-80, +60]  (论文判决窗)
  family:    通道标签映射 Kv/HCN/K2P/Kir/其他; 判决池=VSD 家族(Kv)
产出:
  features/features_corpus_v5r.csv     (v5 + 三列旗标)
  features/per_process_v5r.csv         (判决表官方 robust 版: mid_obs_rob/ratio_rob)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np, pandas as pd

R = Path(__file__).parent
F = R / "features"
d = pd.read_csv(F / "features_corpus_v5.csv")
print("corpus v5:", d.shape)

# ---- 家族映射 ----
def fam(ch):
    ch = str(ch)
    if ch.startswith("Kv") or ch.startswith("KCNQ") or ch.startswith("hERG"): return "Kv"
    if ch.startswith("HCN"): return "HCN"
    if ch.startswith(("TASK", "TREK", "TRAAK", "TWIK", "THIK", "TALK", "K2P")): return "K2P"
    if ch.startswith("Kir"): return "Kir"
    return "other"
d["family"] = d.channel.map(fam)

# ---- 撞界/窗口旗标 (仅对 Vhalf 类参数有意义, 其余 NaN->False) ----
is_vh = d.parameter.str.startswith("Vhalf", na=False)
d["bound_hit"] = is_vh & (d.value.abs() >= 249.0)
d["in_window"] = is_vh & (d.value >= -80) & (d.value <= 60)
print("Vhalf 行:", is_vh.sum(), " 撞界:", int(d.bound_hit.sum()),
      " 出窗(不含撞界):", int((is_vh & ~d.in_window & ~d.bound_hit).sum()))

out_c = F / "features_corpus_v5r.csv"
d.to_csv(out_c, index=False)
print("saved", out_c)

# ---- 官方判决表 robust 版 ----
MIN_C = 15
rows = []
kv = d[(d.family == "Kv") & (d.flag == "active")]
for (ch, temp), g in kv.groupby(["channel", "temp_nominal"]):
    h = g[(g.parameter == "Vhalf_act") & (g.r2 >= 0.8)]
    hw = h[h.in_window]
    k = g[(g.parameter == "k_act") & (g.r2 >= 0.8)].value.median()
    t = g[(g.parameter == "tau_act") & (g.r2 >= 0.7) & (g.value > 0)
          & (g.voltage_mV >= 0) & (g.voltage_mV <= 40)]
    if len(h) < MIN_C or np.isnan(k) or len(t) == 0:
        continue
    tau_obs = t.groupby("voltage_mV").value.apply(lambda x: np.log10(x).std(ddof=1)).median()
    mid_obs = h.value.std(ddof=1) / k                       # 现行口径(含撞界)
    mid_rob = hw.value.std(ddof=1) / k if len(hw) >= MIN_C else np.nan
    rows.append(dict(channel=ch, temp=temp, n_vhalf=len(h), n_bound=int(h.bound_hit.sum()),
                     n_inwin=len(hw), k_med=k, tau_obs=tau_obs,
                     mid_obs=mid_obs, mid_rob=mid_rob,
                     ratio_obs=mid_obs / tau_obs,
                     ratio_rob=mid_rob / tau_obs if not np.isnan(mid_rob) else np.nan))
v = pd.DataFrame(rows)
out_v = F / "per_process_v5r.csv"
v.to_csv(out_v, index=False)
ok = v.ratio_rob.dropna()
print("\n=== v5r 官方 robust 判决 (纯 Kv 池) ===")
print(f"组数 {len(v)}, ratio_rob>1: {(ok > 1).sum()}/{len(ok)} = {100 * (ok > 1).mean():.1f}%, 中位 {ok.median():.2f}x")
print("对照现行:", f"{(v.ratio_obs > 1).sum()}/{len(v)}, 中位 {v.median().median():.2f}x" if len(v) else "")
bad = v[v.ratio_rob <= 1]
print("robust 口径反例:", bad[["channel", "temp", "ratio_rob"]].to_string(index=False) if len(bad) else "无")
print("saved", out_v)
