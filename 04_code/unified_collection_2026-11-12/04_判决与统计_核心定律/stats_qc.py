# -*- coding: utf-8 -*-
"""
QC 分层判决（预注册逻辑）：
  若"V½松/τ紧"是 active 选择假象 → bad/silent 组比值应崩（<80% 组 ratio>1）
  若为真实生物规律 → bad/silent 中比值仍应 >1 为主（绝对散布可变大）
方法: 与 v2/v3.1 完全同口径（tau_act only, r2>=0.7, 窗[0,40], Vhalf/k r2>=0.8）
分组: channel × temp × flag, >=15 细胞
产出: features/verdict_qc.json, qc_split.csv
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
F = ROOT / "features"
LN10 = np.log(10)
MIN_CELLS = 15

df = pd.read_csv(F / "features_corpus_v34.csv")
# 只留 Kv（定律的原始声明范围）
df = df[df.channel.str.startswith("Kv")]

def logsd(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < MIN_CELLS: return np.nan
    return float(np.std(np.log10(v), ddof=1))

rows = []
for (ch, T, flag), g in df.groupby(["channel", "temp_nominal", "flag"]):
    if g.cell_id.nunique() < MIN_CELLS: continue
    ta = g[(g.parameter == "tau_act") & (g.voltage_mV.between(0, 40)) & (g.value > 0) & (g.r2 >= 0.7)]
    sds = [logsd(gv.value) for V, gv in ta.groupby("voltage_mV")]
    sds = [s for s in sds if np.isfinite(s)]
    tau_sd = float(np.median(sds)) if sds else np.nan
    vh = g[(g.parameter == "Vhalf_act") & (g.r2 >= 0.8)].value
    kk = g[(g.parameter == "k_act") & (g.r2 >= 0.8)].value
    mid = float(vh.std() / kk.median()) if len(vh) >= MIN_CELLS and len(kk) >= MIN_CELLS else np.nan
    if np.isfinite(tau_sd) and np.isfinite(mid):
        rows.append(dict(channel=ch, temp=T, flag=flag, n_cells=g.cell_id.nunique(),
                         tau_sd=tau_sd, mid_disp=mid, ratio=mid / tau_sd))
G = pd.DataFrame(rows)
G.to_csv(F / "qc_split.csv", index=False)

# 汇总：每个 flag 下 ratio>1 的组占比、中位比、绝对散布对比
summ = {}
for flag, g in G.groupby("flag"):
    summ[flag] = dict(n_groups=len(g),
                      frac_gt1=float((g.ratio > 1).mean()),
                      median_ratio=float(g.ratio.median()),
                      tau_sd_median=float(g.tau_sd.median()),
                      mid_disp_median=float(g.mid_disp.median()))
# 同通道同温度的 active vs 其他 flag 配对差
pairs = []
for (ch, T), g in G.groupby(["channel", "temp"]):
    if "active" not in set(g.flag): continue
    a = g[g.flag == "active"].iloc[0]
    for flag in ["silent", "bad", "excl"]:
        o = g[g.flag == flag]
        if len(o):
            pairs.append(dict(channel=ch, temp=T, flag=flag,
                              ratio_active=a.ratio, ratio_other=o.iloc[0].ratio,
                              tau_sd_fold=o.iloc[0].tau_sd / a.tau_sd,
                              mid_fold=o.iloc[0].mid_disp / a.mid_disp))
P = pd.DataFrame(pairs)
P.to_csv(F / "qc_pairs.csv", index=False)

verdict = {"by_flag": summ,
           "paired": {"n": len(P),
                      "ratio_other_gt1_frac": float((P.ratio_other > 1).mean()) if len(P) else None,
                      "tau_sd_fold_median": float(P.tau_sd_fold.median()) if len(P) else None,
                      "mid_fold_median": float(P.mid_fold.median()) if len(P) else None}}
json.dump(verdict, open(F / "verdict_qc.json", "w"), indent=1)
print(json.dumps(verdict, ensure_ascii=False, indent=1))
