# -*- coding: utf-8 -*-
"""稳健性重判：V½ 撞界过滤 + 分层贝叶斯 EB，对全 66 组重算判决。
对照三套口径:
  A 官方(现行): mid_obs 不过滤撞界行, SD(V½)/k_med / tau_obs
  B 过滤撞界:    V½∈[-80,+60] 后 SD/k_med / tau_obs
  C 分层贝叶斯:  EB(V½)/k_med / median_V EB(τ)   (两侧都做 EB)
tau_obs 三口径同用现行冻结值(per_process_v5.tau_act 列)。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, brentq

R = Path(__file__).parent
LN10 = np.log(10)

def eb_sigma(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    def nll(sv):
        sv = max(float(sv), 0.0)
        w = 1.0 / (sv * sv + s * s)
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * np.sum(np.log(2 * np.pi * (sv * sv + s * s)) + (y - mu) ** 2 * w)
    hi_b = max(3 * y.std(), 0.3)
    res = minimize_scalar(nll, bounds=(0.0, hi_b), method="bounded")
    return max(res.x, 0.0)

d = pd.read_csv(R / "features" / "features_corpus_v5.csv",
                usecols=["cell_id", "temp_nominal", "parameter", "voltage_mV",
                         "value", "r2", "se", "channel", "flag"])
pp = pd.read_csv(R / "features" / "per_process_v5.csv")
groups = pp.dropna(subset=["mid_obs", "tau_act"])[["channel", "temp"]]
print("判决组数:", len(groups))

rows = []
for _, g in groups.iterrows():
    ch, temp = g.channel, g.temp
    sub = d[(d.channel == ch) & (d.temp_nominal == temp) & (d.flag == "active")]
    h = sub[(sub.parameter == "Vhalf_act") & (sub.r2 >= 0.8)]
    k = sub[(sub.parameter == "k_act") & (sub.r2 >= 0.8)].value.median()
    t = sub[(sub.parameter == "tau_act") & (sub.r2 >= 0.7) & (sub.value > 0)
            & (sub.voltage_mV >= 0) & (sub.voltage_mV <= 40)]
    if len(h) < 15 or np.isnan(k) or len(t) == 0:
        continue
    # --- tau_obs (冻结口径): 逐电压 log10SD 中位 ---
    tau_by_v = t.groupby("voltage_mV").value.apply(lambda x: np.log10(x).std(ddof=1))
    tau_obs = tau_by_v.median()
    # --- tau EB: 逐电压 EB 后中位 ---
    tau_eb_vs = []
    for V, gv in t.groupby("voltage_mV"):
        if len(gv) < 15: continue
        y = np.log10(gv.value.values)
        s = np.clip(gv.se.values / (gv.value.values * LN10), 1e-4, None)
        tau_eb_vs.append(eb_sigma(y, s))
    tau_eb = float(np.median(tau_eb_vs)) if tau_eb_vs else np.nan
    # --- V½ 三口径 ---
    mid_A = h.value.std(ddof=1) / k                                   # 官方不过滤
    hw = h[(h.value >= -80) & (h.value <= 60)]
    n_bound = int(len(h) - len(hw))
    mid_B = hw.value.std(ddof=1) / k if len(hw) >= 15 else np.nan     # 过滤撞界
    if len(hw) >= 15:
        sv = np.clip(hw.se.values, 1e-3, None)
        mid_C = eb_sigma(hw.value.values, sv) / k                     # EB
    else:
        mid_C = np.nan
    rows.append(dict(channel=ch, temp=temp, n_vhalf=len(h), n_bound=n_bound,
                     k_med=k, tau_obs=tau_obs, tau_eb=tau_eb,
                     mid_A=mid_A, mid_B=mid_B, mid_C=mid_C,
                     ratio_A=mid_A / tau_obs,
                     ratio_B=mid_B / tau_obs if not np.isnan(mid_B) else np.nan,
                     ratio_C=mid_C / tau_eb if not np.isnan(mid_C) and tau_eb else np.nan))
df = pd.DataFrame(rows)
out = R / "features" / "robust_verdict_66groups_2026-11-08.csv"
df.to_csv(out, index=False)

def summ(col):
    v = df[col].dropna()
    return f"{(v > 1).mean() * 100:.1f}% 通过 ({(v > 1).sum()}/{len(v)}), 中位 {v.median():.2f}x"
print("\n=== 三口径对照 ===")
print("A 官方(含撞界行):", summ("ratio_A"))
print("B 过滤撞界±80/60:", summ("ratio_B"))
print("C 分层贝叶斯双侧:", summ("ratio_C"))
print("\n含撞界行的组数:", (df.n_bound > 0).sum(), "/", len(df),
      " 其中撞界≥5%细胞的组:", (df.n_bound / df.n_vhalf >= 0.05).sum())
flip = df[(df.ratio_A > 1) & (df.ratio_B <= 1)]
print("A过B不过的翻转组:", len(flip))
if len(flip): print(flip[["channel", "temp", "n_vhalf", "n_bound", "ratio_A", "ratio_B"]].to_string())
print("\nsaved", out)
