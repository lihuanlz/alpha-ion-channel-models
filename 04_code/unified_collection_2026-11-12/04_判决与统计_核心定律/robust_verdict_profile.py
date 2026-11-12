# -*- coding: utf-8 -*-
"""判决第 D 口径：轮廓似然全库重验版。
对 64 纯 Kv 组, 用 profile_verdict_rows.csv 的逐行轮廓 SE 重算:
  tau 侧: 逐电压 log10 EB, s = se_prof/(tau*ln10), 组内中位
  V½ 侧: 窗内 [-80,60] EB(V½, se_prof)/k_med
  ratio_D = mid_prof / tau_eb_prof
并与 A(官方)/B(过滤)/C(EB+pcov) 同表对照。
剖面失败的行回退 pcov SE（计数报告）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

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
    return max(minimize_scalar(nll, bounds=(0.0, hi_b), method="bounded").x, 0.0)

ck = pd.read_csv(R / "features" / "profile_verdict_rows.csv")
c = pd.read_csv(R / "features" / "features_corpus_v5r.csv",
                usecols=["cell_id", "temp_nominal", "parameter", "value", "r2",
                         "channel", "flag"])
pp = pd.read_csv(R / "features" / "per_process_v5r.csv")
groups = pp.dropna(subset=["mid_obs", "tau_obs"])[["channel", "temp"]]
print("判决组数:", len(groups))

# 轮廓等价 SE（失败回退 pcov）
ck["se_prof"] = (ck.prof_hi - ck.prof_lo) / (2 * 1.96)
bad = ck.se_prof.isna() | (ck.se_prof <= 0)
ck.loc[bad, "se_prof"] = ck.loc[bad, "se_pcov"]
print(f"剖面失败回退 pcov: {bad.sum()} 行 ({100*bad.mean():.2f}%)")

rows = []
for _, g in groups.iterrows():
    ch, temp = g.channel, g.temp
    # tau 侧
    t = ck[(ck.channel == ch) & (ck.temp == temp) & (ck.parameter == "tau_act")]
    t = t[(t.value > 0) & (t.voltage_mV >= 0) & (t.voltage_mV <= 40)]
    # V½ 侧（窗内）
    h = ck[(ck.channel == ch) & (ck.temp == temp) & (ck.parameter == "Vhalf_act")]
    hw = h[(h.value >= -80) & (h.value <= 60)]
    # k_med 沿用官方语料口径
    k = c[(c.channel == ch) & (c.temp_nominal == temp) & (c.flag == "active")
          & (c.parameter == "k_act") & (c.r2 >= 0.8)].value.median()
    if len(hw) < 15 or np.isnan(k) or len(t) == 0:
        continue
    tau_obs = t.groupby("voltage_mV").value.apply(lambda x: np.log10(x).std(ddof=1)).median()
    ebv = []
    for V, gv in t.groupby("voltage_mV"):
        if len(gv) < 15: continue
        y = np.log10(gv.value.values)
        s = np.clip(gv.se_prof.values / (gv.value.values * LN10), 1e-4, None)
        ebv.append(eb_sigma(y, s))
    tau_eb = float(np.median(ebv)) if ebv else np.nan
    mid_prof = eb_sigma(hw.value.values, np.clip(hw.se_prof.values, 1e-3, None)) / k
    rows.append(dict(channel=ch, temp=temp, n_tau=len(t), n_vhalf_in=len(hw),
                     tau_obs=tau_obs, tau_eb_prof=tau_eb,
                     mid_prof=mid_prof,
                     ratio_D=mid_prof / tau_eb if tau_eb else np.nan))

df = pd.DataFrame(rows)
# 并入 A/B/C
old = pd.read_csv(R / "features" / "robust_verdict_66groups_2026-11-08.csv")
# 旧表是 66 组（含 2 个 TASK3 杂质），按 channel+temp 左连
mg = df.merge(old[["channel", "temp", "ratio_A", "ratio_B", "ratio_C"]],
              on=["channel", "temp"], how="left")
out = R / "features" / "robust_verdict_profile_2026-11-08.csv"
mg.to_csv(out, index=False)

def summ(col):
    v = mg[col].dropna()
    return f"{(v > 1).mean() * 100:.1f}% ({(v > 1).sum()}/{len(v)}), 中位 {v.median():.3f}x"
print("\n=== 四口径对照（纯 Kv 64 组）===")
print("A 官方:          ", summ("ratio_A"))
print("B 过滤撞界:      ", summ("ratio_B"))
print("C EB+pcov:       ", summ("ratio_C"))
print("D EB+轮廓似然全库:", summ("ratio_D"))
sub1 = mg[(mg.ratio_C > 1) != (mg.ratio_D > 1)]
print("\nC↔D 判决翻转组数:", len(sub1))
if len(sub1): print(sub1.to_string())
d = (mg.ratio_D / mg.ratio_C - 1).dropna()
print(f"\nD/C 相对差: 中位 {100*d.median():+.2f}%  IQR [{100*d.quantile(.25):+.2f}%, {100*d.quantile(.75):+.2f}%]")
print("\nsaved", out)
