# -*- coding: utf-8 -*-
"""四张判决表的轮廓似然全量重判 v2（2026-11-08 深夜）
数据源: features/profile_all_rows.csv（276,039 行全量剖面）
每表三列:
  obs    —— 冻结观测口径（复现存档）
  C      —— EB + pcov 对称 SE
  D      —— EB + 轮廓似然 SE（唯一变量是 SE 来源，隔离新方法的真实效应）
覆盖: kv_main(64) / tau_deact / tau_inact / hcn(5) / rec
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

R = Path(__file__).parent
LN10 = np.log(10)
MIN_CELLS = 15

def eb_sigma(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    def nll(sv):
        sv = max(float(sv), 0.0)
        w = 1.0 / (sv * sv + s * s)
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * np.sum(np.log(2 * np.pi * (sv * sv + s * s)) + (y - mu) ** 2 * w)
    hi_b = max(3 * y.std(), 0.3)
    return max(minimize_scalar(nll, bounds=(0.0, hi_b), method="bounded").x, 0.0)

ck = pd.read_csv(R / "features" / "profile_all_rows.csv", low_memory=False)
ck["se_prof"] = (ck.prof_hi - ck.prof_lo) / (2 * 1.96)
bad = ck.se_prof.isna() | (ck.se_prof <= 0)
ck.loc[bad, "se_prof"] = ck.loc[bad, "se_pcov"]
print(f"剖面失败回退 pcov: {bad.sum()} 行 ({100*bad.mean():.2f}%)\n")

_flag_src = pd.read_csv(R / "features" / "features_corpus_v5r.csv",
                        usecols=["cell_id", "flag"]).drop_duplicates("cell_id")
ck["flag"] = ck.cell_id.map(_flag_src.set_index("cell_id").flag)
cka = ck[ck.flag == "active"]

out_rows = []

def tau_side(t, tag_v=True):
    """返回 (tau_obs, tau_eb_pcov, tau_eb_prof)"""
    if tag_v:
        obs = t.groupby("voltage_mV").value.apply(lambda x: np.log10(x).std(ddof=1)).median()
        ebs_p, ebs_pr = [], []
        for V, gv in t.groupby("voltage_mV"):
            if len(gv) < MIN_CELLS: continue
            y = np.log10(gv.value.values)
            s1 = np.clip(gv.se_pcov.values / (gv.value.values * LN10), 1e-4, None)
            s2 = np.clip(gv.se_prof.values / (gv.value.values * LN10), 1e-4, None)
            ebs_p.append(eb_sigma(y, s1)); ebs_pr.append(eb_sigma(y, s2))
        f = lambda L: float(np.median(L)) if L else np.nan
        return obs, f(ebs_p), f(ebs_pr)
    else:
        y = np.log10(t.value.values)
        s1 = np.clip(t.se_pcov.values / (t.value.values * LN10), 1e-4, None)
        s2 = np.clip(t.se_prof.values / (t.value.values * LN10), 1e-4, None)
        return float(y.std(ddof=1)), eb_sigma(y, s1), eb_sigma(y, s2)

def mid_side(h, kk, window=(-80, 60)):
    """返回 (mid_obs_all 不过滤, mid_obs_win, mid_eb_pcov_win, mid_eb_prof_win)"""
    k_med = kk.value.median()
    obs_all = h.value.std(ddof=1) / k_med
    hw = h[(h.value >= window[0]) & (h.value <= window[1])] if window else h
    if len(hw) < MIN_CELLS:
        return obs_all, np.nan, np.nan, np.nan
    obs_w = hw.value.std(ddof=1) / k_med
    eb_p = eb_sigma(hw.value.values, np.clip(hw.se_pcov.values, 1e-3, None)) / k_med
    eb_pr = eb_sigma(hw.value.values, np.clip(hw.se_prof.values, 1e-3, None)) / k_med
    return obs_all, obs_w, eb_p, eb_pr

# ============ 1) Kv 主判决 64 组 ============
pp = pd.read_csv(R / "features" / "per_process_v5r.csv")
groups = pp.dropna(subset=["mid_obs", "tau_obs"])[["channel", "temp"]]
for _, g in groups.iterrows():
    ch, T = g.channel, g.temp
    t = cka[(cka.channel == ch) & (cka.temp == T) & (cka.parameter == "tau_act")
            & (cka.r2 >= 0.7) & (cka.value > 0) & (cka.voltage_mV >= 0) & (cka.voltage_mV <= 40)]
    h = cka[(cka.channel == ch) & (cka.temp == T) & (cka.parameter == "Vhalf_act") & (cka.r2 >= 0.8)]
    kk = cka[(cka.channel == ch) & (cka.temp == T) & (cka.parameter == "k_act") & (cka.r2 >= 0.8)]
    if len(h) < MIN_CELLS or len(kk) < MIN_CELLS or len(t) == 0: continue
    tau_obs, te_p, te_pr = tau_side(t)
    mid_all, _, me_p, me_pr = mid_side(h, kk)
    out_rows.append(dict(table="kv_main", channel=ch, temp=T,
                         ratio_obs=mid_all / tau_obs,
                         ratio_C=me_p / te_p, ratio_D=me_pr / te_pr))

# ============ 2) 分过程 ============
pv = pd.read_csv(R / "features" / "per_process_v5.csv")
PROC_WIN = {"tau_deact": (-60, -20), "tau_inact": (-20, 40)}
for _, g in pv.iterrows():
    ch, T = g.channel, g.temp
    h = cka[(cka.channel == ch) & (cka.temp == T) & (cka.parameter == "Vhalf_act") & (cka.r2 >= 0.8)]
    kk = cka[(cka.channel == ch) & (cka.temp == T) & (cka.parameter == "k_act") & (cka.r2 >= 0.8)]
    if len(h) < MIN_CELLS or len(kk) < MIN_CELLS: continue
    mid_all, _, me_p, me_pr = mid_side(h, kk)
    for proc, (lo, hi) in PROC_WIN.items():
        t = cka[(cka.channel == ch) & (cka.temp == T) & (cka.parameter == proc)
                & (cka.r2 >= 0.7) & (cka.value > 0)
                & (cka.voltage_mV >= lo) & (cka.voltage_mV <= hi)]
        if len(t) == 0: continue
        tau_obs, te_p, te_pr = tau_side(t)
        if not np.isfinite(tau_obs): continue
        out_rows.append(dict(table=proc, channel=ch, temp=T,
                             ratio_obs=mid_all / tau_obs,
                             ratio_C=me_p / te_p if np.isfinite(te_p) else np.nan,
                             ratio_D=me_pr / te_pr if np.isfinite(te_pr) else np.nan))

# ============ 3) HCN 5 组 ============
hcn_frz = pd.read_csv(R / "features" / "hcn_verdict_v5.csv")
hcn_recon = []
for (ch, T), g0 in hcn_frz.groupby(["channel", "temp"]):
    t = cka[(cka.channel == ch) & (cka.temp == T) & (cka.parameter == "tau_act")
            & (cka.r2 >= 0.7) & (cka.value > 0)]
    h = cka[(cka.channel == ch) & (cka.temp == T) & (cka.parameter == "Vhalf_act") & (cka.r2 >= 0.8)]
    kk = cka[(cka.channel == ch) & (cka.temp == T) & (cka.parameter == "k_act") & (cka.r2 >= 0.8)]
    if len(t) == 0 or len(h) < MIN_CELLS or len(kk) < MIN_CELLS: continue
    tau_obs, te_p, te_pr = tau_side(t)
    mid_all, _, me_p, me_pr = mid_side(h, kk, window=None)
    out_rows.append(dict(table="hcn", channel=ch, temp=T,
                         ratio_obs=mid_all / tau_obs,
                         ratio_C=me_p / te_p, ratio_D=me_pr / te_pr))
    hcn_recon.append(dict(channel=ch, temp=T, tau_obs=tau_obs, mid_obs=mid_all,
                          frz_tau=float(g0.tau_obs.iloc[0]), frz_mid=float(g0.mid_obs.iloc[0])))

# ============ 4) 恢复判决 ============
rv_ck = ck[(ck.source == "rec") & (ck.value > 0) & (ck.r2 >= 0.9)]
for (ch, T), g0 in rv_ck.groupby(["channel", "temp"]):
    if g0.cell_id.nunique() < MIN_CELLS: continue
    h = ck[(ck.channel == ch) & (ck.temp == T) & (ck.parameter == "Vhalf_act") & (ck.r2 >= 0.8)]
    kk = ck[(ck.channel == ch) & (ck.temp == T) & (ck.parameter == "k_act") & (ck.r2 >= 0.8)]
    if len(h) < MIN_CELLS or len(kk) < MIN_CELLS: continue
    tau_obs, te_p, te_pr = tau_side(g0, tag_v=False)
    mid_all, mid_w, me_p, me_pr = mid_side(h, kk)
    out_rows.append(dict(table="rec", channel=ch, temp=T,
                         ratio_obs=mid_w / tau_obs,
                         ratio_C=me_p / te_p if np.isfinite(te_p) else np.nan,
                         ratio_D=me_pr / te_pr if np.isfinite(te_pr) else np.nan))

df = pd.DataFrame(out_rows)
out = R / "features" / "robust_verdict_all_profile_2026-11-08.csv"
df.to_csv(out, index=False)

print("=== HCN 冻结口径复现校验 ===")
rc = pd.DataFrame(hcn_recon)
rc["d_tau"] = (rc.tau_obs - rc.frz_tau).abs()
rc["d_mid"] = (rc.mid_obs - rc.frz_mid).abs()
print(rc.round(4).to_string(index=False))

print("\n=== 各表 obs / C / D 判决对照 ===")
for tb, gd in df.groupby("table"):
    n = len(gd.dropna(subset=["ratio_obs"]))
    p_obs = int((gd.ratio_obs > 1).sum())
    res = [f"{tb:10s}: 组数 {n:3d} | obs {p_obs}/{n} ({100*p_obs/n:.1f}%) 中位 {gd.ratio_obs.median():.2f}x"]
    for col in ["ratio_C", "ratio_D"]:
        d2 = gd.dropna(subset=[col])
        p = int((d2[col] > 1).sum())
        res.append(f"{col[-1]} {p}/{len(d2)} ({100*p/len(d2):.1f}%) 中位 {d2[col].median():.2f}x")
    both = gd.dropna(subset=["ratio_C", "ratio_D"])
    nfl = int(((both.ratio_C > 1) != (both.ratio_D > 1)).sum())
    res.append(f"C↔D翻转 {nfl}")
    print(" | ".join(res))
    if nfl:
        fl = both[(both.ratio_C > 1) != (both.ratio_D > 1)]
        print(fl[["channel", "temp", "ratio_obs", "ratio_C", "ratio_D"]].round(3).to_string(index=False))
print("\nsaved", out)
