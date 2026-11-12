# -*- coding: utf-8 -*-
"""
全库统计 v3.1 —— 口径严格对齐 stats_family_v2.py（冻结）：
  τ 散布: 仅 tau_act, r2>=0.7, 电压窗 [0,40], 逐电压 log10-SD 跨细胞, 中位
  中点散布: Vhalf_act (r2>=0.8) SD / k_act (r2>=0.8) 中位
  方差分解: 同 v2
分组: 主判据 (channel, temp) 与 v2 一致；另出 (channel, temp, species, host) 细分
层 2/3: 跨物种/跨宿主 τ 中位曲线倍数 + V½ 差
产出: features/verdict_v3.json, family_split_v3.csv, cross_species_v3.csv, cross_host_v3.csv
"""
import json, re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
F = ROOT / "features"
LN10 = np.log(10)
MIN_CELLS = 15
TAU_WIN = (0, 40)

df = pd.read_csv(F / "features_full_v3.csv")
cells = pd.read_csv(F / "channelpedia_all_cells.csv")
meta = cells[cells.flag == "active"].drop_duplicates("cell_id")[["cell_id", "combo"]]
def parse_combo(c):
    m = re.match(r"(\S+)\s+FT\s+([hmr])", c)
    if not m: return pd.Series({"host": None, "sp": None})
    return pd.Series({"host": m.group(1), "sp": {"h": "human", "m": "mouse", "r": "rat"}[m.group(2)]})
meta = meta.join(meta.combo.apply(parse_combo))
df = df.merge(meta[["cell_id", "host", "sp"]], on="cell_id", how="left")

def logsd_deconv(vals, ses):
    v = np.asarray(vals, float); s = np.asarray(ses, float)
    m = np.isfinite(v) & (v > 0)
    v = v[m]; s = np.where(np.isfinite(s[m]) & (s[m] > 0), s[m], np.nan)
    v = v[m]
    lv = np.log10(v)
    obs_var = float(np.var(lv, ddof=1)) if len(lv) >= MIN_CELLS else np.nan
    se_log = s / (v * LN10)
    noise_var = float(np.nanmedian(se_log**2)) if np.isfinite(se_log).sum() >= MIN_CELLS else 0.0
    true_var = max(obs_var - noise_var, 0.0) if np.isfinite(obs_var) else np.nan
    return (np.sqrt(obs_var) if np.isfinite(obs_var) else np.nan,
            np.sqrt(true_var) if np.isfinite(true_var) else np.nan)

def group_stats(g):
    """g: 一个分组的数据帧。返回 dict 或 None。口径 = v2。"""
    if g.cell_id.nunique() < MIN_CELLS: return None
    ta = g[(g.parameter == "tau_act") & (g.voltage_mV.between(*TAU_WIN)) & (g.value > 0) & (g.r2 >= 0.7)]
    obs_l, true_l = [], []
    for V, gv in ta.groupby("voltage_mV"):
        if gv.cell_id.nunique() >= MIN_CELLS:
            o, t_ = logsd_deconv(gv.value.values, gv.se.values)
            if np.isfinite(o): obs_l.append(o); true_l.append(t_)
    tau_obs = float(np.median(obs_l)) if obs_l else np.nan
    tau_true = float(np.median(true_l)) if true_l else np.nan
    vh = g[(g.parameter == "Vhalf_act") & (g.r2 >= 0.8)]
    kk = g[(g.parameter == "k_act") & (g.r2 >= 0.8)]
    if len(vh) >= MIN_CELLS and len(kk) >= MIN_CELLS:
        vh_sd = float(vh.value.std())
        vh_noise = float(np.sqrt(np.nanmedian(vh.se.values**2))) if vh.se.notna().sum() >= MIN_CELLS else 0.0
        vh_true = float(np.sqrt(max(vh_sd**2 - vh_noise**2, 0)))
        k_med = float(kk.value.median())
        k_cv = float(kk.value.std() / kk.value.mean())
        mid_obs, mid_true = vh_sd / k_med, vh_true / k_med
    else:
        mid_obs = mid_true = k_cv = np.nan
    if not (np.isfinite(tau_obs) and np.isfinite(mid_obs)): return None
    return dict(n_cells=g.cell_id.nunique(), tau_obs=tau_obs, tau_true=tau_true,
                mid_obs=mid_obs, mid_true=mid_true, k_CV=k_cv,
                ratio_obs=mid_obs / tau_obs, ratio_true=mid_true / tau_true)

# ---- 主判据分组（同 v2：channel × temp）----
rows = []
for (ch, T), g in df.groupby(["channel", "temp_nominal"]):
    st = group_stats(g)
    if st: rows.append(dict(channel=ch, temp=T, **st))
G = pd.DataFrame(rows)

# 细分组（channel × temp × species × host）
rows2 = []
for (ch, T, sp, host), g in df.groupby(["channel", "temp_nominal", "sp", "host"]):
    st = group_stats(g)
    if st: rows2.append(dict(channel=ch, temp=T, species=sp, host=host, **st))
G2 = pd.DataFrame(rows2)
G.to_csv(F / "family_split_v3.csv", index=False)
G2.to_csv(F / "family_split_v3_fine.csv", index=False)

def verdict_of(G_):
    return {"n": len(G_), "frac_gt1": float((G_.ratio_obs > 1).mean()),
            "median_ratio": float(G_.ratio_obs.median()),
            "law": bool((G_.ratio_obs > 1).mean() >= 0.8 and G_.ratio_obs.median() > 2),
            "true_frac": float((G_.ratio_true > 1).mean()),
            "true_median": float(G_.ratio_true.median()),
            "counterexamples": G_[G_.ratio_obs < 1].to_dict("records")}

verdict = {"main_channel_temp": verdict_of(G), "fine": {k: v for k, v in verdict_of(G2).items() if k != "counterexamples"},
           "fine_n_counterexamples": int((G2.ratio_obs < 1).sum())}

# ---- 层 2: 跨物种 ----
tau = df[(df.parameter == "tau_act") & (df.voltage_mV.between(0, 40)) & (df.r2 >= 0.7) & (df.value > 0)]
vh = df[(df.parameter == "Vhalf_act") & (df.r2 >= 0.8)]
xs_rows = []
for (ch, temp), sub in tau.groupby(["channel", "temp_nominal"]):
    sps = {s: g for s, g in sub.groupby("sp") if g.cell_id.nunique() >= 10}
    if len(sps) < 2: continue
    med = {s: g.groupby("voltage_mV").value.median() for s, g in sps.items()}
    keys = list(med)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            common = med[a].index.intersection(med[b].index)
            if len(common) < 3: continue
            fold = float((med[a][common] / med[b][common]).median())
            va = vh[(vh.channel == ch) & (vh.temp_nominal == temp) & (vh.sp == a)].value
            vb = vh[(vh.channel == ch) & (vh.temp_nominal == temp) & (vh.sp == b)].value
            xs_rows.append(dict(channel=ch, temp=temp, sp_a=a, sp_b=b,
                                n_a=sps[a].cell_id.nunique(), n_b=sps[b].cell_id.nunique(),
                                tau_fold=fold,
                                Vhalf_diff=float(va.median() - vb.median()) if len(va) >= 8 and len(vb) >= 8 else np.nan))
XS = pd.DataFrame(xs_rows)
XS.to_csv(F / "cross_species_v3.csv", index=False)
verdict["cross_species"] = {"pairs": len(XS),
    "tau_fold_median": float(XS.tau_fold.median()) if len(XS) else None,
    "within_2x": float(XS.tau_fold.between(0.5, 2).mean()) if len(XS) else None}

# ---- 层 3: 跨宿主 ----
xh_rows = []
for (ch, temp, sp), sub in tau.groupby(["channel", "temp_nominal", "sp"]):
    hosts = {h: g for h, g in sub.groupby("host") if g.cell_id.nunique() >= 10}
    if len(hosts) < 2: continue
    med = {h: g.groupby("voltage_mV").value.median() for h, g in hosts.items()}
    keys = list(med)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            common = med[a].index.intersection(med[b].index)
            if len(common) < 3: continue
            xh_rows.append(dict(channel=ch, temp=temp, species=sp, host_a=a, host_b=b,
                                n_a=hosts[a].cell_id.nunique(), n_b=hosts[b].cell_id.nunique(),
                                tau_fold=float((med[a][common] / med[b][common]).median())))
XH = pd.DataFrame(xh_rows)
XH.to_csv(F / "cross_host_v3.csv", index=False)
verdict["cross_host"] = {"pairs": len(XH),
    "tau_fold_median": float(XH.tau_fold.median()) if len(XH) else None,
    "within_2x": float(XH.tau_fold.between(0.5, 2).mean()) if len(XH) else None}

json.dump(verdict, open(F / "verdict_v3.json", "w"), indent=1)
print(json.dumps({k: v for k, v in verdict.items() if k != "main_channel_temp" or True}, ensure_ascii=False, indent=1, default=str)[:3000])
