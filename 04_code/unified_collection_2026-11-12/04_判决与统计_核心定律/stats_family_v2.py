# -*- coding: utf-8 -*-
# v2 统计分拆：在 v1 判据（冻结）上加方差分解
# 观测散布² = 真实散布² + 拟合噪声² —— 噪声用每次拟合的 SE 中位数估计
# 输出：obs / noise / true 三个散布，主判据仍用 obs（不改冻结口径），true 作增强报告
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
df = pd.read_csv(ROOT / "features" / "features_all_v2.csv")
MIN_CELLS = 15
TAU_WIN = (0, 40)

def logsd_deconv(vals, ses):
    """log10 空间的方差分解：obs_var - median(se_log10^2)，se_log10 = se/(v*ln10)"""
    v = np.asarray(vals, float); s = np.asarray(ses, float)
    m = np.isfinite(v) & (v > 0)
    v = v[m]; s = np.where(np.isfinite(s[m]) & (s[m] > 0), s[m], np.nan)
    v = v[m]
    lv = np.log10(v)
    obs_var = float(np.var(lv, ddof=1)) if len(lv) >= MIN_CELLS else np.nan
    se_log = s / (v * np.log(10))
    noise_var = float(np.nanmedian(se_log**2)) if np.isfinite(se_log).sum() >= MIN_CELLS else 0.0
    true_var = max(obs_var - noise_var, 0.0) if np.isfinite(obs_var) else np.nan
    return np.sqrt(obs_var) if np.isfinite(obs_var) else np.nan, np.sqrt(true_var) if np.isfinite(true_var) else np.nan, noise_var

rows = []
for (ch, T), g in df.groupby(["channel", "temp_nominal"]):
    if g.cell_id.nunique() < MIN_CELLS: continue
    ta = g[(g.parameter == "tau_act") & (g.voltage_mV.between(*TAU_WIN)) & (g.value > 0) & (g.r2 >= 0.7)]
    obs_l, true_l, nv_l = [], [], []
    for V, gv in ta.groupby("voltage_mV"):
        if gv.cell_id.nunique() >= MIN_CELLS:
            o, t_, nv = logsd_deconv(gv.value.values, gv.se.values)
            if np.isfinite(o): obs_l.append(o); true_l.append(t_); nv_l.append(nv)
    tau_obs = float(np.median(obs_l)) if obs_l else np.nan
    tau_true = float(np.median(true_l)) if true_l else np.nan
    vh = g[(g.parameter == "Vhalf_act") & (g.r2 >= 0.8)]
    kk = g[(g.parameter == "k_act") & (g.r2 >= 0.8)]
    if len(vh) >= MIN_CELLS and len(kk) >= MIN_CELLS:
        vh_sd_obs = float(vh.value.std())
        vh_noise = float(np.sqrt(np.nanmedian(vh.se.values**2))) if vh.se.notna().sum() >= MIN_CELLS else 0.0
        vh_sd_true = float(np.sqrt(max(vh_sd_obs**2 - vh_noise**2, 0)))
        k_med = float(kk.value.median())
        k_cv = float(kk.value.std()/kk.value.mean())
        mid_obs = vh_sd_obs/k_med; mid_true = vh_sd_true/k_med
    else:
        mid_obs = mid_true = k_cv = np.nan
    rows.append(dict(channel=ch, temp=T, n_cells=g.cell_id.nunique(),
                     tau_logSD_obs=tau_obs, tau_logSD_true=tau_true,
                     mid_disp_obs=mid_obs, mid_disp_true=mid_true, k_CV=k_cv,
                     ratio_obs=mid_obs/tau_obs if np.isfinite(mid_obs) and np.isfinite(tau_obs) else np.nan,
                     ratio_true=mid_true/tau_true if np.isfinite(mid_true) and np.isfinite(tau_true) and tau_true>0 else np.nan))
s = pd.DataFrame(rows)
s.to_csv(ROOT / "features" / "family_split_v2.csv", index=False)

ok = s.dropna(subset=["ratio_obs"])
okt = s.dropna(subset=["ratio_true"])
verdict = dict(
    version="v2（三参数Boltzmann + m^p模型选择 + 方差分解）",
    n_groups=len(s), n_with_ratio=int(len(ok)),
    obs=dict(frac_gt1=float((ok.ratio_obs>1).mean()), median_ratio=float(ok.ratio_obs.median()),
             law_supported=bool((ok.ratio_obs>1).mean()>=0.8 and ok.ratio_obs.median()>2)),
    true_deconvolved=dict(frac_gt1=float((okt.ratio_true>1).mean()) if len(okt) else None,
                          median_ratio=float(okt.ratio_true.median()) if len(okt) else None),
    k_CV_median=float(s.k_CV.median()),
)
with open(ROOT / "features" / "family_verdict_v2.json", "w", encoding="utf-8") as f:
    json.dump(verdict, f, ensure_ascii=False, indent=2)

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.scatter(ok.ratio_obs, ok.ratio_obs, alpha=0)  # 定轴
ax.scatter(ok.channel.astype("category").cat.codes, ok.ratio_obs, label="观测散布比", alpha=0.7)
ax.scatter(okt.channel.astype("category").cat.codes, okt.ratio_true, label="去噪声后", marker="s", alpha=0.7)
ax.axhline(1, color="r", ls="--", lw=1)
ax.set_xticks(range(len(ok.channel.unique())))
ax.set_xticklabels(ok.channel.unique(), rotation=90, fontsize=7)
ax.set_ylabel("V½ 散布 / τ 散布"); ax.set_title("v2 精密提取：散布比（圆=观测，方=去噪声）")
ax.legend()
fig.tight_layout()
fig.savefig(ROOT / "features" / "headline_family_v2.png", dpi=160, bbox_inches="tight")
print(json.dumps(verdict, ensure_ascii=False, indent=2))
print(s.sort_values("ratio_obs", ascending=False).head(15).to_string(index=False))
print("--- 反例组 ---")
print(s[s.ratio_obs < 1].to_string(index=False))
