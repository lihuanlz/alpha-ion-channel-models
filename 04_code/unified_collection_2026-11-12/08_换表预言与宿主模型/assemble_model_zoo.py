# -*- coding: utf-8 -*-
"""逐通道模型动物园组装 v1（2026-11-09）
从 profile_all_rows.csv（276,039 行，带逐行剖面似然 CI）为每个 (通道, 温度) 组
组装完整门控模型卡：
  - τ_act(V) / τ_deact(V) / τ_inact(V)：逐电压 群体中位 + EB 真散布（剖面 SE）+ 单细胞 CI 中位
  - V½_act / k_act / V½_inact：群体中位 + IQR + EB 真散布
  - τ_rec：群体中位 + EB 真散布
门槛：组内 active 细胞 >= 15（tau 类按电压至少 15 细胞才入电压表）。
产出：
  assembled_models_v1/cards/{channel}_{temp}.json   每通道每温度一张模型卡
  assembled_models_v1/model_zoo_master.csv          参数级汇总
  assembled_models_v1/tau_curves.csv                τ(V) 曲线长表
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar

R = Path(__file__).parent
OUT = R / "assembled_models_v1"
(OUT / "cards").mkdir(parents=True, exist_ok=True)
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
_flag = pd.read_csv(R / "features" / "features_corpus_v5r.csv",
                    usecols=["cell_id", "flag", "family"]).drop_duplicates("cell_id")
ck["flag"] = ck.cell_id.map(_flag.set_index("cell_id").flag)
ck["family"] = ck.cell_id.map(_flag.set_index("cell_id").family)
ck["ci_width"] = ck.prof_hi - ck.prof_lo
a = ck[ck.flag == "active"]

master_rows, curve_rows = [], []
n_cards = 0

for (ch, T), g in a.groupby(["channel", "temp"]):
    fam = g.family.iloc[0]
    n_cells = g.cell_id.nunique()
    if n_cells < MIN_CELLS: continue
    card = dict(channel=ch, temp=int(T), family=fam, n_cells=int(n_cells),
                min_cells_gate=MIN_CELLS, se_source="profile_likelihood_95ci/3.92",
                generated="2026-11-09", source_table="features/profile_all_rows.csv")
    # ---- τ(V) 曲线 ----
    for par in ["tau_act", "tau_deact", "tau_inact"]:
        gp = g[(g.parameter == par) & (g.r2 >= 0.7) & (g.value > 0)]
        curves = []
        for V, gv in gp.groupby("voltage_mV"):
            if len(gv) < MIN_CELLS: continue
            y = np.log10(gv.value.values)
            s = np.clip(gv.se_prof.values / (gv.value.values * LN10), 1e-4, None)
            row = dict(voltage_mV=float(V), n=int(len(gv)),
                       tau_median_ms=float(gv.value.median()),
                       log10sd_obs=float(y.std(ddof=1)),
                       log10sd_true_eb=float(eb_sigma(y, s)),
                       ci95_median_ms=float(np.nanmedian(gv.ci_width)))
            curves.append(row)
            curve_rows.append(dict(channel=ch, temp=T, parameter=par, **row))
        if curves:
            card[par] = sorted(curves, key=lambda r: r["voltage_mV"])
            med_eb = float(np.median([c["log10sd_true_eb"] for c in curves]))
            master_rows.append(dict(channel=ch, temp=T, family=fam, parameter=par,
                                    n_cells=n_cells, n_voltages=len(curves),
                                    tau_median_ms=float(np.median([c["tau_median_ms"] for c in curves])),
                                    dispersion_obs_dex=float(np.median([c["log10sd_obs"] for c in curves])),
                                    dispersion_true_eb_dex=med_eb))
    # ---- 标量参数 ----
    for par in ["Vhalf_act", "k_act", "Vhalf_inact"]:
        gp = g[(g.parameter == par) & (g.r2 >= 0.8)]
        if len(gp) < MIN_CELLS: continue
        s = np.clip(gp.se_prof.values, 1e-3, None)
        card[par] = dict(n=int(len(gp)), median=float(gp.value.median()),
                         iqr=[float(gp.value.quantile(.25)), float(gp.value.quantile(.75))],
                         sd_obs=float(gp.value.std(ddof=1)),
                         sd_true_eb=float(eb_sigma(gp.value.values, s)),
                         ci95_median=float(np.nanmedian(gp.ci_width)))
        master_rows.append(dict(channel=ch, temp=T, family=fam, parameter=par,
                                n_cells=int(len(gp)), n_voltages=1,
                                tau_median_ms=np.nan,
                                dispersion_obs_dex=np.nan,
                                dispersion_true_eb_dex=card[par]["sd_true_eb"]))
    # ---- τ_rec ----
    gr = g[(g.parameter == "tau_rec") & (g.value > 0) & (g.r2 >= 0.9)]
    if len(gr) >= MIN_CELLS:
        y = np.log10(gr.value.values)
        s = np.clip(gr.se_prof.values / (gr.value.values * LN10), 1e-4, None)
        card["tau_rec"] = dict(n=int(len(gr)), median_ms=float(gr.value.median()),
                               log10sd_obs=float(y.std(ddof=1)),
                               log10sd_true_eb=float(eb_sigma(y, s)),
                               ci95_median_ms=float(np.nanmedian(gr.ci_width)))
        master_rows.append(dict(channel=ch, temp=T, family=fam, parameter="tau_rec",
                                n_cells=int(len(gr)), n_voltages=1,
                                tau_median_ms=card["tau_rec"]["median_ms"],
                                dispersion_obs_dex=card["tau_rec"]["log10sd_obs"],
                                dispersion_true_eb_dex=card["tau_rec"]["log10sd_true_eb"]))
    (OUT / "cards" / f"{ch}_{T}.json").write_text(
        json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    n_cards += 1

mz = pd.DataFrame(master_rows)
mz.to_csv(OUT / "model_zoo_master.csv", index=False)
pd.DataFrame(curve_rows).to_csv(OUT / "tau_curves.csv", index=False)
print(f"模型卡: {n_cards} 张 ({mz.channel.nunique()} 通道)")
print(f"master 行: {len(mz)}  tau_curves 行: {len(curve_rows)}")
print("\n按族: ")
print(mz.groupby('family').channel.nunique().to_dict())
print("\n参数覆盖:")
print(mz.parameter.value_counts().to_dict())
