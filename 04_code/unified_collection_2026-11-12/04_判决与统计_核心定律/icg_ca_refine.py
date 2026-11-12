# -*- coding: utf-8 -*-
"""
icg_ca_refine.py — 钙族细化判决: T型用失活V½<-55 分类(不截断激活V½)
输出: icg_out/verdict_icg_Ca_refined.csv + fig_icg_Ca.png
"""
import sys, warnings
import numpy as np, pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "icg_out"
GRID = np.round(np.arange(-100, 81, 2), 1)

df = pd.read_pickle(OUT / "gates_icg_Ca.pkl")
act = df[df.kind == "act"].copy()
ina = df[df.kind == "inact"].copy()

# 模型级分类: 有失活门 -> inact V½<-55 为 T型; 无失活门 -> act V½<-45 兜底
ina_vh = ina.groupby("model").Vhalf.median()
act_vh = act.groupby("model").Vhalf.median()
def classify(m):
    if m in ina_vh.index:
        return "T-type(LVA)" if ina_vh[m] < -55 else "HVA(L/N/P/Q/R)"
    return "T-type(LVA)" if act_vh.get(m, 0) < -45 else "HVA(L/N/P/Q/R)"
act["cls"] = act.model.map(classify)
ina["cls"] = ina.model.map(classify)
print(act.groupby("cls").model.nunique())
print("T型 act V½ 范围:", act[act.cls.str.startswith("T")].Vhalf.min(),
      act[act.cls.str.startswith("T")].Vhalf.max())

def verdict(sub, gate_kind):
    if sub.shape[0] < 8: return None
    M = {}
    for _, rr in sub.iterrows():
        M[rr.model + "|" + rr.gate] = np.interp(GRID, rr.V, rr.tau, left=np.nan, right=np.nan)
    tm = pd.DataFrame(M, index=GRID)
    vh_med = sub.Vhalf.median()
    win = (GRID >= vh_med - 20) & (GRID <= vh_med + 20)
    logsd = tm[win].apply(lambda r: np.nanstd(np.log10(r.values.astype(float)))
                          if np.isfinite(r.values.astype(float)).sum() >= 8 else np.nan, axis=1)
    tau_obs = np.nanmedian(logsd.values)
    mid_obs = sub.Vhalf.std() / sub.k.median()
    # τ@ownV½ 散布
    tau_at_vh = [np.interp(rr.Vhalf, rr.V, rr.tau) for _, rr in sub.iterrows()]
    tau_vh_sd = np.nanstd(np.log10(np.array(tau_at_vh, dtype=float)))
    return dict(gate=gate_kind, cls=sub.cls.iloc[0], n_models=sub.shape[0],
                tau_obs=tau_obs, mid_obs=mid_obs, ratio=mid_obs / tau_obs,
                vh_med=vh_med, vh_sd=sub.Vhalf.std(), k_med=sub.k.median(),
                tau_at_Vhalf_logsd=tau_vh_sd)

rows = [v for v in [verdict(s, "act") for _, s in act.groupby("cls")] +
        [verdict(s, "inact") for _, s in ina.groupby("cls")] if v]
vd = pd.DataFrame(rows)
vd.to_csv(OUT / "verdict_icg_Ca_refined.csv", index=False)
print(vd[["gate", "cls", "n_models", "tau_obs", "mid_obs", "ratio", "tau_at_Vhalf_logsd"]].round(3).to_string(index=False))

# ---- 图 ----
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot; setup_plot()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for j, (cls, col) in enumerate([("HVA(L/N/P/Q/R)", "#c0392b"), ("T-type(LVA)", "#2471a3")]):
    sub = act[act.cls == cls]
    ax = axes[0, j]
    for _, rr in sub.iterrows():
        ax.semilogy(rr.V, rr.tau, color=col, alpha=0.25, lw=0.8)
    ax.set_title(f"{cls} — activation $\\tau_m(V)$  (n={sub.shape[0]} models)")
    ax.set_xlabel("V (mV)"); ax.set_ylabel("$\\tau_m$ (ms)"); ax.set_xlim(-90, 60); ax.set_ylim(0.05, 500)
    ax2 = axes[1, j]
    ax2.hist(sub.Vhalf, bins=np.arange(-80, 25, 4), color=col, alpha=0.8, edgecolor="k", lw=0.4)
    ax2.axvline(sub.Vhalf.median(), color="k", ls="--", lw=1)
    ax2.set_title(f"{cls} — activation $V_{{1/2}}$  (SD={sub.Vhalf.std():.1f} mV, median={sub.Vhalf.median():.1f} mV)")
    ax2.set_xlabel("$V_{1/2,act}$ (mV)"); ax2.set_ylabel("# models")
fig.suptitle("ICGenealogy model library (unique Ca channel models): kinetics vs midpoints", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig_icg_Ca.png", dpi=200, bbox_inches="tight")
print("fig saved")
