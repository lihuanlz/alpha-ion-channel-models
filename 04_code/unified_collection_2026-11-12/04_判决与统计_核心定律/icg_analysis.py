# -*- coding: utf-8 -*-
"""
icg_analysis.py — 在 ICGenealogy 模型库上检验门控定律(τ/k 守恒 vs V½ 漂移)
用法: python icg_analysis.py Ca
输出: icg_out/features_icg_<fam>.csv, verdict_icg_<fam>.csv, fig_icg_<fam>.png
"""
import sys, pickle, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent
ICG = BASE.parent / "icg"   # 克隆在 _rev_2026-10-21/icg
OUT = BASE / "icg_out"; OUT.mkdir(exist_ok=True)

FAM = sys.argv[1] if len(sys.argv) > 1 else "Ca"
PKL = {"Ca": "icg-channels-Ca.pkl", "K": "icg-channels-K.pkl",
       "Na": "icg-channels-Na.pkl", "IH": "icg-channels-IH.pkl",
       "KCa": "icg-channels-KCa.pkl"}[FAM]
UNIQ = {"Ca": "Ca_unique.csv", "K": "K_unique.csv", "Na": "Na_unique.csv",
        "IH": "Ih_unique.csv", "KCa": "KCa_unique.csv"}[FAM]

# 门名白名单(全小写字母、<=2字符)+ 黑名单(解析垃圾/浓度态)
BLACKLIST = {"mM","1e","10","cai","cao","nai","nao","ki","ko","ca","ci","co",
             "oca","mho","cm2","O","C0","C1","C2","C3","C4","I0","I1","I2","I3","I4","Io"}
def gate_ok(g):
    if g in BLACKLIST: return False
    if not g.isalpha() or not g.islower() or len(g) > 2: return False
    return True

def boltz(V, vh, k, s):
    return 1.0 / (1.0 + np.exp(-s * (V - vh) / k))

def fit_boltz(V, ss, s):
    """返回 (vh, k, r2) 或 None"""
    try:
        p0 = [V[np.argmin(np.abs(ss - 0.5))], 6.0]
        p, _ = curve_fit(lambda v, vh, k: boltz(v, vh, k, s), V, ss, p0=p0,
                         bounds=([-100, 0.8], [80, 40]), maxfev=4000)
        pred = boltz(V, p[0], p[1], s)
        r2 = 1 - np.sum((ss - pred) ** 2) / np.sum((ss - ss.mean()) ** 2)
        return p[0], p[1], r2
    except Exception:
        return None

print(f"=== {FAM} ===")
d = pickle.load(open(ICG / "icg-pickles" / PKL, "rb"), encoding="latin1")
uq = pd.read_csv(ICG / "icg-unique" / UNIQ)
# 去重名 -> pickle 键: '184171_Ca_LVA-ID-1' -> '184171_Ca_LVA.mod'
uq["pkey"] = uq["name"].str.replace(r"-ID-\d+$", ".mod", regex=True)
uq = uq[uq["pkey"].isin(d.keys())]
print(f"unique models with pickle: {len(uq)}")

rows = []
for _, r in uq.iterrows():
    e = d[r["pkey"]]
    if e.get("ERROR_FLAGS"): continue
    states = [g for g in e["STATES"] if gate_ok(g)]
    for g in states:
        try:
            V = np.asarray(e["RATE_VALS_V"][g], dtype=float)
            ss = np.asarray(e["RATE_VALS_SS"][g], dtype=float)
            tau = np.asarray(e["RATE_VALS_TAU"][g], dtype=float)
        except Exception:
            continue
        if len(V) < 20 or len(ss) != len(V) or len(tau) != len(V): continue
        if np.nanmin(ss) > 0.1 or np.nanmax(ss) < 0.9: continue      # 完整S形
        # 单调性定向
        rho = pd.Series(ss).corr(pd.Series(V), method="spearman")
        if abs(rho) < 0.9: continue
        s = 1.0 if rho > 0 else -1.0
        fb = fit_boltz(V, ss, s)
        if fb is None: continue
        vh, k, r2 = fb
        if r2 < 0.7: continue
        tau_ok = (tau > 0) & np.isfinite(tau) & (tau < 1e5)
        rows.append(dict(model=r["pkey"], gate=g, kind="act" if s > 0 else "inact",
                         Vhalf=vh, k=k, r2=r2, q10tau=e.get("Q10_TAU", np.nan),
                         V=list(V), tau=list(np.where(tau_ok, tau, np.nan))))
df = pd.DataFrame(rows)
print(f"clean gate rows: {len(df)}  models: {df.model.nunique()}")
df.to_pickle(OUT / f"gates_icg_{FAM}.pkl")

# ---- 分类: 按激活 V½ 分 LVA/HVA(仅Ca; 其他家族单一类) ----
act = df[df.kind == "act"].copy()
if FAM == "Ca":
    act["cls"] = np.where(act.Vhalf < -45, "LVA(T-type-like)", "HVA(L/N/P/Q/R-like)")
else:
    act["cls"] = FAM
print(act.groupby("cls").model.nunique())

# ---- 判决: 逐电压 log10-SD(τ) vs SD(V½)/median(k) ----
GRID = np.round(np.arange(-100, 81, 2), 1)
def tau_matrix(sub):
    M = {}
    for _, rr in sub.iterrows():
        M[rr.model + "|" + rr.gate] = np.interp(GRID, rr.V, rr.tau, left=np.nan, right=np.nan)
    return pd.DataFrame(M, index=GRID)

verdicts = []
for cls, sub in act.groupby("cls"):
    if sub.shape[0] < 8: continue
    tm = tau_matrix(sub)
    vh_med = sub.Vhalf.median()
    win = (GRID >= vh_med - 20) & (GRID <= vh_med + 20)
    logsd = tm[win].apply(lambda row: np.nanstd(np.log10(row.values.astype(float))) if np.sum(np.isfinite(row.values.astype(float))) >= 8 else np.nan, axis=1)
    tau_obs = np.nanmedian(logsd.values)
    mid_obs = sub.Vhalf.std() / sub.k.median()
    verdicts.append(dict(family=FAM, gate="act", cls=cls, n_models=sub.shape[0],
                         tau_obs=tau_obs, mid_obs=mid_obs, ratio=mid_obs / tau_obs,
                         vh_med=vh_med, vh_sd=sub.Vhalf.std(), k_med=sub.k.median()))
ina = df[df.kind == "inact"].copy()
if len(ina):
    if FAM == "Ca":
        amap = act.drop_duplicates("model").set_index("model")["cls"]
        ina["cls"] = ina.model.map(amap)
        ina = ina.dropna(subset=["cls"])
    else:
        ina["cls"] = FAM  # 注意: IH 的"下降门"实为超极化激活门
    for cls, sub in ina.groupby("cls"):
        if sub.shape[0] < 8: continue
        tm = tau_matrix(sub)
        vh_med = sub.Vhalf.median()
        win = (GRID >= vh_med - 20) & (GRID <= vh_med + 20)
        logsd = tm[win].apply(lambda row: np.nanstd(np.log10(row.values.astype(float))) if np.sum(np.isfinite(row.values.astype(float))) >= 8 else np.nan, axis=1)
        tau_obs = np.nanmedian(logsd.values)
        mid_obs = sub.Vhalf.std() / sub.k.median()
        verdicts.append(dict(family=FAM, gate="inact", cls=cls, n_models=sub.shape[0],
                             tau_obs=tau_obs, mid_obs=mid_obs, ratio=mid_obs / tau_obs,
                             vh_med=vh_med, vh_sd=sub.Vhalf.std(), k_med=sub.k.median()))

vd = pd.DataFrame(verdicts)
vd.to_csv(OUT / f"verdict_icg_{FAM}.csv", index=False)
print(vd[["gate","cls","n_models","tau_obs","mid_obs","ratio"]].to_string(index=False))
print("saved:", OUT / f"verdict_icg_{FAM}.csv")
