# -*- coding: utf-8 -*-
"""
四合一分析包 v5 —— 2026-11-08 冻结口径
包1: 斜率因子 k 守恒性(第三支柱) —— k_CV vs mid_obs(V½散布/k) vs tau_obs, 逐(channel,temp)组
包2: 窗口电流共变 —— 同细胞 Vhalf_act vs Vhalf_inact 相关 + 差值散布 (Kv only, HCN无失活)
包3: 分过程守恒度 —— tau_act / tau_deact / tau_inact 各自 tau_obs vs mid_obs
包4: τ指纹分类器 —— τ_act(V∈{0..40})+V½+k -> 随机森林分通道, 5折分层CV
包5: Q10 三温度扩展 —— 逐通道 τ_act 中位曲线, Q10(15/25),Q10(25/35),Q10(15/35) + 前向预言误差
产出: features/analysis_pack_v5.json + 各CSV
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
F = ROOT / "features"
LN10 = np.log(10)
OUT = {}

df = pd.read_csv(F / "features_corpus_v5.csv")
is_hcn = df.channel.str.startswith("HCN")
kv = df[~is_hcn]
hcn = df[is_hcn]

MIN_CELLS = 15

def logsd(v):
    v = np.asarray(v, float); v = v[np.isfinite(v) & (v > 0)]
    if len(v) < MIN_CELLS: return np.nan
    return float(np.std(np.log10(v), ddof=1))

# ============ 包1: k 守恒性(第三支柱) ============
print("=" * 60)
print("包1: 斜率因子 k 守恒性")
fs = pd.read_csv(F / "family_split_v3.csv")  # Kv active, 已含 k_CV, mid_obs, tau_obs
fs = fs.dropna(subset=["k_CV", "mid_obs", "tau_obs"])
cmp1 = dict(
    n_groups=len(fs),
    k_CV_median=float(fs.k_CV.median()),
    mid_obs_median=float(fs.mid_obs.median()),
    tau_obs_median=float(fs.tau_obs.median()),
    frac_k_lt_mid=float((fs.k_CV < fs.mid_obs).mean()),
    frac_tau_lt_k=float((fs.tau_obs < fs.k_CV).mean()),
    frac_tau_lt_mid=float((fs.tau_obs < fs.mid_obs).mean()),
)
OUT["pillar3_k"] = cmp1
print(f"  {len(fs)} 组: k_CV 中位 {cmp1['k_CV_median']:.3f} | V½散布/k 中位 {cmp1['mid_obs_median']:.3f} | tau log10-SD 中位 {cmp1['tau_obs_median']:.3f}")
print(f"  k 比 V½ 更守恒的组占比: {cmp1['frac_k_lt_mid']*100:.1f}%")
print(f"  τ 比 k 更守恒的组占比: {cmp1['frac_tau_lt_k']*100:.1f}%")
print(f"  τ 比 V½ 更守恒的组占比: {cmp1['frac_tau_lt_mid']*100:.1f}%")

# ============ 包2: 窗口电流共变 ============
print("=" * 60)
print("包2: 窗口电流共变 (Vhalf_act vs Vhalf_inact 同细胞配对, active only)")
kva = kv[kv.flag == "active"]
va = kva[(kva.parameter == "Vhalf_act") & (kva.r2 >= 0.8)][["cell_id", "channel", "temp_nominal", "value"]].rename(columns={"value": "vha"})
vi = kva[(kva.parameter == "Vhalf_inact") & (kva.r2 >= 0.8)][["cell_id", "value"]].rename(columns={"value": "vhi"})
both = va.merge(vi, on="cell_id").dropna()
w_rows = []
for (ch, T), g in both.groupby(["channel", "temp_nominal"]):
    if len(g) < 20: continue
    r = float(np.corrcoef(g.vha, g.vhi)[0, 1])
    sd_a, sd_i = float(g.vha.std()), float(g.vhi.std())
    sd_d = float((g.vha - g.vhi).std())
    w_rows.append(dict(channel=ch, temp=T, n=len(g), r=r, sd_act=sd_a, sd_inact=sd_i,
                       sd_diff=sd_d, window_protected=bool(sd_d < min(sd_a, sd_i))))
W = pd.DataFrame(w_rows)
W.to_csv(F / "window_covar_v5.csv", index=False)
OUT["window"] = dict(n_groups=len(W),
                     r_median=float(W.r.median()),
                     frac_r_pos=float((W.r > 0).mean()),
                     frac_protected=float(W.window_protected.mean()),
                     sd_ratio_median=float((W.sd_diff / W[["sd_act", "sd_inact"]].min(axis=1)).median()))
print(f"  {len(W)} 组(配对>=20): 相关 r 中位 {OUT['window']['r_median']:.2f}, r>0 占比 {OUT['window']['frac_r_pos']*100:.0f}%")
print(f"  差值散布 < 各自散布(窗口被保护)组占比: {OUT['window']['frac_protected']*100:.1f}%")
print(f"  SD(diff)/min(SD_a,SD_i) 中位: {OUT['window']['sd_ratio_median']:.2f}")

# ============ 包3: 分过程守恒度 ============
print("=" * 60)
print("包3: 分过程守恒度")
PROC_WIN = {"tau_act": (0, 40), "tau_deact": (-60, -20), "tau_inact": (-20, 40)}
p_rows = []
for (ch, T), g in kva.groupby(["channel", "temp_nominal"]):
    if g.cell_id.nunique() < MIN_CELLS: continue
    vh = g[(g.parameter == "Vhalf_act") & (g.r2 >= 0.8)].value
    kk = g[(g.parameter == "k_act") & (g.r2 >= 0.8)].value
    if len(vh) < MIN_CELLS or len(kk) < MIN_CELLS: continue
    mid = float(vh.std() / kk.median())
    row = dict(channel=ch, temp=T, mid_obs=mid)
    for proc, (lo, hi) in PROC_WIN.items():
        ta = g[(g.parameter == proc) & (g.r2 >= 0.7) & (g.value > 0) & (g.voltage_mV.between(lo, hi))]
        sds = [logsd(gv.value) for _, gv in ta.groupby("voltage_mV")]
        sds = [s for s in sds if np.isfinite(s)]
        row[proc] = float(np.median(sds)) if sds else np.nan
    p_rows.append(row)
P = pd.DataFrame(p_rows)
P.to_csv(F / "per_process_v5.csv", index=False)
OUT["per_process"] = {}
for proc in PROC_WIN:
    sub = P.dropna(subset=[proc])
    OUT["per_process"][proc] = dict(n=len(sub), tau_obs_median=float(sub[proc].median()),
                                    frac_more_conserved_than_Vhalf=float((sub[proc] < sub.mid_obs).mean()))
    print(f"  {proc}: {len(sub)} 组, tau log10-SD 中位 {sub[proc].median():.3f}, 比V½更守恒占比 {(sub[proc]<sub.mid_obs).mean()*100:.1f}%")

# ============ 包4: τ 指纹分类器 ============
print("=" * 60)
print("包4: τ 指纹分类器")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
VSET = [0, 10, 20, 30, 40]
ta = kv[(kv.parameter == "tau_act") & (kv.r2 >= 0.7) & (kv.value > 0) & (kv.voltage_mV.isin(VSET))]
piv = ta.pivot_table(index="cell_id", columns="voltage_mV", values="value", aggfunc="median")
piv.columns = [f"tau_{int(c)}" for c in piv.columns]
vh = kv[(kv.parameter == "Vhalf_act") & (kv.r2 >= 0.8)].set_index("cell_id").value.rename("Vhalf")
kk = kv[(kv.parameter == "k_act") & (kv.r2 >= 0.8)].set_index("cell_id").value.rename("k")
X = piv.join([vh, kk], how="inner").dropna()
meta = kv.drop_duplicates("cell_id").set_index("cell_id")
X["channel"] = meta.loc[X.index, "channel"]
X["flag"] = meta.loc[X.index, "flag"]
Xa = X[X.flag == "active"]
cnt = Xa_count = Xa = X[X.flag == "active"]
ch_counts = Xa.channel.value_counts()
keep = ch_counts[ch_counts >= 150].index
Xa = Xa[Xa.channel.isin(keep)]
y = np.asarray(Xa.channel, dtype=str)
feats_all = [c for c in Xa.columns if c not in ("channel", "flag")]
X_tau = Xa[[c for c in feats_all if c.startswith("tau_")]].apply(np.log10)
X_vh = Xa[["Vhalf", "k"]]
X_all = Xa[feats_all].copy()
for c in [c for c in feats_all if c.startswith("tau_")]:
    X_all[c] = np.log10(X_all[c])
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = {}
for tag, Xx in [("tau_only", X_tau), ("Vhalf_k_only", X_vh), ("all", X_all)]:
    rf = RandomForestClassifier(n_estimators=300, n_jobs=8, random_state=42)
    acc = cross_val_score(rf, Xx.values, y, cv=skf, scoring="accuracy")
    res = dict(mean=float(acc.mean()), sd=float(acc.std()))
    OUT.setdefault("fingerprint", {})[tag] = res
    print(f"  {tag}: 5折CV准确率 {acc.mean():.3f} ± {acc.std():.3f}")
OUT["fingerprint"]["n_cells"] = int(len(Xa))
OUT["fingerprint"]["n_classes"] = int(len(keep))
OUT["fingerprint"]["chance"] = float(1/len(keep))
OUT["fingerprint"]["classes"] = list(keep)
print(f"  细胞数 {len(Xa)}, 类别数 {len(keep)}, 随机基线 {1/len(keep):.3f}")

# ============ 包5: Q10 三温度 ============
print("=" * 60)
print("包5: Q10 三温度扩展 (tau_act, V=+20mV)")
ta = kv[(kv.parameter == "tau_act") & (kv.r2 >= 0.7) & (kv.value > 0) & (kv.voltage_mV == 20) & (kv.flag == "active")]
q_rows = []
for ch, g in ta.groupby("channel"):
    med = g.groupby("temp_nominal").value.median()
    n = g.groupby("temp_nominal").cell_id.nunique()
    row = dict(channel=ch)
    ok = True
    for pair, nm in [((15, 25), "Q10_15_25"), ((25, 35), "Q10_25_35"), ((15, 35), "Q10_15_35")]:
        a, b = pair
        if a in med.index and b in med.index and n.get(a, 0) >= 15 and n.get(b, 0) >= 15:
            row[nm] = float((med[a]/med[b])**(10.0/(b-a)))
        else:
            row[nm] = np.nan; ok = False
    if ok or any(np.isfinite([row.get("Q10_15_25", np.nan), row.get("Q10_25_35", np.nan)])):
        q_rows.append(row)
Q = pd.DataFrame(q_rows)
Q.to_csv(F / "q10_v5.csv", index=False)
OUT["q10"] = dict(n_channels=len(Q),
                  Q10_15_25_median=float(Q.Q10_15_25.median()),
                  Q10_25_35_median=float(Q.Q10_25_35.median()),
                  Q10_15_35_median=float(Q.Q10_15_35.median()) if Q.Q10_15_35.notna().any() else None)
print(f"  {len(Q)} 通道: Q10(15-25) 中位 {Q.Q10_15_25.median():.2f}, Q10(25-35) 中位 {Q.Q10_25_35.median():.2f}, Q10(15-35) 中位 {Q.Q10_15_35.median():.2f}")

json.dump(OUT, open(F / "analysis_pack_v5.json", "w"), indent=1)
print("saved", F / "analysis_pack_v5.json")
