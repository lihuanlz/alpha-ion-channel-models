# 第二遍零记忆审计 H 项：跨物种/Q10/指纹 三抽验
import json, re
import numpy as np, pandas as pd

print('========== H1. cross_species_v3 抽验（stats_v3.py 口径） ==========')
df3 = pd.read_csv('features/features_full_v3.csv', low_memory=False)
cells = pd.read_csv('features/channelpedia_all_cells.csv', low_memory=False)
meta = cells[cells.flag == "active"].drop_duplicates("cell_id")[["cell_id", "combo"]]
def parse_combo(c):
    m = re.match(r"(\S+)\s+FT\s+([hmr])", str(c))
    if not m: return pd.Series({"host": None, "sp": None})
    return pd.Series({"host": m.group(1), "sp": {"h": "human", "m": "mouse", "r": "rat"}[m.group(2)]})
meta = meta.join(meta.combo.apply(parse_combo))
df3 = df3.merge(meta[["cell_id", "host", "sp"]], on="cell_id", how="left")

tau = df3[(df3.parameter == "tau_act") & (df3.voltage_mV.between(0, 40)) & (df3.r2 >= 0.7) & (df3.value > 0)]
vh = df3[(df3.parameter == "Vhalf_act") & (df3.r2 >= 0.8)]
sub = tau[(tau.channel == 'Kv1.1') & (tau.temp_nominal == 25)]
sps = {s: g for s, g in sub.groupby("sp") if g.cell_id.nunique() >= 10}
med = {s: g.groupby("voltage_mV").value.median() for s, g in sps.items()}
print('物种细胞数:', {s: g.cell_id.nunique() for s, g in sps.items()})
common = med['human'].index.intersection(med['mouse'].index)
fold = float((med['human'][common] / med['mouse'][common]).median())
va = vh[(vh.channel == 'Kv1.1') & (vh.temp_nominal == 25) & (vh.sp == 'human')].value
vb = vh[(vh.channel == 'Kv1.1') & (vh.temp_nominal == 25) & (vh.sp == 'mouse')].value
vdiff = float(va.median() - vb.median())
print(f'Kv1.1@25 human/mouse: tau_fold 新算 {fold:.6f} vs 存档 0.867243 (n_h={sps["human"].cell_id.nunique()} vs 96, n_m={sps["mouse"].cell_id.nunique()} vs 64)')
print(f'  Vhalf_diff 新算 {vdiff:.6f} vs 存档 2.316425')

print()
print('========== H2. family_q10 抽验（stats_family.py 口径，features_all.csv，不滤r2） ==========')
dfa = pd.read_csv('features/features_all.csv', low_memory=False)
g = dfa[dfa.channel == 'Kv1.1']
for V in [20, 30, 40]:
    t15 = g[(g.temp_nominal == 15) & (g.parameter == "tau_act") & (g.voltage_mV == V) & (g.value > 0)]["value"].median()
    t35 = g[(g.temp_nominal == 35) & (g.parameter == "tau_act") & (g.voltage_mV == V) & (g.value > 0)]["value"].median()
    q10 = float((t15 / t35) ** (10 / 20))
    print(f'Kv1.1@{V}mV: tau15={t15:.6f} tau35={t35:.6f} Q10={q10:.4f}')

print()
print('========== H3. 指纹分类器（analysis_pack_v5.py 包4 口径，corpus v5） ==========')
fc = pd.read_csv('features/features_corpus_v5.csv', low_memory=False)
kv = fc[~fc.channel.str.startswith("HCN")]
VSET = [0, 10, 20, 30, 40]
ta = kv[(kv.parameter == "tau_act") & (kv.r2 >= 0.7) & (kv.value > 0) & (kv.voltage_mV.isin(VSET))]
Xa = ta.pivot_table(index=["cell_id", "channel"], columns="voltage_mV", values="value", aggfunc="median")
Xa.columns = [f"tau_{int(c)}" for c in Xa.columns]
Xa = Xa.reset_index()
vh5 = kv[(kv.parameter == "Vhalf_act") & (kv.r2 >= 0.8)].groupby("cell_id").value.median().rename("Vhalf")
kk5 = kv[(kv.parameter == "k_act") & (kv.r2 >= 0.8)].groupby("cell_id").value.median().rename("k")
Xa = Xa.merge(vh5, on="cell_id", how="left").merge(kk5, on="cell_id", how="left")
Xa = Xa.dropna()
print('指纹样本:', Xa.shape, ' 类别数:', Xa.channel.nunique())
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
feats_all = [c for c in Xa.columns if c.startswith("tau_")] + ["Vhalf", "k"]
y = np.asarray(Xa.channel, dtype=str)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for tag, cols in [("tau_only", [c for c in feats_all if c.startswith("tau_")]),
                  ("Vhalf_k_only", ["Vhalf", "k"]), ("all", feats_all)]:
    X = Xa[cols].copy()
    for c in cols:
        if c.startswith("tau_") or c == "k": X[c] = np.log10(X[c])
    sc = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1), X, y, cv=cv)
    print(f'{tag}: {sc.mean():.4f} ± {sc.std():.4f}')
