# 第二遍零记忆审计 E 项：ICG 判决链独立复算
# 口径来源：icg_analysis.py 行86-133（K/Na/KCa/IH）与 icg_ca_refine.py 行18-52（Ca refined）
import numpy as np, pandas as pd

OUT = 'icg_out'
GRID = np.round(np.arange(-100, 81, 2), 1)

def verdict(sub, gate_kind, cls_name):
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
    return dict(gate=gate_kind, cls=cls_name, n_models=sub.shape[0],
                tau_obs=tau_obs, mid_obs=mid_obs, ratio=mid_obs / tau_obs,
                vh_med=vh_med, vh_sd=sub.Vhalf.std(), k_med=sub.k.median())

rows = []
for FAM in ['K', 'Na', 'KCa', 'IH']:
    df = pd.read_pickle(f'{OUT}/gates_icg_{FAM}.pkl')
    act = df[df.kind == "act"].copy(); act["cls"] = FAM
    r = verdict(act, "act", FAM)
    if r: rows.append(r)
    ina = df[df.kind == "inact"].copy()
    if len(ina):
        ina["cls"] = FAM
        r = verdict(ina, "inact", FAM)
        if r: rows.append(r)

# Ca refined：有失活门 -> inact V½<-55 为 T型；无失活门 -> act V½<-45 兜底
df = pd.read_pickle(f'{OUT}/gates_icg_Ca.pkl')
act = df[df.kind == "act"].copy(); ina = df[df.kind == "inact"].copy()
ina_vh = ina.groupby("model").Vhalf.median()
act_vh = act.groupby("model").Vhalf.median()
def classify(m):
    if m in ina_vh.index:
        return "T-type(LVA)" if ina_vh[m] < -55 else "HVA(L/N/P/Q/R)"
    return "T-type(LVA)" if act_vh.get(m, 0) < -45 else "HVA(L/N/P/Q/R)"
act["cls"] = act.model.map(classify); ina["cls"] = ina.model.map(classify)
for cls, s in act.groupby("cls"):
    r = verdict(s, "act", cls)
    if r: rows.append(r)
for cls, s in ina.groupby("cls"):
    r = verdict(s, "inact", cls)
    if r: rows.append(r)

new = pd.DataFrame(rows)
old = pd.read_csv(f'{OUT}/verdict_icg_ALL.csv')
m = new.merge(old, on=['gate', 'cls'], suffixes=('_new', '_old'), how='outer', indicator=True)
print('新算行数:', len(new), ' 存档行数:', len(old))
print(m[['gate', 'cls', '_merge']].to_string(index=False))
b = m[m._merge == 'both']
for col in ['n_models', 'tau_obs', 'mid_obs', 'ratio', 'vh_med', 'vh_sd', 'k_med']:
    d = (b[f'{col}_new'].astype(float) - b[f'{col}_old'].astype(float)).abs()
    rel = (d / b[f'{col}_old'].astype(float).abs()).replace([np.inf], np.nan)
    print(f'{col}: max abs {d.max():.3e}, max rel {np.nanmax(rel):.3e}')
