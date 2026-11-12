# 第二遍零记忆审计 B 项 v2：按 analysis_pack_v5.py 包3 的真实冻结口径独立重写
# 口径来源（行21-31, 56, 82-98）：
#   kv  = corpus 中非 HCN 行;  kva = kv[flag=="active"]
#   组 = (channel, temp_nominal), 判据 cell_id.nunique() >= 15
#   mid_obs = std(Vhalf_act 行级 value, r2>=0.8, ddof=1) / median(k_act 行级 value, r2>=0.8), 各需 >=15 行
#   tau_obs(proc) : 行级 r2>=0.7, value>0, voltage in [lo,hi];
#                   逐电压 logsd (内部: finite&>0, 行数<15 -> NaN; std(log10,ddof=1)); 跨电压中位
import pandas as pd, numpy as np

MIN_CELLS = 15
PROC_WIN = {"tau_act": (0, 40), "tau_deact": (-60, -20), "tau_inact": (-20, 40)}

def logsd(v):
    v = np.asarray(v, float); v = v[np.isfinite(v) & (v > 0)]
    if len(v) < MIN_CELLS: return np.nan
    return float(np.std(np.log10(v), ddof=1))

df = pd.read_csv('features/features_corpus_v5.csv', low_memory=False)
kva = df[~df.channel.str.startswith("HCN")]
kva = kva[kva.flag == "active"]

rows = []
for (ch, T), g in kva.groupby(["channel", "temp_nominal"]):
    if g.cell_id.nunique() < MIN_CELLS: continue
    vh = g[(g.parameter == "Vhalf_act") & (g.r2 >= 0.8)].value
    kk = g[(g.parameter == "k_act") & (g.r2 >= 0.8)].value
    if len(vh) < MIN_CELLS or len(kk) < MIN_CELLS: continue
    row = dict(channel=ch, temp=T, mid_obs=float(vh.std() / kk.median()))
    for proc, (lo, hi) in PROC_WIN.items():
        ta = g[(g.parameter == proc) & (g.r2 >= 0.7) & (g.value > 0) & (g.voltage_mV.between(lo, hi))]
        sds = [logsd(gv.value) for _, gv in ta.groupby("voltage_mV")]
        sds = [s for s in sds if np.isfinite(s)]
        row[proc] = float(np.median(sds)) if sds else np.nan
    rows.append(row)
P_new = pd.DataFrame(rows)

pp = pd.read_csv('features/per_process_v5.csv')
m = P_new.merge(pp, on=['channel', 'temp'], how='outer', suffixes=('_new', '_old'), indicator=True)
print('新算组数:', len(P_new), ' 存档组数:', len(pp))
print('仅新算有:', (m._merge == 'left_only').sum(), ' 仅存档有:', (m._merge == 'right_only').sum())
if (m._merge != 'both').any():
    print(m[m._merge != 'both'][['channel', 'temp', '_merge']].to_string())

b = m[m._merge == 'both']
for col in ['mid_obs', 'tau_act', 'tau_deact', 'tau_inact']:
    d = (b[f'{col}_new'] - b[f'{col}_old']).abs()
    rel = (d / b[f'{col}_old'].abs()).replace([np.inf], np.nan)
    both_nan = b[f'{col}_new'].isna() & b[f'{col}_old'].isna()
    one_nan = b[f'{col}_new'].isna() ^ b[f'{col}_old'].isna()
    print(f'{col}: 共有非空 {b[f"{col}_new"].notna().sum()}/{b[f"{col}_old"].notna().sum()}, '
          f'双空 {both_nan.sum()}, 单侧空 {one_nan.sum()}, 最大绝对偏差 {d.max():.3e}, 最大相对偏差 {rel.max():.3e}')

sub = b.dropna(subset=['mid_obs_new', 'tau_act_new'])
r = sub.mid_obs_new / sub.tau_act_new
print(f'\n判决: {len(sub)} 组, ratio>1 通过 {(r>1).sum()} ({(r>1).mean()*100:.2f}%), tau_act 中位 {sub.tau_act_new.median():.4f}')
print('失败组:'); print(sub[r <= 1][['channel', 'temp', 'mid_obs_new', 'tau_act_new']].to_string())
