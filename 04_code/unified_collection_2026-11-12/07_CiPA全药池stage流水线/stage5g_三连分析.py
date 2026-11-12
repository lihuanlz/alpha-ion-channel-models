# -*- coding: utf-8 -*-
"""
Stage5g 三连：结合动力学 kon + 盲药验证 + Rs-τ 稳健性
======================================================
G1 结合动力学：per_segment 全量，k_obs≈|slope|/block_seg (1/sweep)，
   log-log 对浓度——一级动力学预言斜率=1；段1/段2分开。
G2 盲药验证：127 盲细胞 vs 1,419 非盲细胞的 tau_ratio / block_frac 分布对照。
G3 Rs-τ：lab_1 室内 Rs vs ref_tau_ms 相关 + lab_3(无Rs补偿) vs 有补偿室 τ 分布。
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
seg = pd.read_csv(os.path.join(HERE, 'stage5a_per_segment.csv'))
s2 = pd.read_csv(os.path.join(HERE, 'stage2_final_cell_metrics.csv'))
prop = pd.read_csv(os.path.join(HERE, 'stage5b_cell_properties_v2.csv'))

# ============ G1 结合动力学 ============
d = seg[(~seg.e4031) & (seg.block_seg > 0.1) & (seg.seg_slope_per_sweep < 0)].copy()
d['k_obs'] = -d.seg_slope_per_sweep / d.block_seg  # 1/sweep
d = d[(d.k_obs > 0) & np.isfinite(d.k_obs) & (d.conc_uM > 0)]
# 段序号：每细胞内按 conc 排序
d = d.sort_values(['cell', 'conc_uM'])
d['seg_idx'] = d.groupby('cell').cumcount() + 1
print('=== G1 结合动力学 ===')
print('有效段:', len(d), ' 段1:', (d.seg_idx==1).sum(), ' 段2:', (d.seg_idx==2).sum())
res = {}
for idx, name in [(1, '段1'), (2, '段2')]:
    s = d[d.seg_idx == idx]
    x, y = np.log10(s.conc_uM), np.log10(s.k_obs)
    sl, ic = np.polyfit(x, y, 1)
    r, p = spearmanr(s.conc_uM, s.k_obs)
    res[name] = (sl, r, p, len(s))
    print(f'{name}: n={len(s)}  log-log斜率={sl:.2f}  Spearman ρ={r:+.3f}(p={p:.1e})')
s = d
sl, ic = np.polyfit(np.log10(s.conc_uM), np.log10(s.k_obs), 1)
r, p = spearmanr(s.conc_uM, s.k_obs)
print(f'合并: n={len(s)}  log-log斜率={sl:.2f}  ρ={r:+.3f}(p={p:.1e})')
# 双浓度对的直接速率比 vs 浓度比（最干净的检验）
pairs = pd.read_csv(os.path.join(HERE, 'stage5a_two_conc_pairs.csv'))
pp = pairs[(pairs.slope_lo < 0) & (pairs.slope_hi < 0) & (pairs.b_lo > 0.1) & (pairs.b_hi > 0.1)].copy()
pp['rate_lo'] = -pp.slope_lo / pp.b_lo
pp['rate_hi'] = -pp.slope_hi / pp.b_hi
pp['conc_ratio'] = pp.c_hi / pp.c_lo
pp['rate_ratio'] = pp.rate_hi / pp.rate_lo
pp = pp[(pp.rate_ratio > 0) & np.isfinite(pp.rate_ratio)]
r, p = spearmanr(pp.conc_ratio, pp.rate_ratio)
print(f'双浓度对(速率比vs浓度比): n={len(pp)}  速率比中位={pp.rate_ratio.median():.2f}  浓度比中位={pp.conc_ratio.median():.1f}  ρ={r:+.3f}(p={p:.1e})')
d[['cell','lab','drug','conc_uM','block_seg','seg_slope_per_sweep','k_obs','seg_idx']].to_csv(
    os.path.join(HERE, 'stage5g_kon_analysis.csv'), index=False)

# ============ G2 盲药验证 ============
print('\n=== G2 盲药波形级验证 ===')
blind = s2[s2.blind == True]
nblind = s2[s2.blind == False]
t_b = blind.tau_ratio.replace([np.inf, -np.inf], np.nan).dropna()
t_n = nblind.tau_ratio.replace([np.inf, -np.inf], np.nan).dropna()
inwin = lambda x: ((x >= 0.8) & (x <= 1.25)).mean() * 100
print(f'盲细胞 n={len(blind)}（τ比有效 {len(t_b)}）: τ比中位={t_b.median():.3f}  判线内(0.8-1.25)={inwin(t_b):.0f}%')
print(f'非盲   n={len(nblind)}（τ比有效 {len(t_n)}）: τ比中位={t_n.median():.3f}  判线内={inwin(t_n):.0f}%')
u, pu = mannwhitneyu(t_b, t_n)
print(f'盲vs非盲 τ比 Mann-Whitney p={pu:.3f}')
b_b = blind.block_frac.replace([np.inf, -np.inf], np.nan).dropna()
b_n = nblind.block_frac.replace([np.inf, -np.inf], np.nan).dropna()
print(f'block: 盲={b_b.median():.3f} vs 非盲={b_n.median():.3f}')
print('盲细胞按lab τ比中位:', blind.groupby('lab').tau_ratio.median().round(3).to_dict())
blind_out = blind[['cell','lab','drug','tau_ratio','block_frac','qc_pass']]
blind_out = os.path.join(HERE, 'stage5g_blind_cells.csv')
blind.to_csv(blind_out, index=False)

# ============ G3 Rs-τ 稳健性 ============
print('\n=== G3 Rs-τ 稳健性 ===')
prop['Rs_all'] = prop['Rs'].fillna(prop['RS'])  # 合并大小写两列
prop['cell'] = prop['lab'] + '/' + prop['drug'] + '/' + prop['CELLID']
m = s2.merge(prop[['cell', 'Rs_all']], on='cell', how='left')
ok = m[m.qc_pass == True].dropna(subset=['Rs_all', 'ref_tau_ms'])
ok = ok[np.isfinite(ok.ref_tau_ms) & (ok.Rs_all > 0)]
print('Rs×τ 有效细胞:', len(ok), ' 按lab:', ok.groupby('lab').size().to_dict())
for lab, g in ok.groupby('lab'):
    if len(g) < 30:
        continue
    r, p = spearmanr(g.Rs_all, g.ref_tau_ms)
    print(f'{lab}: n={len(g)}  Rs中位={g.Rs_all.median():.2f}MΩ  Rs vs τ: ρ={r:+.3f}(p={p:.3f})')
# 跨室：lab_3(无补偿) vs lab_1(有补偿) 的对照段 τ 分布
t3 = m[(m.lab == 'lab_3') & (m.qc_pass == True)].ref_tau_ms.replace([np.inf, -np.inf], np.nan).dropna()
t1 = m[(m.lab == 'lab_1') & (m.qc_pass == True)].ref_tau_ms.replace([np.inf, -np.inf], np.nan).dropna()
u, pu = mannwhitneyu(t3, t1)
print(f'lab_3(无Rs补偿) τ中位={t3.median():.2f}ms vs lab_1(有补偿) {t1.median():.2f}ms  M-W p={pu:.3f}')
print('各室对照段τ中位:', m[m.qc_pass==True].groupby('lab').ref_tau_ms.median().round(2).to_dict())
ok[['cell','lab','drug','Rs_all','ref_tau_ms']].to_csv(
    os.path.join(HERE, 'stage5g_rs_tau.csv'), index=False)
print('\nsaved: stage5g_kon_analysis.csv / stage5g_blind_cells.csv / stage5g_rs_tau.csv')
