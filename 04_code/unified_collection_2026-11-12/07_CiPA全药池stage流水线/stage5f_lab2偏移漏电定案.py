# -*- coding: utf-8 -*-
"""
Stage5f: lab_2 系统性偏移 × 漏电机制的定量定案
================================================
检验链：
T1 逐细胞相关：lab_2 内部 leak 大的细胞是否阻断偏浅（Spearman）
   ——若 lab_2 内部不相关，漏电解释在室内部就死了
T2 对照：其余四室同样相关（判断是否 lab_2 特异）
T3 幅度账：解释 3.10 倍 IC50 偏移需要 δ/A ≈ 1.05（block 0.5→0.244）
   实测 leak_hold/A 分布离 1.05 多远
数据：stage5c_leak_audit.csv（修复版）+ stage5a_per_segment.csv + stage2_final_cell_metrics.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
leak = pd.read_csv(os.path.join(HERE, 'stage5c_leak_audit.csv'))
seg = pd.read_csv(os.path.join(HERE, 'stage5a_per_segment.csv'))

leak['cell'] = leak['lab'] + '/' + leak['drug'] + '/' + leak['cell']
seg_drug = seg[~seg.e4031 & np.isfinite(seg.block_seg) & (seg.conc_uM > 0)]
# 每细胞取末段（最深阻断段）block
cell_block = seg_drug.sort_values('conc_uM').groupby('cell').agg(
    block_last=('block_seg', 'last'), conc_last=('conc_uM', 'last'),
    block_mean=('block_seg', 'mean')).reset_index()

df = leak.merge(cell_block, on='cell', how='inner')
df = df[np.isfinite(df.leak_hold) & np.isfinite(df.block_last)]
print('连接后细胞数:', len(df), df.groupby('lab').size().to_dict())

print('\n=== T1/T2 逐细胞 Spearman 相关：leak_hold vs block ===')
for lab in ['lab_1', 'lab_2', 'lab_3', 'lab_4', 'lab_5']:
    s = df[df.lab == lab]
    if len(s) < 30:
        continue
    r1, p1 = spearmanr(s.leak_hold.abs(), s.block_last)
    r2, p2 = spearmanr(s.leak_hold.abs(), s.block_mean)
    r3, p3 = spearmanr(s.leak_max_abs, s.block_last)
    print(f'{lab}: n={len(s)}  |leak_hold| vs 末段block ρ={r1:+.3f}(p={p1:.3f})  vs 均段block ρ={r2:+.3f}(p={p2:.3f})  leak_max vs block ρ={r3:+.3f}(p={p3:.3f})')

print('\n=== T3 幅度账：解释 3.10× IC50 偏移所需 δ/A ===')
# block_meas = block_true/(1+δ/A·(1/... )) 推导: δ/A=1.05 使 0.5→0.244
need = 1.05
# 实测：lab_2 信号幅度（用 stage2 的 ref_A）
s2 = pd.read_csv(os.path.join(HERE, 'stage2_final_cell_metrics.csv'))
s2c = s2[s2.qc_pass == True][['cell', 'ref_A_peak']]
df2 = df.merge(s2c, on='cell', how='left')
df2['ratio'] = df2.leak_hold.abs() / df2.ref_A_peak.abs()
for lab in ['lab_1', 'lab_2', 'lab_3', 'lab_4', 'lab_5']:
    s = df2[df2.lab == lab]['ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 30:
        continue
    print(f'{lab}: n={len(s)}  |leak_hold|/|ref_A| 中位={s.median():.3f}  P90={s.quantile(0.9):.3f}  ≥1.05占比={100*(s>=need).mean():.1f}%')
print(f'\n判线：需 δ/A≥{need} 才能解释 3.10× 偏移（block 0.5→0.244）')

# 附加：lab_2 高leak亚组 vs 低leak亚组的 block 差（直接效应量）
s = df2[df2.lab == 'lab_2'].dropna(subset=['ratio'])
hi = s[s.ratio >= s.ratio.quantile(0.75)]
lo = s[s.ratio <= s.ratio.quantile(0.25)]
print(f'\nlab_2 高漏电四分位(n={len(hi)}) block末段中位={hi.block_last.median():.3f} vs 低漏电(n={len(lo)}) {lo.block_last.median():.3f}')
df2[['cell','lab','drug','leak_hold','leak_max_abs','block_last','block_mean','ref_A_peak','ratio']].to_csv(
    os.path.join(HERE, 'stage5f_leak_vs_block.csv'), index=False)
print('saved stage5f_leak_vs_block.csv')
