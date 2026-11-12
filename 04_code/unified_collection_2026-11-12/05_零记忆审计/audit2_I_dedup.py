# 第二遍零记忆审计 I 项补做：v4/v5 两次独立提取对拍（原判词卡"85,591 行/98.85%"无出处，此为正式口径）
# 用法: python audit2_I_dedup.py
import pandas as pd, numpy as np

r4 = pd.read_csv('features/features_rest_v4.csv', low_memory=False)
fc = pd.read_csv('features/features_corpus_v5.csv', low_memory=False)
key = ['cell_id', 'parameter', 'voltage_mV', 'channel']
a = r4[key + ['value']].copy(); b = fc[key + ['value']].copy()
a['value'] = a['value'].astype(float); b['value'] = b['value'].astype(float)
m = a.merge(b, on=key, suffixes=('_v4', '_v5'))
rel = (np.abs(m.value_v4 - m.value_v5) / np.abs(m.value_v5)).replace([np.inf], np.nan)
m['rel'] = rel.values
m['fam'] = np.where(m.channel.str.startswith('HCN'), 'HCN', 'non-HCN')

print(f'对拍键行数: {len(m)}')
print(f'逐字节相同: {(m.rel == 0).mean()*100:.2f}% ({(m.rel==0).sum()}/{len(m)})')
print(f'差异>1% 行数: {(m.rel > 0.01).sum()}')
print(m.groupby('fam').agg(n=('rel', 'size'), n_bad=('rel', lambda s: (s > 0.01).sum()),
                           byte_same_pct=('rel', lambda s: round((s == 0).mean()*100, 2))))
print('\n离群行明细:')
print(m[m.rel > 0.01][key + ['value_v4', 'value_v5', 'rel']].to_string(index=False))
