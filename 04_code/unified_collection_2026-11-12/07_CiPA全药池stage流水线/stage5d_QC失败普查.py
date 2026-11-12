# -*- coding: utf-8 -*-
"""Stage5d QC失败普查: 汇总各 stage ckpt 的失败记录, 按 lab/drug/原因透视。"""
import json, os
import pandas as pd

BASE = r'D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia'
OUT = os.path.join(BASE, '分析_全药池_2026-11-10')

def load(name):
    p = os.path.join(OUT, name + '.jsonl')
    rows = []
    if not os.path.exists(p):
        return pd.DataFrame()
    with open(p, encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return pd.DataFrame(rows)

frames = []
for name, stage in [('stage2_ckpt','2'), ('stage2b_ckpt','2b'), ('stage2c_ckpt','2c'),
                    ('stage2d_ckpt','2d'), ('stage3_ckpt','3'),
                    ('stage4b_ckpt','4b'), ('stage5a_ckpt','5a')]:
    df = load(name)
    if df.empty or 'ok' not in df.columns:
        continue
    bad = df[df['ok'] == False].copy()
    if bad.empty:
        continue
    # 解析 cell 路径补 lab/drug
    if 'cell' in bad.columns:
        sp = bad['cell'].astype(str).str.split('/', expand=True)
        if 'lab' not in bad.columns or bad['lab'].isna().all():
            bad['lab'] = sp[0] if sp.shape[1] > 0 else None
        if 'drug' not in bad.columns or bad['drug'].isna().all():
            bad['drug'] = sp[1] if sp.shape[1] > 1 else None
    bad['stage'] = stage
    # 原因字段
    for cand in ['err', 'reason', 'qc_pass', 'seg_warn']:
        if cand in bad.columns:
            bad['reason'] = bad[cand].astype(str)
            break
    else:
        bad['reason'] = 'unknown'
    frames.append(bad[['stage', 'lab', 'drug', 'cell', 'reason']])

allb = pd.concat(frames, ignore_index=True)
allb.to_csv(os.path.join(OUT, 'stage5d_QC失败普查.csv'), index=False)

print('=== 各 stage 失败数 ===')
print(allb.groupby('stage').size().to_string())
print()
print('=== stage × lab 透视 ===')
print(pd.crosstab(allb['stage'], allb['lab']).to_string())
print()
print('=== 失败原因 top15 ===')
print(allb['reason'].value_counts().head(15).to_string())
print()
print('=== stage2b 失败原因按 lab ===')
b2b = allb[allb['stage'] == '2b']
print(pd.crosstab(b2b['reason'], b2b['lab']).to_string())
print()
print('=== stage2b 失败 top 药物 ===')
print(b2b.groupby('drug').size().sort_values(ascending=False).head(10).to_string())
