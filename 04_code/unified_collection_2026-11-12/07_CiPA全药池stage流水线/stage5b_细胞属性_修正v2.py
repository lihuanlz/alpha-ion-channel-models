# -*- coding: utf-8 -*-
"""
stage5b 细胞属性修正版 v2
==========================
动机：旧 stage5b_cell_properties.csv 只收录 CellProperties 非空的细胞(1,320/1,893)，
lab_3 完全缺席；且 lab_5 温度散落在波形 csv 的 trace_#N_temperature_C 列未并入。

本脚本：
1. 以数据总库 5 lab × subtracted/ted/*.csv.zip 全集(1,893 细胞)为骨架；
2. 左连接旧表属性(方言已核实四套: RESULT_TYPE_* / TEMPERATURE / 首字母大写 / 裸名)；
3. lab_5 从波形列 trace_#1_temperature_C 补温度(逐细胞中位数)；
4. 输出 v2 csv + 覆盖率 json。
"""
import os, glob, json, zipfile
import pandas as pd
import numpy as np

BASE = r'D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia'
WARE = os.path.join(BASE, '数据总库', '01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t')
OUT = os.path.join(BASE, '分析_全药池_2026-11-10')

# ---------- 1. 骨架 ----------
rows = []
for lab in ['lab_1', 'lab_2', 'lab_3', 'lab_4', 'lab_5']:
    lab_d = os.path.join(WARE, lab)
    if not os.path.isdir(lab_d):
        continue
    for phase in sorted(os.listdir(lab_d)):  # 自动枚举 hERG / hERG_Phase_II / hERG_Blinded_Drugs
        d = os.path.join(lab_d, phase)
        if not os.path.isdir(d):
            continue
        for drug in sorted(os.listdir(d)):
            ted = os.path.join(d, drug, 'subtracted', 'ted')
            if not os.path.isdir(ted):
                continue
            for z in sorted(glob.glob(os.path.join(ted, '*.csv.zip'))):
                rows.append({'CELLID': os.path.basename(z)[:-len('.csv.zip')],
                             'lab': lab, 'drug': drug,
                             'zip_path': os.path.relpath(z, BASE)})
skel = pd.DataFrame(rows)
print('骨架细胞数:', len(skel), skel.groupby('lab').size().to_dict())

# ---------- 2. 左连接旧表 ----------
old = pd.read_csv(os.path.join(OUT, 'stage5b_cell_properties.csv'))
old = old.drop(columns=[c for c in ['zip_path'] if c in old.columns])
# 旧表 drug 列与骨架 drug 列一致(小写药名)
v2 = skel.merge(old.drop(columns=['lab', 'drug']), on='CELLID', how='left')
print('连接后:', v2.shape, ' 旧表命中:', int(v2['Cm'].notna().sum() + v2['Temp'].notna().sum() > 0))

# ---------- 3. lab_5 补温度 ----------
lab5_mask = (v2['lab'] == 'lab_5') & (v2['Temp'].isna())
todo = v2[lab5_mask]
print('lab_5 待补温度:', len(todo))
temps = {}
for _, r in todo.iterrows():
    zp = os.path.join(BASE, r['zip_path'])
    try:
        with zipfile.ZipFile(zp) as z:
            name = z.namelist()[0]
            vals = []
            with z.open(name) as fh:
                for i, line in enumerate(fh):
                    if i == 0:
                        continue  # header
                    if i > 200:
                        break
                    parts = line.decode('ascii', 'ignore').split(',')
                    if len(parts) > 2:
                        try:
                            vals.append(float(parts[2]))
                        except ValueError:
                            pass
            temps[r['CELLID']] = float(np.median(vals)) if vals else np.nan
    except Exception:
        temps[r['CELLID']] = np.nan
v2.loc[lab5_mask, 'Temp'] = v2.loc[lab5_mask, 'CELLID'].map(temps)
got = sum(1 for v in temps.values() if v == v)
print('lab_5 温度补到:', got, '/', len(todo))

# ---------- 4. 覆盖率统计 ----------
PARAMS = ['Cm', 'Rm', 'Passage', 'Rseal', 'Rs', 'Temp', 'RS', 'RSEAL', 'CM', 'VM', 'RM']
cov = {}
for lab in ['lab_1', 'lab_2', 'lab_3', 'lab_4', 'lab_5']:
    sub = v2[v2['lab'] == lab]
    cov[lab] = {'n_cells': int(len(sub))}
    for p in PARAMS:
        n = int(sub[p].notna().sum()) if p in sub.columns else 0
        if n:
            cov[lab][p] = n
cov['_dialects'] = {
    'lab_1': 'RESULT_TYPE_* 六参数(Cm/Rm/Passage/Rseal/Rs/Temp)',
    'lab_2': 'TEMPERATURE 独苗(全大写)',
    'lab_3': 'CellProperties 表全空(0行×16文件×2层), 与审稿回应Lab3不做Rs补偿(0%)互洽',
    'lab_4': '首字母大写 Temperature/Rs/Rseal (8/30文件有, 88细胞)',
    'lab_5': '裸名 Cm/Rs/Vm/Rm; 温度在波形列, 本版已并入',
}
with open(os.path.join(OUT, 'stage5b_覆盖率_v2.json'), 'w', encoding='utf-8') as f:
    json.dump(cov, f, ensure_ascii=False, indent=2)
v2.to_csv(os.path.join(OUT, 'stage5b_cell_properties_v2.csv'), index=False)
print(json.dumps(cov, ensure_ascii=False, indent=2))
print('OK -> stage5b_cell_properties_v2.csv / stage5b_覆盖率_v2.json')
