# -*- coding: utf-8 -*-
"""fig12: Stage5 彻底清剿六面板汇总图"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = r'D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia'
OUT = os.path.join(BASE, '分析_全药池_2026-11-10')

seg = pd.read_csv(os.path.join(OUT, 'stage5a_per_segment.csv'))
pairs = pd.read_csv(os.path.join(OUT, 'stage5a_two_conc_pairs.csv'))
prop = pd.read_csv(os.path.join(OUT, 'stage5b_cell_properties_v2.csv'))
leak = pd.read_csv(os.path.join(OUT, 'stage5c_leak_audit.csv'))

LABS = ['lab_1', 'lab_2', 'lab_3', 'lab_4', 'lab_5']
LC = dict(zip(LABS, ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']))

fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

# (a) 全段 Hill 全景
ax = axa = plt.subplot(2, 3, 1)
d = seg[(~seg.e4031) & np.isfinite(seg.block_seg) & (seg.conc_uM > 0)]
for lab in LABS:
    s = d[d.lab == lab]
    ax.scatter(s.conc_uM, s.block_seg, s=7, alpha=0.28, color=LC[lab], label=lab, rasterized=True)
# 全局 Hill 拟合
from scipy.optimize import curve_fit
def hill(C, ic50, h): return C ** h / (C ** h + ic50 ** h)
m = np.isfinite(d.block_seg) & (d.block_seg >= 0) & (d.block_seg <= 1.2)
popt, _ = curve_fit(hill, d.conc_uM[m], d.block_seg[m], p0=[0.5, 1.0], bounds=([1e-4, 0.3], [1e3, 3]))
cc = np.logspace(-4, 2, 200)
ax.plot(cc, hill(cc, *popt), 'k-', lw=2.2, label=f'全局Hill IC50={popt[0]:.2f}µM h={popt[1]:.2f}')
ax.set_xscale('log')
ax.set_xlabel('浓度 (µM)'); ax.set_ylabel('实测阻断分数 block_seg')
ax.set_title('a  全段提取 Hill 全景（2,342 剂量点）\n臂B LOO MAE=0.157 vs 零模型 0.232 → 浓度轴价值翻正', fontsize=10)
ax.legend(fontsize=7, loc='lower right'); ax.set_ylim(-0.08, 1.15)

# (b) 双浓度单调性
ax = plt.subplot(2, 3, 2)
mono = (pairs.b_hi >= pairs.b_lo).mean()
ax.scatter(pairs.b_lo, pairs.b_hi, s=14, alpha=0.4, color='#377eb8', rasterized=True)
mx = max(pairs.b_lo.max(), pairs.b_hi.max()) * 1.05
ax.plot([0, mx], [0, mx], 'k--', lw=1, label='对角线（等阻断）')
ax.fill_between([0, mx], [0, mx], [mx, mx], color='green', alpha=0.06)
ax.text(0.03, 0.93, f'{mono*100:.1f}% 单调（高浓度阻断更深）', transform=ax.transAxes, fontsize=10, color='darkgreen')
ax.set_xlabel('低浓度段阻断 b_lo'); ax.set_ylabel('高浓度段阻断 b_hi')
ax.set_title(f'b  双浓度细胞单调性（651 对，两段 ρ=0.648）', fontsize=10)

# (c) E-4031 残余逐室
ax = plt.subplot(2, 3, 3)
e4 = seg[seg.e4031 == True].copy()
e4['resid'] = 1 - e4.block_seg
data = [e4[e4.lab == lab]['resid'].dropna().values * 100 for lab in LABS]
bp = ax.boxplot(data, labels=LABS, showfliers=False, patch_artist=True, widths=0.55)
for patch, lab in zip(bp['boxes'], LABS):
    patch.set_facecolor(LC[lab]); patch.set_alpha(0.7)
med = np.median(e4['resid']) * 100
ax.axhline(med, color='k', ls='--', lw=1)
ax.text(0.03, 0.93, f'全数据集残余中位 {med:.1f}%（IQR 1.5–3.9%）\n→ 所测电流确为 hERG', transform=ax.transAxes, fontsize=9.5)
ax.set_ylabel('E-4031 1µM 残余电流 (%)')
ax.set_title('c  E-4031 阳性对照残余（逐室）', fontsize=10)

# (d) 段1/段2 未平台化率
ax = plt.subplot(2, 3, 4)
r1 = (pairs.slope_lo < -0.002).groupby(pairs.lab).mean() * 100
r2 = (pairs.slope_hi < -0.002).groupby(pairs.lab).mean() * 100
x = np.arange(len(LABS)); w = 0.36
ax.bar(x - w/2, [r1.get(l, 0) for l in LABS], w, label='段1（低浓度）', color='#fc9272')
ax.bar(x + w/2, [r2.get(l, 0) for l in LABS], w, label='段2（高浓度）', color='#4292c6')
all1 = (pairs.slope_lo < -0.002).mean() * 100
all2 = (pairs.slope_hi < -0.002).mean() * 100
ax.axhline(all1, color='#fc9272', ls=':', lw=1)
ax.axhline(all2, color='#4292c6', ls=':', lw=1)
ax.text(0.02, 0.93, f'全体：段1 {all1:.0f}% / 段2 {all2:.0f}% 未平台化\n（FDA 协议要求监测至稳态，各室未达）', transform=ax.transAxes, fontsize=9.5)
ax.set_xticks(x); ax.set_xticklabels(LABS); ax.set_ylabel('未平台化段占比 (%)')
ax.set_title('d  阻断时间进程未平台化（幅度斜率<−0.002/sweep）\n双浓度段仅存在于 lab_1/4/5；lab_2/3 协议为单浓度段', fontsize=9)
ax.legend(fontsize=8)

# (e) CellProperties 覆盖地图
ax = plt.subplot(2, 3, 5)
params = [('Rs', 'Rs'), ('Temp', 'Temp'), ('Cm/CM', None), ('Rm/RM', None), ('Rseal/RSEAL', None), ('VM', 'VM'), ('Passage', 'Passage')]
M = np.zeros((5, len(params)))
for i, lab in enumerate(LABS):
    sub = prop[prop.lab == lab]
    n = len(sub)
    for j, (pname, _) in enumerate(params):
        cols = [c for c in pname.split('/')]
        cnt = sub[cols].notna().any(axis=1).sum()
        M[i, j] = cnt / n * 100 if n else 0
im = ax.imshow(M, cmap='YlGnBu', vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(len(params))); ax.set_xticklabels([p[0] for p in params], rotation=30, ha='right', fontsize=8)
ax.set_yticks(range(5)); ax.set_yticklabels(LABS)
for i in range(5):
    for j in range(len(params)):
        ax.text(j, i, f'{M[i,j]:.0f}', ha='center', va='center', fontsize=8,
                color='white' if M[i, j] > 50 else 'black')
ax.set_title('e  细胞属性覆盖率（%，n=1,893）\nlab_3 全空 ↔ 审稿回应Lab3不做Rs补偿(0%)', fontsize=10)
plt.colorbar(im, ax=ax, shrink=0.8, label='覆盖率 %')

# (f) 漏电流审计
ax = plt.subplot(2, 3, 6)
ok = leak[leak.get('err').isna()] if 'err' in leak.columns else leak
data = [ok[ok.lab == lab]['leak_hold'].abs().values for lab in LABS]
bp = ax.boxplot(data, labels=LABS, showfliers=False, patch_artist=True, widths=0.55)
for patch, lab in zip(bp['boxes'], LABS):
    patch.set_facecolor(LC[lab]); patch.set_alpha(0.7)
ax.set_ylabel('|保持段漏电| (pA)')
ax.set_title('f  raw−subtracted 保持段漏电（五室全有真实减除）\nlab_2 峰值 1.05 nA 属中等——偏移的漏电解释不能排除', fontsize=9)
meds = [np.median(d) for d in data]
for i, m_ in enumerate(meds):
    ax.text(i + 1, m_, f'{m_:.1f}', ha='center', va='bottom', fontsize=8)

fig.suptitle('Fig.12  Stage5 数据彻底清剿：全段提取 · 细胞属性 · 漏电流审计', fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = os.path.join(OUT, 'fig12_Stage5彻底清剿.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print('saved:', out)
