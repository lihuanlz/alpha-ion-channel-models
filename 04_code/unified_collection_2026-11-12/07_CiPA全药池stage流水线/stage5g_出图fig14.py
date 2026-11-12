# -*- coding: utf-8 -*-
"""fig14 三面板：Stage5g 三连分析
(a) 结合动力学：k_obs vs 浓度 log-log，段1/段2分色 —— 一级动力学预言斜率=1，实测≈0 → 灌注限速
(b) 盲药验证：盲/非盲 τ比 分布 + 判线带(0.8-1.25)
(c) Rs-τ稳健性：lab_1/lab_5 室内 Rs vs ref_tau_ms 散点（log x）—— 零相关=τ对钳制误差稳健
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

HERE = Path(__file__).parent

kon = pd.read_csv(HERE / 'stage5g_kon_analysis.csv')
blind = pd.read_csv(HERE / 'stage5g_blind_cells.csv')
rstau = pd.read_csv(HERE / 'stage5g_rs_tau.csv')
s2 = pd.read_csv(HERE / 'stage2_final_cell_metrics.csv')

fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))

# ---------- (a) kon ----------
ax = axes[0]
c1, c2 = '#2176ae', '#e36414'
for seg, col, lab in [(1, c1, '段1（首次给药）'), (2, c2, '段2（序贯高浓度）')]:
    d = kon[kon.seg_idx == seg]
    ax.scatter(d.conc_uM, d.k_obs, s=7, alpha=0.28, color=col, rasterized=True)
    x = np.log10(d.conc_uM.values); y = np.log10(d.k_obs.values)
    m = np.isfinite(x) & np.isfinite(y)
    sl, ic = np.polyfit(x[m], y[m], 1)
    xs = np.linspace(x[m].min(), x[m].max(), 50)
    ax.plot(10**xs, 10**(sl*xs+ic), color=col, lw=2.2,
            label=f'{lab}：斜率={sl:.2f}（n={m.sum()}）')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('浓度 (µM)'); ax.set_ylabel(r'表观速率 $k_{obs}$ (每sweep归一)')
ax.set_title('(a) 结合速率不随浓度标度\n一级动力学预言斜率=1，实测≈0 → 灌注限速，kon 不可提取', fontsize=11)
ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

# ---------- (b) 盲药 ----------
ax = axes[1]
b_tau = blind.tau_ratio.dropna().values
nb_tau = s2[(~s2.blind.astype(bool)) & s2.qc_pass.astype(bool)].tau_ratio.dropna().values
bins = np.linspace(0, 2.5, 51)
ax.hist(nb_tau, bins=bins, density=True, alpha=0.55, color='#6a994e',
        label=f'非盲（n={len(nb_tau)}，中位={np.median(nb_tau):.3f}）')
ax.hist(b_tau, bins=bins, density=True, alpha=0.60, color='#bc4749',
        label=f'盲药（n={len(b_tau)}，中位={np.median(b_tau):.3f}）')
ax.axvspan(0.8, 1.25, color='gray', alpha=0.15)
ax.axvline(1.0, color='k', ls='--', lw=1)
inb_b = ((b_tau>=0.8)&(b_tau<=1.25)).mean()*100
inb_nb = ((nb_tau>=0.8)&(nb_tau<=1.25)).mean()*100
p = mannwhitneyu(b_tau, nb_tau).pvalue
ax.text(0.03, 0.97, f'判线内：盲 {inb_b:.0f}% vs 非盲 {inb_nb:.0f}%\nMann-Whitney p={p:.3f}（效应量小）',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(fc='white', ec='0.7', alpha=0.9))
ax.set_xlabel('τ比（药后/对照）'); ax.set_ylabel('密度')
ax.set_title('(b) 盲药细胞通过同一管线\n分布与非盲一致 → 盲法封顶有效', fontsize=11)
ax.legend(fontsize=9, loc='upper right')

# ---------- (c) Rs-τ ----------
ax = axes[2]
for lab, col in [('lab_1', '#2176ae'), ('lab_5', '#9d4edd')]:
    d = rstau[rstau.lab == lab]
    ax.scatter(d.Rs_all, d.ref_tau_ms, s=9, alpha=0.35, color=col, rasterized=True)
    r, pp = spearmanr(d.Rs_all, d.ref_tau_ms)
    ax.plot([], [], color=col, lw=3, label=f'{lab}：ρ={r:+.3f}（p={pp:.2f}，n={len(d)}）')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('串联电阻 Rs (MΩ)'); ax.set_ylabel(r'对照段 $\tau$ (ms)')
ax.set_title('(c) τ 与 Rs 室内零相关\n钳制误差（∝Rs）不驱动 τ → τ 守恒稳健', fontsize=11)
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.text(0.03, 0.03, 'lab_4 Rs 全为恒定 80 MΩ（名义值，已登记不计）',
        transform=ax.transAxes, fontsize=8, color='0.45')

fig.suptitle('fig14｜Stage5g 三连审计：结合动力学极限 · 盲法验证 · Rs 稳健性', fontsize=13, y=1.02)
fig.tight_layout()
out = HERE / 'fig14_stage5g三连审计.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print('saved:', out)
print(f'(a) 段1 n={ (kon.seg_idx==1).sum() } 段2 n={ (kon.seg_idx==2).sum() }')
print(f'(b) 盲 n={len(b_tau)} 非盲 n={len(nb_tau)} p={p:.4f} 判线内 {inb_b:.0f}%/{inb_nb:.0f}%')
print(f'(c) lab_1 n={(rstau.lab=="lab_1").sum()} lab_5 n={(rstau.lab=="lab_5").sum()}')
