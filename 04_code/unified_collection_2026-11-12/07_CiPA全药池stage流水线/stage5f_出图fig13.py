# -*- coding: utf-8 -*-
"""fig13: lab_2 偏移×漏电定案三面板"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, 'stage5f_leak_vs_block.csv'))
LABS = ['lab_1', 'lab_2', 'lab_3', 'lab_4', 'lab_5']
LC = dict(zip(LABS, ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# (a) lab_2 逐细胞散点：|leak_hold| vs block
ax = axes[0]
s = df[df.lab == 'lab_2']
ax.scatter(s.leak_hold.abs(), s.block_last, s=18, alpha=0.5, color=LC['lab_2'], rasterized=True)
r, p = spearmanr(s.leak_hold.abs(), s.block_last)
ax.set_xscale('symlog', linthresh=1)
ax.set_xlabel('|保持段漏电| (pA)'); ax.set_ylabel('末段阻断分数')
ax.set_title(f'a  lab_2 逐细胞：漏电 vs 阻断\nSpearman ρ={r:+.3f}（p={p:.2f}，n={len(s)}）→ 零相关', fontsize=10)

# (b) 五室相关条形
ax = axes[1]
rhos, ps = [], []
for lab in LABS:
    s = df[df.lab == lab]
    r, p = spearmanr(s.leak_hold.abs(), s.block_last)
    rhos.append(r); ps.append(p)
bars = ax.bar(LABS, rhos, color=[LC[l] for l in LABS], alpha=0.8)
for b, r, p in zip(bars, rhos, ps):
    sig = '***' if p < 0.001 else ('*' if p < 0.05 else 'ns')
    ax.text(b.get_x() + b.get_width()/2, r + 0.008, f'{r:+.2f}{sig}', ha='center', fontsize=9)
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('Spearman ρ（|漏电| vs 阻断）')
ax.set_title('b  五室逐细胞相关：无一室呈"漏电大→阻断浅"\n（lab_5 弱正相关为反方向，登记不解释）', fontsize=10)

# (c) δ/A 幅度账
ax = axes[2]
for lab in LABS:
    s = df[df.lab == lab]['ratio'].replace([np.inf, -np.inf], np.nan).dropna()
    ax.scatter(np.full(len(s), LABS.index(lab)) + np.random.uniform(-0.15, 0.15, len(s)),
               s, s=6, alpha=0.3, color=LC[lab], rasterized=True)
ax.axhline(1.05, color='red', ls='--', lw=1.5)
ax.text(0.02, 0.95, '红线：解释 3.10× IC50 偏移所需 δ/A=1.05\n（假设保持段漏电全部进峰测量——天真上界）',
        transform=ax.transAxes, fontsize=9, va='top')
ax.set_yscale('symlog', linthresh=0.01)
ax.set_ylabel('|leak_hold| / |对照峰幅度|')
med2 = df[df.lab=='lab_2']['ratio'].median()
ax.set_title(f'c  幅度账：lab_2 中位 {med2:.2f}（他室 0.01–0.03）\n幅度上"够"但差分测量自动抵消 → 与(a)零相关互洽', fontsize=10)

fig.suptitle('Fig.13  lab_2 系统性偏移 × 漏电机制：定案（室内零相关 + 差分抵消）', fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(HERE, 'fig13_lab2偏移漏电定案.png')
fig.savefig(out, dpi=200, bbox_inches='tight')
print('saved:', out)
