# -*- coding: utf-8 -*-
# Generate the per-group verdict long table for the SI.
# Input: frozen four-calibre verdict table (194 evaluable rows of 212).
import math

import pandas as pd

v = pd.read_csv(r'D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia\features\robust_verdict_all_profile_2026-11-08.csv')
BS = chr(92) * 2  # LaTeX row terminator \\
NL = chr(10)

name = {'kv_main': 'Kv activation', 'tau_deact': 'Kv deactivation',
        'tau_inact': 'Kv inactivation', 'rec': 'Kv recovery', 'hcn': 'HCN activation'}

head = (r'Layer & Channel & Temp ($^\circ$C) & Ratio (obs) & '
        r'Ratio (shrunk) & Pass ' + BS)

lines = []
lines.append(r'\begin{longtable}{lllrrl}')
lines.append(r'\caption{Per-group conservation verdicts. Ratio (obs) is the raw '
             r'scatter ratio; Ratio (shrunk) is the headline calibre (hierarchical '
             r'partial pooling with profile-likelihood per-row uncertainties); Pass '
             r'marks shrunk ratio above one. Groups with too few valid midpoint fits '
             r'for shrinkage are marked n/a and do not enter the pass counts. '
             r'$^{a}$Raw ratio off scale: bound-hit Boltzmann fits collapse the '
             r'slope normalisation; calibre B removes exactly such fits.}'
             r'\label{tab:verdictlong}' + BS)
lines.append(r'\toprule')
lines.append(head)
lines.append(r'\midrule')
lines.append(r'\endfirsthead')
lines.append(r'\multicolumn{6}{l}{\footnotesize Table \ref{tab:verdictlong} (continued)}'
             + BS)
lines.append(r'\toprule')
lines.append(head)
lines.append(r'\midrule')
lines.append(r'\endhead')
lines.append(r'\midrule\multicolumn{6}{r}{\footnotesize continued on next page}' + BS)
lines.append(r'\endfoot')
lines.append(r'\bottomrule')
lines.append(r'\endlastfoot')

first = True
for tb, g in v.groupby('table'):
    if not first:
        lines.append(r'\midrule')
    first = False
    g = g.sort_values(['channel', 'temp'])
    for _, r in g.iterrows():
        if math.isnan(r.ratio_obs):
            obs = 'n/a'
        elif r.ratio_obs > 9999:
            obs = r'off-scale$^{a}$'
        else:
            obs = '%.2f' % r.ratio_obs
        if math.isnan(r.ratio_D):
            shr, p = 'n/a', 'n/a'
        else:
            shr, p = '%.2f' % r.ratio_D, ('yes' if r.ratio_D > 1 else 'no')
        ch = str(r.channel).replace('_', '\\_')
        lines.append('%s & %s & %g & %s & %s & %s %s' % (
            name.get(tb, tb), ch, r.temp, obs, shr, p, BS))
lines.append(r'\end{longtable}')

out = r'D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\latex_v2_unified\si_verdict_table.tex'
open(out, 'w', encoding='utf-8').write(NL.join(lines) + NL)
print('rows:', len(v), '->', out)
