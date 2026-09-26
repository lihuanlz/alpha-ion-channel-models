# -*- coding: utf-8 -*-
"""
EDFig4: five-temperature Q10 layer + 37C cross-host check (Lei temperature series)
Numbers: lei211_温度汇总.json (sealed in 判词卡_Lei温度Q10_2026-09-21.md)
Output: 04_细胞线4/文章/figures_v01/EDFig6_temperature_Q10.png/.pdf
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

BASE = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
AM = os.path.join(BASE, "α模型")
OUT = os.path.join(BASE, "文章", "figures_v01")
os.makedirs(OUT, exist_ok=True)

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 5.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': 1.0, 'savefig.dpi': 300, 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})
BLUE, VERM, GREEN, GREY, BLACK = '#0072B2', '#D55E00', '#009E73', '#8A8A8A', '#222222'
ORANGE, SKY = '#E69F00', '#56B4E9'
def MM(x): return x / 25.4

D = json.load(open(os.path.join(AM, "lei211_温度汇总.json"), encoding="utf-8"))
S = D["各批"]
COLS = ["herg25oc1", "herg27oc1", "herg30oc1", "herg33oc1", "herg37合并"]
T = np.array([25, 27, 30, 33, 37], float)

def series(fam, key, sub=None):
    ys = []
    for b in COLS:
        x = S[b][fam][key]
        if sub: x = x[sub]
        ys.append(x["median"])
    return np.array(ys, float)

pc = []
for b in ("herg37oc3", "herg37oc4"):
    d = json.load(open(os.path.join(AM, f"lei211_tauh_逐孔_{b}.json"), encoding="utf-8"))
    pc += [c["-120"] for c in d.values() if "-120" in c]
pc = np.array(pc) * 1000
q25, q75 = np.percentile(pc, [25, 75])

fig, axes = plt.subplots(1, 4, figsize=(MM(183), MM(50)))
Tt = np.linspace(24, 38, 50)

# ---- a: t_rec Arrhenius (direct line labels, no box) ----
ax = axes[0]
for V, col, mk in (("-120", BLUE, 'o'), ("-140", SKY, 's')):
    ys = series("trec", V) * 1000
    k, c = np.polyfit(T, np.log(ys), 1)
    ax.plot(Tt, np.exp(k * Tt + c), color=col, lw=0.9)
    ax.plot(T, ys, mk, color=col, ms=3.2, mfc='white', mew=1.0)
    ax.annotate(f'{V} mV', xy=(37.4, np.exp(k * 37.6 + c)), fontsize=5.5,
                color=col, va='center')
ax.plot([24, 37], [3.8, 1.1], '*', color=VERM, ms=7, mew=0.8)
ax.annotate('Vandenberg 2006\n(CHO, 24 °C)', xy=(24, 3.8), xytext=(25.6, 1.75),
            fontsize=5.5, color=VERM,
            arrowprops=dict(arrowstyle='-', color=VERM, lw=0.5))
ax.annotate('Vandenberg 2006\n(CHO, 37 °C)', xy=(37.2, 1.1), xytext=(37.6, 2.6),
            fontsize=5.5, color=VERM,
            arrowprops=dict(arrowstyle='-', color=VERM, lw=0.5))
ax.text(0.97, 0.96, 'Q$_{10}$ = 2.84\n$R^2$ = 0.991', transform=ax.transAxes,
        ha='right', va='top', fontsize=6)
ax.set_yscale('log'); ax.set_yticks([1, 2, 5, 10])
ax.set_yticklabels(['1', '2', '5', '10'])
ax.set_xlabel('Temperature (°C)'); ax.set_ylabel(r'$\tau_{rec}$ (ms)')
ax.set_title('a', loc='left', fontweight='bold')
ax.set_xlim(23, 41); ax.set_ylim(0.9, 12)

# ---- b: registration-grade tau layer (direct labels) ----
ax = axes[1]
specs = [("bigstep", "tau2_激活", None, r'$\tau_{act}$ Q$_{10}$=1.50', VERM, 'o', (33.5, 700)),
         ("bigstep", "tau1_失活", None, r'$\tau_{inact}$ Q$_{10}$=1.83', ORANGE, 's', (37.6, 45)),
         ("tdeact", "-40", "tau2", r'$\tau_{deact,slow}$ Q$_{10}$=7.6 $\circ$', GREY, '^', (37.6, 320))]
for fam, key, sub, lab, col, mk, tx in specs:
    ys = series(fam, key, sub) * 1000
    ax.plot(T, ys, mk, color=col, ms=3.2, mfc='white', mew=1.0)
    ax.plot(T, ys, color=col, lw=0.8, ls='--')
    ax.annotate(lab, xy=tx, fontsize=5.5, color=col, va='center',
                ha='left' if tx[0] > 37 else 'center')
ax.set_yscale('log'); ax.set_yticks([10, 100, 1000, 3000])
ax.set_yticklabels(['10', '100', '1000', '3000'])
ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('Time constant (ms)')
ax.set_title('b', loc='left', fontweight='bold')
ax.set_xlim(23, 43); ax.set_ylim(8, 6000)

# ---- c: E_rev(T) ----
ax = axes[2]
eys = np.array([S[b]["E_rev"]["median"] for b in COLS])
k, c = np.polyfit(T, eys, 1)
ax.plot(Tt, k * Tt + c, color=BLACK, lw=0.9)
ax.plot(T, eys, 'o', color=GREEN, ms=3.2, mfc='white', mew=1.0)
ax.plot([37], [-92.9], 'x', color=VERM, ms=5, mew=1.2)
ax.text(0.04, 0.96, rf'slope = {k:.2f} mV/°C $\circ$', transform=ax.transAxes, va='top', fontsize=6)
ax.text(0.03, 0.30, 'per-cell $E_{rev}$\nmedian, CV≤0.034', transform=ax.transAxes,
        va='top', fontsize=5.5, color=GREEN)
ax.annotate('model-fitted\n−92.9 mV @37 °C', xy=(37, -92.9), xytext=(26.5, -94.3),
            fontsize=5.5, color=VERM,
            arrowprops=dict(arrowstyle='-', color=VERM, lw=0.5))
ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('$E_{rev}$ (mV)')
ax.set_title('c', loc='left', fontweight='bold')
ax.set_xlim(23, 39); ax.set_ylim(-95, -82)

# ---- d: cross-host 37C ----
ax = axes[3]
ax.axhspan(3.04 * 0.3, 3.04 * 3, color=GREY, alpha=0.15, lw=0)
ax.axhline(3.04, color=BLACK, lw=0.8, ls=':')
ax.errorbar([1], [1.518], yerr=[[1.518 - q25], [q75 - 1.518]], fmt='o',
            color=BLUE, ms=4.5, mfc='white', mew=1.1, capsize=2.5, elinewidth=0.8)
ax.plot([2], [3.04], 'D', color=VERM, ms=4.5, mfc='white', mew=1.1)
ax.annotate('Lei CHO 37 °C\nn=105, median±IQR', xy=(1, 1.518), xytext=(1.0, 2.0),
            fontsize=5.5, color=BLUE, ha='center', va='bottom')
ax.annotate('Beattie CHO 37 °C\n(sealed)', xy=(2, 3.04), xytext=(1.98, 4.4),
            fontsize=5.5, color=VERM, ha='center')
ax.plot([3], [1.1], '*', color=GREEN, ms=7, mfc='white', mew=0.9)
ax.text(3, 1.32, '1.1 ms', fontsize=5.5, color=GREEN, ha='center', va='bottom')
ax.set_yscale('log'); ax.set_yticks([1, 2, 3.04, 10])
ax.set_yticklabels(['1', '2', '3.04', '10'])
ax.set_xlim(0.4, 3.6); ax.set_ylim(0.55, 12)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['Lei\nCHO', 'Beattie\nCHO', 'Vand.\nCHO'])
ax.set_ylabel(r'$\tau_{rec}$(−120) at 37 °C (ms)')
ax.set_title('d', loc='left', fontweight='bold')
ax.text(0.03, 0.97, 'Lei/Beattie 0.50 in [0.3, 3] sealed;\nVandenberg via Li 2016',
        transform=ax.transAxes, ha='left', va='top', fontsize=6)

fig.suptitle("hERG temperature dependence: a single Q10 of 2.84 from 25 to 37 °C",
             fontsize=8.5, fontweight="bold", y=1.0)
fig.tight_layout(w_pad=1.4)
rcParams["svg.fonttype"] = "none"
for ext in ("png", "svg", "pdf"):
    fp = os.path.join(OUT, f"EDFig6_temperature_Q10.{ext}")
    fig.savefig(fp, bbox_inches='tight')
    print("saved", fp)
