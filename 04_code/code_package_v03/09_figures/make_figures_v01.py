# -*- coding: utf-8 -*-
"""
Nature-style figures v01 for:
"Data-extracted gating kinetics yield parameter-efficient, audit-ready models
 of four cardiac ion channels"
All numbers come from the sealed model cards / judgement-card result files.
Output: 04_细胞线4/文章/figures_v01/Fig*.png + .pdf (300 dpi)
"""
import json, csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import stats

BASE = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
AM = os.path.join(BASE, "α模型")
W182 = os.path.join(BASE, "w182", "代码182_model62可识别性谱卡_16713003_2026-09-11",
                    "model62_identifiability")
OUT = os.path.join(BASE, "文章", "figures_v01")
os.makedirs(OUT, exist_ok=True)

# ---------- Nature style ----------
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': 1.0, 'savefig.dpi': 300, 'pdf.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
})
BLUE, VERM, GREEN, PINK = '#0072B2', '#D55E00', '#009E73', '#CC79A7'
ORANGE, SKY, GREY, BLACK = '#E69F00', '#56B4E9', '#8A8A8A', '#222222'

def MM(x): return x / 25.4

def plabel(ax, s, dx=-0.16, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8,
            fontweight='bold', va='top', ha='left')

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('saved', name)

def jload(p):
    with open(p, encoding='utf-8') as f: return json.load(f)

def csvload(p):
    with open(p, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def fnum(x):
    try: return float(x)
    except Exception: return np.nan

# =====================================================================
# Fig. 1 — four channels on the membrane, gating states, measured tables,
#           and the extract–judge–assemble–predict pipeline (schematic)
# =====================================================================
def fig1():
    from matplotlib.patches import Ellipse
    fig = plt.figure(figsize=(MM(183), MM(125)))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    LIPID = '#B9C6D8'
    yT, yB, yM = 0.800, 0.672, 0.736          # lipid head rows + midplane
    chans_x = [(0.06, 0.24), (0.30, 0.47), (0.53, 0.70), (0.76, 0.94)]

    # ---------- lipid bilayer ----------
    ax.axhspan(yB - 0.006, yT + 0.006, color='#8FA3BE', alpha=0.10, zorder=0)
    xs = np.arange(0.008, 1.0, 0.0095)
    def in_chan(x): return any(a - 0.013 < x < b + 0.013 for a, b in chans_x)
    hx = [x for x in xs if not in_chan(x)]
    ax.plot(hx, [yT] * len(hx), 'o', ms=3.1, color='#9AABC4', mec='none', zorder=1)
    ax.plot(hx, [yB] * len(hx), 'o', ms=3.1, color='#9AABC4', mec='none', zorder=1)
    sx, syt, syb = [], [], []
    for x in hx:
        sx += [x, x, np.nan]; syt += [yT, yM, np.nan]; syb += [yB, yM, np.nan]
    ax.plot(sx, syt, '-', color='#9AABC4', lw=0.7, zorder=1)
    ax.plot(sx, syb, '-', color='#9AABC4', lw=0.7, zorder=1)
    ax.text(0.004, 0.845, 'extracellular', fontsize=5.6, color=GREY, style='italic')
    ax.text(0.004, 0.628, 'intracellular', fontsize=5.6, color=GREY, style='italic')

    def tm_rect(x, w=0.016, y0=0.656, y1=0.816, fc='white', ec=BLACK):
        ax.add_patch(FancyBboxPatch((x - w / 2, y0), w, y1 - y0,
                     boxstyle='round,pad=0,rounding_size=0.004',
                     fc=fc, ec=ec, lw=0.9, zorder=4))

    def ion_arrow(x, up, col):
        xy = (x, 0.848) if up else (x, 0.624)
        xyt = (x, 0.624) if up else (x, 0.848)
        ax.annotate('', xy=xy, xytext=xyt,
                    arrowprops=dict(arrowstyle='-|>', mutation_scale=9,
                                    lw=1.1, color=col), zorder=5)

    # ---------- hERG ----------
    xc = 0.15
    ax.text(xc, 0.878, 'hERG (K$_{v}$11.1)', ha='center', fontsize=7,
            fontweight='bold')
    ax.text(xc, 0.848, 'IKr', ha='center', fontsize=5.6, color=GREY)
    for dx in (-0.030, -0.0105, 0.0105, 0.030):
        tm_rect(xc + dx, fc='#DCE9F7', ec=BLUE)
    ion_arrow(xc, True, BLUE)
    ax.plot([xc - 0.004, xc, xc + 0.004], [0.70, 0.755, 0.70], 'o', ms=2.2,
            color=BLUE, zorder=5)
    ax.text(xc + 0.044, 0.745, 'K$^+$', fontsize=6, color=BLUE)

    # ---------- Nav1.5 ----------
    xc = 0.385
    ax.text(xc, 0.878, 'Nav1.5', ha='center', fontsize=7, fontweight='bold')
    ax.text(xc, 0.848, 'INa', ha='center', fontsize=5.6, color=GREY)
    doms = [-0.033, -0.011, 0.011, 0.033]
    for i, dx in enumerate(doms):
        tm_rect(xc + dx, w=0.017, fc='#FBE3D6', ec=VERM)
        ax.text(xc + dx, 0.662, ['I', 'II', 'III', 'IV'][i], ha='center',
                fontsize=4.2, color=VERM, zorder=5)
    ax.plot([xc - 0.033, xc + 0.033], [0.652, 0.652], '-', color=VERM, lw=0.8,
            zorder=4)
    ax.plot([xc + 0.011, xc + 0.004], [0.652, 0.618], ':', color=VERM, lw=0.9,
            zorder=4)
    ax.add_patch(Ellipse((xc + 0.004, 0.608), 0.017, 0.017 * 183 / 125,
                         fc=VERM, ec='none', zorder=5))
    ax.text(xc + 0.020, 0.596, 'IFM', fontsize=4.6, color=VERM)
    ion_arrow(xc, False, VERM)
    ax.text(xc + 0.046, 0.72, 'Na$^+$', fontsize=6, color=VERM)

    # ---------- CaV1.2 ----------
    xc = 0.62
    ax.text(xc, 0.878, 'CaV1.2', ha='center', fontsize=7, fontweight='bold')
    ax.text(xc, 0.848, 'ICaL', ha='center', fontsize=5.6, color=GREY)
    for dx in doms:
        tm_rect(xc + dx, w=0.017, fc='#DFF0E8', ec=GREEN)
    ax.add_patch(Ellipse((xc + 0.033, 0.612), 0.034, 0.034 * 183 / 125,
                         fc='#DFF0E8', ec=GREEN, lw=0.9, zorder=4))
    ax.text(xc + 0.033, 0.612, 'CaM', fontsize=4.2, color=GREEN, ha='center',
            va='center', zorder=5)
    ion_arrow(xc, False, ORANGE)
    ax.text(xc + 0.052, 0.72, r'$\mathrm{Ca^{2+}}$', fontsize=6, color=ORANGE)

    # ---------- IKs ----------
    xc = 0.855
    ax.text(xc, 0.878, 'IKs (KCNQ1 + KCNE1)', ha='center', fontsize=7,
            fontweight='bold')
    for dx in (-0.030, -0.0105, 0.0105, 0.030):
        tm_rect(xc + dx, fc='#F3E4F0', ec=PINK)
    tm_rect(xc + 0.053, w=0.010, fc='#F3E4F0', ec=PINK)
    ax.text(xc + 0.067, 0.736, 'KCNE1', fontsize=4.2, color=PINK, ha='left',
            va='center', rotation=90)
    ion_arrow(xc, True, PINK)
    ax.text(xc - 0.052, 0.745, 'K$^+$', fontsize=6, color=PINK, ha='right')

    # ---------- gating-state schemes ----------
    ax.text(0.004, 0.585, 'channel states &\nmeasured quantities', fontsize=5.6,
            color=GREY, style='italic', va='center')

    def state(x, y, s, fc):
        ax.add_patch(Ellipse((x, y), 0.021, 0.021 * 183 / 125, fc=fc,
                             ec=BLACK, lw=0.7, zorder=4))
        ax.text(x, y, s, ha='center', va='center', fontsize=5.6, zorder=5)

    def sarrow(x0, y0, x1, y1, both=True):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='<|-|>' if both else '-|>',
                                    mutation_scale=7, lw=0.8, color=BLACK),
                    zorder=3)

    yS = 0.545
    # hERG: C <-> O <-> I
    xc = 0.15
    state(xc - 0.045, yS, 'C', '#DCE9F7'); state(xc, yS, 'O', '#DCE9F7')
    state(xc + 0.045, yS, 'I', '#DCE9F7')
    sarrow(xc - 0.032, yS, xc - 0.013, yS); sarrow(xc + 0.013, yS, xc + 0.032, yS)
    ax.text(xc, 0.470, 'τ_deact(V): 3 components, channel constants\n'
                       'h_ss(V): 12-level measured table\n'
                       'E_rev = −88.33 mV (single value)',
            ha='center', va='top', fontsize=5.2, color=BLACK)
    # Nav1.5: C <-> O -> I
    xc = 0.385
    state(xc - 0.045, yS, 'C', '#FBE3D6'); state(xc, yS, 'O', '#FBE3D6')
    state(xc + 0.045, yS, 'I', '#FBE3D6')
    sarrow(xc - 0.032, yS, xc - 0.013, yS); sarrow(xc + 0.013, yS, xc + 0.032, yS,
                                                   both=False)
    ax.text(xc, 0.470, 'τ_h(V): a distribution (CV 0.772)\n'
                       '→ per-cell anchoring · V½h per cell (±4 mV)\n'
                       'k_h constant (CV 0.029)',
            ha='center', va='top', fontsize=5.2, color=BLACK)
    # CaV1.2: O -> I(VDI) ; O --Ca--> I(CDI)
    xc = 0.62
    state(xc - 0.045, yS, 'O', '#DFF0E8')
    state(xc + 0.045, yS + 0.028, 'I', '#DFF0E8')
    state(xc + 0.045, yS - 0.028, 'I', '#DFF0E8')
    sarrow(xc - 0.032, yS + 0.008, xc + 0.030, yS + 0.026, both=False)
    sarrow(xc - 0.032, yS - 0.008, xc + 0.030, yS - 0.026, both=False)
    ax.text(xc + 0.062, yS + 0.030, 'VDI', fontsize=4.6, color=GREEN, ha='left')
    ax.text(xc + 0.062, yS - 0.030, 'CDI', fontsize=4.6, color=GREEN, ha='left')
    ax.text(xc, 0.470, 'f_inact family constants:\n'
                       '$\\mathrm{Ba^{2+}}$ 0.44 · $\\mathrm{Ca^{2+}}$ 0.73\n'
                       'CDI = one additive measured term (w = 0.289)',
            ha='center', va='top', fontsize=5.2, color=BLACK)
    # IKs: C --delay--> O
    xc = 0.855
    state(xc - 0.045, yS, 'C', '#F3E4F0'); state(xc + 0.045, yS, 'O', '#F3E4F0')
    sarrow(xc - 0.032, yS, xc + 0.032, yS, both=False)
    ax.text(xc, yS + 0.035, 'delay d(V)', fontsize=5.0, ha='center', color=PINK)
    ax.text(xc, 0.470, 'τ_act(V): bell-shaped, anchored\n'
                       'delay d(V): measured table\n'
                       'τ_deact: real spread (CV 0.313)',
            ha='center', va='top', fontsize=5.2, color=BLACK)

    # ---------- pipeline strip ----------
    ax.plot([0.02, 0.98], [0.318, 0.318], color='#DDDDDD', lw=0.7, zorder=1)
    ax.text(0.004, 0.290, 'model-building pipeline', fontsize=5.6, color=GREY,
            style='italic', va='center')
    boxes = [
        (0.02, 'EXTRACT', ['protocol-rich recordings', '→ voltage-dependent tables',
                           'no global fit'], BLUE),
        (0.28, 'JUDGE', ['pre-registered constancy test', 'constant / per-cell /',
                         'distributed / unusable'], VERM),
        (0.54, 'ASSEMBLE', ['Hodgkin–Huxley form', 'measured tables inserted',
                            'per-cell free: G (+ΔV½h)'], GREEN),
        (0.78, 'PREDICT', ['held-out protocols & cells', 'sine · AP clamp · CiPA',
                           'failure → one table, one voltage'], PINK),
    ]
    for x0, title, lines, col in boxes:
        bb = FancyBboxPatch((x0, 0.085), 0.185, 0.15,
                            boxstyle='round,pad=0.010,rounding_size=0.012',
                            fc=col + '18', ec=col, lw=1.1, zorder=2)
        ax.add_patch(bb)
        ax.text(x0 + 0.0925, 0.205, title, ha='center', fontsize=7,
                fontweight='bold', color=col, zorder=3)
        for i, ln in enumerate(lines):
            ax.text(x0 + 0.0925, 0.172 - 0.030 * i, ln, ha='center',
                    fontsize=5.2, color=BLACK, zorder=3)
        if x0 < 0.7:
            ax.add_patch(FancyArrowPatch((x0 + 0.200, 0.16), (x0 + 0.262, 0.16),
                                         arrowstyle='-|>', mutation_scale=12,
                                         lw=1.3, color=BLACK, zorder=3))
    # feeder: measured quantities -> pipeline
    ax.add_patch(FancyArrowPatch((0.1125, 0.318), (0.1125, 0.242),
                                 arrowstyle='-|>', mutation_scale=9,
                                 lw=0.8, color='#999999', ls='--', zorder=1))
    ax.text(0.02, 0.048, 'load-bearing property 1:  every table entry is traceable to named cells & sweeps (provenance)',
            fontsize=5.8, color=BLACK)
    ax.text(0.02, 0.020, 'load-bearing property 2:  cell-to-cell differences are measured and reported, not averaged away',
            fontsize=5.8, color=BLACK)
    save(fig, 'Fig1_pipeline')

# =====================================================================
# Fig. 2 — hERG
# =====================================================================
def fig2():
    amp = jload(os.path.join(AM, '2026-09-13_α模型_幅度表提取_结果.json'))
    hss = jload(os.path.join(AM, '2026-09-13_α模型_纯hssV剥离判决_结果.json'))
    sine = jload(os.path.join(AM, '2026-09-13_α模型_hss实测表重跑_sine前向判决_结果.json'))
    ap = jload(os.path.join(AM, '2026-09-13_α模型_hss实测表重跑_AP前向判决_结果.json'))
    c5 = csvload(os.path.join(AM, '2026-09-14_α模型_药物卡5_形状不变性判决_单元表.csv'))
    c6 = csvload(os.path.join(AM, '2026-09-14_α模型_药物卡6_形状变形分解_单元表.csv'))

    fig = plt.figure(figsize=(MM(183), MM(118)))
    gs = fig.add_gridspec(3, 6, hspace=0.85, wspace=0.9,
                          left=0.07, right=0.98, top=0.90, bottom=0.09)

    # a: model equation panel
    axa = fig.add_subplot(gs[0, :2]); axa.axis('off')
    axa.text(0.5, 0.92, r'$\mathrm{hERG}\;\; I = G\cdot m\cdot h\cdot (V - E_{rev})$',
             ha='center', fontsize=9)
    axa.text(0.5, 0.72, r'$E_{rev} = -88.33$ mV  (single value, all protocols & cells)',
             ha='center', fontsize=6.5)
    axa.text(0.5, 0.48, 'every voltage dependence = measured lookup table\n'
                        '(log-linear interpolation, endpoint clamp)\n'
                        'no rate equations · no topology · zero tuning',
             ha='center', va='top', fontsize=6.2, color=BLACK)
    axa.text(0.5, 0.08, 'per-cell free parameters:  G only',
             ha='center', fontsize=7, fontweight='bold', color=BLUE)
    plabel(axa, 'a', dx=-0.06, dy=1.02)

    # b1: deactivation tau tables
    axb = fig.add_subplot(gs[0, 2:4])
    tau = amp['tau_used']
    Vs = [-70, -60, -50, -40]
    for key, col, lab in [('TF', SKY, r'$\tau_f$ (fast)'), ('TM', BLUE, r'$\tau_m$ (middle)'),
                          ('TAU_L', VERM, r'$\tau_{late}$ (slow)')]:
        ys = [tau[key][str(v)] for v in Vs]
        axb.semilogy(Vs, ys, 'o-', color=col, ms=3.5, lw=1.2, label=lab)
    axb.set_xlabel('V (mV)'); axb.set_ylabel(r'deactivation $\tau$ (s)')
    axb.set_xticks(Vs)
    axb.legend(frameon=False, loc='upper left', handlelength=1.2)
    axb.text(0.40, 0.24, 'cross-cell CV 0.13–0.19\n→ channel-level constants',
             transform=axb.transAxes, fontsize=6, color=GREEN,
             ha='left', va='bottom')
    plabel(axb, 'b')

    # b2: h_ss measured 12-level table
    axh = fig.add_subplot(gs[0, 4:])
    vgrid = sorted(int(v) for v in hss['gears'].keys())
    for c, d in hss['cells'].items():
        vv, yy = [], []
        for v in vgrid:
            val = d['curve'].get(str(v))
            if val is not None:
                vv.append(v); yy.append(fnum(val))
        axh.plot(vv, yy, '-', color=GREY, lw=0.5, alpha=0.55, zorder=1)
    med = [hss['gears'][str(v)]['med'] for v in vgrid]
    axh.plot(vgrid, med, 'o-', color=BLUE, ms=3.5, lw=1.4, zorder=3,
             label='population median (measured)')
    axh.axhline(1.0, color=VERM, ls='--', lw=0.9, zorder=2)
    axh.text(-28, 1.045, 'B1 assumption  h = 1  — refuted', color=VERM, fontsize=6)
    axh.annotate('h(−70) = 0.425', xy=(-70, 0.425), xytext=(-38, 0.55),
                 fontsize=6, color=BLUE,
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.7))
    axh.set_xlabel('V (mV)'); axh.set_ylabel(r'$h_{ss}$ (steady-state availability)')
    axh.set_ylim(-0.02, 1.16)
    axh.legend(frameon=False, loc='lower left')
    plabel(axh, 'c')

    # c: forward prediction — predicted vs measured
    axc = fig.add_subplot(gs[1, :3])
    for c, d in sine['cells'].items():
        ok = d.get('pass', False)
        axc.plot(d['mea40'], d['sim40'], 'o', ms=4,
                 color=GREEN if ok else VERM,
                 markerfacecolor='none' if not ok else GREEN, zorder=3)
    for c, d in ap['cells'].items():
        ok = bool(d.get('A1')) and bool(d.get('A2'))
        axc.plot(d['hook_meaA'], d['hook_simA'], 's', ms=4,
                 color=GREEN if ok else VERM,
                 markerfacecolor='none' if not ok else GREEN, zorder=3)
    lo, hi = 0.02, 3.0
    axc.plot([lo, hi], [lo, hi], '-', color=BLACK, lw=0.8)
    axc.set_xscale('log'); axc.set_yscale('log')
    axc.set_xlim(lo, hi); axc.set_ylim(lo, hi)
    axc.set_xlabel('measured (nA)'); axc.set_ylabel('predicted (nA)')
    axc.plot([], [], 'o', color=GREEN, ms=4, label='sinusoid protocol (7/9 pass)')
    axc.plot([], [], 's', color=GREEN, ms=4, label='AP clamp (8/9 pass)')
    axc.plot([], [], 'o', color=VERM, ms=4, mfc='none', label='fail (attributed, on record)')
    axc.legend(frameon=False, loc='upper left', handlelength=1.0)
    plabel(axc, 'd')

    # d: CiPA labs
    axd = fig.add_subplot(gs[1, 3:])
    labs = ['lab 3', 'lab 4', 'lab 5']
    pas = [20, 16, 29]; tot = [20, 16, 32]
    frac = [p / t for p, t in zip(pas, tot)]
    bars = axd.bar(labs, frac, color=[BLUE, BLUE, BLUE], width=0.6)
    for b, p, t in zip(bars, pas, tot):
        axd.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f'{p}/{t}', ha='center', fontsize=7, color=BLACK)
    axd.set_ylim(0, 1.15); axd.set_ylabel('fraction reproduced')
    axd.set_title('CiPA drug-screening traces: 65/68 total', fontsize=7)
    plabel(axd, 'e')

    # e: drug eps_x per drug
    axe = fig.add_subplot(gs[2, :3])
    per = {}
    for r in c5:
        drug = r['dataset'].split('|')[1]
        per.setdefault(drug, []).append(fnum(r['eps_x']))
    drugs = sorted(per, key=lambda d: np.nanmedian(per[d]))
    meds = [np.nanmedian(per[d]) for d in drugs]
    cols = [BLUE if m > 0.030 else GREY for m in meds]
    axe.bar(range(len(drugs)), meds, color=cols, width=0.75)
    axe.axhline(0.030, color=VERM, ls='--', lw=1.0, label='acceptance threshold 0.030')
    axe.axhline(0.004, color=GREY, ls=':', lw=0.9, label='noise floor 0.004')
    axe.legend(frameon=False, loc='upper left', fontsize=5.6, handlelength=1.6)
    axe.set_xticks(range(len(drugs)))
    axe.set_xticklabels([d[:9] for d in drugs], rotation=90, fontsize=4.2)
    axe.set_ylabel(r'median $\epsilon_x$ per drug')
    axe.set_title('pure-block hypothesis rejected: 30/30 drugs above line',
                  fontsize=7)
    plabel(axe, 'f')

    # f: kappa deviation histogram
    axf = fig.add_subplot(gs[2, 3:])
    kap = [fnum(r['kappa']) for r in c6 if str(r.get('gated')) == 'True']
    dev = np.abs(np.array(kap) - 1.0)
    dev = dev[np.isfinite(dev)]
    axf.hist(dev, bins=np.linspace(0, 0.35, 36), color=BLUE, alpha=0.75)
    yl = axf.get_ylim()[1]
    axf.axvline(0.020, color=GREEN, lw=1.2)
    axf.axvline(0.10, color=VERM, ls='--', lw=1.0)
    axf.annotate('median 0.020', xy=(0.020, yl * 0.55), xytext=(0.115, yl * 0.82),
                 fontsize=6, color=GREEN,
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=0.7))
    axf.annotate('threshold 0.10', xy=(0.10, yl * 0.30), xytext=(0.155, yl * 0.52),
                 fontsize=6, color=VERM,
                 arrowprops=dict(arrowstyle='->', color=VERM, lw=0.7))
    axf.set_xlabel(r'steady-state shape deviation  $|\kappa^* - 1|$')
    axf.set_ylabel('cell–concentration units')
    axf.set_title('drug action is amplitude-type (occupancy), not kinetic', fontsize=7)
    plabel(axf, 'g')

    fig.suptitle('hERG: data-extracted constants and tables, tested by held-out forward prediction',
                 fontsize=8.5, fontweight='bold', y=1.005)
    save(fig, 'Fig2_hERG')

# =====================================================================
# Fig. 3 — cross-channel portability verdicts
# =====================================================================
def fig3():
    na4 = csvload(os.path.join(AM, '2026-09-16_Nα4_单通道负40单点锚定判决_结果.csv'))
    bell = csvload(os.path.join(AM, '2026-09-15_Kα2_Fedida2024_Fig3B_wtEQ_数字化.csv'))
    dtj = jload(os.path.join(AM, '2026-09-15_Kα2_Δt登记.json'))
    ka3 = csvload(os.path.join(AM, '2026-09-15_Kα3_IKs前向验证_逐细胞表_结果.csv'))

    fig = plt.figure(figsize=(MM(183), MM(112)))
    gs = fig.add_gridspec(2, 6, hspace=0.9, wspace=1.0,
                          left=0.07, right=0.98, top=0.88, bottom=0.11)

    # a: verdict matrix
    axa = fig.add_subplot(gs[0, :3]); axa.axis('off')
    channels = ['hERG', 'Nav1.5', 'CaV1.2', 'IKs']
    quants = ['kinetic τ', 'steady-state slope', 'steady-state midpoint',
              'reversal potential', 'delay']
    # verdict codes: 0 constant (green), 1 per-cell (orange), 2 distributed (purple), 3 closed (grey)
    M = np.array([
        [0, 2, 1, 0, 0],   # hERG
        [0, 0, 1, 2, 3],   # Nav
        [0, 3, 3, 0, 3],   # CaV
        [0, 0, 0, 2, 0],   # IKs
    ])
    cmap = {0: GREEN, 1: ORANGE, 2: PINK, 3: '#BBBBBB'}
    lab = {0: 'constant', 1: 'per-cell', 2: 'distributed /\nregistered', 3: 'closed / n.a.'}
    for i in range(4):
        for j in range(5):
            axa.add_patch(plt.Rectangle((j, 3 - i), 0.92, 0.86,
                                        fc=cmap[M[i, j]], ec='white', lw=1.5))
    for i, cname in enumerate(channels):
        axa.text(-0.12, 3 - i + 0.43, cname, ha='right', va='center', fontsize=7)
    for j, q in enumerate(quants):
        axa.text(j + 0.46, 4.12, q, ha='center', va='bottom', fontsize=6, rotation=18)
    for k, (code, txt) in enumerate(lab.items()):
        axa.add_patch(plt.Rectangle((0.1 + 1.25 * k, -0.95), 0.18, 0.28,
                                    fc=cmap[code], ec='none'))
        axa.text(0.32 + 1.25 * k, -0.81, txt, fontsize=5.4, va='center')
    axa.text(-1.55, -1.55, 'hERG τ CV 0.13–0.19 · Nav τ_h(−30) CV 0.029 · Nav V½h ±4 mV per cell',
             fontsize=5.6, color=BLACK)
    axa.text(-1.55, -1.88, 'IKs τ_deact CV 0.313 (real spread, reported not averaged)',
             fontsize=5.6, color=BLACK)
    axa.set_xlim(-1.6, 5.1); axa.set_ylim(-2.15, 5.0)
    axa.set_title('portability verdicts, four channels (pre-registered)',
                  fontsize=7, pad=38)
    plabel(axa, 'a', dx=-0.02, dy=1.13)

    # b: Nav tau_h(-40) distribution
    axb = fig.add_subplot(gs[0, 3:])
    taus = [fnum(r['value']) for r in na4 if r['metric'] == 'tau_decay_ms']
    taus = np.array([t for t in taus if np.isfinite(t)])
    bins = np.logspace(np.log10(0.25), np.log10(8.0), 26)
    axb.hist(taus, bins=bins, color=BLUE, alpha=0.8)
    axb.set_xscale('log')
    axb.axvline(1.03, color=BLACK, lw=1.0)
    axb.text(1.07, axb.get_ylim()[1] * 0.92, 'median 1.03 ms', fontsize=6)
    for v in [1.64, 1.68, 3.15]:
        axb.axvline(v, color=VERM, lw=0.8, ls='--')
    axb.text(3.3, axb.get_ylim()[1] * 0.72, 'whole-cell trio\n(sits in upper half)', fontsize=5.6, color=VERM)
    axb.set_xlabel(r'Nav1.5  $\tau_h$(−40 mV) across 69 patches (ms)')
    axb.set_ylabel('patches')
    axb.set_title('a distribution, not a number (CV 0.772) → N-arm closed', fontsize=7)
    plabel(axb, 'b')

    # c: CaV CDI additive component
    axc = fig.add_subplot(gs[1, :2])
    t = np.linspace(0, 100, 400)
    f_vdi = 0.44 / (1 - np.exp(-40 / 41.6))   # calibrated so Ba inactivated fraction at 40 ms = 0.44
    h_ba = 1 - f_vdi * (1 - np.exp(-t / 41.6))
    h_ca = 1 - f_vdi * (1 - np.exp(-t / 50.0)) - 0.289 * (1 - np.exp(-t / 8.0))
    axc.plot(t, 1 - h_ba, color=GREY, lw=1.4, label=r'Ba$^{2+}$ (VDI skeleton)')
    axc.plot(t, 1 - h_ca, color=VERM, lw=1.4, label=r'Ca$^{2+}$ (+ CDI fast term)')
    axc.fill_between(t, 1 - h_ba, 1 - h_ca, color=VERM, alpha=0.18)
    axc.annotate('', xy=(40, 0.73), xytext=(40, 0.44),
                 arrowprops=dict(arrowstyle='<->', color=BLACK, lw=0.8))
    axc.text(50, 0.26, r'$\Delta f$ = 0.293' + '\n' + r'$w_{fast}$ = 0.289',
             fontsize=6)
    axc.set_xlabel('t at +17 mV (ms)'); axc.set_ylabel('inactivated fraction')
    axc.legend(frameon=False, loc='upper left')
    axc.set_title('CaV1.2: CDI = "add one measured term"', fontsize=7)
    plabel(axc, 'c')

    # d: f_inact bars
    axd = fig.add_subplot(gs[1, 2:4])
    g = [r'$\mathrm{Ba^{2+}}$ (n=4)', r'$\mathrm{Ca^{2+}}$ (n=6)']
    med = [0.44, 0.73]; cv = [0.101, 0.141]
    sd = [m * c for m, c in zip(med, cv)]
    axd.bar(g, med, yerr=sd, capsize=3, color=[GREY, VERM], width=0.5,
            error_kw=dict(lw=0.9))
    for i, (m, s) in enumerate(zip(med, sd)):
        axd.text(i, m + s + 0.015, f'{m:.2f} (CV {cv[i]:.2f})', ha='center', fontsize=6.2)
    axd.set_ylabel(r'$f_{inact}$ (+17 mV, 40 ms)')
    axd.set_ylim(0, 0.95)
    axd.set_title('inactivated fraction by charge carrier\n(sealed family constants)', fontsize=7)
    plabel(axd, 'd')

    # e: IKs bell curve + delay inset + per-cell R2
    axe = fig.add_subplot(gs[1, 4:])
    V = [fnum(r['V_mV']) for r in bell]; T = [fnum(r['tau_s']) for r in bell]
    axe.plot(V, T, 'o-', color=BLUE, ms=3, lw=1.2)
    axe.set_yscale('log')
    axe.annotate('peak ≈ 8.9 s @ −10 mV', xy=(-10, 8.92), xytext=(-52, 11.0),
                 fontsize=6, arrowprops=dict(arrowstyle='->', lw=0.7))
    axe.set_xlabel('V (mV)'); axe.set_ylabel(r'IKs $\tau_{act}$ (s)')
    axe.set_title('IKs: bell-shaped activation (Fedida 2024 Fig. 3B, digitised)',
                  fontsize=7)
    axin = axe.inset_axes([0.58, 0.50, 0.38, 0.42])
    dv = sorted((int(k), v) for k, v in dtj['dt_full_read'].items())
    axin.plot([d[0] for d in dv], [d[1] for d in dv], 's-', color=GREEN, ms=2.5, lw=0.9)
    axin.set_xlabel('V (mV)', fontsize=5); axin.set_ylabel('delay (s)', fontsize=5)
    axin.tick_params(labelsize=4.5)
    axin.set_title('activation delay d(V)', fontsize=5.5)
    r2 = [fnum(r['r2_act']) for r in ka3 if r['arm'] == 'A']
    axe.text(0.02, 0.06, f'forward prediction R² ≥ 0.80 in 7/8 cells\n'
                         f'(per-cell R²: {", ".join(f"{x:.2f}" for x in sorted(r2, reverse=True))})',
             transform=axe.transAxes, fontsize=5.6, color=GREEN)
    plabel(axe, 'e')

    save(fig, 'Fig3_cross_channel')

# =====================================================================
# Fig. 4 — drug-action grammar
# =====================================================================
def fig4():
    c6 = csvload(os.path.join(AM, '2026-09-14_α模型_药物卡6_形状变形分解_单元表.csv'))
    nav = csvload(os.path.join(AM, '2026-09-15_Nα药_Nav15药物形状调制判决_逐条件表_结果.csv'))
    cav = jload(os.path.join(AM, '2026-09-15_Cα5_CaV12药物形状调制判决_结果.json'))
    iks = csvload(os.path.join(AM, '2026-09-15_Kα药_IKs药物表分解判决_逐文件表_结果.csv'))

    fig = plt.figure(figsize=(MM(183), MM(100)))
    gs = fig.add_gridspec(1, 4, wspace=0.75, left=0.06, right=0.985,
                          top=0.82, bottom=0.16)

    # a: hERG kappa deviation histogram
    axa = fig.add_subplot(gs[0, 0])
    kap = [fnum(r['kappa']) for r in c6 if str(r.get('gated')) == 'True']
    dev = np.abs(np.array(kap) - 1.0); dev = dev[np.isfinite(dev)]
    axa.hist(dev, bins=np.linspace(0, 0.35, 30), color=BLUE, alpha=0.8)
    yla = axa.get_ylim()[1]
    axa.axvline(0.020, color=GREEN, lw=1.1)
    axa.axvline(0.10, color=VERM, ls='--', lw=0.9)
    axa.annotate('median 0.020', xy=(0.020, yla * 0.55), xytext=(0.10, yla * 0.80),
                 fontsize=6, color=GREEN,
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=0.7))
    axa.annotate('threshold 0.10', xy=(0.10, yla * 0.28), xytext=(0.155, yla * 0.52),
                 fontsize=6, color=VERM,
                 arrowprops=dict(arrowstyle='->', color=VERM, lw=0.7))
    axa.set_xlabel(r'$|\kappa^* - 1|$'); axa.set_ylabel('units')
    axa.set_title('hERG\nsingle amplitude component\n(|κ*−1| = 0.020; warp ≈ 15%)', fontsize=6.5)
    plabel(axa, 'a', dx=-0.28, dy=1.22)

    # b: Nav kappa vs df
    axb = fig.add_subplot(gs[0, 1])
    ks, dfs = [], []
    for r in nav:
        k, d = fnum(r['kappa']), fnum(r['df'])
        if np.isfinite(k) and np.isfinite(d):
            ks.append(k); dfs.append(d)
    axb.scatter(ks, dfs, s=7, color=BLUE, alpha=0.6, edgecolors='none')
    axb.axhline(0, color=BLACK, lw=0.5); axb.axvline(0, color=BLACK, lw=0.5)
    # sealed reference values (from the sealed model card), not raw-table medians
    axb.axvline(0.20, color=GREEN, ls='--', lw=0.8)
    axb.axvline(-0.20, color=GREEN, ls='--', lw=0.8)
    axb.axhline(0.12, color=GREEN, ls=':', lw=0.8)
    x0, x1 = axb.get_xlim(); y0, y1 = axb.get_ylim()
    axb.text(0.215, y1 - 0.04 * (y1 - y0), 'sealed |κ*| = 0.20', fontsize=5.4,
             color=GREEN, va='top', ha='left', rotation=90)
    axb.text(x0 + 0.03 * (x1 - x0), 0.16, 'use-dependence Δf = +0.12',
             fontsize=5.4, color=GREEN, va='bottom', ha='left')
    axb.set_xlabel(r'$\kappa$ (steady-state shape)'); axb.set_ylabel(r'$\Delta f$ (occupancy shift)')
    axb.set_title('Nav1.5\ndual component: amplitude + temporal\n(|κ*| = 0.20; use-dependence Δf = +0.12)',
                  fontsize=6.5)
    plabel(axb, 'b', dx=-0.28, dy=1.22)

    # c: CaV pools
    axc = fig.add_subplot(gs[0, 2])
    ina_pool, open_pool = [], []
    for cell in cav['cells']:
        g = cell['group']
        if g in ('Verapamil_Ca2+_PT', 'Verapamil_Ca2+_RT', 'Diltiazem_Ba2+_PT'):
            ina_pool.append(fnum(cell['df']))
        elif g == 'Norbuprenorphine_Ca2+_PT':
            open_pool.append(fnum(cell['df']))
    ina_pool = np.array(ina_pool); open_pool = np.array(open_pool)
    ina_pool = ina_pool[np.isfinite(ina_pool)]; open_pool = open_pool[np.isfinite(open_pool)]
    jit = np.random.default_rng(3)
    axc.scatter(1 + jit.normal(0, 0.05, len(ina_pool)), ina_pool, s=8,
                color=VERM, alpha=0.65, edgecolors='none')
    axc.scatter(2 + jit.normal(0, 0.05, len(open_pool)), open_pool, s=8,
                color=BLUE, alpha=0.65, edgecolors='none')
    axc.plot([0.75, 1.25], [0.168, 0.168], color=VERM, lw=1.6)
    axc.plot([1.75, 2.25], [-0.088, -0.088], color=BLUE, lw=1.6)
    axc.axhline(0, color=BLACK, lw=0.5)
    axc.set_xticks([1, 2]); axc.set_xticklabels(['inactivated pool\n(n=43)', 'open pool\n(n=12)'], fontsize=6)
    axc.set_ylabel(r'$\Delta f$ (drug − control)')
    axc.set_title('CaV1.2\npools move in opposite directions\n(+0.168 / −0.088)', fontsize=6.5)
    plabel(axc, 'c', dx=-0.28, dy=1.22)

    # d: IKs multi-table
    axd = fig.add_subplot(gs[0, 3])
    mef = [r for r in iks if r['arm'] == 'Mef']
    dids = [r for r in iks if r['arm'] == 'DIDS']
    dvh_m = [fnum(r['dvh']) for r in mef]; inst_m = [fnum(r['dinst']) for r in mef]
    axd.scatter(dvh_m, inst_m, s=10, color=GREEN, alpha=0.75, edgecolors='none')
    if dids:
        axd.scatter([fnum(r['dvh']) for r in dids], [fnum(r['dinst']) for r in dids],
                    s=10, color=GREY, alpha=0.6, edgecolors='none')
    axd.axhline(0, color=BLACK, lw=0.5); axd.axvline(0, color=BLACK, lw=0.5)
    # direct cluster labels instead of a legend (legend covered the points)
    axd.scatter([0.06], [0.93], transform=axd.transAxes, s=10, color=GREEN,
                clip_on=False)
    axd.text(0.11, 0.93, 'Mefloquine (judged)', transform=axd.transAxes,
             fontsize=5.6, color=GREEN, va='center')
    axd.scatter([0.06], [0.84], transform=axd.transAxes, s=10, color=GREY,
                clip_on=False)
    axd.text(0.11, 0.84, 'DIDS (pool-confounded, no ticket)', transform=axd.transAxes,
             fontsize=5.6, color=GREY, va='center')
    axd.set_xlabel(r'$\Delta V_{1/2}$ of $a_{ss}$ (mV)')
    axd.set_ylabel(r'$\Delta$ instantaneous component')
    axd.set_title('IKs\nmulti-table modulation\n($a_{ss}$ shift + inst +0.30 + deact. suppression)',
                  fontsize=6.5)
    plabel(axd, 'd', dx=-0.28, dy=1.22)

    fig.text(0.5, 0.02,
             'Common grammar: drugs move steady-state occupancy / inactivation tables, not time-constant tables.  '
             'Pure conductance block rejected in all four channels.',
             ha='center', fontsize=7, color=BLACK)
    fig.suptitle('Drug grammar: drugs edit steady-state occupancy tables, not time constants',
                 fontsize=8.5, fontweight='bold', y=1.02)
    save(fig, 'Fig4_drug_grammar')

# =====================================================================
# Fig. 5 — host-model integration & CiPA drug scoring
# =====================================================================
def fig5():
    ordj = jload(os.path.join(AM, '2026-09-15_ORd三通道α化组装_结果.json'))
    b1 = csvload(os.path.join(AM, '2026-09-14_α模型_StageB_B1_官方静态臂_24药_结果.csv'))
    b2 = csvload(os.path.join(AM, '2026-09-14_α模型_StageB_B2_α替换_24药_tauA1_Q2.5_ka3_结果.csv'))
    arms_files = {
        '0.3': '2026-09-14_α模型_StageB_B2_α替换_24药_tauA0.3_Q2.5_ka3_结果.csv',
        '0.5': '2026-09-14_α模型_StageB_B2_α替换_24药_tauA0.5_Q2.5_ka3_结果.csv',
        '1':   '2026-09-14_α模型_StageB_B2_α替换_24药_tauA1_Q2.5_ka3_结果.csv',
        '2':   '2026-09-14_α模型_StageB_B2_α替换_24药_tauA2_Q2.5_ka3_结果.csv',
    }

    fig = plt.figure(figsize=(MM(183), MM(105)))
    gs = fig.add_gridspec(2, 6, hspace=1.0, wspace=1.1,
                          left=0.07, right=0.98, top=0.88, bottom=0.12)

    # a: AP overlay ref vs alpha-assembled
    axa = fig.add_subplot(gs[0, :3])
    vref = np.array(ordj['reference']['v_last'][::10])
    vmain = np.array(ordj['main']['v_last'][::10])
    t = np.arange(len(vref)) * 1.0  # v_last dt = 0.1 ms; [::10] → 1 ms per point
    axa.plot(t, vref, color=BLACK, lw=1.1, label='ORd reference (Markov IKr/IKs)')
    axa.plot(t, vmain, color=BLUE, lw=1.0, ls='--',
             label='α-assembled (IKr/IKs tables; INa Markov retained)')
    axa.set_xlabel('t (ms)'); axa.set_ylabel('V (mV)')
    axa.legend(frameon=False, loc='upper right', fontsize=5.6)
    axa.set_title(f"ORd host: APD90 difference {abs(ordj['main']['APD90_ms']/ordj['reference']['APD90_ms']-1)*100:.2f}%",
                  fontsize=7)
    plabel(axa, 'a')

    # b: J6 current peak ratios
    axb = fig.add_subplot(gs[0, 3:])
    j6 = ordj['J6']
    names = list(j6.keys()); ratios = [j6[k]['ratio'] for k in names]
    cols = [VERM if abs(r - 1) > 0.5 else GREEN for r in ratios]
    axb.bar(names, ratios, color=cols, width=0.55)
    axb.axhline(1.0, color=BLACK, lw=0.7)
    for i, r in enumerate(ratios):
        axb.text(i, r + 0.05, f'{r:.2f}', ha='center', fontsize=6)
    ik = names.index('IKs')
    axb.text(ik, ratios[ik] + 0.24, '(small base)', ha='center', fontsize=5,
             color=GREY)
    axb.set_ylabel('peak ratio  α / reference')
    axb.set_ylim(0, max(ratios) * 1.25)
    axb.set_title('J6: current time-structure comparison: IKr ×2.06 flagged as the\n'
                  'substantive difference (data-extracted repolarisation time course)',
                  fontsize=6.5)
    plabel(axb, 'b')

    # c: Stage B per-drug scatter B1 vs B2 (dynamic arm only, sealed convention)
    axc = fig.add_subplot(gs[1, :2])
    m1 = {r['drug']: fnum(r['qNet_ratio']) for r in b1}
    cl1 = {r['drug']: fnum(r['CiPA']) for r in b1}
    xs, ys, cc = [], [], []
    cmap = {2.0: VERM, 1.0: ORANGE, 0.0: BLUE}
    for r in b2:
        if r['mode'] != 'dyn':
            continue
        d = r['drug']
        if d in m1:
            xs.append(m1[d]); ys.append(fnum(r['qNet_ratio'])); cc.append(cmap.get(cl1.get(d, 1.0), GREY))
    axc.scatter(xs, ys, s=14, c=cc, alpha=0.8, edgecolors='none')
    lo = min(xs + ys) * 0.97; hi = max(xs + ys) * 1.03
    axc.plot([lo, hi], [lo, hi], color=BLACK, lw=0.7, ls='--')
    axc.set_xlabel('official Markov arm: qNet ratio')
    axc.set_ylabel('α-model dynamic arm: qNet ratio')
    for cls, col, lab in [(2.0, VERM, 'CiPA high'), (1.0, ORANGE, 'CiPA intermediate'),
                          (0.0, BLUE, 'CiPA low')]:
        axc.plot([], [], 'o', color=col, ms=4, label=lab)
    axc.legend(frameon=False, fontsize=5.6, loc='upper left')
    plabel(axc, 'c')

    # d: arm summary dots (sealed numbers)
    axd = fig.add_subplot(gs[1, 2:4])
    arms = ['B1 official\nMarkov+static', 'B2s α+static', 'B2 α+dynamic']
    rho = [-0.523, -0.506, -0.547]; auc = [0.833, 0.833, 0.857]
    x = np.arange(3)
    axd.axhline(-0.60, color=BLUE, ls='--', lw=0.8)
    axd.text(-0.42, -0.595, 'V1 line −0.60', fontsize=5.4, color=BLUE, ha='left',
             va='bottom')
    axd.plot(x, rho, 'o', color=BLUE, ms=7, zorder=3)
    for xi, r in zip(x, rho):
        axd.text(xi, r - 0.018, f'{r:.3f}', ha='center', va='top', fontsize=5.6,
                 color=BLUE)
    axd.set_xticks(x); axd.set_xticklabels(arms, fontsize=6)
    axd.set_ylabel('V1 ρ', color=BLUE); axd.set_ylim(-0.62, -0.42)
    axd.tick_params(axis='y', colors=BLUE)
    axd.set_xlim(-0.5, 2.5)
    axd2 = axd.twinx()
    axd2.axhline(0.80, color=VERM, ls='--', lw=0.8)
    axd2.text(-0.42, 0.803, 'V2 line 0.80', fontsize=5.4, color=VERM, ha='left',
              va='bottom')
    axd2.plot(x, auc, 's', color=VERM, ms=6.5, zorder=3)
    for xi, a in zip(x, auc):
        axd2.text(xi, a + 0.006, f'{a:.3f}', ha='center', va='bottom',
                  fontsize=5.6, color=VERM)
    axd2.set_ylim(0.72, 0.89); axd2.set_ylabel('V2 AUC', color=VERM)
    axd2.tick_params(axis='y', colors=VERM); axd2.spines['right'].set_visible(True)
    axd.set_title('dynamic-binding criterion: α arm non-inferior\n'
                  '(shared 1×Cmax static ceiling ≈ −0.55)', fontsize=6.5)
    plabel(axd, 'd')

    # e: sensitivity strip - rho per tauA arm (dyn arm only, sealed convention)
    axe = fig.add_subplot(gs[1, 4:])
    rhos = []
    for k, f in arms_files.items():
        rows = [r for r in csvload(os.path.join(AM, f)) if r['mode'] == 'dyn']
        q = [fnum(r['qNet_ratio']) for r in rows]
        cl = [fnum(r['CiPA']) for r in rows]
        r_, _ = stats.spearmanr(q, cl)
        rhos.append(r_)
    xs = [0.3, 0.5, 1, 2]
    axe.plot(xs, rhos, 'o-', color=BLUE, ms=4, lw=1.2)
    for i, (xi, r) in enumerate(zip(xs, rhos)):
        if i == 0:
            axe.text(xi, r + 0.0015, f'{r:.3f}', ha='center', va='bottom',
                     fontsize=5.4, color=BLUE)
        else:
            axe.text(xi, r - 0.0025, f'{r:.3f}', ha='center', va='top', fontsize=5.4,
                     color=BLUE)
    axe.set_xscale('log'); axe.set_xticks(xs)
    axe.set_xticklabels(['×0.3', '×0.5', '×1', '×2']); axe.minorticks_off()
    rng = max(rhos) - min(rhos)
    # recomputation note moved to the manuscript legend (was an in-panel footnote)
    axe.set_xlabel(r'hERG $\tau_{act}$ scaling (full B6 uncertainty band)')
    axe.set_ylabel('V1 ρ')
    axe.set_title(f'sensitivity: inter-arm V1 range {rng:.3f}\n'
                  f'B6 uncertainty does not enter Stage B',
                  fontsize=6.5)
    plabel(axe, 'e')

    fig.suptitle('Host-model integration and CiPA panel scoring',
                 fontsize=8.5, fontweight='bold', y=1.02)
    save(fig, 'Fig5_host_model')

# =====================================================================
# Extended Data Fig. 2 — identifiability audit
# =====================================================================
def edfig2():
    fe = jload(os.path.join(W182, 'fisher_eigen.json'))
    eig = np.array(fe['特征值_升序'])
    lam_max = fe['λmax']; chi = fe['条件数χ']; n_null = fe['零方向数_阈λmax×1e-3']

    fig = plt.figure(figsize=(MM(183), MM(70)))
    gs = fig.add_gridspec(1, 3, wspace=0.8, left=0.06, right=0.985,
                          top=0.80, bottom=0.16)

    axa = fig.add_subplot(gs[0, 0])
    axa.semilogy(range(1, 14), eig, 'o-', color=BLUE, ms=4.5, lw=1.1)
    thr = lam_max * 1e-3
    axa.axhline(thr, color=VERM, ls='--', lw=0.9)
    axa.axhspan(eig.min() * 0.3, thr, color=VERM, alpha=0.07)
    axa.text(8.2, eig.min() * 2.0, f'{n_null} null directions\n'
             + r'($\lambda < \lambda_{\rm max}\times 10^{-3}$)',
             fontsize=6, color=VERM, va='bottom')
    axa.set_xlabel('eigenvalue index (ascending)'); axa.set_ylabel('Fisher eigenvalue λ')
    axa.set_title(f'13-parameter Markov formulation, 4 protocol families\n'
                  f'condition number χ = {chi:.1e}', fontsize=6.5)
    plabel(axa, 'a', dx=-0.25, dy=1.25)

    axb = fig.add_subplot(gs[0, 1]); axb.axis('off')
    txt = ('Exact degeneracies (verified symbolically):\n\n'
           '1. rate–conductance joint scaling:\n'
           '    all rates × s, G ÷ s  →  all four protocols invariant\n\n'
           '2. α_amp · ALPHA · XMAX enter only as a product\n'
           '    (2-dimensional unconstrained fibre)\n\n'
           '3. double-exponential component interchange\n\n'
           'Loop-inversion test on the null manifold:\n'
           '0 / 451 attempts recover the generating point\n\n'
           '→ identifiable coordinates = modal coordinates\n'
           '   (time constants · steady states · delays)\n'
           '→ fitted rate constants carry ≥ 10× uncertainty\n'
           '   along null directions')
    axb.text(0.02, 0.98, txt, transform=axb.transAxes, fontsize=6.3, va='top',
             family='sans-serif')
    plabel(axb, 'b', dx=-0.06, dy=1.25)

    axc = fig.add_subplot(gs[0, 2]); axc.axis('off')
    txt2 = ('Slow-memory audit (hERG, bounds of any\n'
            'memoryless table model):\n\n'
            '• same voltage, different history:\n'
            '   currents differ by 17–112× (9/9 cells)\n\n'
            '• continuous memory spectrum:\n'
            '   4.8 s → beyond 2,000 s\n\n'
            '• present models capture the dominant\n'
            '   discrete modes (fast / middle / late)\n'
            '   and explicitly do not claim the continuum\n\n'
            'Cross-cell stability of modal ordering: 0.074 dex')
    axc.text(0.02, 0.98, txt2, transform=axc.transAxes, fontsize=6.3, va='top')
    plabel(axc, 'c', dx=-0.06, dy=1.25)

    save(fig, 'EDFig2_identifiability')

if __name__ == '__main__':
    fig1(); fig2(); fig3(); fig4(); fig5(); edfig2()
