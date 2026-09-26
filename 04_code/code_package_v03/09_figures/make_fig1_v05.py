# -*- coding: utf-8 -*-
# Fig. 1 v05: extract-judge-assemble pipeline + five-channel membrane topology.
# Realistic subunit composition per channel; Kv4.3 marked registration grade.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle
import figstyle_v05 as fs

fs.apply()

ROOT = os.path.dirname(os.path.abspath(__file__))
FIGD = os.path.join(ROOT, "figures_v01")
os.makedirs(FIGD, exist_ok=True)

fig = plt.figure(figsize=(7.2, 7.2))
ax = fig.add_axes([0, 0.345, 1, 0.655])  # panel a
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
axb = fig.add_axes([0, 0.0, 1, 0.345])   # panel b
axb.set_xlim(0, 100); axb.set_ylim(0, 100); axb.axis("off")

# ---------- helpers ----------
def membrane(a, x0, x1, yc=50.0, hh=9.0, n=None):
    span = x1 - x0
    n = n or max(6, int(span / 1.1))
    xs = np.linspace(x0, x1, n)
    for x in xs:
        for sgn in (1, -1):
            a.add_patch(Circle((x, yc + sgn * hh / 2), 0.55, fc="#c9d4de", ec="#9fb0bd", lw=0.3, zorder=1))
            a.plot([x, x], [yc + sgn * hh / 2 - sgn * 0.5, yc + sgn * 0.8], color="#b6c3ce", lw=0.5, zorder=1)
    a.add_patch(Rectangle((x0 - 0.6, yc - 0.8), span + 1.2, 1.6, fc="#eef3f7", ec="none", zorder=0))

def helix(a, x, yc, w=1.05, hh=13.0, fc="#4C9BD6", ec="#2f6ea3", z=3, alpha=1.0):
    a.add_patch(FancyBboxPatch((x - w / 2, yc - hh / 2), w, hh,
        boxstyle="round,pad=0,rounding_size=0.45", fc=fc, ec=ec, lw=0.5, zorder=z, alpha=alpha))

def pore_loop(a, x, yc, fc="#2f6ea3", z=4):
    t = np.linspace(np.pi, 2 * np.pi, 40)
    a.plot(x + 1.6 * np.cos(t) * 0.4, yc + 2.2 + 1.1 * np.sin(t) * 0.7, color=fc, lw=1.1, zorder=z)

def subunit6tm(a, xc, yc, base, edge, s4="#D64545", alpha=1.0, z=3, scale=1.0):
    w = 0.62 * scale
    for i in range(6):
        x = xc + (i - 2.5) * (w + 0.16 * scale)
        col = s4 if i == 3 else base
        ec = "#a33333" if i == 3 else edge
        helix(a, x, yc, w=w, hh=12.5 * scale, fc=col, ec=ec, z=z, alpha=alpha)
    pore_loop(a, xc + 1.9 * scale, yc, fc=edge, z=z + 1)

def tetramer(a, xc, yc, base, edge, scale=1.0):
    # two ghost subunits behind, two in front
    subunit6tm(a, xc - 2.3 * scale, yc + 1.2, base, edge, alpha=0.35, z=2, scale=scale)
    subunit6tm(a, xc + 2.3 * scale, yc + 1.2, base, edge, alpha=0.35, z=2, scale=scale)
    subunit6tm(a, xc - 2.1 * scale, yc, base, edge, z=3, scale=scale)
    subunit6tm(a, xc + 2.1 * scale, yc, base, edge, z=3, scale=scale)

def ion_arrow(a, x, y0, y1, label, color, dx=2.6):
    a.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=7, color=color, lw=1.2, zorder=6))
    a.text(x + 0.8, (y0 + y1) / 2, label, fontsize=6.5, color=color, va="center", zorder=6)

def ball_chain(a, x, y, r=1.15, fc="#E8913A", ec="#a3621f", label=None, lcol=None):
    a.plot([x, x], [y + r, y + r + 2.2], color=ec, lw=0.9, ls=(0, (2, 1.4)), zorder=5)
    a.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=0.6, zorder=5))
    if label:
        a.text(x + 1.6, y, label, fontsize=5.8, color=lcol or ec, va="center", zorder=6)

def title_block(a, xc, y, name, sub, col="#222222"):
    a.text(xc, y, name, ha="center", fontsize=8, fontweight="bold", color=col)
    a.text(xc, y - 3.4, sub, ha="center", fontsize=6.3, color="#555555")

def state_line(a, xc, y, txt, col="#333333", fsz=5.6):
    a.text(xc, y, txt, ha="center", fontsize=fsz, color=col)

def state_block(a, xc, lines):
    # four stacked short lines, avoids cross-channel text collisions
    ys = (33.5, 30.1, 26.7, 23.3)
    for (txt, col), y in zip(lines, ys):
        state_line(a, xc, y, txt, col)

YC = 52.0  # membrane centre in panel a

# ================= hERG =================
x0, x1 = 2, 20
membrane(ax, x0, x1, YC)
tetramer(ax, (x0 + x1) / 2, YC, "#4C9BD6", "#2f6ea3", scale=0.85)
ion_arrow(ax, (x0 + x1) / 2, YC + 8.5, YC + 15.5, "K$^+$", "#2f6ea3")
title_block(ax, (x0 + x1) / 2, 88, "hERG (K$_V$11.1)", "homotetramer, 6TM subunits")
state_block(ax, (x0 + x1) / 2, [
    ("C ↔ O ↔ I   gates m, h", "#333333"),
    ("τ$_{deact}$(V): constants", "#2f6ea3"),
    ("CV 0.13–0.19", "#2f6ea3"),
    ("h$_{ss}$(V): 12-level table", "#2f6ea3"),
])

# ================= Nav1.5 =================
x0, x1 = 22.5, 40.5
membrane(ax, x0, x1, YC)
xc = (x0 + x1) / 2
for k, dom in enumerate(("I", "II", "III", "IV")):
    x = xc + (k - 1.5) * 4.1
    subunit6tm(ax, x, YC, "#E8913A", "#a3621f", scale=0.55)
    ax.text(x, YC - 9.3, dom, ha="center", fontsize=5.6, color="#a3621f")
ax.plot([xc - 6.2, xc + 6.2], [YC + 6.9, YC + 6.9], color="#a3621f", lw=1.0, zorder=2)  # one peptide
ball_chain(ax, xc + 2.1, YC - 12.4, label="IFM")
ion_arrow(ax, xc, YC + 8.5, YC + 15.5, "Na$^+$", "#a3621f")
title_block(ax, xc, 88, "Na$_V$1.5", "one α, 4 domains (I–IV)")
state_block(ax, xc, [
    ("C ↔ O ↔ I   gates m$^3$, h", "#333333"),
    ("τ$_h$(V): constant mid-band", "#a3621f"),
    ("CV 0.029", "#a3621f"),
    ("τ$_h$(−40 mV): distributed", "#a3621f"),
])

# ================= CaV1.2 =================
x0, x1 = 43, 61
membrane(ax, x0, x1, YC)
xc = (x0 + x1) / 2
for k, dom in enumerate(("I", "II", "III", "IV")):
    x = xc + (k - 1.5) * 4.1
    subunit6tm(ax, x, YC, "#3FA37C", "#27755a", scale=0.55)
    ax.text(x, YC - 9.3, dom, ha="center", fontsize=5.6, color="#27755a")
ax.plot([xc - 6.2, xc + 6.2], [YC + 6.9, YC + 6.9], color="#27755a", lw=1.0, zorder=2)
# alpha2delta extracellular
ax.plot([xc - 4.5, xc - 3.0, xc - 1.5], [YC + 12.0, YC + 14.0, YC + 12.2], color="#7a5ea8", lw=1.6, zorder=5)
ax.text(xc - 3.0, YC + 15.6, "α$_2$δ", fontsize=5.8, color="#7a5ea8", ha="center")
# beta subunit cytosolic
ax.add_patch(FancyBboxPatch((xc - 6.8, YC - 13.8), 3.4, 2.6, boxstyle="round,pad=0,rounding_size=0.8", fc="#bfe3d4", ec="#27755a", lw=0.6, zorder=4))
ax.text(xc - 5.1, YC - 12.5, "β", fontsize=6, color="#27755a", ha="center", va="center", zorder=5)
# CaM
ax.add_patch(Circle((xc + 4.6, YC - 12.2), 1.7, fc="#d9efe5", ec="#27755a", lw=0.6, zorder=4))
ax.text(xc + 4.6, YC - 12.2, "CaM", fontsize=4.8, color="#27755a", ha="center", va="center", zorder=5)
ion_arrow(ax, xc, YC + 8.5, YC + 15.5, "Ca$^{2+}$", "#27755a")
title_block(ax, xc, 88, "Ca$_V$1.2", "α$_1$ (4 domains) + α$_2$δ + β")
state_block(ax, xc, [
    ("O → I$_{VDI}$;  O → I$_{CDI}$ (CaM)", "#333333"),
    ("f$_{inact}$: Ba$^{2+}$ 0.44", "#27755a"),
    ("Ca$^{2+}$ 0.73", "#27755a"),
    ("CDI: one additive term", "#27755a"),
])

# ================= IKs =================
x0, x1 = 63.5, 81.5
membrane(ax, x0, x1, YC)
xc = (x0 + x1) / 2
subunit6tm(ax, xc - 2.3, YC + 1.2, "#C77DBB", "#8e4f86", alpha=0.35, z=2, scale=0.8)
subunit6tm(ax, xc + 2.3, YC + 1.2, "#C77DBB", "#8e4f86", alpha=0.35, z=2, scale=0.8)
subunit6tm(ax, xc - 2.1, YC, "#C77DBB", "#8e4f86", scale=0.8)
subunit6tm(ax, xc + 2.1, YC, "#C77DBB", "#8e4f86", scale=0.8)
helix(ax, xc, YC, w=0.6, hh=12.0, fc="#E8C9E2", ec="#8e4f86", z=5)  # KCNE1 single TM
ax.text(xc, YC + 8.6, "KCNE1", fontsize=5.2, color="#8e4f86", ha="center")
ion_arrow(ax, xc, YC + 10.5, YC + 15.5, "K$^+$", "#8e4f86")
title_block(ax, xc, 88, "IKs", "4× KCNQ1 + KCNE1")
state_block(ax, xc, [
    ("C ↔ O", "#333333"),
    ("a = β·a$_{ss}$ + (1−β)·a$_{dyn}$", "#333333"),
    ("τ$_{act}$(V): bell-shaped", "#8e4f86"),
    ("delay d(V): measured table", "#8e4f86"),
])

# ================= Kv4.3 (registration grade) =================
x0, x1 = 84, 98.5
membrane(ax, x0, x1, YC)
xc = (x0 + x1) / 2
tetramer(ax, xc, YC, "#4DB6AC", "#2e7d76", scale=0.8)
ball_chain(ax, xc - 1.4, YC - 12.2, r=1.0, fc="#4DB6AC", ec="#2e7d76")
ax.add_patch(FancyBboxPatch((xc + 2.4, YC - 13.6), 3.6, 2.8, boxstyle="round,pad=0,rounding_size=0.8", fc="#cdeae6", ec="#2e7d76", lw=0.6, zorder=4))
ax.text(xc + 4.2, YC - 12.2, "KChIP", fontsize=4.6, color="#2e7d76", ha="center", va="center", zorder=5)
ion_arrow(ax, xc, YC + 8.5, YC + 15.5, "K$^+$", "#2e7d76")
title_block(ax, xc, 88, "K$_V$4.3 (Ito)", "homotetramer + KChIP")
state_block(ax, xc, [
    ("C ↔ O ↔ I (N-type)", "#333333"),
    ("anchors 6/6", "#2e7d76"),
    ("reproduced (probe)", "#2e7d76"),
    ("registration grade", "#B23A3A"),
])
# registration-grade frame
ax.add_patch(Rectangle((x0 - 1.2, 18.5), x1 - x0 + 2.4, 76.5, fill=False, ec="#B23A3A", lw=0.8, ls=(0, (4, 2)), zorder=7))

ax.text(0.4, 88, "extracellular", fontsize=6, color="#667788", rotation=90, va="center")
ax.text(0.4, 40, "intracellular", fontsize=6, color="#667788", rotation=90, va="center")
ax.text(-0.5, 97.5, "a", fontsize=11, fontweight="bold", va="top")

# ================= panel b: pipeline =================
def box(a, x0, w, title, lines, fc, ec):
    a.add_patch(FancyBboxPatch((x0, 38), w, 40, boxstyle="round,pad=0,rounding_size=2.2", fc=fc, ec=ec, lw=1.0))
    a.text(x0 + w / 2, 68, title, ha="center", fontsize=7.6, fontweight="bold", color=ec)
    for i, ln in enumerate(lines):
        a.text(x0 + w / 2, 60 - i * 7.2, ln, ha="center", fontsize=6.2, color="#333333")

bw = 20.5
xs = [1.5, 28.0, 54.5, 81.0]
box(axb, xs[0], bw, "EXTRACT", ["protocol-rich recordings", "→ voltage-dependent tables", "no global fit"], "#EAF3FB", "#2f6ea3")
box(axb, xs[1], bw, "JUDGE", ["pre-registered constancy test", "constant / per-cell /", "distributed / unusable"], "#FDF1E4", "#a3621f")
box(axb, xs[2], bw, "ASSEMBLE", ["Hodgkin–Huxley form", "measured tables inserted", "per-cell free: G (+ΔV½h)"], "#E8F5EF", "#27755a")
box(axb, xs[3], bw - 1.5, "PREDICT", ["held-out protocols & cells", "sine · AP clamp · CiPA", "failure → one table, one voltage"], "#F9ECF6", "#8e4f86")
for i in range(3):
    axb.add_patch(FancyArrowPatch((xs[i] + bw + 0.6, 58), (xs[i + 1] - 0.6, 58), arrowstyle="-|>", mutation_scale=10, color="#555555", lw=1.1))
axb.text(0.5, 88, "b", fontsize=11, fontweight="bold", va="top")
axb.text(50, 22, "design property 1:  every table entry is traceable to named cells and sweeps (provenance)", ha="center", fontsize=6.2, color="#333333")
axb.text(50, 13, "design property 2:  cell-to-cell differences are measured and reported, not averaged away", ha="center", fontsize=6.2, color="#333333")

fs.save(fig, os.path.join(FIGD, "Fig1_pipeline_v05"))
