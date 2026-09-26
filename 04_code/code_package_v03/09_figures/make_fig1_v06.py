# -*- coding: utf-8 -*-
# Fig. 1 v06: extract-judge-assemble pipeline + five-channel membrane topology.
# v06: tighter layout (less whitespace), faithful subunit architecture:
#   - tetramers (hERG, IKs, Kv4.3): VSD(S1-S4) + pore(S5-S6) side view + top-view inset
#   - Nav1.5 / CaV1.2: one peptide, 4 domains linked, each domain = VSD + pore module
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

fig = plt.figure(figsize=(7.2, 6.3))
ax = fig.add_axes([0, 0.295, 1, 0.705])  # panel a
ax.set_xlim(0, 100); ax.set_ylim(24, 100); ax.axis("off")
axb = fig.add_axes([0, 0.0, 1, 0.295])   # panel b
axb.set_xlim(0, 100); axb.set_ylim(0, 100); axb.axis("off")

YC = 56.0  # membrane centre, panel a

# ---------- membrane ----------
def membrane(a, x0, x1, yc=YC, hh=11.0):
    span = x1 - x0
    n = max(8, int(span / 1.05))
    xs = np.linspace(x0, x1, n)
    for x in xs:
        for sgn in (1, -1):
            a.add_patch(Circle((x, yc + sgn * hh / 2), 0.72, fc="#c9d4de", ec="#9fb0bd", lw=0.3, zorder=1))
            a.plot([x, x], [yc + sgn * (hh / 2 - 0.6), yc + sgn * 1.1], color="#b6c3ce", lw=0.55, zorder=1)
    a.add_patch(Rectangle((x0 - 0.6, yc - 1.1), span + 1.2, 2.2, fc="#eef3f7", ec="none", zorder=0))

# ---------- helices / domains ----------
def helix(a, x, yc, w=0.62, hh=12.0, fc="#4C9BD6", ec="#2f6ea3", z=3, alpha=1.0):
    a.add_patch(FancyBboxPatch((x - w / 2, yc - hh / 2), w, hh,
        boxstyle="round,pad=0,rounding_size=0.4", fc=fc, ec=ec, lw=0.5, zorder=z, alpha=alpha))

def vsd(a, xc, yc, s, base, edge, s4="#D64545", z=3, alpha=1.0, mirror=False):
    # S1-S4 voltage-sensor domain: 4 helices, S4 highlighted (pore-proximal)
    w, gap = 0.58 * s, 0.20 * s
    for i in range(4):
        x = xc + (i - 1.5) * (w + gap)
        is_s4 = (i == 0) if mirror else (i == 3)
        col = s4 if is_s4 else base
        ec = "#a33333" if is_s4 else edge
        helix(a, x, yc, w=w, hh=11.5 * s, fc=col, ec=ec, z=z, alpha=alpha)

def pore_mod(a, xc, yc, s, base, edge, z=3, alpha=1.0, loop_dir=1):
    # S5-S6 pore module: 2 helices + re-entrant pore loop
    w, gap = 0.58 * s, 0.20 * s
    for i in range(2):
        x = xc + (i - 0.5) * (w + gap)
        helix(a, x, yc, w=w, hh=11.5 * s, fc=base, ec=edge, z=z, alpha=alpha)
    t = np.linspace(0, np.pi, 30)
    a.plot(xc + 0.55 * s * np.cos(t), yc + 5.75 * s - loop_dir * 1.7 * s * np.sin(t),
           color=edge, lw=1.0, zorder=z + 1, alpha=alpha)

def subunit6tm(a, xc, yc, s, base, edge, z=3, alpha=1.0):
    vsd(a, xc - 1.35 * s, yc, s, base, edge, z=z, alpha=alpha)
    pore_mod(a, xc + 1.55 * s, yc, s, base, edge, z=z, alpha=alpha)

def tetramer_side(a, xc, yc, s, base, edge):
    # ghost pair behind (the other two subunits of the tetramer)
    subunit6tm(a, xc - 0.9 * s, yc + 1.6, s * 0.96, base, edge, z=2, alpha=0.30)
    subunit6tm(a, xc + 0.9 * s, yc + 1.6, s * 0.96, base, edge, z=2, alpha=0.30)
    # front pair: left subunit mirrored (VSD outside, pore toward centre)
    vsd(a, xc - 3.3 * s, yc, s, base, edge)
    pore_mod(a, xc - 0.85 * s, yc, s, base, edge)
    pore_mod(a, xc + 0.85 * s, yc, s, base, edge)
    vsd(a, xc + 3.3 * s, yc, s, base, edge, mirror=True)

def topview(a, x, y, r, base, edge, label="top"):
    a.add_patch(Circle((x, y), r, fc="white", ec=edge, lw=0.6, zorder=6))
    for ang in (45, 135, 225, 315):
        dx = r * 0.44 * np.cos(np.radians(ang)); dy = r * 0.44 * np.sin(np.radians(ang))
        a.add_patch(Circle((x + dx, y + dy), r * 0.36, fc=base, ec=edge, lw=0.5, zorder=7))
    a.add_patch(Circle((x, y), r * 0.13, fc="white", ec=edge, lw=0.5, zorder=8))
    a.text(x, y - r - 1.6, label, fontsize=4.8, color=edge, ha="center", va="top", zorder=8)

def ion_arrow(a, x, y0, y1, label, color):
    a.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=8, color=color, lw=1.3, zorder=6))
    a.text(x + 1.0, (y0 + y1) / 2, label, fontsize=7, color=color, va="center", zorder=6)

def ball_chain(a, x, y, r=1.2, fc="#E8913A", ec="#a3621f", label=None, lcol=None):
    a.plot([x, x], [y + r, y + r + 2.4], color=ec, lw=1.0, ls=(0, (2, 1.4)), zorder=5)
    a.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=0.6, zorder=5))
    if label:
        a.text(x + 1.8, y, label, fontsize=6, color=lcol or ec, va="center", zorder=6)

def title_block(a, xc, name, sub):
    a.text(xc, 95, name, ha="center", fontsize=8.5, fontweight="bold", color="#222222")
    a.text(xc, 91.2, sub, ha="center", fontsize=6.4, color="#555555")

def state_block(a, xc, lines):
    ys = (40.5, 36.8, 33.1, 29.4)
    for (txt, col), y in zip(lines, ys):
        a.text(xc, y, txt, ha="center", fontsize=5.8, color=col)

# ================= hERG =================
x0, x1 = 2, 20
membrane(ax, x0, x1)
xc = (x0 + x1) / 2
tetramer_side(ax, xc, YC, 1.0, "#4C9BD6", "#2f6ea3")
topview(ax, x0 + 3.2, YC + 15.5, 2.6, "#4C9BD6", "#2f6ea3")
ion_arrow(ax, xc, YC + 8.0, YC + 15.0, "K$^+$", "#2f6ea3")
title_block(ax, xc, "hERG (K$_V$11.1)", "homotetramer, 6TM subunits")
state_block(ax, xc, [
    ("C ↔ O ↔ I   gates m, h", "#333333"),
    ("τ$_{deact}$(V): constants", "#2f6ea3"),
    ("CV 0.13–0.19", "#2f6ea3"),
    ("h$_{ss}$(V): 12-level table", "#2f6ea3"),
])

# ================= Nav1.5 =================
x0, x1 = 22.5, 40.5
membrane(ax, x0, x1)
xc = (x0 + x1) / 2
ds = 0.62
doms = [xc + (k - 1.5) * 4.35 for k in range(4)]
for k, (x, dom) in enumerate(zip(doms, ("I", "II", "III", "IV"))):
    subunit6tm(ax, x, YC, ds, "#E8913A", "#a3621f")
    ax.text(x, YC - 8.6, dom, ha="center", fontsize=5.8, color="#a3621f")
# single polypeptide: extracellular linkers between domains
for k in range(3):
    ax.plot([doms[k] + 1.2, doms[k + 1] - 1.2], [YC + 6.6, YC + 6.6], color="#a3621f", lw=1.1, zorder=2)
ball_chain(ax, xc + 2.2, YC - 13.0, label="IFM")
ion_arrow(ax, xc, YC + 8.0, YC + 15.0, "Na$^+$", "#a3621f")
title_block(ax, xc, "Na$_V$1.5", "one α, 4 domains (I–IV)")
state_block(ax, xc, [
    ("C ↔ O ↔ I   gates m$^3$, h", "#333333"),
    ("τ$_h$(V): constant mid-band", "#a3621f"),
    ("CV 0.029", "#a3621f"),
    ("τ$_h$(−40 mV): distributed", "#a3621f"),
])

# ================= CaV1.2 =================
x0, x1 = 43, 61
membrane(ax, x0, x1)
xc = (x0 + x1) / 2
ds = 0.62
doms = [xc + (k - 1.5) * 4.35 for k in range(4)]
for k, (x, dom) in enumerate(zip(doms, ("I", "II", "III", "IV"))):
    subunit6tm(ax, x, YC, ds, "#3FA37C", "#27755a")
    ax.text(x, YC - 8.6, dom, ha="center", fontsize=5.8, color="#27755a")
for k in range(3):
    ax.plot([doms[k] + 1.2, doms[k + 1] - 1.2], [YC + 6.6, YC + 6.6], color="#27755a", lw=1.1, zorder=2)
# alpha2delta extracellular, GPI-anchored
ax.plot([xc - 4.6, xc - 3.1, xc - 1.6], [YC + 11.0, YC + 13.2, YC + 11.2], color="#7a5ea8", lw=1.7, zorder=5)
ax.text(xc - 3.1, YC + 14.8, "α$_2$δ", fontsize=6, color="#7a5ea8", ha="center")
# beta subunit cytosolic
ax.add_patch(FancyBboxPatch((xc - 7.2, YC - 13.0), 3.6, 2.8, boxstyle="round,pad=0,rounding_size=0.8", fc="#bfe3d4", ec="#27755a", lw=0.6, zorder=4))
ax.text(xc - 5.4, YC - 11.6, "β", fontsize=6.5, color="#27755a", ha="center", va="center", zorder=5)
# CaM
ax.add_patch(Circle((xc + 5.0, YC - 11.4), 1.8, fc="#d9efe5", ec="#27755a", lw=0.6, zorder=4))
ax.text(xc + 5.0, YC - 11.4, "CaM", fontsize=5, color="#27755a", ha="center", va="center", zorder=5)
ion_arrow(ax, xc, YC + 8.0, YC + 15.0, "Ca$^{2+}$", "#27755a")
title_block(ax, xc, "Ca$_V$1.2", "α$_1$ (4 domains) + α$_2$δ + β")
state_block(ax, xc, [
    ("O → I$_{VDI}$;  O → I$_{CDI}$ (CaM)", "#333333"),
    ("f$_{inact}$: Ba$^{2+}$ 0.44", "#27755a"),
    ("Ca$^{2+}$ 0.73", "#27755a"),
    ("CDI: one additive term", "#27755a"),
])

# ================= IKs =================
x0, x1 = 63.5, 81.5
membrane(ax, x0, x1)
xc = (x0 + x1) / 2
tetramer_side(ax, xc, YC, 1.0, "#C77DBB", "#8e4f86")
topview(ax, x0 + 3.2, YC + 15.5, 2.6, "#C77DBB", "#8e4f86")
# KCNE1 single-TM accessory subunit at the periphery
helix(ax, xc + 5.9, YC, w=0.62, hh=11.5, fc="#E8C9E2", ec="#8e4f86", z=5)
ax.text(xc + 5.9, YC + 7.6, "KCNE1", fontsize=5.4, color="#8e4f86", ha="center")
ion_arrow(ax, xc, YC + 8.0, YC + 15.0, "K$^+$", "#8e4f86")
title_block(ax, xc, "IKs", "4× KCNQ1 + KCNE1")
state_block(ax, xc, [
    ("C ↔ O", "#333333"),
    ("a = β·a$_{ss}$ + (1−β)·a$_{dyn}$", "#333333"),
    ("τ$_{act}$(V): bell-shaped", "#8e4f86"),
    ("delay d(V): measured table", "#8e4f86"),
])

# ================= Kv4.3 (registration grade) =================
x0, x1 = 84, 98.5
membrane(ax, x0, x1)
xc = (x0 + x1) / 2
tetramer_side(ax, xc, YC, 0.85, "#4DB6AC", "#2e7d76")
topview(ax, x0 + 3.0, YC + 15.5, 2.4, "#4DB6AC", "#2e7d76")
ball_chain(ax, xc - 1.5, YC - 11.4, r=1.05, fc="#4DB6AC", ec="#2e7d76")
ax.add_patch(FancyBboxPatch((xc + 1.6, YC - 12.8), 3.8, 2.9, boxstyle="round,pad=0,rounding_size=0.8", fc="#cdeae6", ec="#2e7d76", lw=0.6, zorder=4))
ax.text(xc + 3.5, YC - 11.35, "KChIP", fontsize=4.6, color="#2e7d76", ha="center", va="center", zorder=5)
ion_arrow(ax, xc, YC + 8.0, YC + 15.0, "K$^+$", "#2e7d76")
title_block(ax, xc, "K$_V$4.3 (Ito)", "homotetramer + KChIP")
state_block(ax, xc, [
    ("C ↔ O ↔ I (N-type)", "#333333"),
    ("anchors 6/6", "#2e7d76"),
    ("reproduced (probe)", "#2e7d76"),
    ("registration grade", "#B23A3A"),
])
ax.add_patch(Rectangle((x0 - 1.4, 24.5), x1 - x0 + 2.8, 69.5, fill=False, ec="#B23A3A", lw=0.9, ls=(0, (4, 2)), zorder=7))

ax.text(0.5, 84, "extracellular", fontsize=6.2, color="#667788", rotation=90, va="center")
ax.text(0.5, 42, "intracellular", fontsize=6.2, color="#667788", rotation=90, va="center")
ax.text(-0.5, 99, "a", fontsize=11, fontweight="bold", va="top")

# ================= panel b: pipeline =================
def box(a, x0, w, title, lines, fc, ec):
    a.add_patch(FancyBboxPatch((x0, 26), w, 56, boxstyle="round,pad=0,rounding_size=2.2", fc=fc, ec=ec, lw=1.0))
    a.text(x0 + w / 2, 73, title, ha="center", fontsize=7.8, fontweight="bold", color=ec)
    for i, ln in enumerate(lines):
        a.text(x0 + w / 2, 63 - i * 8.6, ln, ha="center", fontsize=6.4, color="#333333")

bw = 20.5
xs = [1.5, 28.0, 54.5, 81.0]
box(axb, xs[0], bw, "EXTRACT", ["protocol-rich recordings", "→ voltage-dependent tables", "no global fit"], "#EAF3FB", "#2f6ea3")
box(axb, xs[1], bw, "JUDGE", ["pre-registered constancy test", "constant / per-cell /", "distributed / unusable"], "#FDF1E4", "#a3621f")
box(axb, xs[2], bw, "ASSEMBLE", ["Hodgkin–Huxley form", "measured tables inserted", "per-cell free: G (+ΔV½h)"], "#E8F5EF", "#27755a")
box(axb, xs[3], bw - 1.5, "PREDICT", ["held-out protocols & cells", "sine · AP clamp · CiPA", "failure → one table, one voltage"], "#F9ECF6", "#8e4f86")
for i in range(3):
    axb.add_patch(FancyArrowPatch((xs[i] + bw + 0.6, 54), (xs[i + 1] - 0.6, 54), arrowstyle="-|>", mutation_scale=10, color="#555555", lw=1.1))
axb.text(0.5, 95, "b", fontsize=11, fontweight="bold", va="top")
axb.text(50, 13.5, "design property 1:  every table entry is traceable to named cells and sweeps (provenance)", ha="center", fontsize=6.4, color="#333333")
axb.text(50, 5.0, "design property 2:  cell-to-cell differences are measured and reported, not averaged away", ha="center", fontsize=6.4, color="#333333")

fig.suptitle("Data-extracted channel modelling: extract, judge, assemble, predict",
             fontsize=9, fontweight="bold", y=1.01)
fs.save(fig, os.path.join(FIGD, "Fig1_pipeline_v06"))
print("saved Fig1_pipeline_v06")
