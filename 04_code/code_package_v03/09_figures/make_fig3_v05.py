# -*- coding: utf-8 -*-
# Fig. 3 v05: portability verdict matrix (five channels) + channel verdict panels.
# Fixes vs v01: Kv4.3 registration row added; annotation collisions fixed;
# inset enlarged; unified figstyle; svg output.
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import make_figures_v01 as M
import figstyle_v05 as fs

fs.apply()

BLUE, VERM, GREEN, PINK = M.BLUE, M.VERM, M.GREEN, M.PINK
ORANGE, GREY, BLACK = M.ORANGE, M.GREY, M.BLACK
MM, plabel = M.MM, M.plabel
csvload, jload, fnum = M.csvload, M.jload, M.fnum
AM = M.AM
OUT = os.path.join(M.OUT)

na4 = csvload(os.path.join(AM, "2026-09-16_Nα4_单通道负40单点锚定判决_结果.csv"))
bell = csvload(os.path.join(AM, "2026-09-15_Kα2_Fedida2024_Fig3B_wtEQ_数字化.csv"))
dtj = jload(os.path.join(AM, "2026-09-15_Kα2_Δt登记.json"))
ka3 = csvload(os.path.join(AM, "2026-09-15_Kα3_IKs前向验证_逐细胞表_结果.csv"))

fig = plt.figure(figsize=(MM(183), MM(118)))
gs = fig.add_gridspec(2, 6, hspace=0.95, wspace=1.0,
                      left=0.07, right=0.98, top=0.88, bottom=0.11)

# ---------------- a: verdict matrix, five channels ----------------
axa = fig.add_subplot(gs[0, :3]); axa.axis("off")
channels = ["hERG", "Nav1.5", "CaV1.2", "IKs", "Kv4.3"]
quants = ["kinetic τ", "steady-state slope", "steady-state midpoint",
          "reversal potential", "delay"]
# verdict codes: 0 constant, 1 per-cell, 2 distributed, 3 closed, 4 registration
Mat = np.array([
    [0, 2, 1, 0, 0],
    [0, 0, 1, 2, 3],
    [0, 3, 3, 0, 3],
    [0, 0, 0, 2, 0],
    [4, 4, 4, 4, 4],
])
cmap = {0: GREEN, 1: ORANGE, 2: PINK, 3: "#BBBBBB", 4: "white"}
lab = {0: "constant", 1: "per-cell", 2: "distributed", 3: "closed / n.a.",
       4: "registration"}
NR = 5
for i in range(NR):
    for j in range(5):
        kw = dict(fc=cmap[Mat[i, j]], ec="white", lw=1.5)
        if Mat[i, j] == 4:
            kw = dict(fc="white", ec="#999999", lw=0.7, hatch="///")
        axa.add_patch(plt.Rectangle((j, NR - 1 - i), 0.92, 0.86, **kw))
for i, cname in enumerate(channels):
    axa.text(-0.12, NR - 1 - i + 0.43, cname, ha="right", va="center", fontsize=7)
for j, q in enumerate(quants):
    axa.text(j + 0.46, NR + 0.12, q, ha="center", va="bottom", fontsize=6, rotation=18)
for k, (code, txt) in enumerate(lab.items()):
    fc = cmap[code]
    kw = dict(fc=fc, ec="none") if code != 4 else dict(fc="white", ec="#999999",
                                                       lw=0.6, hatch="///")
    axa.add_patch(plt.Rectangle((0.1 + 1.0 * k, -0.95), 0.18, 0.28, **kw))
    axa.text(0.32 + 1.0 * k, -0.81, txt, fontsize=5.2, va="center")
axa.text(-1.55, -1.62, "hERG τ CV 0.13–0.19 · Nav τ$_h$(−30) CV 0.029 · Nav V½h ±4 mV per cell",
         fontsize=5.6, color=BLACK)
axa.text(-1.55, -1.95, "IKs τ$_{deact}$ CV 0.313 (real spread, reported) · Kv4.3: probe only, not sealed",
         fontsize=5.6, color=BLACK)
axa.set_xlim(-1.6, 5.1); axa.set_ylim(-2.2, NR + 0.95)
axa.set_title("portability verdicts, five channels (pre-registered)",
              fontsize=7, pad=40)
plabel(axa, "a", dx=-0.02, dy=1.10)

# ---------------- b: Nav tau_h(-40) distribution ----------------
axb = fig.add_subplot(gs[0, 3:])
taus = [fnum(r["value"]) for r in na4 if r["metric"] == "tau_decay_ms"]
taus = np.array([t for t in taus if np.isfinite(t)])
bins = np.logspace(np.log10(0.25), np.log10(8.0), 26)
axb.hist(taus, bins=bins, color=BLUE, alpha=0.8)
axb.set_xscale("log")
axb.axvline(1.03, color=BLACK, lw=1.0)
axb.annotate("median 1.03 ms", xy=(1.03, axb.get_ylim()[1] * 0.96),
             xytext=(0.32, axb.get_ylim()[1] * 0.86), fontsize=6,
             arrowprops=dict(arrowstyle="->", lw=0.6, color=BLACK))
for v in [1.64, 1.68, 3.15]:
    axb.axvline(v, color=VERM, lw=0.8, ls="--")
axb.text(3.45, axb.get_ylim()[1] * 0.70, "whole-cell trio\n(sits in upper half)",
         fontsize=5.6, color=VERM)
axb.set_xlabel(r"Nav1.5  $\tau_h$(−40 mV) across 69 patches (ms)")
axb.set_ylabel("patches")
axb.set_title("a distribution, not a number (CV 0.772) → N-arm closed", fontsize=7)
plabel(axb, "b")

# ---------------- c: CaV CDI additive component ----------------
axc = fig.add_subplot(gs[1, :2])
t = np.linspace(0, 100, 400)
f_vdi = 0.44 / (1 - np.exp(-40 / 41.6))
h_ba = 1 - f_vdi * (1 - np.exp(-t / 41.6))
h_ca = 1 - f_vdi * (1 - np.exp(-t / 50.0)) - 0.289 * (1 - np.exp(-t / 8.0))
axc.plot(t, 1 - h_ba, color=GREY, lw=1.4, label=r"Ba$^{2+}$ (VDI skeleton)")
axc.plot(t, 1 - h_ca, color=VERM, lw=1.4, label=r"Ca$^{2+}$ (+ CDI fast term)")
axc.fill_between(t, 1 - h_ba, 1 - h_ca, color=VERM, alpha=0.18)
axc.annotate("", xy=(40, 0.73), xytext=(40, 0.44),
             arrowprops=dict(arrowstyle="<->", color=BLACK, lw=0.8))
axc.text(50, 0.26, r"$\Delta f$ = 0.293" + "\n" + r"$w_{fast}$ = 0.289", fontsize=6)
axc.set_xlabel("t at +17 mV (ms)"); axc.set_ylabel("inactivated fraction")
axc.legend(frameon=False, loc="upper left")
axc.set_title('CaV1.2: CDI = "add one measured term"', fontsize=7)
plabel(axc, "c")

# ---------------- d: f_inact bars ----------------
axd = fig.add_subplot(gs[1, 2:4])
g = [r"Ba$^{2+}$ (n=4)", r"Ca$^{2+}$ (n=6)"]
med = [0.44, 0.73]; cv = [0.101, 0.141]
sd = [m * c for m, c in zip(med, cv)]
axd.bar(g, med, yerr=sd, capsize=3, color=[GREY, VERM], width=0.5,
        error_kw=dict(lw=0.9))
for i, (m_, s_) in enumerate(zip(med, sd)):
    axd.text(i, m_ + s_ + 0.015, f"{med[i]:.2f} (CV {cv[i]:.2f})",
             ha="center", fontsize=6.2)
axd.set_ylabel(r"$f_{inact}$ (+17 mV, 40 ms)")
axd.set_ylim(0, 0.95)
axd.set_title("inactivated fraction by charge carrier\n(sealed family constants)",
              fontsize=7)
plabel(axd, "d")

# ---------------- e: IKs bell + delay inset ----------------
axe = fig.add_subplot(gs[1, 4:])
V = [fnum(r["V_mV"]) for r in bell]; T = [fnum(r["tau_s"]) for r in bell]
axe.plot(V, T, "o-", color=BLUE, ms=3, lw=1.2)
axe.set_yscale("log")
axe.annotate("peak ≈ 8.9 s\n@ −10 mV", xy=(-10, 8.92), xytext=(-56, 5.2),
             fontsize=6, arrowprops=dict(arrowstyle="->", lw=0.7))
axe.set_xlabel("V (mV)"); axe.set_ylabel(r"IKs $\tau_{act}$ (s)")
axe.set_title("IKs: bell-shaped activation (Fedida 2024 Fig. 3B, digitised)",
              fontsize=7)
axin = axe.inset_axes([0.57, 0.36, 0.40, 0.40])
dv = sorted((int(k), v) for k, v in dtj["dt_full_read"].items())
axin.plot([d[0] for d in dv], [d[1] for d in dv], "s-", color=GREEN, ms=2.5, lw=0.9)
axin.set_xlabel("V (mV)", fontsize=5.5, labelpad=1.5)
axin.set_ylabel("delay (s)", fontsize=5.5, labelpad=1.5)
axin.tick_params(labelsize=5)
axin.set_title("activation delay d(V)", fontsize=6)
r2 = [fnum(r["r2_act"]) for r in ka3 if r["arm"] == "A"]
axe.text(0.02, 0.03, "forward prediction R² ≥ 0.80 in 7/8 cells\n"
         "(per-cell R²: " + ", ".join(f"{x:.2f}" for x in sorted(r2, reverse=True)) + ")",
         transform=axe.transAxes, fontsize=5.4, color=GREEN)
plabel(axe, "e")

fig.suptitle("Constancy verdicts across four cardiac channels: constants, per-cell coordinates, distributions",
             fontsize=8.5, fontweight="bold", y=1.0)
fs.save(fig, os.path.join(OUT, "Fig3_cross_channel"))
