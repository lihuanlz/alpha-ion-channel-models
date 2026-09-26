# -*- coding: utf-8 -*-
# ED Fig. 7 v05: population validation on the 211-cell automated-patch panel (25 C).
# English re-render of the former EDFig3_Lei211群体验证 (identical sealed numbers).
# a E_rev histogram; b h_ss(V); c m_ss(V); d tau_rec(V) ladder; e raw ap1hz trace;
# f J5 AP-clamp forward scores (A1/A2 per frequency).
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle_v05 as fs

fs.apply()

ROOT = os.path.dirname(os.path.abspath(__file__))
LINE = os.path.dirname(ROOT)
RES = os.path.join(LINE, "结果")
AM = os.path.join(LINE, "α模型")
DATA = os.path.join(LINE, "数据", "Lei全量")
FIGD = os.path.join(ROOT, "figures_v01")

BLUE, ORANGE, GREEN, GREY, BLACK = "#0072B2", "#E69F00", "#009E73", "#8A8A8A", "#222222"
VERM = "#D55E00"

df = pd.read_csv(os.path.join(RES, "逐细胞参数总表_四通道_2026-09-21.csv"))
h = df[(df.channel == "hERG") & (df.ok == 1) & (df.temp_C.astype(str) == "25")]

tauh = json.load(open(os.path.join(AM, "lei211_tauh_表_herg25oc1.json"), encoding="utf-8"))
j5 = json.load(open(os.path.join(AM, "lei211_J5_结果_herg25oc1.json"), encoding="utf-8"))

fig = plt.figure(figsize=(7.2, 4.8))
gs = fig.add_gridspec(2, 3, hspace=0.85, wspace=0.55,
                      left=0.075, right=0.985, top=0.90, bottom=0.13)


def plab(ax, s):
    ax.text(-0.24, 1.10, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")


# ---- a: E_rev histogram ----
ax = fig.add_subplot(gs[0, 0])
erev = h[h.parameter == "E_rev"]["value"].to_numpy(float)
erev = erev[np.isfinite(erev)]
ax.hist(erev, bins=np.arange(-88.5, -71.5, 0.5), color=GREY, alpha=0.85)
med = np.median(erev)
ax.axvline(med, color=VERM, lw=1.0)
ax.text(med + 0.35, ax.get_ylim()[1] * 0.72,
        f"median {med:.1f} mV\nCV {np.std(erev)/abs(np.mean(erev)):.3f}\n"
        f"n = {len(erev)} (sealed)",
        fontsize=6, color=VERM, ha="left")
ax.set_xlabel("per-cell $E_{rev}$ (mV)", fontsize=7)
ax.set_ylabel("cells", fontsize=7)
ax.set_title("reversal potential is constant", fontsize=7)
plab(ax, "a")

# ---- b: h_ss(V) ----
ax = fig.add_subplot(gs[0, 1])
d = h[h.parameter == "h_ss"]
Vs = sorted(d.voltage_mV.unique())
meds, allv = [], []
for v in Vs:
    x = d[d.voltage_mV == v]["value"].to_numpy(float)
    x = x[np.isfinite(x)]
    meds.append(np.median(x))
    allv.append(x)
for i, x in enumerate(allv):
    xj = Vs[i] + np.random.default_rng(3).uniform(-1.5, 1.5, len(x))
    ax.scatter(xj, x, s=1.2, color=GREY, alpha=0.25, lw=0)
ax.plot(Vs, meds, "o-", color=BLUE, ms=3.5, lw=1.1)
ax.set_ylim(-0.05, 1.25)
ax.set_xlabel("V (mV)", fontsize=7)
ax.set_ylabel("$h_{ss}$ (median ± cells)", fontsize=7)
ax.set_title("$h_{ss}$(V): limbs sealed, mid registered", fontsize=7)
ax.text(0.03, 0.30, "1.000 @ −140 mV\n0.955 @ −120 mV\n(CV ≤ 0.065)",
        transform=ax.transAxes, fontsize=6, color=BLUE)
plab(ax, "b")

# ---- c: m_ss(V) ----
ax = fig.add_subplot(gs[0, 2])
d = h[h.parameter == "m_ss"]
Vs = sorted(d.voltage_mV.unique())
meds = []
for i, v in enumerate(Vs):
    x = d[d.voltage_mV == v]["value"].to_numpy(float)
    x = x[np.isfinite(x)]
    meds.append(np.median(x))
    xj = v + np.random.default_rng(3).uniform(-1.5, 1.5, len(x))
    ax.scatter(xj, x, s=1.2, color=GREY, alpha=0.25, lw=0)
ax.plot(Vs, meds, "s-", color=ORANGE, ms=3.5, lw=1.1)
ax.set_ylim(-0.05, 1.25)
ax.set_xlabel("V (mV)", fontsize=7)
ax.set_ylabel("$m_{ss}$ (1 s apparent)", fontsize=7)
ax.set_title("$m_{ss}$(V): lower bound, +25/+40 sealed", fontsize=7)
plab(ax, "c")

# ---- d: tau_rec(V) ladder ----
ax = fig.add_subplot(gs[1, 0])
Vd = ["-140", "-120", "-100", "-80", "-60", "-40", "-20"]
xs = [int(v) for v in Vd]
ys = [tauh[v]["median"] * 1000 for v in Vd]
ns = [tauh[v]["n"] for v in Vd]
ax.plot(xs, ys, "o-", color=GREEN, ms=3.5, lw=1.1)
for x, y, n in zip(xs, ys, ns):
    ax.text(x, y + 1.2, f"n={n}", ha="center", fontsize=5.2, color=GREEN)
ax.set_xlabel("V (mV)", fontsize=7)
ax.set_ylabel(r"$\tau_{rec}$ (ms)", fontsize=7)
ax.set_ylim(0, 26)
ax.set_title("recovery ladder: ordering preserved", fontsize=7)
ax.text(0.03, 0.96, "deactivation slow layer:\n0.74 s @ −60 mV vs 3.03 s @ −40 mV",
        transform=ax.transAxes, fontsize=6, color=BLACK, va="top")
plab(ax, "d")

# ---- e: raw ap1hz trace (cell A01) ----
ax = fig.add_subplot(gs[1, 1])
fp = os.path.join(DATA, "ap1hz", "herg25oc1-ap1hz-A01.csv")
I = np.genfromtxt(fp, delimiter=",")
I = np.ravel(I)
I = I[np.isfinite(I)]
dt = 1e-4
n4 = int(4.0 / dt)
t = np.arange(min(n4, len(I))) * dt
ax.plot(t, I[: len(t)], lw=0.4, color="0.55")
ax.set_xlabel("t (s)", fontsize=7)
ax.set_ylabel("I (pA)", fontsize=7)
ax.set_title("measured AP clamp 1 Hz (cell A01):\nper-beat repolarisation rebound visible",
             fontsize=6.5)
plab(ax, "e")

# ---- f: J5 AP forward scores ----
ax = fig.add_subplot(gs[1, 2])
protos = [("ap05hz", "0.5 Hz"), ("ap1hz", "1 Hz"), ("ap2hz", "2 Hz")]
a1, a2 = [], []
cells = [c for c in j5.values() if isinstance(c, dict) and "ap1hz" in c]
for p, _ in protos:
    a1.append(np.mean([bool(c[p].get("A1")) for c in cells]))
    a2.append(np.mean([bool(c[p].get("A2")) for c in cells]))
x = np.arange(3)
ax.bar(x - 0.18, a1, width=0.34, color=BLUE, label="A1 rebound per cycle")
ax.bar(x + 0.18, a2, width=0.34, color="#9467BD", label="A2 magnitude RMS in [0.3, 3]")
for i, (v1, v2) in enumerate(zip(a1, a2)):
    ax.text(i - 0.18, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=5.6, color=BLUE)
    ax.text(i + 0.18, v2 + 0.02, f"{v2:.2f}", ha="center", fontsize=5.6, color="#9467BD")
ax.text(-0.18, a1[0] + 0.10, "A1", ha="center", fontsize=6.5, color=BLUE, fontweight="bold")
ax.text(0.18, a2[0] + 0.10, "A2", ha="center", fontsize=6.5, color="#9467BD", fontweight="bold")
ax.axhline(0.6, color=VERM, ls="--", lw=0.8)
ax.text(-0.45, 0.64, "pass line 0.6", fontsize=5.6, color=VERM, ha="left", va="bottom")
ax.set_xticks(x)
ax.set_xticklabels([lab for _, lab in protos], fontsize=7)
ax.set_ylim(0, 1.12)
ax.set_ylabel("fraction of cells", fontsize=7)
ax.text(0.5, -0.34, "A1: repolarisation rebound scored per cycle; A2: magnitude RMS ratio in [0.3, 3]",
        transform=ax.transAxes, fontsize=5.6, color="0.35", ha="center")
ax.set_title("AP-clamp forward: A3 direction 209/209;\nA1/A2 combined 45.8% (registered)",
             fontsize=6.5)
plab(ax, "f")

fig.suptitle("Population validation: 211 QC-passed CHO cells, 25 °C, nine protocols per cell (Lei panel)",
             fontsize=7.5, y=0.985)

fs.save(fig, os.path.join(FIGD, "EDFig7_Lei211_population_validation"))
