# -*- coding: utf-8 -*-
# ED Fig. 1 v05: hERG per-cell forward-prediction panels, all nine cells.
# Left: held-out sinusoidal protocol; right: held-out AP clamp.
# Computation reuses the sealed verdict modules verbatim (import, no edits).
import os, json, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
LINE = os.path.dirname(ROOT)
AM = os.path.join(LINE, "α模型")
FIGD = os.path.join(ROOT, "figures_v01")


def load_mod(fname, alias):
    spec = importlib.util.spec_from_file_location(alias, os.path.join(AM, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sine = load_mod("2026-09-13_α模型_hss实测表重跑_sine前向判决.py", "sealed_sine")
ap = load_mod("2026-09-13_α模型_hss实测表重跑_AP前向判决.py", "sealed_ap")

import figstyle_v05 as fs  # re-apply after the modules set their own rcParams
fs.apply()

amp = json.load(open(sine.F_AMP, encoding="utf-8"))
hook = json.load(open(sine.F_HOOK, encoding="utf-8"))
inact = json.load(open(sine.F_INACT, encoding="utf-8"))
hss = json.load(open(sine.F_HSS, encoding="utf-8"))

# measured sine rebound targets (verbatim logic from the sealed module main)
meas = {}
for r in hook["B_rows"]:
    if r["valid"] and r["A"] > 0:
        meas.setdefault(r["cell"], {})[r["which"]] = r["A"]
for r in hook["B_rows"]:
    d = meas.setdefault(r["cell"], {})
    if r["which"] not in d and r["A"] != 0:
        d[r["which"]] = abs(r["A"])
meas = {c: (d[0], d[1]) for c, d in meas.items() if 0 in d and 1 in d}

CELLS = ap.CELLS
sine_res, ap_res = {}, {}
for c in CELLS:
    V, I = sine.load_mat("sine_wave_protocol.mat", c, "sine_wave")
    Vi, Ii = sine.load_mat("inactivation_protocol.mat", c, "inactivation")
    smea40 = None
    if I is not None:
        stc = sine.find_sine_structure(V)
        if stc["long40"]:
            v, s0, nseg = stc["long40"][0]
            smea40 = float(np.mean(I[s0 + nseg - 2000: s0 + nseg]))
    tabs = sine.build_tabs(c, amp, hook, inact, hss, Ii, Vi, sine_mea40=smea40)
    r = sine.score_cell(c, V, I, tabs, meas, None)
    sine_res[c] = (V, I, r)

    Va, Ia = ap.load_mat("ap_protocol.mat", c, "ap")
    tabsa = ap.build_tabs(c, amp, hook, inact, hss, Ii, Vi)
    ra = ap.score_cell(c, Va, Ia, tabsa)
    ap_res[c] = (Va, Ia, ra)

npass_sine = sum(1 for _, _, r in sine_res.values() if r["pass"])
npass_ap = sum(1 for _, _, r in ap_res.values() if r["pass"])
print("sine pass:", npass_sine, " AP pass:", npass_ap)
assert npass_sine == 7 and npass_ap == 8, "sealed verdicts must reproduce"

# ---------------- figure ----------------
fig, axes = plt.subplots(9, 2, figsize=(7.2, 10.8), sharex=False)
plt.subplots_adjust(left=0.075, right=0.99, top=0.965, bottom=0.03,
                    hspace=0.55, wspace=0.16)
DT = 1e-4

for i, c in enumerate(CELLS):
    for j, (res, name) in enumerate(((sine_res, "sinusoidal protocol"),
                                     (ap_res, "AP clamp"))):
        ax = axes[i, j]
        V, I, r = res[c]
        tt = np.arange(len(V)) * DT
        q = 5  # decimate for file size
        if I is not None:
            ax.plot(tt[::q], I[::q], lw=0.3, color="0.55")
        ax.plot(tt[::q], r["_sim"][::q], lw=0.4, color="#C03030", alpha=0.85)
        verdict = "pass" if r["pass"] else "FAIL"
        col = "#1a7a4a" if r["pass"] else "#B23A3A"
        if j == 0:
            detail = " ".join(f"{k}{'+' if r.get(k) else '-'}" for k in ("S1", "S2", "S3"))
        else:
            detail = " ".join(f"{k}{'+' if r.get(k) else '-'}" for k in ("A1", "A2", "A3"))
        ax.set_title(f"cell {c}   {verdict} ({detail})", fontsize=5.6,
                     color=col, loc="left", pad=1.5)
        ax.tick_params(labelsize=5, length=1.5)
        if i == 8:
            ax.set_xlabel("t (s)", fontsize=6)
        if j == 0:
            ax.set_ylabel("I (nA)", fontsize=6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

fig.text(0.305, 0.972, "held-out sinusoidal protocol", fontsize=7,
         fontweight="bold", ha="center")
fig.text(0.76, 0.972, "held-out AP clamp", fontsize=7,
         fontweight="bold", ha="center")

fig.text(0.075, 0.995, "hERG forward prediction with one free per-cell number (G): "
         "7/9 sine, 8/9 AP pass the pre-registered thresholds; "
         "residual whiteness 28/33 tail segments (SI S2). Grey, measured; red, model.",
         fontsize=6.4, va="top")

fig.suptitle("hERG forward predictions: all nine cells, all held-out protocols (28/33 pass)",
             fontsize=8.5, fontweight="bold", y=1.01)
fs.save(fig, os.path.join(FIGD, "EDFig1_hERG_forward_panels"))
