# -*- coding: utf-8 -*-
# EDFig10: IKs mechanistic anchoring and gold-standard pipeline test (merged Kα-5/Kα-6 four-panel figure)
# a) Δt vs F2 τ2 (physical identity)  b) z_a vs Boltzmann z (effective window)
# c) gold-standard G-V recover      d) gold-standard τ_act/Δt recover
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot; setup_plot()
import numpy as np
import matplotlib.pyplot as plt

RES = Path(r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_归档\结果")
FIG = Path(r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\投稿包\02_figures")

k5 = json.load(open(RES / "判词组件_Kα5_VCF物理锚_2026-09-26.json", encoding="utf-8"))
k6 = json.load(open(RES / "判词组件_Kα6_金标准_2026-09-26.json", encoding="utf-8"))

# per-voltage comparison pairs from the Kα-5 CSV
import csv
tau2_v, tau2_m, tau2_s, dt_v, dt_m = [], [], [], [], []
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
with open(RES / "对拍_Kα5_VCF物理锚_2026-09-26.csv", encoding="utf-8-sig") as f:
    for r in csv.reader(f):
        if r and _num(r[0]) is not None:
            tau2_v.append(_num(r[0])); tau2_m.append(float(r[1])); tau2_s.append(float(r[2]))
            if r[3]:
                dt_v.append(_num(r[0])); dt_m.append(float(r[3]))
# gold-standard extractions from the Kα-6 CSV
g6 = {"V": [], "d": [], "tau": [], "rms": [], "tau_expt": [], "dt_expt": []}
with open(RES / "金标准_Kα6_SchemeV管线测试_2026-09-26.csv", encoding="utf-8-sig") as f:
    for r in csv.reader(f):
        if r and _num(r[0]) is not None:
            g6["V"].append(_num(r[0])); g6["d"].append(float(r[1])); g6["tau"].append(float(r[2]))
            g6["rms"].append(float(r[3]))
            g6["tau_expt"].append(float(r[4]) if r[4] else np.nan)
            g6["dt_expt"].append(float(r[5]) if r[5] else np.nan)

za = k5["za"]
fig = plt.figure(figsize=(10.4, 8.6))
axa = fig.add_subplot(221); axb = fig.add_subplot(222)
axc = fig.add_subplot(223); axd = fig.add_subplot(224)

# a) Δt = F2 synchronous step
axa.errorbar(tau2_v, tau2_m, yerr=tau2_s, fmt="o-", ms=5, c="C0", capsize=3,
             label=r"VCF F2 $\tau_2=1/(\delta+\gamma)$ (per oocyte)")
axa.plot(dt_v, dt_m, "s--", ms=5, c="C3", mfc="none",
         label=r"current delay $d(V)$ (author table)")
axa.set_xlabel("V (mV)"); axa.set_ylabel("time scale (s)")
axa.set_title("a  Delay $d$ = concerted VS/pore step (F2),\nmedian $d/\\tau_2$ = 0.89", fontsize=9.5)
axa.legend(fontsize=7.5)

# b) gating charge
names = list(za); zm = [za[k][0] for k in names]; ze = [za[k][1] for k in names]
xb = np.arange(len(names))
axb.bar(xb, zm, 0.55, yerr=ze, capsize=4, color="C0", alpha=0.85,
        label="limiting-slope $z_a$ (true charge)")
axb.axhspan(1.1, 1.9, color="C3", alpha=0.18,
            label="single-Boltzmann $z=RT/Fk$ ($I_{Ks}$)")
axb.axhline(25.4 / 7.5, c="C3", ls="--", lw=1.2)
axb.text(3.05, 25.4 / 7.5 + 0.08, "hERG $\\alpha$ $z\\approx$3.4", fontsize=7, c="C3")
axb.set_xticks(xb); axb.set_xticklabels(names, fontsize=8)
axb.set_ylabel("effective gating charge ($e_0$)")
axb.set_title("b  True charge vs apparent: $a_{ss}$ valid window\n$P_o \\gtrsim 0.01$ (registered)", fontsize=9.5)
axb.legend(fontsize=7.5)

# c) gold-standard G-V
gv = k6["GV"]
def boltz(V, Vh, k): return 1.0 / (1.0 + np.exp(-(V - Vh) / k))
Vf = np.linspace(-80, 110, 200)
axc.plot(Vf, boltz(Vf, gv["Vh"], gv["k"]), c="C0", lw=1.6,
         label=f"extracted from gold-standard traces\n$V_{{1/2}}$={gv['Vh']:.1f}, k={gv['k']:.1f}")
axc.plot(Vf, boltz(Vf, 13.2, 13.5), "k--", lw=1.2, label="experiment, 10 s isochronal 13.2/13.5")
axc.plot(Vf, boltz(Vf, 7.5, 17.5), ":", c="0.4", lw=1.2, label="Scheme VA equilibrium 7.5/17.5")
axc.set_xlabel("V (mV)"); axc.set_ylabel("G/Gmax")
axc.set_title("c  Gold-standard G-V recovery\n($\\Delta V_{1/2}$ = 0.9 mV vs experiment)", fontsize=9.5)
axc.legend(fontsize=7, loc="center right")

# d) gold-standard kinetics
axd.plot(g6["V"], np.array(g6["tau_expt"]), "o-", ms=4, c="0.2", label=r"expt $\tau_{act}$")
axd.plot(g6["V"], np.array(g6["tau"]) / 1000, "s--", ms=4, c="C0", mfc="none",
         label=r"$\alpha$-extracted $\tau_{act}$")
axd.plot(g6["V"], np.array(g6["dt_expt"]), "^-", ms=4, c="0.55", label=r"expt $d$")
axd.plot(g6["V"], np.array(g6["d"]) / 1000, "v--", ms=4, c="C3", mfc="none",
         label=r"$\alpha$-extracted $d$")
axd.set_xlabel("V (mV)"); axd.set_ylabel("time (s)"); axd.set_yscale("log")
axd.set_title("d  Gold-standard kinetics recovery\n(delay ratio 1.06; low-V $\\tau$ deviation = model's own)", fontsize=9.5)
axd.legend(fontsize=7)

fig.suptitle("Extended Data Fig. 10 | Mechanistic identity and gold-standard pipeline test, $I_{Ks}$",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = FIG / "EDFig10_IKs_mechanistic_goldstandard.png"
fig.savefig(out, bbox_inches="tight", dpi=300)
fig.savefig(str(out).replace(".png", ".svg"), bbox_inches="tight")
print("saved", out)
