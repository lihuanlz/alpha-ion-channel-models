# kv43_judge.py — Kv4.3 探针判决与图版（登记层，非封卷）
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import chi2
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    pass
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})

vi = pd.read_csv(os.path.join(ROOT, "veh_inact.csv"))
li = pd.read_csv(os.path.join(ROOT, "les_inact.csv"))
va = pd.read_csv(os.path.join(ROOT, "veh_act.csv"))
la = pd.read_csv(os.path.join(ROOT, "les_act.csv"))

# QC: R2>=0.95 且 Vh 拟合误差 < 5 mV（失活）；脚部 R2>=0.9（激活）
viq = vi[(vi.R2 >= 0.95) & (vi.Vh_err < 5)]
liq = li[(li.R2 >= 0.95) & (li.Vh_err < 5)]
vaq = va[va.foot_R2 >= 0.9]
laq = la[la.foot_R2 >= 0.9]

def cochran(x, s):
    w = 1.0 / np.asarray(s) ** 2
    x = np.asarray(x)
    xw = (w * x).sum() / w.sum()
    Q = (w * (x - xw) ** 2).sum()
    return Q, len(x) - 1

def stats(df, col, errcol=None):
    d = df[col].to_numpy(float)
    out = dict(n=len(d), med=np.median(d), cv=np.std(d, ddof=1) / abs(np.mean(d)))
    if errcol is not None:
        Q, dfree = cochran(d, df[errcol].to_numpy(float))
        out.update(Q=Q, df=dfree, Qcrit=chi2.ppf(0.95, dfree))
    return out

S = {
    "veh_Vh": stats(viq, "Vh_inact", "Vh_err"), "les_Vh": stats(liq, "Vh_inact", "Vh_err"),
    "veh_k": stats(viq, "k_inact"), "les_k": stats(liq, "k_inact"),
    "veh_s": stats(vaq, "s_ref"), "les_s": stats(laq, "s_ref"),
    "veh_tau": stats(viq.dropna(subset=["tau_inact_s"]), "tau_inact_s"),
    "les_tau": stats(liq.dropna(subset=["tau_inact_s"]), "tau_inact_s"),
}
dVh = abs(S["les_Vh"]["med"] - S["veh_Vh"]["med"]) / abs(S["veh_Vh"]["med"])
ds = abs(S["les_s"]["med"] - S["veh_s"]["med"]) / abs(S["veh_s"]["med"])
dt_ = abs(S["les_tau"]["med"] - S["veh_tau"]["med"]) / abs(S["veh_tau"]["med"])

VC, LC = "#4C72B0", "#C44E52"

def box_strip(ax, d, x, color, w=0.52):
    d = np.asarray(d, float); d = d[np.isfinite(d)]; n = len(d)
    if n == 0: return
    xj = x + np.random.default_rng(7).uniform(-0.15, 0.15, n)
    ax.scatter(xj, d, s=4, c="k", alpha=0.45, linewidths=0, zorder=3)
    med = np.median(d)
    if n >= 5:
        q1, q3 = np.percentile(d, [25, 75]); p10, p90 = np.percentile(d, [10, 90])
        ax.add_patch(Rectangle((x - w / 2, q1), w, q3 - q1, facecolor=color, alpha=0.20,
                               edgecolor=color, lw=0.9, zorder=2))
        for y0, y1 in [(p10, q1), (q3, p90)]:
            ax.plot([x, x], [y0, y1], color=color, lw=0.9, zorder=2)
        ax.plot([x - w / 2, x + w / 2], [med, med], color=color, lw=2.0, zorder=4)
    else:
        ax.plot([x - 0.26, x + 0.26], [med, med], color=color, lw=1.8, zorder=4)

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.11, hspace=0.55, wspace=0.35)

# a: 失活中位 V½ 两组
ax = axes[0, 0]
box_strip(ax, viq.Vh_inact, 1, VC); box_strip(ax, liq.Vh_inact, 2, LC)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"vehicle\n({len(viq)})", f"6-OHDA\n({len(liq)})"], fontsize=6.5)
ax.set_ylabel(r"per-cell $V_{1/2}$ inact (mV)")
q1, q2 = S["veh_Vh"], S["les_Vh"]
ax.set_title(f"avail. $V_{{1/2}}$: CV {q1['cv']:.3f}/{q2['cv']:.3f}; "
             f"Δ med {dVh*100:.1f}% across groups", fontsize=6.6)

# b: k_inact 两组
ax = axes[0, 1]
box_strip(ax, viq.k_inact, 1, VC); box_strip(ax, liq.k_inact, 2, LC)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"vehicle\n({len(viq)})", f"6-OHDA\n({len(liq)})"], fontsize=6.5)
ax.set_ylabel(r"per-cell $k_{inact}$ (mV)")
ax.set_title(f"avail. slope k: med {S['veh_k']['med']:.1f}/{S['les_k']['med']:.1f} mV, "
             f"CV {S['veh_k']['cv']:.2f}/{S['les_k']['cv']:.2f}", fontsize=6.6)

# c: 激活脚部 s_ref 两组
ax = axes[1, 0]
box_strip(ax, vaq.s_ref, 1, VC); box_strip(ax, laq.s_ref, 2, LC)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"vehicle\n({len(vaq)})", f"6-OHDA\n({len(laq)})"], fontsize=6.5)
ax.set_ylabel(r"activation-foot $s_{ref}$ (mV$^{-1}$)")
ax.set_title(f"activation foot (−60..−15 mV): med {S['veh_s']['med']:.3f}/{S['les_s']['med']:.3f}, "
             f"Δ {ds*100:.0f}%", fontsize=6.6)

# d: τ_inact(−20 mV) 两组
ax = axes[1, 1]
vt = viq.tau_inact_s.dropna() * 1e3; lt = liq.tau_inact_s.dropna() * 1e3
box_strip(ax, vt, 1, VC); box_strip(ax, lt, 2, LC)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"vehicle\n({len(vt)})", f"6-OHDA\n({len(lt)})"], fontsize=6.5)
ax.set_ylabel(r"$\tau_{inact}$(−20 mV) (ms)")
ax.set_title(f"inactivation kinetics: med {S['veh_tau']['med']*1e3:.0f}/{S['les_tau']['med']*1e3:.0f} ms, "
             f"Δ {dt_*100:.0f}%", fontsize=6.6)

for i, ax in enumerate(axes.flat):
    ax.text(-0.22, 1.06, "abcd"[i], transform=ax.transAxes, fontsize=9, fontweight="bold")
fig.suptitle("Kv4.3 probe · Dryad 76hdr7t6z (SNc DA neurons, vehicle vs 6-OHDA lesion)", fontsize=8)
fig.text(0.10, 0.008, "Exploratory registration-grade probe (not sealed). QC: fit R²≥0.95 & σ(V½)<5 mV "
                      "(availability), foot R²≥0.9 (activation).", fontsize=6, color="0.35")
fig.savefig(os.path.join(ROOT, "Kv43_probe_2026-09-21.png"), dpi=300, bbox_inches="tight")

print("== inact QC ==  veh %d/%d  les %d/%d" % (len(viq), len(vi), len(liq), len(li)))
for k, v in S.items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
print("dVh%% = %.1f  ds%% = %.1f  dtau%% = %.1f" % (dVh * 100, ds * 100, dt_ * 100))
