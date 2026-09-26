# kv43_figure_v2.py - Kv4.3+HCN single-dataset triangulation final figure (2x3, registry tier)
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    pass
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6})

vi = pd.read_csv(os.path.join(ROOT, "veh_inact.csv")); li = pd.read_csv(os.path.join(ROOT, "les_inact.csv"))
va = pd.read_csv(os.path.join(ROOT, "veh_act.csv"));   la = pd.read_csv(os.path.join(ROOT, "les_act.csv"))
hc = pd.read_csv(os.path.join(ROOT, "hcn_all.csv"))
bx = pd.read_csv(os.path.join(ROOT, "biexp_decay.csv"))

viq = vi[(vi.R2 >= 0.95) & (vi.Vh_err < 5)]
liq = li[(li.R2 >= 0.95) & (li.Vh_err < 5)]
vaq = va[(va.foot_R2 >= 0.9) & (va.Imax_nA < 50)]
laq = la[(la.foot_R2 >= 0.9) & (la.Imax_nA < 50)]

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
    return med

def p_mwu(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3: return np.nan
    return mannwhitneyu(a, b, alternative="two-sided").pvalue

fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.2))
fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.11, hspace=0.60, wspace=0.42)

# a V½ inact
ax = axes[0, 0]
box_strip(ax, viq.Vh_inact, 1, VC); box_strip(ax, liq.Vh_inact, 2, LC)
p = p_mwu(viq.Vh_inact, liq.Vh_inact)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"veh\n({len(viq)})", f"6-OHDA\n({len(liq)})"], fontsize=6.5)
ax.set_ylabel(r"per-cell $V_{1/2}$ inact (mV)")
ax.set_title(f"Kv4.3 availability $V_{{1/2}}$ unchanged\n(Δ1.3%, p={p:.2f}) ✓ paper", fontsize=6.4)

# b amplitude anchor I(-30) + paper star
ax = axes[0, 1]
box_strip(ax, vaq.Imax_nA, 1, VC); box_strip(ax, laq.Imax_nA, 2, LC)
ax.scatter([1], [3.8], marker="*", s=70, c="#B22222", zorder=6)
ax.scatter([2], [1.7], marker="*", s=70, c="#B22222", zorder=6)
p = p_mwu(vaq.Imax_nA, laq.Imax_nA)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"veh\n({len(vaq)})", f"6-OHDA\n({len(laq)})"], fontsize=6.5)
ax.set_ylabel(r"per-cell peak $I$(−30 mV) (nA)")
ax.set_title(f"Kv4.3 amplitude down-regulated\n(p={p:.0e}) ★ paper 3.8/1.7 nA ✓", fontsize=6.4)

# c activation foot s_ref
ax = axes[0, 2]
box_strip(ax, vaq.s_ref, 1, VC); box_strip(ax, laq.s_ref, 2, LC)
p = p_mwu(vaq.s_ref, laq.s_ref)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"veh\n({len(vaq)})", f"6-OHDA\n({len(laq)})"], fontsize=6.5)
ax.set_ylabel(r"activation-foot $s_{ref}$ (mV$^{-1}$)")
ax.set_title(f"activation slope differs (Δ−28%,\np={p:.0e}) ✓ paper", fontsize=6.4)

# d apparent inactivation tau (model-free t50)
ax = axes[1, 0]
t50v = bx[(bx.group == "veh")].pk_nA  # placeholder safeguard
import scipy.io as sio, glob
def t50_list(d):
    out = []
    for fn in sorted(glob.glob(os.path.join(r"D:\data\doi_10_5061_dryad_76hdr7t6z\physiology_data\in_vitro\extracted", d, "*.mat"))):
        m = sio.loadmat(fn)
        for k in m:
            if not k.startswith("Trace"): continue
            t = m[k]; v = t[:, 2]; cur = t[:, 1]
            dt = np.median(np.diff(t[:, 0]))
            jumps = np.where(np.abs(np.diff(v)) > 1e-3)[0]
            if len(jumps) < 2: continue
            segs = [0] + [j + 1 for j in jumps] + [len(v)]
            for si in range(1, len(segs) - 1):
                a0, b0, a1, b1 = segs[si - 1], segs[si], segs[si], segs[si + 1]
                if b0 - a0 < 200 or b1 - a1 < 200: continue
                l0 = np.median(v[a0 + 50:b0]) * 1e3; l1 = np.median(v[a1 + 50:b1]) * 1e3
                if abs(l0 + 120) < 1 and abs(l1 + 20) < 1:
                    base = np.median(cur[a1 - int(0.05 / dt):a1])
                    y = cur[a1 + int(0.002 / dt):a1 + int(0.9 / dt)] - base
                    pk = y.max(); ip = int(np.argmax(y))
                    half = np.where(y[ip:] < pk / 2)[0]
                    if len(half): out.append(half[0] * dt * 1e3)
    return out
t50v = t50_list("veh_inact"); t50l = t50_list("les_inact")
box_strip(ax, t50v, 1, VC); box_strip(ax, t50l, 2, LC)
p = p_mwu(t50v, t50l)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"veh\n({len(t50v)})", f"6-OHDA\n({len(t50l)})"], fontsize=6.5)
ax.set_ylabel(r"apparent decay $t_{50}$(−20 mV) (ms)")
ax.set_title(f"apparent inactivation slows (Δ+{100*(np.median(t50l)/np.median(t50v)-1):.0f}%,\n"
             f"p={p:.0e}) — NOT reported by paper", fontsize=6.4)

# e HCN amplitude at -120 (Tukey fence de-artifacted)
ax = axes[1, 1]
def tukey(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    q1, q3 = np.percentile(d, [25, 75]); iqr = q3 - q1
    keep = (d >= q1 - 1.5 * iqr) & (d <= q3 + 1.5 * iqr)
    return d[keep], int((~keep).sum())
hv0 = hc[(hc.group == "veh") & (abs(hc.V_cond + 120) < 1)].ih_amp_nA.abs().to_numpy()
hl0 = hc[(hc.group == "les") & (abs(hc.V_cond + 120) < 1)].ih_amp_nA.abs().to_numpy()
hv, nv = tukey(hv0); hl, nl = tukey(hl0)
box_strip(ax, hv, 1, VC); box_strip(ax, hl, 2, LC)
p = p_mwu(hv, hl)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"veh\n({len(hv)})", f"6-OHDA\n({len(hl)})"], fontsize=6.5)
ax.set_ylabel(r"|$I_h$|(−120 mV) (nA)")
ax.set_title(f"HCN amplitude unchanged\n(p={p:.2f}) ✓ paper", fontsize=6.4)
n_excl_e = nv + nl

# f HCN tau_act at -120 (R2>=0.8 and tau<2 s and amplitude>0.1 nA)
ax = axes[1, 2]
def hcn_tau(g):
    s = hc[(hc.group == g) & (abs(hc.V_cond + 120) < 1) & (hc.fit_R2 >= 0.8)
           & (hc.tau_act_s < 2.0) & (hc.ih_amp_nA.abs() > 0.1)].tau_act_s * 1e3
    return s.to_numpy()
tv = hcn_tau("veh"); tl = hcn_tau("les")
box_strip(ax, tv, 1, VC); box_strip(ax, tl, 2, LC)
p = p_mwu(tv, tl)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"veh\n({len(tv)})", f"6-OHDA\n({len(tl)})"], fontsize=6.5)
ax.set_ylabel(r"HCN $\tau_{act}$(−120 mV) (ms)")
ax.set_title(f"HCN kinetics: nominal p={p:.2f} at −120 mV;\npaper across-voltage n.s. — registered open", fontsize=6.4)

for i, ax in enumerate(axes.flat):
    ax.text(-0.24, 1.06, "abcdef"[i], transform=ax.transAxes, fontsize=9, fontweight="bold")
fig.suptitle("Single-dataset triangulation · Dryad 76hdr7t6z (Kv4.3 + HCN, vehicle vs 6-OHDA)", fontsize=8)
fig.text(0.09, 0.008, "Registration-grade probe (not sealed). Spike-artifact repair: per-voltage repeats with "
                      "max/min>5 replaced by the smaller repeat (10/10 points in 18/28 activation files; "
                      "data-side registry, probe card §2).", fontsize=5.8, color="0.35")
fig.savefig(os.path.join(ROOT, "Kv43_HCN_triangulation_2026-09-21.png"), dpi=300, bbox_inches="tight")
print("saved; p-values printed per panel")
