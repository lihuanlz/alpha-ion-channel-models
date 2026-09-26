# make_percell_violin_v02_2026-09-21.py
# ED8 v02: small-sample panels de-violinized (n<20 plots points + median/IQR only), all linear axes,
# IKs gains the Chan source anchor star (25.4 mV), CaV gains the literature E_rev +46 mV dashed line.
# Data source: 结果\逐细胞参数总表_四通道_2026-09-21.csv (single source of truth, figure-table consistency guaranteed)
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
LINE = os.path.dirname(ROOT)
RES = os.path.join(LINE, "结果")
FIGD = os.path.join(ROOT, "figures_v01")
os.makedirs(FIGD, exist_ok=True)

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    pass
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

df = pd.read_csv(os.path.join(RES, "逐细胞参数总表_四通道_2026-09-21.csv"))
df = df[df["ok"] == 1]

def L(channel, param, voltage=None, group=None, temp=None, dataset=None):
    d = df[(df.channel == channel) & (df.parameter == param)]
    if voltage is not None:
        d = d[d.voltage_mV == voltage]
    if group is not None:
        d = d[d.group == group]
    if temp is not None:
        d = d[d.temp_C.astype(str) == str(temp)]
    if dataset is not None:
        d = d[d.dataset == dataset]
    return d["value"].to_numpy(float)

VIOLIN_MIN_N = 20   # rule: density violin only for n>=20, otherwise points only

def panel(ax, data_list, labels, color, star=None, star_label="", tick_fs=6.0,
          special_marker=None):
    """data_list: list of arrays; special_marker: dict {index: marker_dict} to swap markers for a whole group"""
    pos = np.arange(1, len(data_list) + 1)
    for i, d in enumerate(data_list):
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
        if len(d) == 0:
            continue
        if len(d) >= VIOLIN_MIN_N:
            vp = ax.violinplot([d], positions=[pos[i]], widths=0.72,
                               showmeans=False, showmedians=False, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(color); b.set_alpha(0.25)
                b.set_edgecolor(color); b.set_linewidth(0.8)
        mk = dict(s=3.5, c="k", alpha=0.45, linewidths=0, zorder=3)
        if special_marker and i in special_marker:
            mk.update(special_marker[i])
        xj = pos[i] + np.random.default_rng(7).uniform(-0.16, 0.16, len(d))
        ax.scatter(xj, d, **mk)
        ax.hlines(np.median(d), pos[i]-0.28, pos[i]+0.28, colors=color, lw=1.6, zorder=4)
        if 2 <= len(d) < VIOLIN_MIN_N:
            q1, q3 = np.percentile(d, [25, 75])
            ax.vlines(pos[i]+0.28, q1, q3, colors=color, lw=1.0, zorder=4)
            ax.hlines([q1, q3], pos[i]+0.22, pos[i]+0.34, colors=color, lw=1.0, zorder=4)
    ax.set_xticks(pos)
    ax.set_xticklabels(["{}\n({})".format(l, len(d)) for l, d in zip(labels, data_list)],
                       fontsize=tick_fs)
    if star is not None:
        ax.scatter([star[1]], [star[0]], marker="*", s=80, c="#B22222", zorder=6)
        ax.annotate(star_label, (star[1], star[0]), textcoords="offset points",
                    xytext=(7, 3), fontsize=6, color="#B22222")
    ax.spines[["top", "right"]].set_visible(False)

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
TEMPS = [25, 27, 30, 33, 37]

# a: hERG E_rev × T
ax = axes[0, 0]
panel(ax, [L("hERG", "E_rev", temp=t) for t in TEMPS], [str(t) for t in TEMPS],
      "#4C72B0", star=(-92.9, 5.0), star_label="model62-chain fit −92.9", tick_fs=5.5)
ax.set_ylabel("per-cell $E_{rev}$ (mV)")
ax.set_xlabel("temperature (°C)")
ax.set_title("hERG · Lei CHO 670 wells", fontsize=7)

# b: hERG h_ss limb 25 °C
ax = axes[0, 1]
panel(ax, [L("hERG", "h_ss", voltage=v, temp=25) for v in [-140, -120, -100]],
      ["−140", "−120", "−100"], "#4C72B0")
ax.set_ylabel("per-cell $h_{ss}$")
ax.set_xlabel("voltage (mV)")
ax.set_title("hERG · $h_{ss}$ negative limb, 25 °C", fontsize=7)

# c: hERG tau_rec(-120) x T - linear axis
ax = axes[0, 2]
panel(ax, [L("hERG", "tau_rec", voltage=-120, temp=t) for t in TEMPS],
      [str(t) for t in TEMPS], "#4C72B0", star=(3.04, 5.0), star_label="Beattie HEK 3.04 ms", tick_fs=5.5)
ax.set_ylabel(r"per-cell $\tau_{rec}$(−120 mV) (ms)")
ax.set_xlabel("temperature (°C)")
ax.set_ylim(0, 18)
ax.set_yticks([0, 3, 6, 9, 12, 15, 18])
ax.set_title("hERG · recovery τ travels: $Q_{10}$=2.84, $R^2$=0.991", fontsize=7)

# d: Nav1.5 tau(-40) by modality - linear axis; whole-cell 3 points as red diamonds
ax = axes[1, 0]
d1 = L("Nav1.5", "tau_decay", group="多通道膜片")
d2 = L("Nav1.5", "tau_decay", group="单通道膜片")
d3 = L("Nav1.5", "tau_h", voltage=-40.0, dataset="Nα2-Lei-Nav35")
panel(ax, [d1, d2, d3], ["multi\nch", "single\nch", "whole\ncell"],
      "#DD8452", tick_fs=6.0,
      special_marker={2: dict(marker="D", s=14, c="#B22222", alpha=0.9)})
cv1 = np.std(d1) / np.mean(d1)
ax.set_ylim(0, 7.5)
ax.set_title(f"Nav1.5 · τ(−40 mV): a distribution (CV={cv1:.2f})", fontsize=6.5)
ax.set_ylabel("per-patch / per-cell τ (ms)")
ax.set_xlabel("recording modality")

# e: IKs V1/2 / k / tau_app - all 8 cells; Chan source anchor star 25.4 mV
ax = axes[1, 1]
e1 = L("IKs", "V_half_act"); e2 = L("IKs", "k_act")
panel(ax, [e1, e2], ["$V_{1/2}$ act", "$k$ act"], "#55A868", tick_fs=6.5)
ax.set_ylabel("mV")
ax.set_ylim(10, 35)
ax.scatter([1], [25.4], marker="*", s=80, c="#B22222", zorder=6)
ax.text(0.62, 11.5, "★ Chan 2023 anchor 25.4", fontsize=6, color="#B22222")
ax2 = ax.twinx()
e3 = L("IKs", "tau_app")
xj = 3 + np.random.default_rng(7).uniform(-0.16, 0.16, len(e3))
ax2.scatter(xj, e3, s=10, c="#2d5a3a", alpha=0.8, linewidths=0)
ax2.hlines(np.median(e3), 3-0.28, 3+0.28, colors="#55A868", lw=1.6)
ax2.set_ylabel(r"$\tau_{app}$(−40 mV) (s)", color="#2d5a3a")
ax2.set_ylim(0, 1.0)
ax2.spines["top"].set_visible(False)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([f"$V_{{1/2}}$\n({len(e1)})", f"$k$\n({len(e2)})",
                    f"$\\tau_{{app}}$\n({len(e3)})"], fontsize=6.2)
ax.set_title("IKs · Chan 2023 WT pool, complete record", fontsize=6.5)

# f: CaV1.2 E_chord / Q10(tau_slow) - all points; literature anchor +46 mV dashed line
ax = axes[1, 2]
f1 = L("CaV1.2", "E_chord")
panel(ax, [f1], ["$E_{chord}$"], "#8172B3", tick_fs=6.5)
ax.axhline(46, ls="--", lw=0.9, color="#B22222", zorder=2)
ax.text(0.62, 41.5, "literature anchor +46 mV", fontsize=6, color="#B22222")
ax.set_ylabel("mV")
ax.set_ylim(28, 66)
ax2 = ax.twinx()
f2 = L("CaV1.2", "Q10_tau_slow")
xj = 2 + np.random.default_rng(7).uniform(-0.10, 0.10, len(f2))
ax2.scatter(xj, f2, s=12, c="#4a3d6b", alpha=0.85, linewidths=0)
ax2.hlines(np.median(f2), 2-0.28, 2+0.28, colors="#8172B3", lw=1.6)
ax2.set_ylabel(r"$Q_{10}(\tau_{slow})$", color="#4a3d6b")
ax2.set_ylim(0.5, 2.5)
ax2.spines["top"].set_visible(False)
ax.set_xticks([1, 2])
ax.set_xticklabels([f"$E_{{chord}}$\n({len(f1)})", f"$Q_{{10}}$($\\tau_{{slow}}$)\n({len(f2)})"],
                   fontsize=6.5)
ax.set_title("CaV1.2 · Ren 2022 (Ca$^{2+}$), complete record", fontsize=6.5)

for i, ax in enumerate(axes.flat):
    ax.text(-0.22, 1.04, "abcdef"[i], transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")
fig.tight_layout(w_pad=2.2, h_pad=1.6)

# figure footnote
fig.text(0.01, 0.005,
         "Violin density shown only for groups with n ≥ 20; smaller groups show every cell/patch. "
         "d–f display the complete QC-passed public record for each channel (SI S9).",
         fontsize=6, color="0.35")

png = os.path.join(FIGD, "EDFig_percell_violin_v02.png")
pdf = os.path.join(FIGD, "EDFig_percell_violin_v02.pdf")
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
print("PNG pixels:", Image.open(png).size)
print("saved:", png)
