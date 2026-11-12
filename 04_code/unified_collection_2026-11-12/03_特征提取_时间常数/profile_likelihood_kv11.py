# -*- coding: utf-8 -*-
"""Tier-1 轮廓似然展示: 12 个 Kv1.1@25 细胞 tau_act(+20mV) 的不对称 CI vs pcov 对称 SE,
并量化"换用轮廓似然权重后组散布变多少"——闭合方法学评估卡第 1 层。
波形读取与拟合严格复用冻结提取器 extract_features_v2 的逻辑(rep2, 0-700ms, 基线<100ms)。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np, pandas as pd, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from daimon_runtime import setup_plot
setup_plot()

R = Path(__file__).parent
sys.path.insert(0, str(R))
from extract_features_v2 import fit_tau_rise_v2, _rise, ACT_V

corpus = pd.read_csv(R / "features" / "features_corpus_v5r.csv", low_memory=False)
pool = corpus[(corpus.channel == "Kv1.1") & (corpus.temp_nominal == 25) & (corpus.flag == "active")
              & (corpus.parameter == "tau_act") & (corpus.voltage_mV == 20)
              & (corpus.r2 >= 0.9)].head(12)
print("展示细胞:", pool.cell_id.tolist())

def profile_ci(tt, yy, p_fix, tau_hat, t0, n_grid=60):
    """固定 p, 剖面 tau: 每个 tau 网格点只优化 (Iss, c)。"""
    n = len(yy)
    def rss_of(tau):
        f = lambda t_, Iss, c: _rise(t_, Iss, tau, p_fix, c, t0)
        try:
            pb, _ = curve_fit(f, tt, yy, p0=[yy.max(), yy[0]], maxfev=8000)
            return float(np.sum((yy - f(tt, *pb)) ** 2))
        except Exception:
            return np.nan
    rss0 = rss_of(tau_hat)
    lo_g = np.linspace(max(tau_hat * 0.4, 0.05), tau_hat, n_grid // 2)
    hi_g = np.linspace(tau_hat, tau_hat * 2.5, n_grid // 2)
    thr = rss0 * np.exp(2 * 1.92 / n)   # ΔnLL = n/2 * ln(RSS/RSS0) = 1.92
    lo = hi = np.nan
    rl = [rss_of(t) for t in lo_g]
    for t, r in zip(lo_g[::-1], rl[::-1]):
        if np.isfinite(r) and r > thr: lo = t; break
    rh = [rss_of(t) for t in hi_g]
    for t, r in zip(hi_g, rh):
        if np.isfinite(r) and r > thr: hi = t; break
    return rss0, lo, hi

rows = []
for cid in pool.cell_id:
    fp = R / "nwb_full" / f"c{int(cid)}.nwb"
    with h5py.File(fp, "r") as f:
        d = f["acquisition/timeseries/Activation/repetitions/repetition2/data"][:]
        if d.ndim == 1: d = d[:, None]
        d = d.T
        n = d.shape[1]; t = np.linspace(0, 700, n)
        si = ACT_V.index(20)
        i0 = d[si].astype(float) - d[si][t < 100].mean()
        ft = fit_tau_rise_v2(t, i0, 100, 600)
        if ft is None: continue
        m = (t >= 100) & (t <= 600); tt = t[m]; ii = i0[m]
        pk = int(np.argmax(ii)); tt = tt[:pk + 1]; yy = ii[:pk + 1]
        rss0, lo, hi = profile_ci(tt, yy, ft["p"], ft["tau"], 100)
        se_sym = ft["tau_se"]
        rows.append(dict(cell_id=int(cid), tau=ft["tau"], p=ft["p"], r2=ft["r2"],
                         se_pcov=se_sym, prof_lo=lo, prof_hi=hi,
                         asym=(hi - ft["tau"]) / max(ft["tau"] - lo, 1e-9) if np.isfinite(lo) and np.isfinite(hi) else np.nan))
        print(f"  c{int(cid)}: tau={ft['tau']:.3f} p={ft['p']} r2={ft['r2']:.3f}  "
              f"pcov±1.96se=[{ft['tau']-1.96*se_sym:.3f},{ft['tau']+1.96*se_sym:.3f}]  "
              f"profile95=[{lo:.3f},{hi:.3f}]  不对称度(上/下)={rows[-1]['asym']:.2f}")

df = pd.DataFrame(rows)
df.to_csv(R / "features" / "profile_likelihood_kv11_demo.csv", index=False)

# ---- 组散布影响: pcov-SE 权重 vs 轮廓-SE 权重 喂给同一 EB ----
LN10 = np.log(10)
def eb_sigma(y, s):
    def nll(sv):
        sv = max(float(sv), 0.0)
        w = 1.0 / (sv * sv + s * s)
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * np.sum(np.log(2 * np.pi * (sv * sv + s * s)) + (y - mu) ** 2 * w)
    from scipy.optimize import minimize_scalar
    return max(minimize_scalar(nll, bounds=(0, max(3 * np.std(y), 0.3)), method="bounded").x, 0.0)

t20 = corpus[(corpus.channel == "Kv1.1") & (corpus.temp_nominal == 25) & (corpus.flag == "active")
             & (corpus.parameter == "tau_act") & (corpus.voltage_mV == 20)
             & (corpus.r2 >= 0.7) & (corpus.value > 0)]
y = np.log10(t20.value.values)
s_pcov = np.clip(t20.se.values / (t20.value.values * LN10), 1e-4, None)
sig_pcov = eb_sigma(y, s_pcov)
# 用展示细胞估典型不对称放大因子, 修正全组 SE
f_asym = np.nanmedian((df.prof_hi - df.prof_lo) / (2 * 1.96 * df.se_pcov))
s_prof = s_pcov * f_asym
sig_prof = eb_sigma(y, s_prof)
print(f"\nKv1.1@25 tau_act(+20mV) 组散布 (n={len(y)}):")
print(f"  观测 SD = {y.std(ddof=1):.4f}")
print(f"  EB(pcov SE)   = {sig_pcov:.4f}")
print(f"  EB(轮廓似然SE, 典型展宽因子 {f_asym:.2f}) = {sig_prof:.4f}")
print(f"  相对差 = {100 * (sig_prof - sig_pcov) / sig_pcov:+.2f}%")

# ---- 图: 3x4 小图阵, 每细胞 pcov 对称区间 vs 轮廓区间 ----
C = dict(blue="#3A6FF7", red="#E63946", teal="#12B5A5", dark="#22223B", grey="#8D99AE")
fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharex=True)
for ax, (_, r) in zip(axes.flat, df.iterrows()):
    ax.errorbar([0], [r.tau], yerr=[[1.96 * r.se_pcov], [1.96 * r.se_pcov]], fmt="o",
                color=C["grey"], capsize=5, lw=2, ms=7, label="pcov 对称 ±1.96SE")
    ax.errorbar([0.35], [r.tau], yerr=[[r.tau - r.prof_lo], [r.prof_hi - r.tau]], fmt="s",
                color=C["red"], capsize=5, lw=2, ms=7, label="轮廓似然 95%")
    ax.set_title(f"c{r.cell_id}  p={r.p}, R²={r.r2:.2f}", fontsize=10)
    ax.set_xlim(-0.3, 0.65); ax.set_xticks([])
h_, l_ = axes[0][0].get_legend_handles_labels()
fig.legend(h_, l_, fontsize=10, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 0.005))
fig.suptitle("轮廓似然 vs pcov 对称标准误：τ_act(+20 mV) 的真实不确定度是不对称的（Kv1.1@25，12 细胞）",
             fontsize=13.5, fontweight="bold", color=C["dark"])
fig.supylabel("τ_act（ms）")
fig.tight_layout(rect=(0, 0.05, 1, 0.95))
out = R / "features" / "figS_profile_likelihood_2026-11-08.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print("saved", out)
