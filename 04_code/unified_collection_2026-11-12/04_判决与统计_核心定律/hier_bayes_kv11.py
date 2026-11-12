# -*- coding: utf-8 -*-
"""分层贝叶斯（经验贝叶斯·边际似然）稳健性检验 —— Kv1.1 三温度展示组
模型: 每电压 v, 细胞 c:
  y_cv = log10(tau_cv),  s_cv = se/(tau*ln10)   (delta 法转到 log10)
  y_cv ~ N( mu_v, sigma_v^2 + s_cv^2 )          (法-法层级, 细胞层解析积出)
对 (mu, sigma) 最大边际似然; sigma 的 95% CI 用剖面似然 (LR 降 1.92)。
对照现行管线: sigma_obs = SD(y); sigma_true_wd = sqrt(max(var - mean(s^2), 0))。
冻结口径: tau_act, r2>=0.7, value>0, V in [0,40]。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from daimon_runtime import setup_plot
setup_plot()

R = Path(__file__).parent
LN10 = np.log(10)

def eb_fit(y, s):
    """返回 mu, sigma, (sigma_lo, sigma_hi) —— 剖面似然 95% CI。"""
    y = np.asarray(y, float); s = np.asarray(s, float)
    def nll(sv):
        sv = max(float(sv), 0.0)
        w = 1.0 / (sv * sv + s * s)
        mu = np.sum(w * y) / np.sum(w)
        return 0.5 * np.sum(np.log(2 * np.pi * (sv * sv + s * s)) + (y - mu) ** 2 * w)
    res = minimize_scalar(nll, bounds=(0.0, max(3 * y.std(), 0.3)), method="bounded")
    sig = max(res.x, 0.0); nll0 = res.fun
    def prof(sv):
        return nll(sv) - nll0 - 1.92
    lo = hi = np.nan
    if sig > 1e-6:
        if prof(0.0) < 0:  # 边界仍在CI内
            from scipy.optimize import brentq
            try: lo = brentq(prof, 0.0, sig)
            except Exception: lo = 0.0
        else:
            lo = 0.0
        from scipy.optimize import brentq
        try: hi = brentq(prof, sig, max(4 * sig, sig + 0.3))
        except Exception: hi = np.nan
    else:
        lo = 0.0
        from scipy.optimize import brentq
        try: hi = brentq(prof, 1e-9, max(3 * y.std(), 0.3))
        except Exception: hi = np.nan
    w = 1.0 / (sig ** 2 + s ** 2)
    mu = np.sum(w * y) / np.sum(w)
    return mu, sig, lo, hi

d = pd.read_csv(R / "features" / "features_corpus_v5.csv",
                usecols=["cell_id", "temp_nominal", "parameter", "voltage_mV",
                         "value", "r2", "se", "channel", "flag"])
rows = []
for temp in [15, 25, 35]:
    sub = d[(d.channel == "Kv1.1") & (d.temp_nominal == temp) & (d.flag == "active")]
    # --- tau 侧 ---
    t = sub[(sub.parameter == "tau_act") & (sub.r2 >= 0.7) & (sub.value > 0)
            & (sub.voltage_mV >= 0) & (sub.voltage_mV <= 40)]
    for V, g in t.groupby("voltage_mV"):
        if len(g) < 15: continue
        y = np.log10(g.value.values)
        s = (g.se.values / (g.value.values * LN10))
        s = np.clip(s, 1e-4, None)
        mu, sig, lo, hi = eb_fit(y, s)
        sig_obs = y.std(ddof=1)
        sig_wd = np.sqrt(max(y.var(ddof=1) - np.mean(s ** 2), 0.0))
        rows.append(dict(temp=temp, V=V, n=len(g), sig_obs=sig_obs,
                         sig_true_wd=sig_wd, sig_EB=sig, EB_lo=lo, EB_hi=hi,
                         mu_EB=mu))
    # --- V½ 侧 (层级同样做法, 线性空间) ---
    h = sub[(sub.parameter == "Vhalf_act") & (sub.r2 >= 0.8)
            & (sub.value >= -80) & (sub.value <= 60)]
    k = sub[(sub.parameter == "k_act") & (sub.r2 >= 0.8)].value.median()
    if len(h) >= 15:
        yv = h.value.values; sv = np.clip(h.se.values, 1e-3, None)
        muV, sigV, loV, hiV = eb_fit(yv, sv)
        rows.append(dict(temp=temp, V=np.nan, n=len(h), sig_obs=yv.std(ddof=1),
                         sig_true_wd=np.sqrt(max(yv.var(ddof=1) - np.mean(sv ** 2), 0)),
                         sig_EB=sigV, EB_lo=loV, EB_hi=hiV, mu_EB=muV,
                         kind="Vhalf", k_med=k))
df = pd.DataFrame(rows)
df["kind"] = df.get("kind", pd.Series(["tau"] * len(df))).fillna("tau")
out_csv = R / "features" / "hier_bayes_kv11_2026-11-08.csv"
df.to_csv(out_csv, index=False)

# ---- 汇总判读 ----
print("=== Kv1.1 分层贝叶斯 vs 现行方差分解 ===")
for temp in [15, 25, 35]:
    g = df[(df.temp == temp) & (df.kind == "tau")]
    vh = df[(df.temp == temp) & (df.kind == "Vhalf")]
    if len(g) == 0: continue
    rel = (g.sig_EB - g.sig_true_wd) / g.sig_true_wd
    print(f"\n[Kv1.1@{temp}°C]  tau 侧 {len(g)} 电压点:")
    for _, r in g.iterrows():
        print(f"  V={r.V:>4.0f} n={r.n:>3d}  obs={r.sig_obs:.3f}  wd={r.sig_true_wd:.3f}"
              f"  EB={r.sig_EB:.3f} [{r.EB_lo:.3f},{r.EB_hi:.3f}]  相对差 {100*(r.sig_EB-r.sig_true_wd)/r.sig_true_wd:+.1f}%")
    print(f"  窗内中位: obs={g.sig_obs.median():.4f}  wd={g.sig_true_wd.median():.4f}  EB={g.sig_EB.median():.4f}  相对差中位 {100*rel.median():+.1f}%")
    if len(vh):
        r = vh.iloc[0]
        print(f"  Vhalf 侧: n={r.n}  obs_SD={r.sig_obs:.2f} mV  wd={r.sig_true_wd:.2f}  EB={r.sig_EB:.2f} [{r.EB_lo:.2f},{r.EB_hi:.2f}]  k_med={r.k_med:.2f}")
        print(f"    -> 归一化中点散布: wd={r.sig_true_wd/r.k_med:.3f}  EB={r.sig_EB/r.k_med:.3f}")
        ratio_wd = (r.sig_true_wd / r.k_med) / g.sig_true_wd.median()
        ratio_eb = (r.sig_EB / r.k_med) / g.sig_EB.median()
        print(f"    -> 散布比: 现行={ratio_wd:.2f}x   分层={ratio_eb:.2f}x")

# ---- 图 ----
C = dict(red="#E63946", orange="#F4842B", teal="#12B5A5", blue="#3A6FF7", dark="#22223B", grey="#8D99AE")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
for ax, temp in zip(axes, [15, 25, 35]):
    g = df[(df.temp == temp) & (df.kind == "tau")].sort_values("V")
    ax.plot(g.V, g.sig_obs, "o-", color=C["grey"], lw=1.6, label="观测散布（含噪声）")
    ax.plot(g.V, g.sig_true_wd, "s-", color=C["orange"], lw=2.0, label="现行方差分解")
    ax.fill_between(g.V, g.EB_lo, g.EB_hi, color=C["blue"], alpha=0.18)
    ax.plot(g.V, g.sig_EB, "D-", color=C["blue"], lw=2.2, label="分层贝叶斯（95% CI）")
    ax.set_title(f"Kv1.1 @ {temp} °C", fontweight="bold")
    ax.set_xlabel("电压（mV）")
axes[0].set_ylabel("τ_act 跨细胞散布（log$_{10}$ SD）")
axes[0].legend(fontsize=9.5, loc="upper left")
fig.suptitle("分层贝叶斯交叉验证：真实散布估计两条独立路线互证", fontsize=14, fontweight="bold", color=C["dark"])
fig.tight_layout(rect=(0, 0, 1, 0.93))
out_png = R / "features" / "fig_hier_bayes_kv11_2026-11-08.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
print("\nsaved", out_csv)
print("saved", out_png)
