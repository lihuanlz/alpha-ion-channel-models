# -*- coding: utf-8 -*-
# B+E 补充分析：全族温度定律 + 跨温度前向预言
# B1: 激活中点随温度的方向（15/25->35°C），逐通道
# B2: 失活中点随温度的方向（有失活拟合的通道）
# B3: 过程特异 Q10（tau_act / tau_deact / tau_inact 分开）
# E:  跨温度前向预言——25°C 群体 τ(V) 表 × 过程Q10 -> 预言 35°C 群体 τ(V)，算命中率
# 判据（跑前钉死）：
#   B1 支持"激活左移": 中位 dVhalf < 0 且 <0 的通道占比 >=70%
#   B2 支持"失活右移": 中位 dVhalf > 0 且 >0 的通道占比 >=70%
#   E  支持"可预言": |log2(预言/实测)| 中位 < 0.5（即误差中位 < 41%），且 80% 电压点在 ±1 内
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "features" / "features_all.csv")
OUT = ROOT / "features"

# ---------- B1/B2: 中点温度漂移（每通道：25->35 与 15->25 的群体 Vhalf 中位差） ----------
def midpoint_shifts(par):
    rows = []
    v = df[(df.parameter == par) & (df.r2 >= 0.8)]
    for ch, g in v.groupby("channel"):
        med = g.groupby("temp_nominal")["value"].median()
        med = med[med.index.isin([15, 25, 35])]
        ns = g.groupby("temp_nominal")["cell_id"].nunique()
        for a, b in [(15, 25), (25, 35)]:
            if a in med.index and b in med.index and ns.get(a, 0) >= 10 and ns.get(b, 0) >= 10:
                dT = b - a
                rows.append(dict(channel=ch, seg=f"{a}->{b}", dVhalf=float(med[b]-med[a]),
                                 per_C=float((med[b]-med[a])/dT), n_a=int(ns[a]), n_b=int(ns[b])))
    return pd.DataFrame(rows)

b1 = midpoint_shifts("Vhalf_act")
b2 = midpoint_shifts("Vhalf_inact")
b1.to_csv(OUT / "B1_act_shifts.csv", index=False)
b2.to_csv(OUT / "B2_inact_shifts.csv", index=False)

def dir_verdict(b, expect):
    if len(b) == 0: return "无法判定", {}
    frac = float((np.sign(b.dVhalf) == expect).mean())
    med = float(b.per_C.median())
    ok = frac >= 0.7 and np.sign(med) == expect
    return ("支持" if ok else "不支持"), dict(n_pairs=len(b), frac_expected_dir=frac,
                                              median_per_C=med)

b1v, b1d = dir_verdict(b1, -1)
b2v, b2v_detail = (("无法判定", {}) if len(b2)==0 else dir_verdict(b2, +1))

# ---------- B3: 过程特异 Q10（群体池化：每 通道×过程×电压 的 15/25/35 中位 τ -> Arrhenius） ----------
R = 8.314
q10_rows = []
for par in ["tau_act", "tau_deact", "tau_inact"]:
    t = df[(df.parameter == par) & (df.value > 0) & (df.r2 >= 0.7)]
    for (ch, V), g in t.groupby(["channel", "voltage_mV"]):
        med = g.groupby("temp_nominal")["value"].median()
        med = med[med.index.isin([15, 25, 35])]
        if len(med) == 3 and (med > 0).all():
            Tk = med.index.values + 273.15
            x = 1.0/Tk; y = np.log(med.values)
            sl = np.polyfit(x, y, 1)
            yhat = np.polyval(sl, x)
            r2 = 1 - np.sum((y-yhat)**2)/np.sum((y-y.mean())**2)
            Tbar = Tk.mean()
            q10 = float(np.exp(sl[0]*R/R*10.0/Tbar**2))
            q10_rows.append(dict(channel=ch, parameter=par, voltage_mV=V, Q10=q10, R2=float(r2)))
q3 = pd.DataFrame(q10_rows)
q3.to_csv(OUT / "B3_q10_by_process.csv", index=False)
q3s = q3[q3.R2 >= 0.8].groupby("parameter")["Q10"].agg(["count","median","std"])
q3_summary = {p: dict(n=int(r["count"]), median=float(r["median"]),
                      cv=float(r["std"]/r["median"]) if r["median"] else None)
              for p, r in q3s.iterrows()}

# ---------- E: 跨温度前向预言 ----------
# 预言: tau_35(V) = tau_25(V) / Q10_proc^2  (25->35 = 10°C -> Q10^1; 注意是除以)
# 用每 通道×过程×电压 自己的池化 Q10（B3），无拟合自由度之外的参数
pred_rows = []
q10_map = {(r.channel, r.parameter, r.voltage_mV): r.Q10 for r in q3.itertuples() if r.R2 >= 0.8 and 1.2 < r.Q10 < 10}
for (ch, par, V), q10 in q10_map.items():
    t25 = df[(df.channel==ch)&(df.parameter==par)&(df.voltage_mV==V)&(df.temp_nominal==25)&(df.value>0)]["value"].median()
    t35 = df[(df.channel==ch)&(df.parameter==par)&(df.voltage_mV==V)&(df.temp_nominal==35)&(df.value>0)]["value"].median()
    if np.isfinite(t25) and np.isfinite(t35) and t35 > 0:
        pred = t25 / q10
        pred_rows.append(dict(channel=ch, parameter=par, voltage_mV=V, Q10=q10,
                              tau25=t25, tau35_meas=t35, tau35_pred=pred,
                              log2err=float(np.log2(pred/t35))))
p = pd.DataFrame(pred_rows)
p.to_csv(OUT / "E_forward_prediction.csv", index=False)
if len(p):
    med_ae = float(p.log2err.abs().median())
    frac_in1 = float((p.log2err.abs() <= 1).mean())
    e_verdict = "支持" if (med_ae < 0.5 and frac_in1 >= 0.8) else "不支持"
    e_detail = dict(n_points=len(p), n_channels=int(p.channel.nunique()),
                    median_abs_log2err=med_ae, frac_within_2fold=frac_in1)
else:
    e_verdict, e_detail = "无法判定", {}

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
ax = axes[0]
if len(b1):
    for seg, mk in [("15->25","o"),("25->35","s")]:
        s_ = b1[b1.seg==seg]
        ax.scatter(s_.channel, s_.per_C, marker=mk, label=f"激活 {seg}°C", alpha=0.8)
    if len(b2):
        for seg, mk in [("15->25","^"),("25->35","v")]:
            s_ = b2[b2.seg==seg]
            ax.scatter(s_.channel, s_.per_C, marker=mk, label=f"失活 {seg}°C", alpha=0.8)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("dV½/dT (mV/°C)"); ax.set_title("B1/B2 全族中点漂移方向")
ax.legend(fontsize=7); plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)

ax = axes[1]
for par, c in [("tau_act","tab:blue"),("tau_deact","tab:orange"),("tau_inact","tab:green")]:
    s_ = q3[(q3.parameter==par)&(q3.R2>=0.8)]
    ax.hist(s_.Q10, bins=15, alpha=0.6, label=f"{par} (n={len(s_)}, 中位{s_.Q10.median():.2f})")
ax.set_xlabel("Q10"); ax.set_title("B3 过程特异 Q10 分布"); ax.legend(fontsize=8)

ax = axes[2]
if len(p):
    ax.scatter(p.tau35_meas, p.tau35_pred, c=(p.parameter=="tau_act"), cmap="coolwarm", alpha=0.6)
    lim = [p.tau35_meas.min()/2, p.tau35_meas.max()*2]
    ax.plot(lim, lim, "k--", lw=1)
    ax.plot(lim, [l*2 for l in lim], "r:", lw=1); ax.plot(lim, [l/2 for l in lim], "r:", lw=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("35°C 实测 τ (ms)"); ax.set_ylabel("25°C表×Q10 预言 (ms)")
    ax.set_title(f"E 跨温度前向预言（中位误差 {2**med_ae-1:.0%}，{frac_in1:.0%} 在2倍内）")
fig.tight_layout()
fig.savefig(OUT / "B_E_supplement.png", dpi=160, bbox_inches="tight")

verdict = dict(B1_act_left_shift=[b1v, b1d], B2_inact_right_shift=[b2v, b2v_detail],
               B3_q10_by_process=q3_summary, E_forward=[e_verdict, e_detail])
with open(OUT / "BE_verdict.json", "w", encoding="utf-8") as f:
    json.dump(verdict, f, ensure_ascii=False, indent=2)
print(json.dumps(verdict, ensure_ascii=False, indent=2))
