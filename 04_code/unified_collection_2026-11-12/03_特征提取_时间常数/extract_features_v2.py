# -*- coding: utf-8 -*-
# Channelpedia Kv 提取管线 v2 —— 工程精度升级
# 升级点（相对 v1）：
#  1) 激活 τ：I = Iss*(1-exp(-t/τ))^p，p∈{1,2,3} AIC 模型选择；记录失活污染旗标
#  2) 不确定性传播：每次拟合从协方差取参数 SE，写入输出
#  3) Boltzmann 三参数（Gmax 自由）：G = Gmax/(1+exp(-(V-Vh)/k)) + c
#  4) 去激活：单/双指数 AIC 选择
# 输入/输出结构同 v1；判据不变（冻结）
import sys, re
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
CELLS = ROOT / "data" / "cells"
OUT = ROOT / "features"
OUT.mkdir(exist_ok=True)

ACT_V = list(range(-90, 90, 10))
DEA_V = list(range(-80, 40, 10))
INA_V = list(range(-40, 80, 10))
E_REV = -90.0
LN10 = np.log(10)

def _se(pcov, i):
    try:
        return float(np.sqrt(pcov[i][i])) if np.isfinite(pcov[i][i]) else np.nan
    except Exception:
        return np.nan

# ---------- 3 参数 Boltzmann ----------
def boltz3_inc(V, A, Vh, k, c): return A/(1.0+np.exp(-(V-Vh)/k)) + c
def boltz3_dec(V, A, Vh, k, c): return A/(1.0+np.exp((V-Vh)/k)) + c

def fit_boltz3(V, y, inc=True, min_pts=6, r2_min=0.8):
    try:
        V = np.asarray(V, float); y = np.asarray(y, float)
        m = np.isfinite(V) & np.isfinite(y)
        if m.sum() < min_pts: return None
        V, y = V[m], y[m]
        A0 = max(y.max()-y.min(), 1e-12)
        f = boltz3_inc if inc else boltz3_dec
        p0 = [A0, np.median(V), 8.0, float(y.min())]
        bounds = ([0, -250, 0.5, -np.inf], [np.inf, 250, 100, np.inf])
        p, pcov = curve_fit(f, V, y, p0=p0, bounds=bounds, maxfev=20000)
        pred = f(V, *p)
        ss = np.sum((y-pred)**2); st = np.sum((y-y.mean())**2)
        r2 = 1-ss/st if st > 0 else 0.0
        if r2 < r2_min: return None
        return dict(Vh=float(p[1]), Vh_se=_se(pcov,1), k=float(p[2]), k_se=_se(pcov,2),
                    A=float(p[0]), r2=float(r2))
    except Exception:
        return None

# ---------- 激活上升相：m^p 模型选择 ----------
def _rise(t, Iss, tau, p, c, t0):
    return Iss*np.clip(1-np.exp(-(t-t0)/tau), 0, None)**p + c

def fit_tau_rise_v2(t, i, t0, t1):
    m = (t >= t0) & (t <= t1)
    if m.sum() < 20: return None
    tt = t[m]; ii = i[m]
    pk = int(np.argmax(ii))
    if pk < 10: return None
    tt = tt[:pk+1]; yy = ii[:pk+1]
    npts = len(yy)
    best = None
    for p in (1, 2, 3):
        try:
            f = lambda t_, Iss, tau, c: _rise(t_, Iss, tau, p, c, t0)
            p0 = [yy.max(), 50.0, yy[0]]
            pb, pcov = curve_fit(f, tt, yy, p0=p0,
                                 bounds=([0, 0.05, -np.inf], [np.inf, 2000, np.inf]),
                                 maxfev=15000)
            pred = f(tt, *pb)
            rss = float(np.sum((yy-pred)**2))
            k_par = 3
            aic = npts*np.log(rss/npts + 1e-30) + 2*k_par
            st = float(np.sum((yy-yy.mean())**2))
            r2 = 1-rss/st if st > 0 else 0.0
            if best is None or aic < best["aic"]:
                best = dict(tau=float(pb[1]), tau_se=_se(pcov,1), p=p, r2=float(r2), aic=aic)
        except Exception:
            continue
    if best is None or best["r2"] < 0.7 or best["tau"] <= 0: return None
    # 失活污染旗标：峰值出现在窗的前 60% 以内（说明上升相被失活截断）
    best["inact_contam"] = bool(pk < 0.6*(m.sum()-1))
    return best

# ---------- 去激活：单/双指数 AIC ----------
def fit_tau_decay_v2(t, i, t0, t1):
    m = (t >= t0) & (t <= t1)
    if m.sum() < 20: return None
    tt = t[m]; ii = i[m]
    npts = len(tt)
    def mono(t_, A, tau, c): return A*np.exp(-(t_-t0)/tau) + c
    def bi(t_, A1, tau1, A2, tau2, c): return A1*np.exp(-(t_-t0)/tau1)+A2*np.exp(-(t_-t0)/tau2)+c
    out = None
    try:
        p1, pc1 = curve_fit(mono, tt, ii, p0=[ii.max()-ii.min(), 50, ii[-1]],
                            bounds=([0,0.05,-np.inf],[np.inf,5000,np.inf]), maxfev=15000)
        rss1 = float(np.sum((ii-mono(tt,*p1))**2))
        aic1 = npts*np.log(rss1/npts+1e-30) + 2*3
        out = dict(tau=float(p1[1]), tau_se=_se(pc1,1), model="mono", aic=aic1,
                   r2=1-rss1/np.sum((ii-ii.mean())**2))
        try:
            p0b = [p1[0]*0.6, max(p1[1]/5,1), p1[0]*0.4, p1[1]*5, p1[2]]
            p2, pc2 = curve_fit(bi, tt, ii, p0=p0b,
                                bounds=([0,0.05,0,0.05,-np.inf],[np.inf,5000,np.inf,5000,np.inf]),
                                maxfev=20000)
            rss2 = float(np.sum((ii-bi(tt,*p2))**2))
            aic2 = npts*np.log(rss2/npts+1e-30) + 2*5
            if aic2 < aic1 - 6:  # 显著偏好双指数才换
                taus = sorted([(float(p2[1]), _se(pc2,1)), (float(p2[3]), _se(pc2,3))])
                out = dict(tau=taus[0][0], tau_se=taus[0][1], tau2=taus[1][0],
                           tau2_se=taus[1][1], model="bi", aic=aic2,
                           r2=1-rss2/np.sum((ii-ii.mean())**2))
        except Exception:
            pass
    except Exception:
        return None
    if out is None or out["r2"] < 0.7 or out["tau"] <= 0: return None
    return out

def parse_cell(fp):
    m = re.match(r"rCell(\d+)_(kv\d+)_(\d+)\.nwb$", fp.name)
    if not m: return []
    cid, ch_code, temp = m.group(1), m.group(2), int(m.group(3))
    channel = "Kv" + ch_code[2:-1] + "." + ch_code[-1]
    rows = []
    def add(par, V, val, unit, r2, se=np.nan, extra=None):
        r = dict(cell_id=cid, temp_C=temp_real, temp_nominal=temp, parameter=par,
                 voltage_mV=V, value=val, unit=unit, r2=r2, se=se)
        if extra: r.update(extra)
        rows.append(r)
    try:
        with h5py.File(fp, "r") as f:
            ts = f.get("acquisition/timeseries")
            if ts is None: return []
            try:
                temp_real = float(ts["Activation/repetitions/repetition2/head_temp"][0][0])
            except Exception:
                temp_real = float(temp)
            # ---- Activation ----
            if "Activation" in ts:
                d = ts["Activation/repetitions/repetition2/data"][:]
                if d.ndim == 1: d = d[:, None]
                d = d.T
                n = d.shape[1]; t = np.linspace(0, 700, n)
                G, Gv = [], []
                for si, V in enumerate(ACT_V):
                    if si >= d.shape[0]: break
                    i0 = d[si].astype(float) - d[si][t < 100].mean()
                    seg_end = i0[(t > 550) & (t <= 600)].mean()
                    seg_pk = i0[(t >= 100) & (t <= 600)].max()
                    amp = seg_pk if abs(seg_pk) > abs(seg_end)*1.2 else seg_end
                    if abs(V - E_REV) >= 1:
                        G.append(amp/(V-E_REV)); Gv.append(V)
                    ft = fit_tau_rise_v2(t, i0, 100, 600)
                    if ft and V >= -40:
                        add("tau_act", V, ft["tau"], "ms", ft["r2"], se=ft["tau_se"],
                            extra=dict(p_model=ft["p"], inact_contam=ft["inact_contam"]))
                fh = fit_boltz3(Gv, G, inc=True)
                if fh:
                    add("Vhalf_act", np.nan, fh["Vh"], "mV", fh["r2"], se=fh["Vh_se"])
                    add("k_act", np.nan, fh["k"], "mV", fh["r2"], se=fh["k_se"])
            # ---- Deactivation ----
            if "Deactivation" in ts:
                d = ts["Deactivation/repetitions/repetition1/data"][:]
                if d.ndim == 1: d = d[:, None]
                d = d.T
                n = d.shape[1]; t = np.linspace(0, 700, n)
                for si, V in enumerate(DEA_V):
                    if si >= d.shape[0]: break
                    i0 = d[si].astype(float) - d[si][t < 100].mean()
                    ft = fit_tau_decay_v2(t, i0, 402, 600)
                    if ft:
                        add("tau_deact", V, ft["tau"], "ms", ft["r2"], se=ft["tau_se"],
                            extra=dict(model=ft.get("model")))
                        if ft.get("model") == "bi":
                            add("tau_deact_slow2", V, ft["tau2"], "ms", ft["r2"], se=ft["tau2_se"])
            # ---- Inactivation ----
            if "Inactivation" in ts:
                d = ts["Inactivation/repetitions/repetition1/data"][:]
                if d.ndim == 1: d = d[:, None]
                d = d.T
                n = d.shape[1]; t = np.linspace(0, 1750, n)
                peaks = []
                for si, V in enumerate(INA_V):
                    if si >= d.shape[0]: break
                    i0 = d[si].astype(float) - d[si][t < 100].mean()
                    test = i0[(t >= 1600) & (t <= 1700)]
                    peaks.append(test.max() if len(test) else np.nan)
                    cond_t = (t >= 200) & (t <= 1500)
                    cond = i0[cond_t]
                    if len(cond) > 50 and cond.max() > 0 and (cond.max()-cond[-1])/(cond.max()+1e-12) > 0.3:
                        ft = fit_tau_decay_v2(t[cond_t], cond, 200, 1500)
                        if ft:
                            add("tau_inact", V, ft["tau"], "ms", ft["r2"], se=ft["tau_se"],
                                extra=dict(model=ft.get("model")))
                fh = fit_boltz3(INA_V, peaks, inc=False)
                if fh:
                    add("Vhalf_inact", np.nan, fh["Vh"], "mV", fh["r2"], se=fh["Vh_se"])
    except Exception as e:
        print(f"FAIL {fp.name}: {e}", file=sys.stderr)
    for r in rows: r["channel"] = channel
    return rows

if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    files = sorted(CELLS.glob("*.nwb"))
    if limit: files = files[:limit]
    print(f"{len(files)} nwb files ...", flush=True)
    all_rows = []
    for i, fp in enumerate(files):
        all_rows.extend(parse_cell(fp))
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(files)} cells, {len(all_rows)} rows", flush=True)
    df = pd.DataFrame(all_rows)
    out = OUT / ("features_all_v2.csv" if not limit else "features_smoke_v2.csv")
    df.to_csv(out, index=False)
    print("saved", out, len(df), "rows")
