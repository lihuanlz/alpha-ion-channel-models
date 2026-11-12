# -*- coding: utf-8 -*-
# Channelpedia Kv 全族提取管线 v1
# 输入: data/cells/<CHANNEL>/<cellfile>.nwb  (文件名形如 rCell<ID>_<x>_<temp>C.nwb)
# 输出: features.csv 每行 = 细胞×参数×电压
# 协议结构(来自 SciMLHHModels.jl loader_functions.jl):
#   Activation:   repetition2, 18 sweeps, t=0..700ms, step -90..+80mV at t=100..600
#   Deactivation: repetition1, 12 sweeps, t=0..700ms, +70mV at 100..400, tail -80..+30 at 400..600
#   Inactivation: repetition1, 12 sweeps, t=0..1750ms, cond -40..+70 at 100..1600, test +30 at 1600..1700
import sys, json, re
import numpy as np
import h5py
from pathlib import Path
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent
CELLS = ROOT / "data" / "cells"
OUT = ROOT / "features"
OUT.mkdir(exist_ok=True)

ACT_V = list(range(-90, 90, 10))          # 18
DEA_V = list(range(-80, 40, 10))          # 12
INA_V = list(range(-40, 80, 10))          # 12
E_REV = -90.0  # mV, K  reversal (Ranjan 2019 溶液条件)

def boltz_inc(V, Vh, k):  return 1.0/(1.0+np.exp(-(V-Vh)/k))
def boltz_dec(V, Vh, k):  return 1.0/(1.0+np.exp((V-Vh)/k))

def fit_boltz(V, y, inc=True, min_pts=6, r2_min=0.8):
    try:
        V = np.asarray(V, float); y = np.asarray(y, float)
        m = np.isfinite(V) & np.isfinite(y)
        if m.sum() < min_pts: return None
        f = boltz_inc if inc else boltz_dec
        p, _ = curve_fit(f, V[m], y[m], p0=[np.median(V[m]), 6.0],
                         bounds=([-250, 0.5], [250, 100]), maxfev=10000)
        ss = np.sum((y[m]-f(V[m], *p))**2); st = np.sum((y[m]-y[m].mean())**2)
        r2 = 1-ss/st if st > 0 else 0.0
        if r2 < r2_min: return None
        return float(p[0]), float(p[1]), float(r2)
    except Exception:
        return None

def fit_tau_rise(t, i, t0, t1):
    """激活上升相单指数: I = Iss*(1-exp(-(t-t0)/tau)) + c; 只取到峰值的上升段"""
    m = (t >= t0) & (t <= t1)
    if m.sum() < 20: return None
    tt = t[m]; ii = i[m]
    pk = int(np.argmax(ii))
    if pk < 10: return None
    tt = tt[:pk+1]; yy = ii[:pk+1]
    try:
        def rise(t, Iss, tau, c): return Iss*(1-np.exp(-(t-t0)/tau)) + c
        p, _ = curve_fit(rise, tt, yy, p0=[yy.max(), 50.0, yy[0]],
                         bounds=([0, 0.1, -np.inf], [np.inf, 2000, np.inf]), maxfev=10000)
        pred = rise(tt, *p)
        ss = np.sum((yy-pred)**2); st = np.sum((yy-yy.mean())**2)
        r2 = 1-ss/st if st > 0 else 0.0
        if r2 < 0.7 or p[1] <= 0: return None
        return float(p[1]), float(r2)
    except Exception:
        return None

def fit_tau_decay(t, i, t0, t1):
    m = (t >= t0) & (t <= t1)
    if m.sum() < 20: return None
    tt = t[m]; ii = i[m]
    try:
        def dec(t, A, tau, c): return A*np.exp(-(t-t0)/tau) + c
        p0 = [ii.max()-ii.min(), 50.0, ii[-1]]
        p, _ = curve_fit(dec, tt, ii, p0=p0,
                         bounds=([0, 0.05, -np.inf], [np.inf, 5000, np.inf]), maxfev=10000)
        pred = dec(tt, *p)
        ss = np.sum((ii-pred)**2); st = np.sum((ii-ii.mean())**2)
        r2 = 1-ss/st if st > 0 else 0.0
        if r2 < 0.7 or p[1] <= 0: return None
        return float(p[1]), float(r2)
    except Exception:
        return None

def parse_cell(fp):
    """返回 dict 行列表"""
    m = re.match(r"rCell(\d+)_(kv\d+)_(\d+)\.nwb$", fp.name)
    if not m: return []
    cid, ch_code, temp = m.group(1), m.group(2), int(m.group(3))
    channel = "Kv" + ch_code[2:-1] + "." + ch_code[-1]
    rows = []
    try:
        with h5py.File(fp, "r") as f:
            ts = f.get("acquisition/timeseries")
            if ts is None: return []
            # 真实头台温度
            try:
                temp_real = float(ts["Activation/repetitions/repetition2/head_temp"][0][0])
            except Exception:
                temp_real = float(temp)
            # ---- Activation ----
            if "Activation" in ts:
                d = ts["Activation/repetitions/repetition2/data"][:]
                if d.ndim == 1: d = d[:, None]
                d = d.T  # (sweep, time)
                n = d.shape[1]
                t = np.linspace(0, 700, n)
                G = []
                for si, V in enumerate(ACT_V):
                    if si >= d.shape[0]: break
                    i = d[si].astype(float)
                    base = i[t < 100].mean()
                    i0 = i - base
                    # 电导: 用步阶末端 50ms 均值(非失活)或峰值(失活)——取较大绝对值处
                    seg_end = i0[(t > 550) & (t <= 600)].mean()
                    seg_pk = i0[(t >= 100) & (t <= 600)].max()
                    amp = seg_pk if abs(seg_pk) > abs(seg_end)*1.2 else seg_end
                    if abs(V - E_REV) < 1: 
                        G.append(np.nan)
                        continue
                    G.append(amp / (V - E_REV))
                    ft = fit_tau_rise(t, i0, 100, 600)
                    if ft and V >= -40:   # 低电压太噪声, 只留 -40 以上
                        rows.append(dict(cell_id=cid, temp_C=temp_real, temp_nominal=temp, parameter="tau_act",
                                         voltage_mV=V, value=ft[0], unit="ms", r2=ft[1]))
                Gv = np.array([v for v, g_ in zip(ACT_V, G) if np.isfinite(g_)])
                Gg = np.array([g_ for g_ in G if np.isfinite(g_)])
                fh = None
                if len(Gg) >= 6 and Gg.max() > 0:
                    fh = fit_boltz(Gv, Gg / Gg.max(), inc=True)
                if fh:
                    rows.append(dict(cell_id=cid, temp_C=temp_real, temp_nominal=temp, parameter="Vhalf_act",
                                     voltage_mV=np.nan, value=fh[0], unit="mV", r2=fh[2]))
                    rows.append(dict(cell_id=cid, temp_C=temp_real, temp_nominal=temp, parameter="k_act",
                                     voltage_mV=np.nan, value=fh[1], unit="mV", r2=fh[2]))
            # ---- Deactivation ----
            if "Deactivation" in ts:
                d = ts["Deactivation/repetitions/repetition1/data"][:]
                if d.ndim == 1: d = d[:, None]
                d = d.T
                n = d.shape[1]
                t = np.linspace(0, 700, n)
                for si, V in enumerate(DEA_V):
                    if si >= d.shape[0]: break
                    i = d[si].astype(float)
                    i0 = i - i[t < 100].mean()
                    ft = fit_tau_decay(t, i0, 402, 600)
                    if ft:
                        rows.append(dict(cell_id=cid, temp_C=temp_real, temp_nominal=temp, parameter="tau_deact",
                                         voltage_mV=V, value=ft[0], unit="ms", r2=ft[1]))
            # ---- Inactivation ----
            if "Inactivation" in ts:
                d = ts["Inactivation/repetitions/repetition1/data"][:]
                if d.ndim == 1: d = d[:, None]
                d = d.T
                n = d.shape[1]
                t = np.linspace(0, 1750, n)
                peaks = []
                for si, V in enumerate(INA_V):
                    if si >= d.shape[0]: break
                    i = d[si].astype(float)
                    i0 = i - i[t < 100].mean()
                    test = i0[(t >= 1600) & (t <= 1700)]
                    peaks.append(test.max() if len(test) else np.nan)
                    # 条件脉冲期间的失活衰减(只在有明显衰减的电压拟合)
                    cond = i0[(t >= 200) & (t <= 1500)]
                    tcond = t[(t >= 200) & (t <= 1500)]
                    if len(cond) > 50 and cond.max() > 0 and (cond.max()-cond[-1])/ (cond.max()+1e-12) > 0.3:
                        ft = fit_tau_decay(tcond, cond, 200, 1500)
                        if ft:
                            rows.append(dict(cell_id=cid, temp_C=temp_real, temp_nominal=temp, parameter="tau_inact",
                                             voltage_mV=V, value=ft[0], unit="ms", r2=ft[1]))
                pk = np.asarray(peaks, float)
                if np.isfinite(pk).sum() >= 6 and np.nanmax(pk) > 0:
                    fh = fit_boltz(INA_V, pk/np.nanmax(pk), inc=False)
                    if fh:
                        rows.append(dict(cell_id=cid, temp_C=temp_real, temp_nominal=temp, parameter="Vhalf_inact",
                                         voltage_mV=np.nan, value=fh[0], unit="mV", r2=fh[2]))
    except Exception as e:
        print(f"FAIL {fp.name}: {e}", file=sys.stderr)
    for r in rows:
        r["channel"] = channel
    return rows

if __name__ == "__main__":
    import pandas as pd
    only = sys.argv[1] if len(sys.argv) > 1 else None
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    all_rows = []
    files = sorted(CELLS.glob("*.nwb"))
    if limit: files = files[:limit]
    print(f"{len(files)} nwb files ...", flush=True)
    for i, fp in enumerate(files):
        all_rows.extend(parse_cell(fp))
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{len(files)} cells, {len(all_rows)} rows", flush=True)
    df = pd.DataFrame(all_rows)
    out = OUT / ("features_%s.csv" % (only or "all"))
    df.to_csv(out, index=False)
    print("saved", out, len(df), "rows")
