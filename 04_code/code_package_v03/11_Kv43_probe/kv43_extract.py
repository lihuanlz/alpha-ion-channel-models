# kv43_extract.py - Kv4.3 probe: per-cell voltage-clamp extraction from Dryad 10.5061/dryad.76hdr7t6z
# usage: python kv43_extract.py <folder> <protocol:act|inact> <out_csv>
# data source: 6-OHDA lesioned / vehicle two groups of SNc DA neurons, voltage clamp (.mat, Trace_t_s_r, 20 kHz, col0=t/s, col1=I/A, col2=V/V)
import sys, os
import numpy as np
import scipy.io as sio
from scipy.optimize import curve_fit

folder, proto, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]

def segments(v, dt, jump_thr=1e-3, min_len=200):
    jumps = np.where(np.abs(np.diff(v)) > jump_thr)[0]
    cuts = [0] + [j + 1 for j in jumps] + [len(v)]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= min_len:
            out.append((a, b, float(np.median(v[a + 50:b])) * 1e3, (b - a) * dt))
    return out

def boltz_down(V, Imax, Vh, k):
    return Imax / (1.0 + np.exp((V - Vh) / k))

rows = []
for fn in sorted(os.listdir(folder)):
    if not fn.endswith(".mat"):
        continue
    m = sio.loadmat(os.path.join(folder, fn))
    names = [k for k in m if k.startswith("Trace")]
    if proto == "inact":
        Vc, Ip = [], []
        tau_inact = np.nan
        for n in names:
            t = m[n]
            tt, cur, v = t[:, 0], t[:, 1], t[:, 2]
            dt = float(np.median(np.diff(tt)))
            segs = segments(v, dt)
            # find the test segment: level -25..-15 mV, duration 0.5-2.5 s, not the first segment
            for si, (a, b, lvl, dur) in enumerate(segs):
                if -25 <= lvl <= -15 and 0.5 <= dur <= 2.5 and si > 0:
                    pa, pb, plvl, pdur = segs[si - 1]
                    cond = plvl if pdur >= 0.5 else np.nan
                    if not np.isfinite(cond):
                        continue
                    base = float(np.median(cur[max(a - int(0.05 / dt), 0):a]))
                    i0 = a + int(0.002 / dt)  # 2 ms after the step to avoid the capacitive spike
                    win = cur[i0:i0 + int(0.15 / dt)] - base
                    ipk = float(win.max())
                    Vc.append(cond); Ip.append(ipk)
                    # inactivation kinetics: single-exponential fit of the test segment on the cond=-120 trace
                    if abs(cond - (-120)) < 1 and dur >= 0.9:
                        i1 = a + int(0.002 / dt)
                        i2 = min(a + int(0.9 / dt), b)
                        y = cur[i1:i2] - base
                        x = (np.arange(len(y)) * dt)
                        try:
                            popt, _ = curve_fit(lambda x, A, tau, C: A * np.exp(-x / tau) + C,
                                                x, y, p0=[y.max(), 0.1, 0.0],
                                                bounds=([0, 1e-3, -np.inf], [np.inf, 5, np.inf]), maxfev=4000)
                            tau_inact = float(popt[1])
                        except Exception:
                            pass
                    break
        if len(Vc) >= 5:
            Vc = np.array(Vc); Ip = np.array(Ip) * 1e9  # nA units, avoiding Imax~1e-9 vs Vh/k scale mismatch stalling LM
            Vu = np.unique(Vc)
            Iu = np.array([Ip[Vc == u].mean() for u in Vu])
            try:
                popt, pcov = curve_fit(boltz_down, Vu, Iu, p0=[max(Iu.max(), 0.5), -55, 5],
                                       bounds=([0, -120, 0.1], [1e4, 0, 60]), maxfev=20000)
                pred = boltz_down(Vu, *popt)
                ss = 1 - ((Iu - pred) ** 2).sum() / ((Iu - Iu.mean()) ** 2).sum()
                perr = np.sqrt(np.diag(pcov))
                rows.append(dict(cell=fn.replace(".mat", ""), n_pts=len(Vu),
                                 Vh_inact=popt[1], Vh_err=perr[1], k_inact=popt[2], k_err=perr[2],
                                 Imax_nA=popt[0], R2=ss, tau_inact_s=tau_inact))
            except Exception as e:
                rows.append(dict(cell=fn.replace(".mat", ""), n_pts=len(Vu), Vh_inact=np.nan,
                                 Vh_err=np.nan, k_inact=np.nan, k_err=np.nan, Imax_nA=np.nan,
                                 R2=np.nan, tau_inact_s=tau_inact))
    else:  # act
        Vt, Ip = {}, []
        t10_90 = np.nan
        for n in names:
            t = m[n]
            tt, cur, v = t[:, 0], t[:, 1], t[:, 2]
            dt = float(np.median(np.diff(tt)))
            segs = segments(v, dt)
            # test segment = the segment after the prepulse (-80)
            for si in range(1, len(segs)):
                a, b, lvl, dur = segs[si]
                pa, pb, plvl, pdur = segs[si - 1]
                if abs(plvl - (-80)) < 2 and pdur >= 0.5:
                    base = float(np.median(cur[max(a - int(0.05 / dt), 0):a]))
                    i0 = a + int(0.002 / dt)
                    ipk = float(win.max()) * 1e9 if False else float((cur[i0:i0 + int(0.3 / dt)] - base).max()) * 1e9  # nA
                    Vt.setdefault(round(lvl), []).append(ipk)
                    break
        if len(Vt) >= 5:
            Vu = np.array(sorted(Vt.keys()), float)
            # robust merge of repeated sweeps: max/min>5 is judged a spike artifact -> take the smaller value, else the mean
            Iu = []
            n_spike = 0
            for u in Vu:
                vals = np.asarray(Vt[u], float)
                if len(vals) >= 2 and vals.max() / max(vals.min(), 1e-12) > 5:
                    Iu.append(vals.min()); n_spike += 1
                else:
                    Iu.append(vals.mean())
            Iu = np.array(Iu)
            imax = Iu.max()
            # foot slope: ln I vs V, absolute band 0.25 nA..80% Imax; if fewer than 4 points, fall back to the 5%..80% relative band
            band = (Iu >= 0.25) & (Iu <= 0.8 * imax)
            if band.sum() < 4:
                band = (Iu >= 0.05 * imax) & (Iu <= 0.8 * imax)
            if band.sum() >= 4:
                sl, ic = np.polyfit(Vu[band], np.log(Iu[band]), 1)
                pred = sl * Vu[band] + ic
                r2 = 1 - ((np.log(Iu[band]) - pred) ** 2).sum() / ((np.log(Iu[band]) - np.log(Iu[band]).mean()) ** 2).sum()
                s_ref, foot_r2 = float(sl), float(r2)
            else:
                s_ref, foot_r2 = np.nan, np.nan
            rows.append(dict(cell=fn.replace(".mat", ""), n_pts=len(Vu),
                             V_min=Vu.min(), V_max=Vu.max(), Imax_nA=imax,
                             s_ref=s_ref, foot_R2=foot_r2, n_foot=int(band.sum()),
                             n_spike_repaired=n_spike))

import csv
if rows:
    keys = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
print(proto, folder.split("/")[-1], "cells:", len(rows))
for r in rows:
    print(r)
