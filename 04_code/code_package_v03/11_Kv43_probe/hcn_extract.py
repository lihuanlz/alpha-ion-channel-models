# hcn_extract.py - HCN(Ih) per-cell extraction: Ih amplitude and activation tau at 3 conditioning voltages (-80/-100/-120)
import scipy.io as sio, numpy as np, glob, os, csv, sys
from scipy.optimize import curve_fit

BASE = r"D:\data\doi_10_5061_dryad_76hdr7t6z\physiology_data\in_vitro\extracted"
OUT = os.path.dirname(os.path.abspath(__file__))

def segments(v, dt, jump_thr=1e-3, min_len=200):
    jumps = np.where(np.abs(np.diff(v)) > jump_thr)[0]
    cuts = [0] + [j + 1 for j in jumps] + [len(v)]
    return [(a, b, float(np.median(v[a + 50:b])) * 1e3, (b - a) * dt)
            for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= min_len]

rows = []
for grp, d in [("veh", "veh_hcn"), ("les", "les_hcn")]:
    for fn in sorted(glob.glob(os.path.join(BASE, d, "*.mat"))):
        m = sio.loadmat(fn)
        cell = os.path.basename(fn).replace(".mat", "")
        for k in m:
            if not k.startswith("Trace"):
                continue
            t = m[k]; v = t[:, 2]; cur = t[:, 1]
            dt = np.median(np.diff(t[:, 0]))
            segs = segments(v, dt)
            for si in range(1, len(segs)):
                a, b, lvl, dur = segs[si]
                pa, pb, plvl, pdur = segs[si - 1]
                if lvl <= -70 and dur >= 0.8 and abs(plvl - (-40)) < 2:
                    base = float(np.median(cur[max(a - int(0.05 / dt), 0):a]))
                    y = (cur[a:b] - base) * 1e9  # nA, inward is negative
                    x = np.arange(len(y)) * dt
                    i_inst = float(np.median(y[int(0.005 / dt):int(0.015 / dt)]))
                    i_ss = float(np.median(y[-int(0.1 / dt):]))
                    ih_amp = i_ss - i_inst
                    tau = np.nan; r2 = np.nan
                    try:
                        i0 = int(0.005 / dt)
                        yy = y[i0:]; xx = x[i0:] - x[i0]
                        popt, _ = curve_fit(lambda x, A, ta, C: A * (1 - np.exp(-x / ta)) + C,
                                            xx, yy, p0=[ih_amp, 0.2, i_inst],
                                            bounds=([-50, 0.01, -50], [50, 5, 50]), maxfev=8000)
                        pred = popt[0] * (1 - np.exp(-xx / popt[1])) + popt[2]
                        r2 = 1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
                        tau = float(popt[1])
                    except Exception:
                        pass
                    rows.append(dict(group=grp, cell=cell, V_cond=lvl,
                                     ih_amp_nA=ih_amp, i_inst=i_inst, i_ss=i_ss,
                                     tau_act_s=tau, fit_R2=r2))
                    break

with open(os.path.join(OUT, "hcn_all.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

import pandas as pd
df = pd.DataFrame(rows)
for grp in ["veh", "les"]:
    sub = df[df.group == grp]
    for V in [-80, -100, -120]:
        s2 = sub[np.abs(sub.V_cond - V) < 1]
        amp = s2.ih_amp_nA; tq = s2[s2.fit_R2 >= 0.8].tau_act_s
        print(grp, "%d mV: n=%d  Ih med %.2f nA (IQR %.2f..%.2f)  tau med %.0f ms (n=%d)" %
              (V, len(s2), amp.median(), *np.percentile(amp, [25, 75]), tq.median() * 1e3, len(tq)))
