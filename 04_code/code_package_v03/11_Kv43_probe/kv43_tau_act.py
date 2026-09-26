# kv43_tau_act.py - Kv4.3 activation time constant @ -20 mV (same-voltage control as inactivation)
import scipy.io as sio, numpy as np, glob, os, csv
from scipy.optimize import curve_fit

BASE = r"D:\data\doi_10_5061_dryad_76hdr7t6z\physiology_data\in_vitro\extracted"
OUT = os.path.dirname(os.path.abspath(__file__))

def segments(v, dt, jump_thr=1e-3, min_len=200):
    jumps = np.where(np.abs(np.diff(v)) > jump_thr)[0]
    cuts = [0] + [j + 1 for j in jumps] + [len(v)]
    return [(a, b, float(np.median(v[a + 50:b])) * 1e3, (b - a) * dt)
            for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= min_len]

rows = []
for grp, d in [("veh", "veh_act"), ("les", "les_act")]:
    for fn in sorted(glob.glob(os.path.join(BASE, d, "*.mat"))):
        m = sio.loadmat(fn)
        taus, t1090s = [], []
        for k in m:
            if not k.startswith("Trace"):
                continue
            t = m[k]; v = t[:, 2]; cur = t[:, 1]
            dt = np.median(np.diff(t[:, 0]))
            segs = segments(v, dt)
            for si in range(1, len(segs)):
                a, b, lvl, dur = segs[si]
                pa, pb, plvl, pdur = segs[si - 1]
                if abs(plvl - (-80)) < 2 and pdur >= 0.5 and abs(lvl - (-20)) < 1.5:
                    base = float(np.median(cur[max(a - int(0.05 / dt), 0):a]))
                    y = (cur[a:a + int(0.15 / dt)] - base) * 1e9
                    x = np.arange(len(y)) * dt
                    pk = y.max(); ip = int(np.argmax(y))
                    if pk < 0.2:  # too small for a meaningful fit
                        break
                    seg_x, seg_y = x[:ip + 1], y[:ip + 1]
                    # t10-90
                    try:
                        i10 = np.where(seg_y >= 0.1 * pk)[0][0]
                        i90 = np.where(seg_y >= 0.9 * pk)[0][0]
                        t1090s.append((i90 - i10) * dt * 1e3)
                    except Exception:
                        pass
                    # rising-edge single exponential (10%..peak)
                    try:
                        i10 = np.where(seg_y >= 0.1 * pk)[0][0]
                        xx = seg_x[i10:] - seg_x[i10]; yy = seg_y[i10:]
                        popt, _ = curve_fit(lambda x, A, ta: A * (1 - np.exp(-x / ta)), xx, yy,
                                            p0=[pk, 0.005], bounds=([0, 0.0005], [np.inf, 0.2]),
                                            maxfev=8000)
                        pred = popt[0] * (1 - np.exp(-xx / popt[1]))
                        r2 = 1 - ((yy - pred) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
                        if r2 >= 0.8:
                            taus.append(popt[1] * 1e3)
                    except Exception:
                        pass
                    break
        if taus or t1090s:
            rows.append(dict(group=grp, cell=os.path.basename(fn).replace(".mat", ""),
                             tau_act_ms=np.median(taus) if taus else np.nan,
                             t1090_ms=np.median(t1090s) if t1090s else np.nan,
                             n_used=len(taus)))

with open(os.path.join(OUT, "tau_act20.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

import pandas as pd
from scipy.stats import mannwhitneyu
df = pd.DataFrame(rows)
for col in ["tau_act_ms", "t1090_ms"]:
    a = df[df.group == "veh"][col].dropna(); b = df[df.group == "les"][col].dropna()
    print(col, " veh med %.2f ms (n=%d, CV %.2f) | les med %.2f ms (n=%d, CV %.2f) | p=%.3f" %
          (a.median(), len(a), a.std(ddof=1) / a.mean(),
           b.median(), len(b), b.std(ddof=1) / b.mean(), mannwhitneyu(a, b).pvalue))
