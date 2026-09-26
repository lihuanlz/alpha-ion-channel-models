# kv43_pure_subtraction.py - within-protocol subtraction purifying Kv4.3: I(-120 cond) - I(-30 cond) ~ pure Kv4.3
# adjudicates whether the "apparent slower inactivation" is component mixing or true plasticity; also extracts the Kv4.3 activation tau_act
import scipy.io as sio, numpy as np, glob, os, csv
from scipy.optimize import curve_fit

BASE = r"D:\data\doi_10_5061_dryad_76hdr7t6z\physiology_data\in_vitro\extracted"
OUT = os.path.dirname(os.path.abspath(__file__))

def test_segment_current(m, cond_target):
    """Find the trace with cond~cond_target followed by a -20 mV test segment; return (t, I-base) nA, t zeroed at the test step"""
    for k in m:
        if not k.startswith("Trace"):
            continue
        t = m[k]; v = t[:, 2]; cur = t[:, 1]
        dt = np.median(np.diff(t[:, 0]))
        jumps = np.where(np.abs(np.diff(v)) > 1e-3)[0]
        if len(jumps) < 2:
            continue
        segs = [0] + [j + 1 for j in jumps] + [len(v)]
        for si in range(1, len(segs) - 1):
            a0, b0, a1, b1 = segs[si - 1], segs[si], segs[si], segs[si + 1]
            if b0 - a0 < 200 or b1 - a1 < 200:
                continue
            l0 = np.median(v[a0 + 50:b0]) * 1e3
            l1 = np.median(v[a1 + 50:b1]) * 1e3
            if abs(l0 - cond_target) < 1.5 and abs(l1 + 20) < 1.5:
                base = np.median(cur[a1 - int(0.05 / dt):a1])
                n = int(0.9 / dt)
                y = (cur[a1:a1 + n] - base) * 1e9
                x = np.arange(n) * dt
                return x, y
    return None, None

def fit_tau(x, y, tmax=0.5):
    m = x <= tmax
    x, y = x[m], y[m]
    try:
        popt, _ = curve_fit(lambda x, A, ta, C: A * np.exp(-x / ta) + C, x, y,
                            p0=[y.max(), 0.03, 0.0],
                            bounds=([0, 0.002, -np.inf], [np.inf, 2, np.inf]), maxfev=20000)
        pred = popt[0] * np.exp(-x / popt[1]) + popt[2]
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return popt[1] * 1e3, r2  # ms
    except Exception:
        return np.nan, np.nan

def t50(y, dt):
    pk = y.max(); ip = int(np.argmax(y))
    h = np.where(y[ip:] < pk / 2)[0]
    return h[0] * dt * 1e3 if len(h) else np.nan

rows = []
for grp, d in [("veh", "veh_inact"), ("les", "les_inact")]:
    for fn in sorted(glob.glob(os.path.join(BASE, d, "*.mat"))):
        m = sio.loadmat(fn)
        x120, y120 = test_segment_current(m, -120.0)
        x30, y30 = test_segment_current(m, -30.0)
        if x120 is None or x30 is None:
            continue
        dt = x120[1] - x120[0]
        pure = y120 - y30
        tau_app, r2_app = fit_tau(x120, y120)
        tau_pure, r2_pure = fit_tau(x120, pure)
        rows.append(dict(group=grp, cell=os.path.basename(fn).replace(".mat", ""),
                         tau_app_ms=tau_app, r2_app=r2_app,
                         tau_pure_ms=tau_pure, r2_pure=r2_pure,
                         t50_pure_ms=t50(pure, dt),
                         pk_pure_nA=pure.max()))

with open(os.path.join(OUT, "pure_kv43.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

from scipy.stats import mannwhitneyu
import pandas as pd
df = pd.DataFrame(rows)
q = df[(df.r2_pure >= 0.9) & (df.r2_app >= 0.9)]
for grp in ["veh", "les"]:
    s = q[q.group == grp]
    print(grp, "n=%d  apparent tau med %.1f ms | pure tau med %.1f ms (IQR %.1f-%.1f) | pure t50 med %.1f ms | CV_pure %.2f" %
          (len(s), s.tau_app_ms.median(), s.tau_pure_ms.median(),
           *np.percentile(s.tau_pure_ms, [25, 75]), s.t50_pure_ms.median(),
           s.tau_pure_ms.std(ddof=1) / s.tau_pure_ms.mean()))
a = q[q.group == "veh"].tau_pure_ms; b = q[q.group == "les"].tau_pure_ms
print("pure tau MWU p =", "%.3f" % mannwhitneyu(a, b).pvalue,
      " med %.1f vs %.1f ms (Δ %+.0f%%)" % (a.median(), b.median(), 100 * (b.median() / a.median() - 1)))
