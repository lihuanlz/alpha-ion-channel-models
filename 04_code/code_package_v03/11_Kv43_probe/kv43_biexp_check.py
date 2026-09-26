# kv43_biexp_check.py — 双指数重拟合检验"τ 变慢是否成分假象" + Imax 组间对拍
import scipy.io as sio, numpy as np, glob, os, csv
from scipy.optimize import curve_fit

BASE = r"D:\data\doi_10_5061_dryad_76hdr7t6z\physiology_data\in_vitro\extracted"

def biexp(x, Af, tf, As, ts, C):
    return Af * np.exp(-x / tf) + As * np.exp(-x / ts) + C

def analyze(fn):
    m = sio.loadmat(fn)
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
            a0, b0 = segs[si - 1], segs[si]
            a1, b1 = segs[si], segs[si + 1]
            if b0 - a0 < 200 or b1 - a1 < 200:
                continue
            l0 = np.median(v[a0 + 50:b0]) * 1e3
            l1 = np.median(v[a1 + 50:b1]) * 1e3
            if abs(l0 + 120) < 1 and abs(l1 + 20) < 1:
                base = np.median(cur[a1 - int(0.05 / dt):a1])
                y = (cur[a1 + int(0.002 / dt):a1 + int(0.9 / dt)] - base) * 1e9
                x = np.arange(len(y)) * dt
                pk = y.max()
                try:
                    p0 = [pk * 0.7, 0.03, pk * 0.3, 0.3, 0.0]
                    bnd = ([0, 0.003, 0, 0.03, -np.inf], [np.inf, 0.5, np.inf, 3, np.inf])
                    popt, _ = curve_fit(biexp, x, y, p0=p0, bounds=bnd, maxfev=20000)
                    Af, tf, As, ts, C = popt
                    pred = biexp(x, *popt)
                    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
                    return dict(pk_nA=pk, Af=Af, tf_ms=tf * 1e3, As=As, ts_ms=ts * 1e3,
                                fast_frac=Af / (Af + As) if Af + As > 0 else np.nan, R2=r2)
                except Exception:
                    return dict(pk_nA=pk)
    return {}

rows = []
for grp, d in [("veh", "veh_inact"), ("les", "les_inact")]:
    for fn in sorted(glob.glob(os.path.join(BASE, d, "*.mat"))):
        r = analyze(fn)
        if r:
            r["group"] = grp
            r["cell"] = os.path.basename(fn).replace(".mat", "")
            rows.append(r)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "biexp_decay.csv"), "w", newline="") as f:
    keys = ["group", "cell", "pk_nA", "Af", "tf_ms", "As", "ts_ms", "fast_frac", "R2"]
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

for grp in ["veh", "les"]:
    sub = [r for r in rows if r["group"] == grp and "tf_ms" in r and r.get("R2", 0) >= 0.9]
    pk = [r["pk_nA"] for r in sub]
    tf = [r["tf_ms"] for r in sub]
    ff = [r["fast_frac"] for r in sub]
    ts = [r["ts_ms"] for r in sub]
    print(grp, "n=%d  pk med %.2f nA (IQR %.2f-%.2f)  tau_fast med %.1f ms (IQR %.1f-%.1f)  "
          "fast_frac med %.2f  tau_slow med %.0f ms" %
          (len(sub), np.median(pk), *np.percentile(pk, [25, 75]),
           np.median(tf), *np.percentile(tf, [25, 75]), np.median(ff), np.median(ts)))
