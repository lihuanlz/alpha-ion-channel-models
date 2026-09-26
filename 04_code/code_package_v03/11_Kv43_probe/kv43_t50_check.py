# kv43_t50_check.py - model-free half-time recheck of the tau_inact between-group difference
import scipy.io as sio, numpy as np, glob, os

def halflife(fn):
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
                y = cur[a1 + int(0.002 / dt):a1 + int(0.9 / dt)] - base
                pk = y.max(); ip = int(np.argmax(y))
                half = np.where(y[ip:] < pk / 2)[0]
                if len(half):
                    return half[0] * dt * 1e3
    return np.nan

BASE = r"D:\data\doi_10_5061_dryad_76hdr7t6z\physiology_data\in_vitro\extracted"
for grp, d in [("veh", "veh_inact"), ("les", "les_inact")]:
    vals = []
    for fn in sorted(glob.glob(os.path.join(BASE, d, "*.mat"))):
        h = halflife(fn)
        if np.isfinite(h):
            vals.append(h)
    vals = np.array(vals)
    print(grp, "n=%d  t50 med %.1f ms  IQR %.1f-%.1f" %
          (len(vals), np.median(vals), *np.percentile(vals, [25, 75])))
