# kv43_amp30_check.py - peak current at -30 mV, between-group comparison (paper anchor 3.8->1.7 nA), reusing the verified extraction logic
import scipy.io as sio, numpy as np, glob, os

BASE = r"D:\data\doi_10_5061_dryad_76hdr7t6z\physiology_data\in_vitro\extracted"

def segments(v, dt, jump_thr=1e-3, min_len=200):
    jumps = np.where(np.abs(np.diff(v)) > jump_thr)[0]
    cuts = [0] + [j + 1 for j in jumps] + [len(v)]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= min_len:
            out.append((a, b, float(np.median(v[a + 50:b])) * 1e3, (b - a) * dt))
    return out

def ipk_at(fn, target=-30.0):
    m = sio.loadmat(fn)
    vals = []
    for k in m:
        if not k.startswith("Trace"):
            continue
        t = m[k]; v = t[:, 2]; cur = t[:, 1]
        dt = np.median(np.diff(t[:, 0]))
        segs = segments(v, dt)
        for si in range(1, len(segs)):
            a, b, lvl, dur = segs[si]
            pa, pb, plvl, pdur = segs[si - 1]
            if abs(plvl - (-80)) < 2 and pdur >= 0.5 and abs(lvl - target) < 1.5:
                base = float(np.median(cur[max(a - int(0.05 / dt), 0):a]))
                i0 = a + int(0.002 / dt)
                vals.append(float((cur[i0:i0 + int(0.3 / dt)] - base).max()) * 1e9)
                break
    if not vals:
        return np.nan
    vals = np.asarray(vals, float)
    # spike-artifact repair: for repeated sweeps with max/min>5 take the smaller value
    if len(vals) >= 2 and vals.max() / max(vals.min(), 1e-12) > 5:
        return float(vals.min())
    return float(vals.mean())

for grp, d in [("veh", "veh_act"), ("les", "les_act")]:
    out = []
    for fn in sorted(glob.glob(os.path.join(BASE, d, "*.mat"))):
        val = ipk_at(fn)
        if np.isfinite(val):
            out.append(val)
    out = np.array(out)
    print(grp, "n=%d  I(-30 mV) med %.2f nA  mean %.2f ± %.2f (SEM)" %
          (len(out), np.median(out), out.mean(), out.std(ddof=1) / np.sqrt(len(out))))
