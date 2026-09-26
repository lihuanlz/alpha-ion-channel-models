# kv43_amp30_dump.py — dump per-cell I(−30 mV) to amp30.csv (CSV twin of kv43_amp30_check.py)
import scipy.io as sio, numpy as np, glob, os, csv

BASE = r"D:\data\doi_10_5061_dryad_76hdr7t6z\physiology_data\in_vitro\extracted"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "amp30.csv")

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
    if len(vals) >= 2 and vals.max() / max(vals.min(), 1e-12) > 5:
        return float(vals.min())
    return float(vals.mean())

rows = []
for grp, d in [("veh", "veh_act"), ("les", "les_act")]:
    for fn in sorted(glob.glob(os.path.join(BASE, d, "*.mat"))):
        val = ipk_at(fn)
        if np.isfinite(val):
            rows.append((grp, os.path.splitext(os.path.basename(fn))[0], val))

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["group", "cell", "I_minus30_nA"])
    w.writerows(rows)

for grp in ("veh", "les"):
    v = np.array([r[2] for r in rows if r[0] == grp])
    print(grp, "n=%d med %.2f mean %.2f ± %.2f (SEM)" %
          (len(v), np.median(v), v.mean(), v.std(ddof=1) / np.sqrt(len(v))))
print("saved:", OUT)
