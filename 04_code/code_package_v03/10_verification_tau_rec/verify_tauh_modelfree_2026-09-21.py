# verify_tauh_modelfree_2026-09-21.py
# tau_rec temperature trend, triple check:
#  (1) SSE improvement of continuous refinement over the grid (is refinement just wandering in a flat valley?)
#  (2) fraction piled at the grid lower edge (are high-temperature medians pinned by the grid edge?)
#  (3) model-free clocks: time-to-extremum t_ext / half-amplitude time t_half / 10-90% rise time
#      - no exponential fit involved; directly tests whether Q10 survives
import os, json
import numpy as np
from scipy.optimize import least_squares

AM = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(AM, "..", "数据", "Lei全量")
DT_I = 2e-4
SIN_V = [-140, -120, -100, -80, -60, -40, -20]
CHECK_V = {-140: 0, -120: 1}
BATCH_T = {"herg25oc1": 25, "herg27oc1": 27, "herg30oc1": 30,
           "herg33oc1": 33, "herg37oc3": 37, "herg37oc4": 37}

d = np.genfromtxt(os.path.join(DATA, "protocol", "protocol-sinactiv.csv"),
                  delimiter=",", skip_header=1)
V10 = d[:, 1:]
def segments(v10):
    r = np.round(v10).astype(int)
    edges = np.where(np.diff(r) != 0)[0] + 1
    return [(int(r[s[0]]), s[0], s[-1] + 1) for s in np.split(np.arange(len(v10)), edges)]
TEST_IDX = {}
for sw, V in enumerate(SIN_V):
    for vv, a, b in segments(V10[:, sw]):
        if vv == V and (b - a) * 1e-4 >= 0.4:
            TEST_IDX[sw] = (a // 2, b // 2)
            break

tt = np.arange(750) * DT_I
GRID = []
for tr in np.exp(np.linspace(np.log(0.001), np.log(0.08), 22)):
    for td in np.exp(np.linspace(np.log(max(0.004, tr * 1.5)), np.log(2.0), 26)):
        GRID.append((tr, td))
PINV = []
for tr, td in GRID:
    x = np.exp(-tt / td) - np.exp(-tt / tr)
    X = np.column_stack([np.ones(750), x])
    PINV.append(np.linalg.pinv(X))
PINV = np.array(PINV)
TR_GRID_MS = np.exp(np.linspace(np.log(1.0), np.log(80.0), 22))  # grid tr values (ms)

def grid_best(seg):
    best = None
    for gi, (tr, td) in enumerate(GRID):
        sol = PINV[gi] @ seg
        pred = sol[1] * (np.exp(-tt / td) - np.exp(-tt / tr)) + sol[0]
        sse = float(np.sum((seg - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, tr, td, sol[1], sol[0])
    return best

def refine(seg, tr0, td0, c0):
    def resid(p):
        c, A, ltr, ltd = p
        tr, td = np.exp(ltr), np.exp(ltd)
        return c + A * (np.exp(-tt / td) - np.exp(-tt / tr)) - seg
    p0 = [c0, (seg.max() - seg.min()) or 1.0, np.log(tr0), np.log(td0)]
    lb = [-np.inf, -np.inf, np.log(3e-4), np.log(1e-3)]
    ub = [np.inf, np.inf, np.log(0.2), np.log(3.0)]
    try:
        sol = least_squares(resid, p0, bounds=(lb, ub), max_nfev=300)
        return float(np.exp(sol.x[2])), float(np.sum(sol.fun ** 2))
    except Exception:
        return None, None

def clocks(seg):
    """Model-free clocks: returns t_half, t_ext, t1090 (ms). Returns (None,None,None) on failure."""
    base = float(np.median(seg[:5]))
    dev = seg - base
    i = int(np.argmax(np.abs(dev)))
    e = dev[i]
    if abs(e) < 1e-12:
        return None, None, None
    frac = dev[: i + 1] / e
    def cross(f):
        idx = np.where(frac >= f)[0]
        return idx[0] * DT_I * 1000.0 if len(idx) else None
    t10, t90 = cross(0.1), cross(0.9)
    t_half = cross(0.5)
    t_ext = i * DT_I * 1000.0
    t1090 = (t90 - t10) if (t10 is not None and t90 is not None) else None
    return t_half, t_ext, t1090

out = {}
for batch, T in BATCH_T.items():
    wells = [l.strip() for l in open(os.path.join(DATA, "qc", f"selected-{batch}.txt"), encoding="utf-8")
             if l.strip() and not l.startswith("#")]
    folder = "sinactiv" if batch == "herg25oc1" else os.path.join(batch, "sinactiv")
    for w in wells:
        fp = os.path.join(DATA, folder, f"{batch}-sinactiv-{w}.csv")
        if not os.path.exists(fp):
            continue
        In = np.genfromtxt(fp, delimiter=",", skip_header=1)
        if In.ndim == 1:
            In = In[:, None]
        for sw, V in enumerate(SIN_V):
            if V not in CHECK_V or sw not in TEST_IDX:
                continue
            a, b = TEST_IDX[sw]
            seg = In[a: a + 750, sw]
            if len(seg) < 750:
                continue
            sig = float(np.std(In[: int(0.09 / DT_I), sw]))
            gb = grid_best(seg)
            if gb is None or abs(gb[3]) < 4 * sig:
                continue
            rf_tr, rf_sse = refine(seg, gb[1], gb[2], gb[4])
            th, te, t19 = clocks(seg)
            edge = int(np.argmin(np.abs(TR_GRID_MS - gb[1] * 1000.0)))  # nearest grid-point index
            out.setdefault(str(T), {}).setdefault(str(V), []).append([
                w, gb[1] * 1000.0, gb[0],
                (rf_tr * 1000.0) if rf_tr else None, rf_sse,
                th, te, t19, edge])
    print("done", batch, flush=True)

fp = os.path.join(AM, "verify_tauh_无模型时钟_2026-09-21.json")
json.dump(out, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
print("saved", fp)

def q10r2(meds):
    x = np.array([t for t, m in meds], float)
    y = np.log(np.array([m for t, m in meds]))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(np.exp(-slope * 10)), float(r2)

print("\n=== (1) SSE improvement of refinement over grid ===")
for T in ["25", "27", "30", "33", "37"]:
    for V in ["-140", "-120"]:
        rows = out.get(T, {}).get(V, [])
        imp = [(r[2] - r[4]) / r[2] * 100 for r in rows if r[4] is not None and r[2] > 0]
        if imp:
            print(f"  {T}C {V}mV: n={len(imp)} SSE improvement median {np.median(imp):.2f}%  p90 {np.percentile(imp, 90):.2f}%")

print("\n=== (2) fraction piled at the grid lower edge (1.0/1.23ms) ===")
for T in ["25", "27", "30", "33", "37"]:
    for V in ["-140", "-120"]:
        rows = out.get(T, {}).get(V, [])
        if rows:
            e = np.array([r[8] for r in rows])
            print(f"  {T}C {V}mV: n={len(e)} fraction in lowest 2 bins {np.mean(e <= 1)*100:.1f}%")

print("\n=== (3) model-free clock five-temperature medians and Q10 ===")
for V in ["-140", "-120"]:
    for name, idx in [("t_half", 5), ("t_ext", 6), ("t10-90", 7)]:
        meds = []
        for T in ["25", "27", "30", "33", "37"]:
            rows = out.get(T, {}).get(V, [])
            vals = [r[idx] for r in rows if r[idx] is not None]
            if vals:
                meds.append((int(T), float(np.median(vals))))
        if len(meds) == 5:
            q, r2 = q10r2(meds)
            print(f"  {V}mV {name}: Q10={q:.2f} R2={r2:.3f} medians={[round(m,2) for _, m in meds]}")
