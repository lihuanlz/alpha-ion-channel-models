# lei211_temperature_summary.py
# five-temperature (25/27/30/33/37) summary + T1 (37 degC structure replication) + T2 (cross-host table comparison) + T3 (Q10 layer)
# criteria source: pre-registered verdict card Lei temperature Q10 2026-09-21 (frozen 2026-09-21 13:30)
# conventions verbatim identical to the 25 degC verdict card; 37 degC = herg37oc3 + herg37oc4 merged
import os, json, math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "数据", "Lei全量")
DT_I, DT_V = 2e-4, 1e-4

BATCHES = {"herg25oc1": 25, "herg27oc1": 27, "herg30oc1": 30,
           "herg33oc1": 33, "herg37oc3": 37, "herg37oc4": 37}
B37 = ["herg37oc3", "herg37oc4"]
SIN_V = [-140, -120, -100, -80, -60, -40, -20, 0, 20, 40]
ACT_V = [-50, -35, -20, -5, 10, 25, 40]
BEATTIE_TREC120 = 3.04e-3  # sealed value (2026-09-13 campaign)

def J(b, kind):
    fp = os.path.join(ROOT, f"lei211_{kind}_{b}.json")
    return json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else None

def cv_ok(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return None, 0, None
    m = float(np.median(a))
    cv = float(a.std() / abs(a.mean())) if abs(a.mean()) > 1e-12 else None
    return m, len(a), cv

def wilson(k, n, z=1.959964):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h

SUM = {}   # per-batch summary
for b, T in BATCHES.items():
    d12, d3, dtau = J(b, "J1J2_结果"), J(b, "J3_结果"), J(b, "tauh_表")
    cells = d12["cells"]
    out = {"T": T}
    # J4 E_rev
    e = [c["E_rev"] for c in cells.values() if c.get("E_rev") is not None]
    out["E_rev"] = dict(zip(("median", "n", "cv"), cv_ok(e)))
    # J1 h_ss
    hss = {}
    for i, V in enumerate(SIN_V):
        v = [c["J1"]["h"][i] for c in cells.values()
             if c.get("J1") and c["J1"]["ok"][i] and np.isfinite(c["J1"]["h"][i])]
        m, n, cv = cv_ok(v)
        hss[str(V)] = {"median": m, "n": n, "cv": cv}
    out["h_ss"] = hss
    # J2 m_ss
    mss = {}
    for i, V in enumerate(ACT_V):
        v = [c["J2"]["m"][i] for c in cells.values()
             if c.get("J2") and c["J2"]["ok_m"][i] and np.isfinite(c["J2"]["m"][i])]
        m, n, cv = cv_ok(v)
        mss[str(V)] = {"median": m, "n": n, "cv": cv}
    out["m_ss"] = mss
    # J3 tdeact / bigstep (per well: median over episodes)
    tdeact = {}
    for V in ("-60", "-40"):
        t1 = [np.median([e["tau1"] for e in c["J3"]["tdeact"][V]])
              for c in d3.values() if V in c["J3"]["tdeact"]]
        t2 = [np.median([e["tau2"] for e in c["J3"]["tdeact"][V]])
              for c in d3.values() if V in c["J3"]["tdeact"]]
        m1, n1, c1 = cv_ok(t1); m2, n2, c2 = cv_ok(t2)
        tdeact[V] = {"tau1": {"median": m1, "n": n1, "cv": c1},
                     "tau2": {"median": m2, "n": n2, "cv": c2}}
    out["tdeact"] = tdeact
    bs = [c["J3"]["tact"]["40_bigstep"] for c in d3.values() if "40_bigstep" in c["J3"]["tact"]]
    m1, n1, c1 = cv_ok([x["tau1"] for x in bs]); m2, n2, c2 = cv_ok([x["tau2"] for x in bs])
    out["bigstep"] = {"tau1_失活": {"median": m1, "n": n1, "cv": c1},
                      "tau2_激活": {"median": m2, "n": n2, "cv": c2}}
    # trec (tauh table)
    out["trec"] = {k: dtau[k] for k in dtau}
    # J5
    d5 = J(b, "J5_结果")
    if d5:
        j5 = {"per_freq": {}, "A3": None}
        kk = nn = 0
        for w, c in d5.items():
            for f in ("ap05hz", "ap1hz", "ap2hz"):
                r = c.get(f)
                if not r or "A1" not in r:
                    continue
                pf = j5["per_freq"].setdefault(f, {"A1": 0, "A2": 0, "n": 0, "reb": []})
                pf["n"] += 1
                pf["A1"] += bool(r["A1"]); pf["A2"] += bool(r["A2"])
                if np.isfinite(r.get("reb_ratio") or np.nan):
                    pf["reb"].append(r["reb_ratio"])
                nn += 1
                kk += bool(r["A1"]) and bool(r["A2"])
        a3v = [c["A3"] for c in d5.values() if c.get("A3") is not None]
        j5["A3"] = {"pass": sum(a3v), "n": len(a3v)}
        p, lo, hi = wilson(kk, nn)
        j5["合并"] = {"k": kk, "n": nn, "p": p, "lo": lo, "hi": hi}
        for f in j5["per_freq"]:
            rb = j5["per_freq"][f].pop("reb")
            j5["per_freq"][f]["reb_median"] = float(np.median(rb)) if rb else None
        out["J5"] = j5
    SUM[b] = out

# ---------- 37 degC merge ----------
def pooled(key_path, batches=B37):
    vals = []
    for b in batches:
        d12 = J(b, "J1J2_结果")
        for c in d12["cells"].values():
            v = key_path(c)
            if v is not None and np.isfinite(v):
                vals.append(v)
    return vals

M37 = {"T": 37}
e37 = pooled(lambda c: c.get("E_rev"))
M37["E_rev"] = dict(zip(("median", "n", "cv"), cv_ok(e37)))
hss = {}
for i, V in enumerate(SIN_V):
    v = pooled(lambda c, i=i: c["J1"]["h"][i] if c.get("J1") and c["J1"]["ok"][i] else None)
    m, n, cv = cv_ok(v); hss[str(V)] = {"median": m, "n": n, "cv": cv}
M37["h_ss"] = hss
mss = {}
for i, V in enumerate(ACT_V):
    v = pooled(lambda c, i=i: c["J2"]["m"][i] if c.get("J2") and c["J2"]["ok_m"][i] else None)
    m, n, cv = cv_ok(v); mss[str(V)] = {"median": m, "n": n, "cv": cv}
M37["m_ss"] = mss
tdeact = {}
for V in ("-60", "-40"):
    t1, t2 = [], []
    for b in B37:
        d3 = J(b, "J3_结果")
        for c in d3.values():
            if V in c["J3"]["tdeact"]:
                t1.append(np.median([e["tau1"] for e in c["J3"]["tdeact"][V]]))
                t2.append(np.median([e["tau2"] for e in c["J3"]["tdeact"][V]]))
    m1, n1, c1 = cv_ok(t1); m2, n2, c2 = cv_ok(t2)
    tdeact[V] = {"tau1": {"median": m1, "n": n1, "cv": c1},
                 "tau2": {"median": m2, "n": n2, "cv": c2}}
M37["tdeact"] = tdeact
bs = []
for b in B37:
    d3 = J(b, "J3_结果")
    bs += [c["J3"]["tact"]["40_bigstep"] for c in d3.values() if "40_bigstep" in c["J3"]["tact"]]
m1, n1, c1 = cv_ok([x["tau1"] for x in bs]); m2, n2, c2 = cv_ok([x["tau2"] for x in bs])
M37["bigstep"] = {"tau1_失活": {"median": m1, "n": n1, "cv": c1},
                  "tau2_激活": {"median": m2, "n": n2, "cv": c2}}
# trec per-well merge
tr37 = {}
for V in ("-140", "-120", "-100", "-80", "-60", "-40", "-20"):
    vv = []
    for b in B37:
        fp = os.path.join(ROOT, f"lei211_tauh_逐孔_{b}.json")
        pc = json.load(open(fp, encoding="utf-8"))
        vv += [c[V] for c in pc.values() if V in c]
    m, n, cv = cv_ok(vv)
    tr37[V] = {"median": m, "n": n, "cv": cv}
M37["trec"] = tr37
# J5 merge
kk = nn = 0; j5f = {}; a3v = []
for b in B37:
    d5 = J(b, "J5_结果") or {}
    for w, c in d5.items():
        if c.get("A3") is not None:
            a3v.append(c["A3"])
        for f in ("ap05hz", "ap1hz", "ap2hz"):
            r = c.get(f)
            if not r or "A1" not in r:
                continue
            pf = j5f.setdefault(f, {"A1": 0, "A2": 0, "n": 0, "reb": []})
            pf["n"] += 1
            pf["A1"] += bool(r["A1"]); pf["A2"] += bool(r["A2"])
            if np.isfinite(r.get("reb_ratio") or np.nan):
                pf["reb"].append(r["reb_ratio"])
            nn += 1; kk += bool(r["A1"]) and bool(r["A2"])
p, lo, hi = wilson(kk, nn)
for f in j5f:
    rb = j5f[f].pop("reb")
    j5f[f]["reb_median"] = float(np.median(rb)) if rb else None
M37["J5"] = {"per_freq": j5f, "A3": {"pass": sum(a3v), "n": len(a3v)},
             "合并": {"k": kk, "n": nn, "p": p, "lo": lo, "hi": hi}}
SUM["herg37合并"] = M37

# ---------- C1' pipeline identity (37oc3-A03 sactiv +40 segment) ----------
def releak(i5, v5):
    v0 = v5[0]
    wa, wb = int(0.175 / DT_I), int(0.2 / DT_I)
    mv = np.mean(v5[wa:wb] - v0)
    g = 0.0 if abs(mv) < 1e-9 else -np.mean(i5[wa:wb]) / mv
    return i5 + g * (v5 - v0)

c1p = None
try:
    dp = np.genfromtxt(os.path.join(DATA, "protocol", "protocol-sactiv.csv"), delimiter=",", skip_header=1)
    Vs = dp[:, 1:]
    dI = np.genfromtxt(os.path.join(DATA, "herg37oc3", "sactiv", "herg37oc3-sactiv-A03.csv"),
                       delimiter=",", skip_header=1)
    if dI.ndim == 1:
        dI = dI[:, None]
    sw = 6  # +40
    v10 = Vs[:, sw]
    i5 = dI[:, sw]
    ic = releak(i5, v10[::2][: dI.shape[0]])
    sig = float(np.std(i5[: int(0.1 / DT_I)]))  # noise of this sweep's first -80 holding segment
    v_ds = v10[::2][: dI.shape[0]]
    wa, wb = int(0.175 / DT_I), int(0.2 / DT_I)
    mv = float(np.mean(v_ds[wa:wb] - v_ds[0]))
    g = 0.0 if abs(mv) < 1e-9 else float(-np.mean(i5[wa:wb]) / mv)
    r = np.round(v10).astype(int)
    edges = np.where(np.diff(r) != 0)[0] + 1
    segs = [(int(r[s[0]]), s[0], s[-1] + 1) for s in np.split(np.arange(len(v10)), edges)]
    for vv, a, b in segs:
        if vv == 40 and (b - a) * DT_V >= 0.9:
            xa, xb = a // 2, b // 2
            end = float(np.mean(ic[xb - int(0.05 / DT_I): xb]))
            raw_end = float(np.mean(i5[xb - int(0.05 / DT_I): xb]))
            c1p = {"I_end_pA": end, "raw_end_pA": raw_end, "sigma_pA": sig,
                   "g_releak_pA每mV": g, "mv_窗": mv,
                   "过": bool(end > 0 or abs(end) < 4 * sig)}
            break
except Exception as ex:
    c1p = {"错误": str(ex)}

# ---------- T1 verdict (37 degC merged) ----------
verd = {}
e = M37["E_rev"]
verd["T1-J4 E_rev"] = "封卷" if (e["cv"] is not None and e["cv"] < 0.10 and -100 <= e["median"] <= -80) else "登记"
for V in ("-140", "-120"):
    h = M37["h_ss"][V]
    verd[f"T1-J1 h_ss({V})"] = "封卷" if (h["n"] >= 53 and h["cv"] is not None and h["cv"] < 0.3) else "登记"
for V in ("25", "40"):
    m = M37["m_ss"][V]
    verd[f"T1-J2 m_ss(+{V})"] = "封卷" if (m["n"] >= 53 and m["cv"] is not None and m["cv"] < 0.3) else "登记"
d60, d40 = M37["tdeact"]["-60"]["tau2"], M37["tdeact"]["-40"]["tau2"]
verd["T1-J3 τdeact方向"] = "方向成立" if (d60["median"] and d40["median"] and d40["median"] > d60["median"]) else "方向不成立"
j5m = M37["J5"]["合并"]
verd["T1-J5 合并"] = "封卷" if (j5m["lo"] is not None and j5m["lo"] >= 0.6) else "登记"

# ---------- T2 ----------
tr120 = M37["trec"]["-120"]["median"]
ratio2 = tr120 / BEATTIE_TREC120 if tr120 else None
verd["T2 τrec(-120)比值"] = "封卷" if (ratio2 and 0.3 <= ratio2 <= 3) else "登记"

# ---------- T3 Q10 ----------
def med(b, fam, key, sub=None):
    s = SUM[b][fam]
    x = s[key] if sub is None else s[key][sub]
    return x["median"]

Ts = np.array([25, 27, 30, 33, 37], float)
bcols = ["herg25oc1", "herg27oc1", "herg30oc1", "herg33oc1", "herg37合并"]
Q = {}
quants = {
    "τ_rec(-140)": lambda b: med(b, "trec", "-140"),
    "τ_rec(-120)": lambda b: med(b, "trec", "-120"),
    "τ_deact τ2(-40)": lambda b: med(b, "tdeact", "-40", "tau2"),
    "τ_act(+40)大补跳τ2": lambda b: med(b, "bigstep", "tau2_激活"),
    "τ_inact(+40)大补跳τ1": lambda b: med(b, "bigstep", "tau1_失活"),
}
for name, f in quants.items():
    ys = np.array([f(b) for b in bcols], float)
    ok = np.isfinite(ys) & (ys > 0)
    q10 = float((ys[0] / ys[-1]) ** (10 / 12)) if (ok[0] and ok[-1]) else None
    r2 = None
    if ok.sum() >= 3:
        A = np.vstack([Ts[ok], np.ones(ok.sum())]).T
        sol, *_ = np.linalg.lstsq(A, np.log(ys[ok]), rcond=None)
        k, c = sol
        pred = A @ np.array([k, c])
        ss_res = float(np.sum((np.log(ys[ok]) - pred) ** 2))
        ss_tot = float(np.sum((np.log(ys[ok]) - np.mean(np.log(ys[ok]))) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    Q[name] = {"medians_s": dict(zip([str(int(t)) for t in Ts], [None if not o else float(y) for y, o in zip(ys, ok)])),
               "Q10": q10, "Arrhenius_R2": r2,
               "判决": "封卷" if (q10 and 1.2 <= q10 <= 4.0) else "登记",
               "线性": "成立" if (r2 is not None and r2 >= 0.9) else "不成立"}
# E_rev(T)
eys = np.array([SUM[b]["E_rev"]["median"] for b in bcols], float)
ok = np.isfinite(eys)
A = np.vstack([Ts[ok], np.ones(ok.sum())]).T
sol, *_ = np.linalg.lstsq(A, eys[ok], rcond=None)
k, c = sol
pred = A @ np.array([k, c])
r2e = 1 - float(np.sum((eys[ok] - pred) ** 2)) / float(np.sum((eys[ok] - eys[ok].mean()) ** 2))
Q["E_rev(T)"] = {"medians_mV": dict(zip([str(int(t)) for t in Ts], [float(y) for y in eys])),
                 "slope_mV每C": float(k), "R2": r2e,
                 "判决": "封卷" if (k < 0 and 0.05 <= abs(k) <= 0.35) else "登记"}

# ---------- C2' completion rate ----------
C2 = {}
for b in BATCHES:
    sel = [l.strip() for l in open(os.path.join(DATA, "qc", f"selected-{b}.txt"), encoding="utf-8")
           if l.strip() and not l.startswith("#")]
    d12 = J(b, "J1J2_结果")
    n_ext = len(d12["cells"])
    C2[b] = {"selected": len(sel), "extracted": n_ext, "rate": n_ext / len(sel) if sel else None}

OUT = {"各批": SUM, "C1p": c1p, "C2完成率": C2, "T1判决": verd,
       "T2": {"Lei37_τrec(-120)_ms": None if tr120 is None else tr120 * 1000,
              "Beattie_ms": BEATTIE_TREC120 * 1000, "比值": ratio2},
       "T3": Q}
fp = os.path.join(ROOT, "lei211_温度汇总.json")
json.dump(OUT, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

# ---------- summary printout ----------
print("== C1' 37oc3-A03 sactiv+40:", c1p)
print("\n== core medians per temperature ==")
for b in bcols:
    s = SUM[b]
    print(f"[{s['T']}°C {b}] E_rev {s['E_rev']['median']:.2f}mV CV{s['E_rev']['cv']:.4f} n{s['E_rev']['n']} | "
          f"h140 {s['h_ss']['-140']['median']:.3f} CV{s['h_ss']['-140']['cv']:.3f} n{s['h_ss']['-140']['n']} | "
          f"h120 {s['h_ss']['-120']['median']:.3f} CV{s['h_ss']['-120']['cv']:.3f} n{s['h_ss']['-120']['n']} | "
          f"m25 {s['m_ss']['25']['median']:.3f} CV{s['m_ss']['25']['cv']:.3f} n{s['m_ss']['25']['n']} | "
          f"τd2(-60/-40) {s['tdeact']['-60']['tau2']['median']:.3f}/{s['tdeact']['-40']['tau2']['median']:.3f}s | "
          f"τrec120 {s['trec']['-120']['median']*1000:.2f}ms n{s['trec']['-120']['n']} | "
          f"大补跳τ1/τ2 {s['bigstep']['tau1_失活']['median']*1000:.1f}/{s['bigstep']['tau2_激活']['median']*1000:.1f}ms n{s['bigstep']['tau1_失活']['n']}")
print("\n== T1 (37 degC merged) ==")
for k_, v_ in verd.items():
    print(" ", k_, "->", v_)
print("  J5 merged:", j5m, " A3:", M37["J5"]["A3"], " per_freq:", M37["J5"]["per_freq"])
print("\n== T2 ==", OUT["T2"])
print("\n== T3 ==")
for k_, v_ in Q.items():
    print(" ", k_, "->", {kk2: vv2 for kk2, vv2 in v_.items() if not kk2.startswith("medians")})
    print("    ", v_.get("medians_s") or v_.get("medians_mV"))
print("\nsaved", fp)
