# lei211_J5_AP_forward.py
# criteria source: pre-registered verdict card section 2-J5 (frozen) + section 1 patch (AP re-leak-correction window 0.065-0.085 s)
# model: dm/dt = (m_ss - m)/tau_m, dh/dt = (h_ss - h)/tau_h(V), I = G*m*h*(V - E_rev), zero tuning
# table sources: m_ss/h_ss = this batch's population median tables (J1/J2 result files); tau_h(V) = sinactiv all-negative-level
#       tau_m = 432 ms flat (big-step tau2 registered value); G = |A(-140)|/(0.9*|-140 - E_rev|); E_rev per cell J4
# criteria: per cell per protocol A1 (repolarisation rebound >= 80% cycles and peak ratio in [0.3,3]) + A2 (RMS ratio in [0.3,3])
#       A3 frequency direction judged across protocols; three-protocol 633-trial pass-rate Wilson lower bound >= 0.6 -> SEAL
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "数据", "Lei全量")
SMOKE = os.environ.get("SMOKE", "0") == "1"

BATCH = os.environ.get("BATCH", "herg25oc1")
WELLS = [l.strip() for l in open(os.path.join(DATA, "qc", f"selected-{BATCH}.txt"), encoding="utf-8")
         if l.strip() and not l.startswith("#")]
if SMOKE:
    WELLS = ["A01", "B03", "C01"]

SIN_V = [-140, -120, -100, -80, -60, -40, -20, 0, 20, 40]
ACT_V = [-50, -35, -20, -5, 10, 25, 40]
DT_I = 2e-4
TAU_M = 0.432  # s, big-step tau2 registered value, flat

# ---------- population median tables ----------
J12 = json.load(open(os.path.join(ROOT, f"lei211_J1J2_结果_{BATCH}.json"), encoding="utf-8"))["cells"]
H_TAB = np.full(len(SIN_V), np.nan)
for i in range(len(SIN_V)):
    vals = [c["J1"]["h"][i] for c in J12.values() if c["J1"] and c["J1"]["ok"][i]]
    if len(vals) >= 3:
        H_TAB[i] = float(np.median(vals))
M_TAB = np.full(len(ACT_V), np.nan)
for i in range(len(ACT_V)):
    vals = [c["J2"]["m"][i] for c in J12.values() if c["J2"] and c["J2"]["ok_m"][i]]
    if len(vals) >= 3:
        M_TAB[i] = float(np.median(vals))
# gap filling: h_ss(-80) linearly interpolated from -100 and -60
HV = np.array(SIN_V, float)
ok = np.isfinite(H_TAB)
H_FILL = np.interp(HV, HV[ok], H_TAB[ok])
MV = np.array(ACT_V, float)
okm = np.isfinite(M_TAB)
M_FILL = np.interp(MV, MV[okm], M_TAB[okm])


def h_ss(v):
    return float(np.interp(v, HV, H_FILL, left=H_FILL[0], right=H_FILL[-1]))


def m_ss(v):
    return float(np.interp(v, MV, M_FILL, left=0.0, right=1.0))


def load_protocol(name):
    d = np.genfromtxt(os.path.join(DATA, "protocol", f"protocol-{name}.csv"),
                      delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1:]


def load_current(proto, well):
    d = np.genfromtxt(os.path.join(DATA, proto if BATCH=="herg25oc1" else os.path.join(BATCH, proto), f"{BATCH}-{proto}-{well}.csv"),
                      delimiter=",", skip_header=1)
    if d.ndim == 1:
        d = d[:, None]
    return d


def releak_ap(i, v, t):
    """AP official re-leak-correction window 0.065-0.085 s"""
    v0 = v[0]
    mask = (t >= 0.065) & (t <= 0.085)
    mv = np.mean(v[mask] - v0)
    g = 0.0 if abs(mv) < 1e-9 else -np.mean(i[mask]) / mv
    return i + g * (v - v0), g


def segments(v10):
    r = np.round(v10).astype(int)
    edges = np.where(np.diff(r) != 0)[0] + 1
    return [(int(r[s[0]]), s[0], s[-1] + 1) for s in np.split(np.arange(len(v10)), edges)]


def fit_doe(tt, y, sig):
    best = None
    for tr in np.exp(np.linspace(np.log(0.001), np.log(0.08), 22)):
        for td in np.exp(np.linspace(np.log(max(0.004, tr * 1.5)), np.log(2.0), 26)):
            x = np.exp(-tt / td) - np.exp(-tt / tr)
            X = np.column_stack([np.ones(len(tt)), x])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ sol) ** 2))
            if best is None or sse < best[0]:
                best = (sse, tr, td, sol)
    sse, tr, td, sol = best
    if abs(sol[1]) < 4 * sig:
        return None
    return float(tr), float(td)


# ---------- tau_h(V) population table (lei211_tauh_表.json, log-linear interpolation, +40 anchor 90 ms registered) ----------
_TAUR = json.load(open(os.path.join(ROOT, f"lei211_tauh_表_{BATCH}.json"), encoding="utf-8"))
_TH_X = np.array([float(k) for k in _TAUR.keys()] + [40.0])
_TH_Y = np.array([_TAUR[k]["median"] for k in _TAUR.keys()] + [0.090])
_th_order = np.argsort(_TH_X)
_TH_X, _TH_Y = _TH_X[_th_order], _TH_Y[_th_order]


def tau_h_arr(v):
    return np.interp(v, _TH_X, _TH_Y, left=_TH_Y[0], right=_TH_Y[-1])


# ---------- per-protocol cache: voltage axis + m_ss/h_ss/tau_h arrays (cell-independent) ----------
_PROTO_CACHE = {}


def proto_arrays(proto):
    if proto in _PROTO_CACHE:
        return _PROTO_CACHE[proto]
    t_v, Vp = load_protocol(proto)
    v = Vp[:, 0]
    mss = np.array([m_ss(x) for x in v[::50]])  # downsampled table lookup then interpolation for speed
    hss = np.array([h_ss(x) for x in v[::50]])
    th = tau_h_arr(v[::50])
    vv = v[::50]
    mss_f = np.interp(np.arange(len(v)), np.arange(0, len(v), 50)[: len(vv)], mss)
    hss_f = np.interp(np.arange(len(v)), np.arange(0, len(v), 50)[: len(vv)], hss)
    th_f = np.interp(np.arange(len(v)), np.arange(0, len(v), 50)[: len(vv)], th)
    _PROTO_CACHE[proto] = (t_v, v, mss_f, hss_f, th_f)
    return _PROTO_CACHE[proto]


def forward(proto, well):
    t_v, v, mss_arr, hss_arr, th_arr = proto_arrays(proto)
    n = len(v)
    Iraw = load_current(proto, well)[:, 0]
    if len(Iraw) != n:
        n2 = min(len(Iraw), n)
        Iraw = Iraw[:n2]
        v = v[:n2]
        t_v = t_v[:n2]
        mss_arr = mss_arr[:n2]
        hss_arr = hss_arr[:n2]
        th_arr = th_arr[:n2]
    Ic, g = releak_ap(Iraw, v, t_v[: len(Iraw)])
    e_rev = J12[well]["E_rev"]
    if e_rev is None:
        return None
    j1 = J12[well]["J1"]
    A140 = abs(j1["A"][0]) if j1 else np.nan
    if not np.isfinite(A140) or A140 < 1e-9:
        return None
    G = A140 / (0.9 * abs(-140 - e_rev))
    sig = float(np.std(Ic[: int(0.05 / DT_I)]))
    dt = 1e-4
    m = mss_arr[0]
    h = hss_arr[0]
    Isim = np.empty(len(v))
    dfac = v - e_rev
    for k in range(len(v)):
        m += dt * (mss_arr[k] - m) / TAU_M
        h += dt * (hss_arr[k] - h) / th_arr[k]
        Isim[k] = G * m * h * dfac[k]
    return {"v": v, "t": t_v[: len(v)], "Idata": Ic[: len(v)], "Isim": Isim, "sig": sig, "G": G}


def judge_cycle(res):
    """A1/A2 verdict (fix: repolarisation window starts at voltage peak +5 ms to avoid capacitive artefacts; sigma from the -80 silent segment between peaks)"""
    v, Id, Is = res["v"], res["Idata"], res["Isim"]
    above = v > -20
    rises = np.where(np.diff(above.astype(int)) == 1)[0]
    n_cyc = len(rises)
    # sigma: -80 segment between the last two peaks (else the first-10%-quantile segment outside the first 0.05 s)
    if n_cyc >= 2:
        a_q, b_q = rises[-2], rises[-1]
        quiet = Id[a_q + int(0.3 / 1e-4): b_q - int(0.05 / 1e-4)] if b_q - a_q > int(0.4 / 1e-4) else Id[:500]
    else:
        quiet = Id[:500]
    sig = float(np.std(quiet)) if len(quiet) > 100 else float(np.std(Id[:500]))
    cyc_hit = 0
    ratios = []
    for r in rises:
        # voltage peak position
        seg_v = v[r: r + int(0.15 / 1e-4)]
        if len(seg_v) < 50:
            continue
        pk_idx = r + int(np.argmax(seg_v))
        start = pk_idx + int(0.005 / 1e-4)  # peak +5 ms
        tail = np.where(v[start:] < -40)[0]
        if len(tail) == 0:
            continue
        w_end = start + tail[0]
        if w_end - start < 20:
            continue
        seg_s = Is[start:w_end]
        seg_d = Id[start:w_end]
        pk_s = np.max(seg_s)
        pk_d = np.max(seg_d)
        if pk_s > 3 * sig:
            cyc_hit += 1
            if pk_d > 3 * sig:
                ratios.append(pk_s / pk_d)
    a1a = (cyc_hit >= 0.8 * n_cyc) if n_cyc > 0 else False
    med_ratio = float(np.median(ratios)) if ratios else np.nan
    a1b = np.isfinite(med_ratio) and 0.3 <= med_ratio <= 3
    rms_d = float(np.sqrt(np.mean(Id ** 2)))
    rms_s = float(np.sqrt(np.mean(Is ** 2)))
    a2 = rms_d > 1e-9 and 0.3 <= (rms_s / rms_d) <= 3
    return {"n_cyc": n_cyc, "cyc_hit": cyc_hit, "A1": bool(a1a and a1b),
            "reb_ratio": med_ratio, "A2": bool(a2), "rms_ratio": rms_s / (rms_d + 1e-12),
            "sig": sig}


def main():
    out = {}
    for w in WELLS:
        cell = {}
        means = {}
        for proto in ["ap05hz", "ap1hz", "ap2hz"]:
            res = forward(proto, w)
            if res is None:
                cell[proto] = {"fail": "forward_none"}
                continue
            j = judge_cycle(res)
            cell[proto] = j
            tail_d = res["Idata"][-int(1.0 / 1e-4):]
            tail_s = res["Isim"][-int(1.0 / 1e-4):]
            means[proto] = (float(np.mean(tail_d)), float(np.mean(tail_s)))
            if SMOKE:
                res["proto"] = proto
                out.setdefault("_plot", []).append(res)
        # A3: 0.5 -> 2 Hz mean-change direction agreement
        if "ap05hz" in means and "ap2hz" in means:
            d_dir = np.sign(means["ap2hz"][0] - means["ap05hz"][0])
            s_dir = np.sign(means["ap2hz"][1] - means["ap05hz"][1])
            cell["A3"] = bool(d_dir == s_dir)
        else:
            cell["A3"] = None
        out[w] = cell
        line = {p: (cell[p].get("A1"), cell[p].get("A2")) for p in ["ap05hz", "ap1hz", "ap2hz"]}
        print(w, line, "A3=", cell["A3"], flush=True)
    fp = os.path.join(ROOT, f"lei211_J5_结果_{BATCH}.json" if not SMOKE else f"lei211_J5_冒烟_{BATCH}.json")
    json.dump({k: v for k, v in out.items() if k != "_plot"}, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", fp)
    if SMOKE and "_plot" in out:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9))
        for ax, res in zip(axes, out["_plot"]):
            t = res["t"][: len(res["Idata"])]
            ax.plot(t, res["Idata"], lw=0.5, color="0.6", label="data (re-leak-corr)")
            ax.plot(t, res["Isim"], lw=0.7, color="C0", label="alpha forward (zero tuning)")
            ax.set_title(f"A01 {res['proto']}  G={res['G']:.1f}nS")
            ax.legend(fontsize=8)
        fig.tight_layout()
        png = os.path.join(ROOT, "lei211_J5_冒烟.png")
        fig.savefig(png, dpi=120, bbox_inches="tight")
        print("saved", png)


if __name__ == "__main__":
    main()
