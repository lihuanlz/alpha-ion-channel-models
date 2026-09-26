# -*- coding: utf-8 -*-
"""
N-alpha-2: Nav1.5 whole-channel one-shot validation (2026-09-15)
Pre-registered item: results/pre-registration N-alpha-2 Nav1.5 whole-channel one-shot validation 2026-09-15.md (criteria pinned
  - A3/A4 (NaInact family) main pool = 40CP (the only level complete in all three cells); NaIV family = 80CP
  - v1.2: A0/A1 = I_pk(v) = G*m_inf^3*(v - E_rev) four-parameter joint fit (E_rev free in [50,75], apparent coordinates);
    static V_eff correction path abandoned (failure registered); C1/C2 forward uses command-voltage coordinates, E_rev from
    C2 V_test grid [-30,+60]; B2 cites the N-alpha-1 sealed 314 tau_h (not recomputed)

Modules:
  A0 E_rev | A1 m_inf(V) (I-V joint fit) | A2 tau_m(V) (m^3 rise x sealed tau_h decay x full chain) | A3 h_inf(V)
  A4 tau_h downward extension -50/-40/-30 | A5 tau_deact(-120)
  B1 008 CP staircase V1/2 stability | B2 314 Q10 (cites N-alpha-1)
  C1 NaIV forward (one mean table against three cells, one G per cell) | C2 NaInact forward (V_test single scalar)
  C3 hipsccm exploration (not judged)
Smoke: S1 tau_m recovery <= 15% | S2 Boltzmann V1/2 <= 1 mV / k <= 10% | S2b I-V recovery | S3 forward self-consistency

Environment variables: SMOKE=1 smoke; NREAL (default 10 smoke / 20 formal).
Discipline: the judge side only ast.parse + SMOKE=1; the formal run is done by the user in Spyder:
  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-15_Nα2_Nav15整通道一次验证.py' --wdir
Output: _结果.json/.csv/.png next to this script (smoke carries the _冒烟 suffix).
"""
import os, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.optimize import curve_fit

SMOKE = os.environ.get("SMOKE", "0") == "1"
NREAL = int(os.environ.get("NREAL", "10" if SMOKE else "20"))

DT_MS = 0.04
DT_S = DT_MS * 1e-3
DATA = Path(__file__).resolve().parent.parent / "公开数据" / "Nav1.5_27193878"
NAV = DATA / "nav"
CELLS = {"008": "batch2/medium_res_data/220502_008_ch2",
         "011": "batch2/medium_res_data/220502_011_ch3",
         "005": "batch2/medium_res_data/220503_005_ch2"}
CP_IV = 80            # NaIV family main pool
CP_IN = 40            # NaInact family main pool (v1.1: the only level complete in all three cells)
V_TM = [-20, -10, 0, 10, 20, 30, 40]       # A2 tau_m seven levels
V_TH_EXT = [-50, -40, -30]                 # A4 tau_h downward-extension three levels

# N-alpha-1 sealed input (verdict card 2026-09-15)
TAUH_NA1 = {-20: 0.494, -10: 0.613, 0: 0.511, 10: 0.414, 20: 0.335, 30: 0.303, 40: 0.270}
#   the -20 level was judged a protocol boundary by N-alpha-1, table value registered only; C1/C2 forward still uses it,
TAUH_314 = {0: (0.4129, 0.2293), 10: (0.3371, 0.2076), 20: (0.2739, 0.1942)}  # (25C, 35C)

rng = np.random.default_rng(20260915)


# ---------- observation chain (same method as N-alpha-1, modal-expansion FIR, do not change) ----------
def _cascade_modes(secs):
    R = None; P = None
    for bi, ai in secs:
        c = float(np.atleast_1d(bi)[0]) / float(ai[0])
        for p in np.roots(ai):
            if R is None:
                R = np.array([1.0 + 0j]); P = np.array([p])
            else:
                w = R / (P - p)
                R = np.concatenate([w, -w]); P = np.concatenate([P, np.full(len(P), p)])
        R = R * c
    return R, P


def chain_discrete(tau_res_s):
    secs = [([1.0], np.array([tau_res_s, 1.0]))]
    for N, fc in ((6, 1e4), (4, 5e3)):
        bb, aa = signal.bessel(N, 2 * np.pi * fc, analog=True, norm="phase")
        secs.append((bb, aa))
    R, P = _cascade_modes(secs)
    lam = np.exp(P * DT_S)
    n = np.arange(1, 256)
    hd = np.zeros(256)
    hd[1:] = np.sum((R / P)[:, None] * (lam - 1)[:, None] * lam[:, None] ** (n - 1)[None, :], axis=0).real
    s = float(hd.sum())
    hd /= s
    return hd, np.array([1.0])


def meta_iv(sub, cp):
    m = pd.read_csv(NAV / sub / f"NaIV_35C_{cp}CP_meta.csv")
    return float(m["rseries_Mohm"].iloc[0]), float(m["capacitance_pF"].iloc[0])


def meta_in(sub, cp):
    m = pd.read_csv(NAV / sub / f"NaInact_35C_{cp}CP_meta.csv")
    return float(m["rseries_Mohm"].iloc[0]), float(m["capacitance_pF"].iloc[0])


def tau_res_of(rs, cm, cp):
    return rs * (1 - cp / 100.0) * cm * 1e-6     # s


# ---------- data loading ----------
def load_naiv(sub, cp):
    df = pd.read_csv(NAV / sub / f"NaIV_35C_{cp}CP.csv")
    volts = [float(c) for c in df.columns]
    n1, n2 = {975: (225, 500), 1000: (250, 500)}[len(df)]
    cur = {v: df[f"{v:.1f}"].to_numpy() * 1e3 for v in volts}    # pA
    return volts, cur, n1, n2


def load_nainact(sub, cp):
    df = pd.read_csv(NAV / sub / f"NaInact_35C_{cp}CP.csv")
    volts = [float(c) for c in df.columns]
    cur = {v: df[f"{v:.1f}"].to_numpy() for v in volts}          # nA
    # test-pulse start: position of max |diff| of the column-averaged trace
    m = np.mean(np.abs(np.diff(np.column_stack([cur[v] for v in volts]), axis=0)), axis=1)
    i_test = int(np.argmax(m)) + 1
    return volts, cur, i_test


def mad_sig(x):
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# ---------- general estimators ----------
def boltz_act(V, vh, k):
    return 1.0 / (1.0 + np.exp(-(V - vh) / k))


def boltz_inact(V, vh, k):
    return 1.0 / (1.0 + np.exp((V - vh) / k))


def fit_boltz(V, y, kind):
    V = np.asarray(V, float); y = np.asarray(y, float)
    if kind == "act":
        f = boltz_act; p0 = [-35.0, 5.0]; bd = ([-80.0, 0.5], [0.0, 30.0])
    else:
        f = boltz_inact; p0 = [-80.0, 6.0]; bd = ([-140.0, 0.5], [-30.0, 30.0])
    p, _ = curve_fit(f, V, y, p0=p0, bounds=bd, maxfev=8000)
    yhat = f(V, *p)
    r2 = 1.0 - float(np.sum((y - yhat) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-30))
    return float(p[0]), float(p[1]), r2


def cv_mm(xs):
    xs = np.asarray(xs, float)
    return float(xs.std(ddof=1) / xs.mean()), float(xs.max() / xs.min())


def const_tc(xs):
    cv, mm = cv_mm(xs)
    return (cv < 0.3 and mm < 2.0), cv, mm


# ---------- A0/A1: peak I-V joint fit (v1.2 convention) ----------
def peak_iv(cur, n1, n2):
    """Per-voltage test-segment peak current (3-point smoothed, pA, signed: inward negative)."""
    out = {}
    for v, I in cur.items():
        seg = I[n1:n1 + n2]
        sm = np.convolve(seg, [1 / 3, 1 / 3, 1 / 3], "same")
        out[v] = float(sm.min()) if -float(sm.min()) > float(sm.max()) else float(sm.max())
    return out


def fit_iv(volts, pk):
    """I_pk(v) = G*m_inf(v; V1/2, k)^3*(v - E_rev) four-parameter joint fit (v1.2).
    Returns dict(G pS, vh_m, k_m, erev, r2). E_rev free in [50,75]; G > 0; m_inf is the apparent curve in command coordinates."""
    vs = np.array(sorted(volts), float)
    i = np.array([pk[v] for v in vs], float)

    def mdl(v, G, vh, k, er):
        return G * boltz_act(v, vh, k) ** 3 * (v - er)

    g0 = max(float(np.abs(i).max()) / 80.0, 100.0)
    p, _ = curve_fit(mdl, vs, i, p0=[g0, -35.0, 6.0, 60.0],
                     bounds=([10.0, -80.0, 0.5, 50.0], [1e7, 0.0, 30.0, 75.0]), maxfev=20000)
    r2 = 1.0 - float(np.sum((i - mdl(vs, *p)) ** 2) / (np.sum((i - i.mean()) ** 2) + 1e-30))
    return {"G": float(p[0]), "vh_m": float(p[1]), "k_m": float(p[2]),
            "erev": float(p[3]), "r2_iv": float(r2),
            "iv_vs": [float(x) for x in vs], "iv_i": [float(x) for x in i]}


# ---------- A2: tau_m (m^3 rise x sealed tau_h x full chain) ----------
def fit_taum(seg_pa, bd, ad, tau_h_ms):
    """Model A*chain(x)[(1-exp(-t/tau_m))^3 * exp(-t/tau_h)] + Ip; window = [2, n2) (skips 2 samples of step-edge capacitive spike, registered).
    tau_h fixed to the N-alpha-1 mean table; multi-start best SSE."""
    n2 = len(seg_pa)
    t = np.arange(n2) * DT_MS
    idx = np.arange(2, n2)
    y = -seg_pa[2:]
    A0 = max(float(y.max()), 1.0)

    def mdl(ii, A, tm, Ip):
        x = (1 - np.exp(-t / tm)) ** 3 * np.exp(-t / tau_h_ms)
        return A * signal.lfilter(bd, ad, x)[np.asarray(ii, dtype=int)] + Ip

    best = None
    for t0 in (0.03, 0.08, 0.15, 0.3, 0.6):
        try:
            p, _ = curve_fit(mdl, idx.astype(float), y, p0=[A0, t0, 0.0],
                             bounds=([0, 0.02, -0.3 * A0], [100 * A0, 2.0, 0.5 * A0]), maxfev=9000)
            ss = float(np.sum((y - mdl(idx, *p)) ** 2))
            if best is None or ss < best[1]:
                best = (ss, p)
        except Exception:
            pass
    if best is None:
        return None
    ss, (A, tm, Ip) = best
    return {"tm": float(tm), "A": float(A), "Ip": float(Ip),
            "rms": float(np.sqrt(ss / len(y))),
            "edge": bool(abs(tm - 0.02) < 1e-3 or abs(tm - 2.0) < 1e-3)}


# ---------- A4: NaInact conditioning-segment tau_h downward extension ----------
def fit_cond_decay(seg_na, bd, ad, sig):
    """Single exponential after the conditioning-segment transient peak (chain-corrected apparent convention). Window = [ipk+4, min(ipk+2500, len-10)]."""
    seg_pa = seg_na * 1e3
    n = len(seg_pa)
    iwin = min(int(200 / DT_MS), n - 1)
    sm = np.convolve(seg_pa[:iwin], [1 / 3, 1 / 3, 1 / 3], "same")
    ipk = int(sm.argmin())
    amp = -float(sm[ipk])
    if amp < max(8 * sig * 1e3, 50.0):
        return {"skipped": f"transient amp {amp:.0f}pA below threshold"}
    a = ipk + 4
    b = min(ipk + 2500, n - 10)
    if b - a < 200:
        return {"skipped": "window too short"}
    idx = np.arange(a, b)
    y = -seg_pa[a:b]
    tloc = np.arange(n) * DT_MS

    def mdl(ii, A, tau, Ip):
        x = np.exp(-tloc / tau)
        return A * signal.lfilter(bd, ad, x)[np.asarray(ii, dtype=int)] + Ip

    best = None
    for t0 in (1.0, 5.0, 20.0, 60.0):
        try:
            p, _ = curve_fit(mdl, idx.astype(float), y, p0=[y.max(), t0, 0.0],
                             bounds=([0, 0.5, -0.3 * y.max()], [100 * y.max(), 300.0, 0.5 * y.max()]),
                             maxfev=9000)
            ss = float(np.sum((y - mdl(idx, *p)) ** 2))
            if best is None or ss < best[1]:
                best = (ss, p)
        except Exception:
            pass
    if best is None:
        return {"skipped": "fit failed"}
    ss, (A, tau, Ip) = best
    return {"tau": float(tau), "A": float(A), "rms": float(np.sqrt(ss / len(y))), "skipped": "",
            "edge": bool(abs(tau - 0.5) < 1e-3 or abs(tau - 300.0) < 1e-3)}


# ---------- A5: tail-current tau_deact ----------
def fit_tail(tail_pa, bd, ad, sig):
    """Tail-segment single exponential (chain-corrected). Window = [2, n3) (skips 2 samples of down-step capacitive spike)."""
    n3 = len(tail_pa)
    sm = np.convolve(tail_pa, [1 / 3, 1 / 3, 1 / 3], "same")
    amp = -float(sm[:8].min())
    if amp < max(8 * sig, 100.0):
        return None
    idx = np.arange(2, n3)
    y = -tail_pa[2:]
    t = np.arange(n3) * DT_MS

    def mdl(ii, A, tau, C):
        x = np.exp(-t / tau)
        return A * signal.lfilter(bd, ad, x)[np.asarray(ii, dtype=int)] + C

    best = None
    for t0 in (0.05, 0.1, 0.2, 0.5):
        try:
            p, _ = curve_fit(mdl, idx.astype(float), y, p0=[y.max(), t0, 0.0],
                             bounds=([0, 0.02, -0.5 * y.max()], [100 * y.max(), 2.0, 0.5 * y.max()]),
                             maxfev=9000)
            ss = float(np.sum((y - mdl(idx, *p)) ** 2))
            if best is None or ss < best[1]:
                best = (ss, p)
        except Exception:
            pass
    return None if best is None else {"tau": float(best[1][1]), "rms": float(np.sqrt(best[0] / len(y)))}


# ---------- forward simulation core (shared by C1/C2/S3) ----------
def sim_trace(v_eff, tm_ms, th_ms, vh_m, k_m, vh_h, k_h, erev, bd, ad, n):
    """m^3 h step response (m(0)=0, h(0)=1) through the chain -> unit-conductance current shape (with driving force)."""
    t = np.arange(n) * DT_MS
    mi = boltz_act(v_eff, vh_m, k_m)
    hi = boltz_inact(v_eff, vh_h, k_h)
    x = (mi * (1 - np.exp(-t / tm_ms))) ** 3 * (hi + (1 - hi) * np.exp(-t / th_ms))
    return signal.lfilter(bd, ad, x) * (v_eff - erev)


# ---------- smoke S1/S2/S3 ----------
def smoke():
    print("=" * 72, flush=True)
    print(" [smoke] S1 tau_m recovery | S2 Boltzmann recovery | S3 forward self-consistency", flush=True)
    print("=" * 72, flush=True)
    ok_all = True
    # S1: synthetic m^3 h (known tau_m table), three pseudo-cells (three residual-pole levels) -> A2 estimator recovery
    tm_true = {-20: 0.50, -10: 0.30, 0: 0.20, 10: 0.14, 20: 0.10, 30: 0.08, 40: 0.07}
    for tr in (7.7e-6, 10.1e-6, 18.1e-6):
        bd, ad = chain_discrete(tr)
        errs = []
        for v in V_TM:
            for _ in range(NREAL):
                n2 = 500
                t = np.arange(n2) * DT_MS
                x = (0.9 * (1 - np.exp(-t / tm_true[v]))) ** 3 * np.exp(-t / TAUH_NA1[v])
                y = -signal.lfilter(bd, ad, x) * 20000.0 + rng.normal(0, 25.0, n2)
                f = fit_taum(y, bd, ad, TAUH_NA1[v])
                if f and not f["edge"]:
                    errs.append(abs(f["tm"] - tm_true[v]) / tm_true[v])
        err = float(np.median(errs)) * 100
        ok = err <= 15.0
        ok_all &= ok
        print(f"  S1 residual pole {tr*1e6:5.1f}us: tau_m recovery median error {err:5.1f}% (n={len(errs)}, criterion <=15%) {'pass' if ok else '**FAIL**'}", flush=True)
    # S2: synthetic Boltzmann + noise -> V1/2/k recovery
    for kind, vh0, k0, vgrid in (("act", -30.0, 6.0, np.arange(-80, 61, 10)),
                                 ("inact", -85.0, 7.0, np.arange(-150, -29, 10))):
        e_vh, e_k = [], []
        f = boltz_act if kind == "act" else boltz_inact
        for _ in range(NREAL):
            y = f(vgrid, vh0, k0) + rng.normal(0, 0.02, len(vgrid))
            vh, k, _ = fit_boltz(vgrid, y, kind)
            e_vh.append(abs(vh - vh0)); e_k.append(abs(k - k0) / k0)
        ok = np.median(e_vh) <= 1.0 and np.median(e_k) <= 0.10
        ok_all &= ok
        print(f"  S2 {kind}: |dV1/2| median {np.median(e_vh):.2f}mV (<=1) |dk|/k median {np.median(e_k)*100:.1f}% (<=10%) {'pass' if ok else '**FAIL**'}", flush=True)
    # S2b: I-V joint-fit recovery (v1.2 new estimator)
    e_vh, e_k, e_er = [], [], []
    vg = np.arange(-80, 61, 10.0)
    for _ in range(NREAL):
        it = 2000.0 * boltz_act(vg, -32.0, 6.5) ** 3 * (vg - 62.0) + rng.normal(0, 30.0, len(vg))
        f = fit_iv(vg, dict(zip(vg.tolist(), it.tolist())))
        e_vh.append(abs(f["vh_m"] + 32.0)); e_k.append(abs(f["k_m"] - 6.5) / 6.5)
        e_er.append(abs(f["erev"] - 62.0))
    ok = np.median(e_vh) <= 1.0 and np.median(e_k) <= 0.10 and np.median(e_er) <= 1.5
    ok_all &= ok
    print(f"  S2b I-V: |ΔV½| {np.median(e_vh):.2f}mV（≤1） |Δk|/k {np.median(e_k)*100:.1f}%（≤10%） "
          f"|dE_rev| {np.median(e_er):.2f}mV (<=1.5) {'pass' if ok else '**FAIL**'}", flush=True)
    # S3: C1 pipeline self-consistency - synthetic (known G, no noise, code-path identity verification) -> forward replay R^2
    bd, ad = chain_discrete(10.1e-6)
    vh_m, k_m, vh_h, k_h, erev = -30.0, 6.0, -85.0, 7.0, 55.0
    tm_v = {v: 0.15 for v in range(-80, 61, 10)}
    G0 = 300.0
    obs_all, sim_all = [], []
    for v in range(-80, 61, 10):
        s = sim_trace(v, tm_v[v], 0.35, vh_m, k_m, vh_h, k_h, erev, bd, ad, 500)
        obs_all.append(G0 * s)                      # noiseless: verifies the G least-squares + replay code path
        sim_all.append(G0 * s)
    o = np.concatenate(obs_all)
    # same method as C1: least-squares re-estimate G, then replay
    s_cat = np.concatenate([sim_trace(v, tm_v[v], 0.35, vh_m, k_m, vh_h, k_h, erev, bd, ad, 500)
                            for v in range(-80, 61, 10)])
    G_hat = float(np.sum(o * s_cat) / np.sum(s_cat * s_cat))
    r2 = 1.0 - float(np.sum((o - G_hat * s_cat) ** 2) / np.sum((o - o.mean()) ** 2))
    gerr = abs(G_hat - G0) / G0 * 100
    ok = r2 >= 0.999
    ok_all &= ok
    print(f"  S3 forward self-consistency (noiseless) R^2={r2:.6f} G recovery error={gerr:.3f}% (criterion R^2>=0.999) {'pass' if ok else '**FAIL**'}", flush=True)
    # S3b registered: R^2 under realistic SNR with added noise sigma=25pA (not judged; reference for C1's 8% line)
    obs_n = o + rng.normal(0, 25.0, o.shape)
    r2n = 1.0 - float(np.sum((obs_n - G_hat * s_cat) ** 2) / np.sum((obs_n - obs_n.mean()) ** 2))
    print(f"  S3b registered (added noise sigma=25pA): R^2={r2n:.4f} (not judged)", flush=True)
    return ok_all


# ---------- module A ----------
def run_A(pool_cells):
    print("\n[module A] E_rev / m_inf / tau_m / h_inf / tau_h ext / tau_deact", flush=True)
    res = {"cells": {}}
    for cell in pool_cells:
        sub = CELLS[cell]
        rs, cm = meta_iv(sub, CP_IV)
        bd, ad = chain_discrete(tau_res_of(rs, cm, CP_IV))
        volts, cur, n1, n2 = load_naiv(sub, CP_IV)
        sig_hold = mad_sig(cur[0.0][:n1])
        pk = peak_iv(cur, n1, n2)
        # A0+A1: I-V joint fit (v1.2: E_rev and m_inf come out together)
        iv = fit_iv(volts, pk)
        erev, vh_m, k_m, r2_m = iv["erev"], iv["vh_m"], iv["k_m"], iv["r2_iv"]
        # A2
        tm = {}
        for v in V_TM:
            seg = cur[float(v)][n1:n1 + n2]
            f = fit_taum(seg, bd, ad, TAUH_NA1[v])
            tm[v] = None if (f is None or f["edge"]) else f["tm"]
        # A5
        td = []
        for v in volts:
            if v < -10:
                continue
            tail = cur[v][n1 + n2:]
            f = fit_tail(tail, bd, ad, sig_hold)
            if f:
                td.append(f["tau"])
        res["cells"][cell] = {"erev": erev, "sig_hold": sig_hold,
                              "vh_m": vh_m, "k_m": k_m, "r2_m": r2_m, "G_iv": iv["G"],
                              "iv_vs": iv["iv_vs"], "iv_i": iv["iv_i"],
                              "tm": tm, "tau_deact_med": float(np.median(td)) if td else None,
                              "n_tail": len(td), "rs": rs, "cm": cm}
        print(f"  {cell}: E_rev={erev:.1f}mV V½m={vh_m:.1f} k_m={k_m:.2f} R²={r2_m:.4f} "
              f"τ_deact={res['cells'][cell]['tau_deact_med'] and round(res['cells'][cell]['tau_deact_med'],3)}ms(n={len(td)})", flush=True)
    # A3/A4：NaInact @40CP
    for cell in pool_cells:
        sub = CELLS[cell]
        rs_i, cm_i = meta_in(sub, CP_IN)
        bd_i, ad_i = chain_discrete(tau_res_of(rs_i, cm_i, CP_IN))
        v_in, cur_in, i_test = load_nainact(sub, CP_IN)
        sig_cond = mad_sig(cur_in[v_in[0]][:i_test - 2600])   # v1.4: reference column = most negative column of each file (-120 for 011/005)
        # A3: test peak (local baseline correction: mean of the conditioning segment's last 100 ms)
        pk_in = {}
        for v in v_in:
            base = float(np.mean(cur_in[v][i_test - 2500:i_test]))
            seg = cur_in[v][i_test:] - base
            sm = np.convolve(seg, [1 / 3, 1 / 3, 1 / 3], "same")
            pk_in[v] = -float(sm.min())
        ref = pk_in[v_in[0]]                              # v1.4: normalisation reference = most negative column (h_inf ~= 1)
        avail = [pk_in[v] / ref for v in v_in]
        vh_h, k_h, r2_h = fit_boltz(v_in, avail, "inact")
        # A4
        th_ext = {}
        for v in V_TH_EXT:
            f = fit_cond_decay(cur_in[float(v)][:i_test], bd_i, ad_i, sig_cond)
            th_ext[v] = None if f["skipped"] else f["tau"]
        res["cells"][cell].update({"vh_h": vh_h, "k_h": k_h, "r2_h": r2_h,
                                   "th_ext": th_ext, "avail": [float(a) for a in avail],
                                   "v_in": [float(v) for v in v_in], "i_test": i_test,
                                   "rs_in": rs_i, "cm_in": cm_i})
        print(f"  {cell}: V½h={vh_h:.1f} k_h={k_h:.2f} R²={r2_h:.3f} "
              f"tau_h_ext={ {v: (None if th_ext[v] is None else round(th_ext[v],1)) for v in V_TH_EXT} }", flush=True)
    return res


# ---------- module B ----------
def run_B():
    print("\n[module B] B1 008 CP staircase | B2 314 Q10 (cites N-alpha-1)", flush=True)
    out = {"B1": {}, "B2": {}}
    sub = CELLS["008"]
    vh_m_l, vh_h_l = [], []
    for cp in (0, 20, 40, 60, 80):
        try:
            volts, cur, n1, n2 = load_naiv(sub, cp)
            pk = peak_iv(cur, n1, n2)
            iv = fit_iv(volts, pk)
            vh_m, k_m = iv["vh_m"], iv["k_m"]
        except Exception:
            vh_m, k_m = None, None
        try:
            v_in, cur_in, i_test = load_nainact(sub, cp)
            pk_in = {}
            for v in v_in:
                base = float(np.mean(cur_in[v][i_test - 2500:i_test]))
                sm = np.convolve(cur_in[v][i_test:] - base, [1 / 3, 1 / 3, 1 / 3], "same")
                pk_in[v] = -float(sm.min())
            avail = [pk_in[v] / pk_in[v_in[0]] for v in v_in]
            vh_h, k_h, _ = fit_boltz(v_in, avail, "inact")
        except Exception:
            vh_h, k_h = None, None
        out["B1"][cp] = {"vh_m": vh_m, "vh_h": vh_h, "k_m": k_m, "k_h": k_h}
        vh_m_l.append(vh_m); vh_h_l.append(vh_h)
        print(f"  B1 CP{cp:2d}: V½m={vh_m and round(vh_m,1)} V½h={vh_h and round(vh_h,1)}", flush=True)
    rng_m = max(x for x in vh_m_l if x) - min(x for x in vh_m_l if x)
    rng_h = max(x for x in vh_h_l if x) - min(x for x in vh_h_l if x)
    out["B1"]["range_m"] = float(rng_m); out["B1"]["range_h"] = float(rng_h)
    out["B1"]["pass"] = bool(rng_m <= 3.0 and rng_h <= 3.0)
    print(f"  B1 verdict: V1/2m cross-CP range={rng_m:.1f}mV V1/2h range={rng_h:.1f}mV (criterion <=3) "
          f"{'pass' if out['B1']['pass'] else '**FAIL**'}", flush=True)
    for v, (t25, t35) in TAUH_314.items():
        q = t25 / t35
        out["B2"][v] = {"t25": t25, "t35": t35, "ratio": float(q), "ok": bool(1.5 <= q <= 2.5)}
    n_ok = sum(1 for d in out["B2"].values() if isinstance(d, dict) and d["ok"])
    out["B2"]["summary"] = f"{n_ok}/3 levels in [1.5,2.5]"
    print(f"  B2 Q10: " + " ".join(f"{v}mV:{d['ratio']:.2f}" for v, d in out["B2"].items()
          if isinstance(d, dict)) + f" → {out['B2']['summary']}", flush=True)
    return out


# ---------- module C: whole-channel forward ----------
def mean_tables(res, pool_cells):
    cs = [res["cells"][c] for c in pool_cells]
    vh_m = float(np.mean([c["vh_m"] for c in cs])); k_m = float(np.mean([c["k_m"] for c in cs]))
    vh_h = float(np.mean([c["vh_h"] for c in cs])); k_h = float(np.mean([c["k_h"] for c in cs]))
    tm_raw = {}
    for v in V_TM:
        xs = [c["tm"][v] for c in cs if c["tm"][v] is not None]
        if xs:
            tm_raw[v] = float(np.mean(xs))
    tm_v = {}
    if tm_raw:
        valid = sorted(tm_raw)
        for v in V_TM:
            tm_v[v] = tm_raw.get(v) or tm_raw[min(valid, key=lambda u: abs(u - v))]  # v1.3: missing measured level clamped to nearest measurable
    else:
        tm_v = {v: 0.15 for v in V_TM}
    th_ext = {}
    for v in V_TH_EXT:
        xs = [c["th_ext"][v] for c in cs if c["th_ext"][v] is not None]
        th_ext[v] = float(np.mean(xs)) if xs else 20.0
    return vh_m, k_m, vh_h, k_h, tm_v, th_ext


def run_C(res, pool_cells):
    print("\n[module C] C1 NaIV forward | C2 NaInact forward | C3 hipsccm exploration", flush=True)
    vh_m, k_m, vh_h, k_h, tm_v, th_ext = mean_tables(res, pool_cells)
    out = {"tables": {"vh_m": vh_m, "k_m": k_m, "vh_h": vh_h, "k_h": k_h,
                      "tm_v": tm_v, "th_ext": th_ext}, "C1": {}, "C2": {}}
    tm_tbl = lambda v: tm_v[int(np.clip(round(v / 10) * 10, -20, 40))]
    th_tbl = lambda v: (th_ext[int(np.clip(round(v / 10) * 10, -50, -30))] if v < -20
                        else TAUH_NA1[int(np.clip(round(v / 10) * 10, -20, 40))])
    # ---- C1 ----
    for cell in pool_cells:
        sub = CELLS[cell]
        rs, cm = meta_iv(sub, CP_IV)
        bd, ad = chain_discrete(tau_res_of(rs, cm, CP_IV))
        volts, cur, n1, n2 = load_naiv(sub, CP_IV)
        erev = res["cells"][cell]["erev"]
        sig_hold = res["cells"][cell]["sig_hold"]
        pk = peak_iv(cur, n1, n2)
        S, O = {}, {}
        for v in volts:
            S[v] = sim_trace(v, tm_tbl(v), th_tbl(v), vh_m, k_m, vh_h, k_h, erev, bd, ad, n2)
            O[v] = cur[v][n1:n1 + n2]
        s_cat = np.concatenate([S[v] for v in volts]); o_cat = np.concatenate([O[v] for v in volts])
        G = float(np.sum(o_cat * s_cat) / (np.sum(s_cat * s_cat) + 1e-30))
        res_cat = o_cat - G * s_cat
        rms = float(np.sqrt(np.mean(res_cat ** 2)))
        pko = float(np.max(np.abs(o_cat)))
        pkmax = max(abs(pk[v]) for v in volts)
        sig_v = [v for v in volts if abs(pk[v]) >= max(5 * sig_hold, 0.05 * pkmax)]  # signal levels registered (v1.2)
        r2v = {v: 1 - float(np.sum((O[v] - G * S[v]) ** 2) / (np.sum((O[v] - O[v].mean()) ** 2) + 1e-30))
               for v in sig_v}
        ok = rms / pko <= 0.08
        out["C1"][cell] = {"G": G, "rms_over_pk": rms / pko, "pass": bool(ok),
                           "r2_min": float(min(r2v.values())), "r2_med": float(np.median(list(r2v.values()))),
                           "r2v": {str(v): float(r2v[v]) for v in r2v}}
        print(f"  C1 {cell}: G={G:.0f}pS RMS/peak={rms/pko*100:.1f}% (<=8%) "
              f"R^2 median={np.median(list(r2v.values())):.3f} worst={min(r2v.values()):.3f} ({len(sig_v)} signal levels) "
              f"{'pass' if ok else '**FAIL**'}", flush=True)
    n1ok = sum(1 for c in pool_cells if out["C1"][c]["pass"])
    out["C1"]["verdict"] = f"{n1ok}/{len(pool_cells)} cells pass"
    print(f"  C1 verdict: {out['C1']['verdict']} (criterion >=2/3) {'**FORWARD HOLDS**' if n1ok >= 2 else 'fail'}", flush=True)
    # ---- C2 ----
    for cell in pool_cells:
        sub = CELLS[cell]
        rs_i, cm_i = meta_in(sub, CP_IN)
        bd_i, ad_i = chain_discrete(tau_res_of(rs_i, cm_i, CP_IN))
        v_in = res["cells"][cell]["v_in"]; avail_obs = np.array(res["cells"][cell]["avail"])
        n_test = len(load_nainact(sub, CP_IN)[1][v_in[0]]) - res["cells"][cell]["i_test"]
        erev = res["cells"][cell]["erev"]
        best = None
        for vtest in np.arange(-30, 61, 2):
            tm_t = tm_tbl(vtest); th_t = th_tbl(vtest)
            hi_t = boltz_inact(vtest, vh_h, k_h)          # h_inf at the test voltage (v1.3: h relaxes from h0 toward hi_t)
            sim = []
            for vc in v_in:
                h0 = boltz_inact(vc, vh_h, k_h)
                t = np.arange(n_test) * DT_MS
                x = (boltz_act(vtest, vh_m, k_m) * (1 - np.exp(-t / tm_t))) ** 3 \
                    * (hi_t + (h0 - hi_t) * np.exp(-t / th_t)) * (vtest - erev)
                y = signal.lfilter(bd_i, ad_i, x)
                sim.append(-float(y.min()))
            sim = np.array(sim); sim /= sim.max() + 1e-30
            obs = avail_obs / (avail_obs.max() + 1e-30)
            ss = float(np.sum((obs - sim) ** 2))
            if best is None or ss < best[0]:
                best = (ss, vtest, sim, obs)
        ss, vtest, sim, obs = best
        r2 = 1 - float(np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2))
        ok = r2 >= 0.99
        out["C2"][cell] = {"v_test": float(vtest), "r2": float(r2), "pass": bool(ok),
                           "sim": [float(x) for x in sim], "obs": [float(x) for x in obs]}
        print(f"  C2 {cell}: V_test={vtest:.0f}mV availability curve R^2={r2:.4f} (>=0.99) {'pass' if ok else '**FAIL**'}", flush=True)
    n2ok = sum(1 for c in pool_cells if out["C2"][c]["pass"])
    out["C2"]["verdict"] = f"{n2ok}/{len(pool_cells)} cells pass"
    print(f"  C2 verdict: {out['C2']['verdict']} (criterion >=2/3) {'**HOLDS**' if n2ok >= 2 else 'fail'}", flush=True)
    # ---- C3 exploration ----
    try:
        hp = DATA / "hipsccm"
        f1 = sorted(hp.glob("*.csv"))[0]
        h = pd.read_csv(f1, comment="#", header=None)
        tcol = h.iloc[:, 0].to_numpy(float)
        out["C3"] = {"file": f1.name, "rows": int(len(h)),
                     "t_span": [float(tcol[0]), float(tcol[-1])],
                     "dt_med": float(np.median(np.diff(tcol)))}
        print(f"  C3 exploration: {f1.name} rows={len(h)} t in [{tcol[0]:.1f},{tcol[-1]:.1f}] dt={out['C3']['dt_med']:.4f} (registered, not judged)", flush=True)
    except Exception as e:
        out["C3"] = {"error": str(e)}
        print(f"  C3 exploration read failed (registered): {e}", flush=True)
    return out


# ---------- verdict summary + save ----------
def aggregate(res, pool_cells):
    print("\n" + "=" * 72, flush=True)
    print(" [verdict summary]", flush=True)
    print("=" * 72, flush=True)
    cs = res["cells"]
    vd = {}
    # A0
    er = [cs[c]["erev"] for c in pool_cells]
    r = max(er) - min(er)
    vd["A0"] = {"vals": er, "range": float(r), "pass": bool(r <= 5.0)}
    print(f"  A0 E_rev cross-cell range={r:.1f}mV (<=5) {'constant' if vd['A0']['pass'] else '**not constant**'}", flush=True)
    # A1
    vh = [cs[c]["vh_m"] for c in pool_cells]; kk = [cs[c]["k_m"] for c in pool_cells]
    r = max(vh) - min(vh); cvk = float(np.std(kk, ddof=1) / np.mean(kk))
    vd["A1"] = {"vh_m": vh, "k_m": kk, "range": float(r), "cv_k": cvk,
                "pass": bool(r <= 5.0 and cvk < 0.3)}
    print(f"  A1 m_inf: V1/2 range={r:.1f}mV (<=5) k CV={cvk:.2f} (<0.3) {'constant' if vd['A1']['pass'] else '**not constant**'}", flush=True)
    # A2
    n_const = 0; tab = {}
    for v in V_TM:
        xs = [cs[c]["tm"][v] for c in pool_cells if cs[c]["tm"][v] is not None]
        if len(xs) == 3:
            ok, cv, mm = const_tc(xs)
            tab[v] = {"taus": xs, "CV": cv, "maxmin": mm, "verdict": "constant" if ok else "not constant"}
            n_const += int(ok)
        else:
            tab[v] = {"taus": xs, "verdict": f"not judgeable (n={len(xs)})"}
    vd["A2"] = {"table": tab, "n_const": n_const, "pass": bool(n_const >= 5)}
    print(f"  A2 tau_m: constant levels {n_const}/7 (>=5) {'constant picture holds' if vd['A2']['pass'] else '**does not hold**'}", flush=True)
    for v in V_TM:
        d = tab[v]
        s = " ".join(f"{x:.3f}" for x in d["taus"]) if len(d["taus"]) == 3 else str(d["taus"])
        print(f"      {v:>4}mV: [{s}] {d['verdict']}" + (f" CV={d.get('CV',0):.2f} ratio={d.get('maxmin',0):.2f}" if "CV" in d else ""), flush=True)
    # A3
    vh = [cs[c]["vh_h"] for c in pool_cells]; kk = [cs[c]["k_h"] for c in pool_cells]
    r = max(vh) - min(vh); cvk = float(np.std(kk, ddof=1) / np.mean(kk))
    vd["A3"] = {"vh_h": vh, "k_h": kk, "range": float(r), "cv_k": cvk,
                "pass": bool(r <= 5.0 and cvk < 0.3)}
    print(f"  A3 h_inf: V1/2 range={r:.1f}mV (<=5) k CV={cvk:.2f} (<0.3) {'constant' if vd['A3']['pass'] else '**not constant**'}", flush=True)
    # A4
    n4 = 0; tab4 = {}
    for v in V_TH_EXT:
        xs = [cs[c]["th_ext"][v] for c in pool_cells if cs[c]["th_ext"][v] is not None]
        if len(xs) == 3:
            ok, cv, mm = const_tc(xs)
            tab4[v] = {"taus": xs, "CV": cv, "maxmin": mm, "verdict": "constant" if ok else "not constant"}
            n4 += int(ok)
        else:
            tab4[v] = {"taus": xs, "verdict": f"not judgeable (n={len(xs)})"}
    vd["A4"] = {"table": tab4, "n_const": n4, "pass": bool(n4 == 3)}
    print(f"  A4 tau_h ext: constant levels {n4}/3 (need 3/3) {'constant' if vd['A4']['pass'] else '**not constant/partially unjudgeable**'}", flush=True)
    # A5
    xs = [cs[c]["tau_deact_med"] for c in pool_cells if cs[c]["tau_deact_med"] is not None]
    if len(xs) == 3:
        ok, cv, mm = const_tc(xs)
        vd["A5"] = {"vals": xs, "CV": cv, "maxmin": mm, "pass": bool(ok)}
        print(f"  A5 tau_deact(-120): [{ ' '.join(f'{x:.3f}' for x in xs)}]ms CV={cv:.2f} ratio={mm:.2f} "
              f"{'constant' if ok else '**not constant**'}", flush=True)
    else:
        vd["A5"] = {"vals": xs, "pass": False, "note": f"n={len(xs)}"}
        print(f"  A5 not judgeable (n={len(xs)})", flush=True)
    return vd


def make_png(res, outC, pool_cells, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.5))
    cs = res["cells"]
    # 1 I-V + joint fit (A0/A1)
    ax = axs[0, 0]
    for c in pool_cells:
        ax.plot(cs[c]["iv_vs"], cs[c]["iv_i"], "o", ms=4, label=c)
        vv = np.linspace(-80, 75, 300)
        ax.plot(vv, cs[c]["G_iv"] * boltz_act(vv, cs[c]["vh_m"], cs[c]["k_m"]) ** 3 * (vv - cs[c]["erev"]),
                "-", lw=1)
    ax.set_title("A0/A1 I-V joint fit (apparent coordinates)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    # 2 h∞
    ax = axs[0, 1]
    for c in pool_cells:
        ax.plot(cs[c]["v_in"], cs[c]["avail"], "o", ms=4, label=c)
        vv = np.linspace(-150, -30, 200)
        ax.plot(vv, boltz_inact(vv, cs[c]["vh_h"], cs[c]["k_h"]), "-", lw=1)
    ax.set_title("A3 h∞(V) @40CP"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    # 3 τ_m
    ax = axs[0, 2]
    for c in pool_cells:
        xs = [cs[c]["tm"][v] for v in V_TM]
        ax.plot(V_TM, [x if x is not None else np.nan for x in xs], "o-", ms=4, label=c)
    ax.set_title("A2 τ_m(V)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    # 4 tau_h full table
    ax = axs[1, 0]
    for c in pool_cells:
        xs = [cs[c]["th_ext"][v] for v in V_TH_EXT]
        ax.semilogy(V_TH_EXT, [x if x is not None else np.nan for x in xs], "s", ms=5, label=c + " ext")
    ax.semilogy(sorted(TAUH_NA1), [TAUH_NA1[v] for v in sorted(TAUH_NA1)], "k^-", label="N-alpha-1 sealed")
    ax.set_title("tau_h(V) full table (A4 ext + N-alpha-1)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    # 5 C1 forward per-level R^2
    ax = axs[1, 1]
    for c in pool_cells:
        d = outC["C1"][c]
        vs = sorted(float(v) for v in d["r2v"])
        ax.plot(vs, [d["r2v"][str(v)] if str(v) in d["r2v"] else d["r2v"][f"{v:.1f}"] for v in vs],
                "o-", ms=3, label=f"{c} RMS/peak={d['rms_over_pk']*100:.1f}%")
    ax.axhline(0.9, color="r", ls="--", lw=1)
    ax.set_title("C1 forward per-level R^2"); ax.set_ylim(-0.2, 1.05); ax.legend(fontsize=7); ax.grid(alpha=.3)
    # 6 C2 availability
    ax = axs[1, 2]
    for c in pool_cells:
        d = outC["C2"][c]
        ax.plot(res["cells"][c]["v_in"], d["obs"], "o", ms=4, label=f"{c} measured")
        ax.plot(res["cells"][c]["v_in"], d["sim"], "-", lw=1, label=f"{c} forward")
    ax.set_title("C2 NaInact availability: measured vs forward"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle("N-alpha-2 Nav1.5 whole-channel one-shot validation")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 72, flush=True)
    print(" N-alpha-2: Nav1.5 whole-channel one-shot validation (pre-registered 2026-09-15, v1.1/v1.2)", flush=True)
    print("=" * 72, flush=True)
    ok = smoke()
    if not ok:
        print("\ncontrols not seated -> statistics void, halt.", flush=True)
        return
    pool = ["008"] if SMOKE else ["008", "011", "005"]
    if SMOKE:
        print("\n[smoke mode] single-cell 008 full-module rehearsal (verdict needs n=3; smoke only verifies the pipeline)", flush=True)
    res = run_A(pool)
    vd = aggregate(res, pool) if len(pool) == 3 else {"note": "smoke n=1 not judged"}
    outB = run_B()
    outC = run_C(res, pool)
    tag = "_冒烟" if SMOKE else ""
    base = Path(__file__).resolve().with_suffix("")
    # CSV: main table
    rows = []
    for c in pool:
        d = res["cells"][c]
        row = {"cell": c, "erev": d["erev"], "vh_m": d["vh_m"], "k_m": d["k_m"], "r2_m": d["r2_m"],
               "vh_h": d["vh_h"], "k_h": d["k_h"], "r2_h": d["r2_h"],
               "tau_deact": d["tau_deact_med"], "G": outC["C1"].get(c, {}).get("G"),
               "C1_rms_over_pk": outC["C1"].get(c, {}).get("rms_over_pk"),
               "C2_vtest": outC["C2"].get(c, {}).get("v_test"),
               "C2_r2": outC["C2"].get(c, {}).get("r2")}
        for v in V_TM:
            row[f"tm_{v}"] = d["tm"][v]
        for v in V_TH_EXT:
            row[f"th_{v}"] = d["th_ext"][v]
        rows.append(row)
    pd.DataFrame(rows).to_csv(str(base) + f"_结果{tag}.csv", index=False, encoding="utf-8-sig")
    # JSON
    js = {"meta": {"smoke": SMOKE, "card": "Nα-2", "cp_iv": CP_IV, "cp_in": CP_IN},
          "A_verdicts": vd, "B": outB,
          "C": {k: v for k, v in outC.items() if k != "tables"}, "tables": outC["tables"],
          "cells": {c: {k: v for k, v in res["cells"][c].items() if k != "minf_rows"}
                    for c in pool}}
    with open(str(base) + f"_结果{tag}.json", "w", encoding="utf-8") as f:
        json.dump(js, f, ensure_ascii=False, indent=1,
                  default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    make_png(res, outC, pool, str(base) + f"_结果{tag}.png")
    print("\n" + "=" * 72, flush=True)
    print(f" saved: {base}_结果{tag}.json/.csv/.png", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
