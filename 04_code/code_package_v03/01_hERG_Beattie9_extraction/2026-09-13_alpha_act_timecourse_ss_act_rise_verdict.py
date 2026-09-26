# -*- coding: utf-8 -*-
# 2026-09-13_alpha model_activation time course_steady-state activation rising-edge adjudication.py
# Purpose: rescue measurement for the activation block (route B) - direct measurement of the
#   activation time course from the steady_activation test-pulse rising edge.
#   Background: the envelope method had valid n=3<4 at two steps (insufficient data). This protocol
#   exists for all nine cells, with 7 voltage steps (-60..+60) and 5-6s long windows.
#   Segment structure (verified programmatically): each period -120x50ms pre-pulse -> -80x0.2s ->
#   h handling (finalized after three smoke rounds, fixed before run): the model contains h
#     explicitly, but whether h is identifiable is decided by a contribution gate -
#     h_contrib = max_t |G*m(t)*(1-h_ss)*e^{-t/tau_h}| > 8*sigma_dec to report h_ss/tau_h.
#     Smoke measurements: the h-dip falls inside the m~0 window at most steps (contribution <8*sigma,
#     not identifiable, not reported); at +60 the h-bump contribution >8*sigma (identifiable, gate
#   Model (pre-registered):
#     I(t) = G_eff · [m0 + (1-m0)·R(t)] · [h_ss + (1-h_ss)·e^{-t/τ_h}]
#     R(t)  = 1 - s1*e^{-t/tau1} - (1-s1)*e^{-t/tau2}     (R(0)=0, R(inf)=1; s1<0 => S-shaped)
#     Parameters: G_eff (linear projection), m0 in [0,0.5], s1 in [-5,1), tau1 in [5ms,10s], tau2>=3*tau1,
#           h_ss in [0,1], tau_h in [1ms,0.5s]. Baseline = mean of last 100ms of the pre-test -80x0.2s segment (subtracted first).
#     Fit: decimate to 1ms, drop first 5ms; coarse grid (tau1 x tau2 x tau_h) with linear evaluation, top3
#           -> least_squares variable projection (tau2>=3*tau1 penalty constraint).
#   Smoke data observations (important; the model was finalized from these):
#     (1) each step starts with a large spike (instantaneous current = G*m0*h(0)*DF): the reset
#        sequence cannot clear the slow layer, m0~0.13-0.15 nonzero (the free-m0 design proved necessary);
#        (2) the positive-voltage h decay is huge (+40 swing ~2nA), measurable; (3) steady-state h_ss is
#        extremely deep on the 5s scale (only ~0.02 at 0mV), so the m-ride amplitude = G*h_ss*(1-m0) is only
#        0.05-0.2 nA -> the row QC must gate on the m-ride visible amplitude >4*sigma (not G>8*sigma: G is extrapolated from a nearly fully inactivated floor and would be inflated);
#     (4) the -60 step contains NaN points (masked; drop if valid points <80%).
#   QC (pre-registered): (1) tau1/tau2/m0 not at grid edge (tau_h at edge only suppresses the h report,
#     does not void the row), (2) RMS/sigma_dec<1.5, (3) noise-envelope gate whitening (20ms binned ACF
#     at lags 1/5/10/20, violations <=2; envelope = second-largest magnitude per order from this cell's silent segments, floor 0.2 - method established in v5),
#     (4) m-ride amplitude m_ride = G*h_ss_fit*(1-m0) > 4*sigma_dec (the 4*sigma operating point is certified by C6).
#   Criteria (pre-registered): each voltage step: valid n>=4 and both CV(tau1), CV(tau2) <0.3 =>
#     activation mode at that step = discrete constants; some steps pass => graded entry; none pass => open a new card.
#     h_ss/tau_h is reported only at gate-open steps (registered for the inactivation block); h is not judged in this card.
#   Controls (pre-registered; noise = block-bootstrap reshuffle of the median cell's silent residuals, ACF preserved):
#     C1 +40-like (G=4, m0=0.05, s1=-0.6, tau1=80ms, tau2=400ms, h_ss=0.16, tau_h=12ms) x20:
#       tau1/tau2 inversion error medians <15% each;
#     C2 single-component m+h (s1=0, tau2=100ms, h as C1) x20: correct collapse (min(|s1|,|1-s1|)<0.08 or at edge) >=16;
#     C3 deep-h invisible step (G=1.5, s1=-0.5, tau1=300ms, tau2=1.5s, h_ss=0.44, tau_h=13.4ms) x20:
#       tau1/tau2 error medians <15% each AND h report gate-rejected >=16/20;
#     C4 tight-separation +60-like (G=12, m0=0.05, s1=-0.5, tau1=40ms, tau2=160ms, h_ss=0.156, tau_h=8ms) x20:
#       tau1/tau2 error medians <20% each (tight-separation criterion 20%, declared before run);
#     C5 no-h (h_ss=1, rest as C1) x20: h report gate-rejected >=16/20 AND tau1 error median <15%.
#     C6 low-SNR step (G=1.3, m-ride ~0.2nA ~ 4*sigma, rest as C1) x20: valid >=16 AND
#       tau1/tau2 error medians <15% each (certifies the 4*sigma row-QC operating point).
# Run: python this file (full nine-cell set); SMOKE=1 for the 16713003 single-cell smoke test.
import os
import json
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("SMOKE", "0") == "1"
CELLS = ["16713003"] if SMOKE else ["16704007", "16704047", "16707014", "16708016",
                                    "16708060", "16708118", "16713003", "16713110", "16715049"]
DT = 1e-4
DEC = 10                       # decimate to 1ms
SKIP_MS = 5                    # drop the first 5ms of the segment
CV_PASS = 0.3
H_GATE = 8.0                   # h contribution gate (in units of sigma_dec)
V_STEPS = [-60, -40, -20, 0, 20, 40, 60]
LN3 = float(np.log(3.0))
BND = dict(m0=(0.0, 0.5), s1=(-5.0, 0.999), u1=(np.log(0.005), np.log(10.0)),
           u2=(np.log(0.005), np.log(10.0)), h=(0.0, 1.0), uh=(np.log(0.001), np.log(0.5)))


def load_cell(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/steady_activation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/steady_activation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def segments(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    return [(float(V[s[0]]), int(s[0]), len(s)) for s in np.split(np.arange(len(V)), edges)]


def find_steps(info):
    """test-step segment: duration >=4s, preceded by -80x0.2s and -120x50ms pre-pulse segments"""
    out = []
    for k, (v, s0, n) in enumerate(info):
        if n * DT >= 4.0 and any(abs(v - vv) < 2.0 for vv in V_STEPS) and k >= 2:
            pv, ps, pn = info[k - 1]
            ppv, pps, ppn = info[k - 2]
            if abs(pv + 80.0) < 2.0 and 0.1 < pn * DT < 0.4 and abs(ppv + 120.0) < 2.0:
                out.append(dict(v=v, s0=s0, n=n, base_s=ps, base_n=pn))
    return out


def quiet_pool(info, I):
    """-80 silent-segment residuals (1ms decimation, linearly detrended)"""
    pool = []
    for v, s0, n in info:
        if abs(v + 80.0) < 2.0 and n * DT >= 0.15:
            seg = I[s0:s0 + n].astype(float)[::DEC]
            t = np.arange(len(seg))
            tr = np.polyfit(t, seg, 1)
            pool.append(seg - np.polyval(tr, t))
    return np.concatenate(pool) if pool else None


def acf_envelope(info, I, lags=(1, 5, 10, 20), bin_n=20):
    """per-cell silent-segment ACF envelope (1ms decimation, bin_n bins, second-largest per order, floor 0.2) - the v5 method"""
    env = {L: 0.2 for L in lags}
    vals = {L: [] for L in lags}
    for v, s0, n in info:
        if abs(v + 80.0) < 2.0 and n * DT >= 0.3:
            seg = I[s0:s0 + n].astype(float)[::DEC]
            t = np.arange(len(seg))
            tr = np.polyfit(t, seg, 1)
            r = seg - np.polyval(tr, t)
            nb = len(r) // bin_n
            if nb < 12:
                continue
            b = r[:nb * bin_n].reshape(nb, bin_n).mean(axis=1)
            b = b - b.mean()
            denom = float(b @ b)
            if denom <= 0:
                continue
            for L in lags:
                if nb > L + 8:
                    vals[L].append(float(b[L:] @ b[:-L]) / denom)
    for L in lags:
        if len(vals[L]) >= 2:
            env[L] = max(0.2, float(sorted(np.abs(vals[L]))[-2]))
        elif vals[L]:
            env[L] = max(0.2, float(abs(vals[L][0])))
    return env


def white_gate(res, env, bin_n=20, lags=(1, 5, 10, 20), max_viol=2):
    nb = len(res) // bin_n
    if nb < 12:
        return False, 99
    b = (res[:nb * bin_n].reshape(nb, bin_n).mean(axis=1))
    b = b - b.mean()
    denom = float(b @ b)
    if denom <= 0:
        return False, 99
    viol = 0
    for L in lags:
        r = float(b[L:] @ b[:-L]) / denom if nb > L + 8 else 0.0
        if abs(r) > env.get(L, 0.2):
            viol += 1
    return viol <= max_viol, viol


def model_f(t, m0, s1, t1, t2, h, th):
    R = 1.0 - s1 * np.exp(-t / t1) - (1.0 - s1) * np.exp(-t / t2)
    hh = h + (1.0 - h) * np.exp(-t / th)
    return (m0 + (1.0 - m0) * R) * hh


def m_part(t, m0, s1, t1, t2):
    R = 1.0 - s1 * np.exp(-t / t1) - (1.0 - s1) * np.exp(-t / t2)
    return m0 + (1.0 - m0) * R


def fit_step(t, y):
    """variable projection: G_eff linear; w parameterization tau2=3*tau1*e^w (w>=0, strict constraint);
    grid starts -> least_squares. Returns a parameter dict or None."""
    scale = max(float(np.ptp(y)), 1e-6)

    def unpack(p):
        m0, s1, u1, w, h, uh = p
        return m0, s1, np.exp(u1), np.exp(u1 + LN3 + w), h, np.exp(uh)

    def proj(p):
        m0, s1, t1, t2, h, th = unpack(p)
        f = model_f(t, m0, s1, t1, t2, h, th)
        num = float(f @ y); den = float(f @ f)
        if den <= 0:
            return None, None
        G = num / den
        return G, y - G * f

    def resid(p):
        G, r = proj(p)
        if G is None or G <= 0:
            r = np.ones_like(y) * scale
        return r

    g1 = [0.015, 0.05, 0.15, 0.5, 1.5]
    g2 = [0.1, 0.3, 0.9, 2.5, 6.0]
    gh = [0.003, 0.010, 0.030, 0.100]
    cand = []
    for a in g1:
        for b in g2:
            if b < 3.0 * a:
                continue
            for hh in gh:
                p0 = np.array([0.05, -0.5, np.log(a), np.log(b / a) - LN3, 0.5, np.log(hh)])
                G, r = proj(p0)
                if G is not None and G > 0:
                    cand.append((float(r @ r), p0))
    if not cand:
        return None
    cand.sort(key=lambda z: z[0])
    lo = np.array([BND['m0'][0], BND['s1'][0], BND['u1'][0], 0.0, BND['h'][0], BND['uh'][0]])
    hi = np.array([BND['m0'][1], BND['s1'][1], BND['u1'][1],
                   BND['u2'][1] - BND['u1'][0] - LN3, BND['h'][1], BND['uh'][1]])
    best = None
    for _, p0 in cand[:3]:
        p0 = np.clip(p0, lo, hi)
        try:
            fq = least_squares(resid, p0, bounds=(lo, hi),
                               xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=3000)
        except Exception:
            continue
        G, r = proj(fq.x)
        if G is None or G <= 0:
            continue
        sse = float(r @ r)
        if best is None or sse < best[0]:
            best = (sse, fq.x, G, r)
    if best is None:
        return None
    sse, p, G, res = best
    m0, s1, t1, t2, h, th = unpack(p)
    m0, s1, t1, t2, h, th = float(m0), float(s1), float(t1), float(t2), float(h), float(th)
    u2 = float(p[2] + LN3 + p[3])
    edge_m = bool(p[2] <= BND['u1'][0] + 0.05 or u2 >= BND['u2'][1] - 0.05
                  or m0 >= BND['m0'][1] - 1e-3)
    h_edge = bool(p[5] <= BND['uh'][0] + 0.05 or p[5] >= BND['uh'][1] - 0.05)
    contrib = float(np.max(np.abs(G * m_part(t, m0, s1, t1, t2) * (1.0 - h) * np.exp(-t / th))))
    return dict(G=G, m0=m0, s1=s1, tau1=t1, tau2=t2, h_ss=h, tau_h=th,
                contrib=contrib, h_edge=h_edge,
                rms=float(np.sqrt(np.mean(res ** 2))), res=res, edge=edge_m)


def synth(t, G, m0, s1, t1, t2, h, th, noise):
    R = 1.0 - s1 * np.exp(-t / t1) - (1.0 - s1) * np.exp(-t / t2)
    hh = h + (1.0 - h) * np.exp(-t / th)
    return G * (m0 + (1.0 - m0) * R) * hh + noise


def resample(pool, n, rng, blk=2000):
    return np.concatenate([pool[i:i + blk]
                           for i in rng.integers(0, len(pool) - blk, n // blk + 1)])[:n]


def main():
    rng = np.random.default_rng(23)
    print("=" * 80)
    print(" activation time course / steady-state activation rising-edge adjudication (I=G*m(t)h(t), h reported via contribution gate)"
          + (" (smoke)" if SMOKE else " (full nine-cell set)"))
    print(f" criteria: each voltage valid n>=4 and CV(tau1),CV(tau2)<{CV_PASS}")
    print("=" * 80, flush=True)

    # ---------- noise pool and envelopes (all cells) ----------
    pools, envs, sigs = {}, {}, {}
    steps_by_cell = {}
    for cell in CELLS:
        V, I = load_cell(cell)
        if I is None:
            print(f"  cell {cell}: file missing", flush=True)
            continue
        info = segments(V)
        pools[cell] = quiet_pool(info, I)
        envs[cell] = acf_envelope(info, I)
        sigs[cell] = float(np.std(pools[cell])) if pools[cell] is not None else np.nan
        steps_by_cell[cell] = find_steps(info)
        print(f"  cell {cell}: sigma_dec={sigs[cell]:.4f}nA  test steps "
              f"{[s['v'] for s in steps_by_cell[cell]]}", flush=True)
    ok_cells = [c for c in CELLS if c in pools and pools[c] is not None]
    if not ok_cells:
        print("  no usable silent segments, halt.")
        return
    med_cell = sorted(ok_cells, key=lambda c: sigs[c])[len(ok_cells) // 2]
    pool_med = pools[med_cell]
    sig_med = sigs[med_cell]
    print(f"  control noise cell (median): {med_cell}  sigma_dec={sig_med:.4f} nA", flush=True)

    # ---------- controls ----------
    print("\n[controls]", flush=True)
    T5 = np.arange(int(SKIP_MS), 5000, 1) * 1e-3
    n5 = len(T5)

    def ctrl(truth, nrep=20):
        outs = []
        for _ in range(nrep):
            y = synth(T5, noise=resample(pool_med, n5, rng), **truth)
            outs.append(fit_step(T5, y))
        return outs

    def h_rejected(r):
        return r["h_edge"] or r["contrib"] <= H_GATE * sig_med

    t1e, t2e = [], []
    for r in ctrl(dict(G=4.0, m0=0.05, s1=-0.6, t1=0.080, t2=0.400, h=0.16, th=0.012)):
        if r is not None and not r["edge"]:
            t1e.append(abs(r["tau1"] - 0.080) / 0.080)
            t2e.append(abs(r["tau2"] - 0.400) / 0.400)
    c1 = len(t1e) >= 16 and np.median(t1e) < 0.15 and np.median(t2e) < 0.15
    print(f"  C1 +40-like x20: valid {len(t1e)}  tau1 error median "
          f"{np.median(t1e) * 100:.1f}% tau2 {np.median(t2e) * 100:.1f}% (<15%) {'pass' if c1 else 'fail'}", flush=True)
    col = 0
    for r in ctrl(dict(G=4.0, m0=0.05, s1=0.0, t1=0.030, t2=0.100, h=0.16, th=0.012)):
        if r is not None and (min(abs(r["s1"]), abs(1.0 - r["s1"])) < 0.08 or r["edge"]):
            col += 1
    c2 = col >= 16
    print(f"  C2 single-component m+h x20: correct collapse {col}/20 (>=16) {'pass' if c2 else 'fail'}", flush=True)
    t1g, t2g, rej3 = [], [], 0
    for r in ctrl(dict(G=1.5, m0=0.05, s1=-0.5, t1=0.300, t2=1.500, h=0.44, th=0.0134)):
        if r is not None:
            if h_rejected(r):
                rej3 += 1
            if not r["edge"]:
                t1g.append(abs(r["tau1"] - 0.300) / 0.300)
                t2g.append(abs(r["tau2"] - 1.500) / 1.500)
    c3 = len(t1g) >= 16 and np.median(t1g) < 0.15 and np.median(t2g) < 0.15 and rej3 >= 16
    print(f"  C3 deep-h invisible step x20: valid {len(t1g)}  tau1 error median {np.median(t1g) * 100:.1f}% "
          f"tau2 {np.median(t2g) * 100:.1f}% (<15%) h report gate-rejected {rej3}/20 (>=16) {'pass' if c3 else 'fail'}", flush=True)
    t1f, t2f = [], []
    for r in ctrl(dict(G=12.0, m0=0.05, s1=-0.5, t1=0.040, t2=0.160, h=0.156, th=0.008)):
        if r is not None and not r["edge"]:
            t1f.append(abs(r["tau1"] - 0.040) / 0.040)
            t2f.append(abs(r["tau2"] - 0.160) / 0.160)
    c4 = len(t1f) >= 16 and np.median(t1f) < 0.20 and np.median(t2f) < 0.20
    print(f"  C4 tight-separation +60-like x20: valid {len(t1f)}  tau1 error median "
          f"{np.median(t1f) * 100:.1f}% tau2 {np.median(t2f) * 100:.1f}% (<20%) {'pass' if c4 else 'fail'}", flush=True)
    rej5, t1h = 0, []
    for r in ctrl(dict(G=4.0, m0=0.05, s1=-0.6, t1=0.080, t2=0.400, h=1.0, th=0.012)):
        if r is not None:
            if h_rejected(r):
                rej5 += 1
            if not r["edge"]:
                t1h.append(abs(r["tau1"] - 0.080) / 0.080)
    c5 = rej5 >= 16 and len(t1h) >= 16 and np.median(t1h) < 0.15
    print(f"  C5 no-h x20: h report gate-rejected {rej5}/20 (>=16) tau1 error median "
          f"{np.median(t1h) * 100:.1f}% (<15%) {'pass' if c5 else 'fail'}", flush=True)
    t1x, t2x = [], []
    for r in ctrl(dict(G=1.3, m0=0.05, s1=-0.6, t1=0.080, t2=0.400, h=0.16, th=0.012)):
        if r is not None and not r["edge"]:
            t1x.append(abs(r["tau1"] - 0.080) / 0.080)
            t2x.append(abs(r["tau2"] - 0.400) / 0.400)
    c6 = len(t1x) >= 16 and np.median(t1x) < 0.15 and np.median(t2x) < 0.15
    print(f"  C6 low-SNR step (m-ride ~4 sigma) x20: valid {len(t1x)}  tau1 error median "
          f"{np.median(t1x) * 100:.1f}% tau2 {np.median(t2x) * 100:.1f}% (<15%) {'pass' if c6 else 'fail'}", flush=True)
    if not (c1 and c2 and c3 and c4 and c5 and c6):
        print("  controls not returned to baseline -> statistic voided, halt.")
        return
    print("  controls passed.", flush=True)

    # ---------- real data ----------
    rows = []
    for cell in ok_cells:
        V, I = load_cell(cell)
        for st in steps_by_cell[cell]:
            s0, n = st["s0"], st["n"]
            base = float(np.mean(I[st["base_s"] + st["base_n"] - int(0.1 / DT):
                                   st["base_s"] + st["base_n"]]))
            idx = np.arange(int(SKIP_MS / 1000 / DT), n, DEC)
            t = (idx - idx[0]) * DT
            y = I[s0 + idx].astype(float) - base
            mask = ~np.isnan(y)
            if mask.mean() < 0.8:
                print(f"  cell {cell} @{st['v']:>4}mV: NaN fraction {1 - mask.mean():.0%}, dropped", flush=True)
                continue
            t, y = t[mask], y[mask]
            r = fit_step(t, y)
            if r is None:
                print(f"  cell {cell} @{st['v']:>4}mV: fit failed", flush=True)
                continue
            white, viol = white_gate(r["res"], envs[cell])
            snr = r["G"] / sigs[cell]
            m_ride = r["G"] * r["h_ss"] * (1.0 - r["m0"])
            h_rep = bool((not r["h_edge"]) and r["contrib"] > H_GATE * sigs[cell])
            valid = bool((not r["edge"]) and r["rms"] < 1.5 * sigs[cell]
                         and white and m_ride > 4.0 * sigs[cell])
            rows.append(dict(cell=cell, v=st["v"], valid=valid, viol=viol, rms=r["rms"],
                             snr=snr, m_ride=m_ride, G=r["G"], m0=r["m0"], s1=r["s1"], tau1=r["tau1"],
                             tau2=r["tau2"], h_report=h_rep, h_ss=(r["h_ss"] if h_rep else None),
                             tau_h=(r["tau_h"] if h_rep else None), contrib=r["contrib"],
                             t=t[::20].tolist(), y=y[::20].tolist(),
                             fit=(r["G"] * model_f(t, r["m0"], r["s1"], r["tau1"],
                                                   r["tau2"], r["h_ss"], r["tau_h"]))[::20].tolist()))
            hs = f"h_ss={r['h_ss']:.2f} tau_h={r['tau_h'] * 1000:.1f}ms" if h_rep else "h=not reported"
            print(f"  cell {cell} @{st['v']:>4}mV: tau1={r['tau1'] * 1000:.0f}ms "
                  f"τ2={r['tau2'] * 1000:.0f}ms s1={r['s1']:.2f} m0={r['m0']:.3f} "
                  f"{hs} ride={m_ride:.3f}({m_ride / sigs[cell]:.1f}σ) "
                  f"rms/sigma={r['rms'] / sigs[cell]:.2f} violations {viol} {'pass' if valid else 'drop'}", flush=True)

    # ---------- constancy ----------
    print("\n" + "-" * 80)
    print("[constancy] activation modes across cells (QC-valid only)")
    summ = {}
    for vv in V_STEPS:
        rs = [r for r in rows if r["v"] == vv and r["valid"]]
        if len(rs) < 4:
            summ[vv] = dict(n=len(rs), ok=False, note="insufficient data")
            print(f"  {vv:>4}mV: valid n={len(rs)}<4, insufficient data")
            continue
        t1 = np.array([r["tau1"] for r in rs]); t2 = np.array([r["tau2"] for r in rs])
        cv1 = float(t1.std(ddof=1) / t1.mean()); cv2 = float(t2.std(ddof=1) / t2.mean())
        ok = cv1 < CV_PASS and cv2 < CV_PASS
        summ[vv] = dict(n=len(rs), t1_med=float(np.median(t1)), cv1=cv1,
                        t2_med=float(np.median(t2)), cv2=cv2, ok=ok)
        print(f"  {vv:>4}mV: n={len(rs)}  τ1 {np.median(t1) * 1000:.0f}ms CV {cv1:.2f} | "
              f"tau2 {np.median(t2) * 1000:.0f}ms CV {cv2:.2f}  {'constant' if ok else 'not constant'}")

    # ---------- h registry ----------
    print("\n[register] steps with the h contribution gate open (reserved for the inactivation block, not judged here)")
    for r in rows:
        if r["h_report"]:
            print(f"  {r['cell']} @{r['v']:>4}mV: h_ss={r['h_ss']:.3f} "
                  f"τ_h={r['tau_h'] * 1000:.2f}ms contrib={r['contrib']:.3f}nA")

    print("\n" + "=" * 80)
    print(" overall verdict:")
    n_ok = sum(1 for s in summ.values() if s.get("ok"))
    if n_ok == len(V_STEPS):
        final = "activation mode = discrete constants (all 7 steps pass) -> second block enters the table"
    elif n_ok > 0:
        final = f"constant at some steps ({n_ok}/{len(V_STEPS)}): constant steps enter the table, the rest registered"
    else:
        final = "activation mode is not discrete constants (or insufficient data) -> open a new card"
    print("  " + final)
    print("=" * 80)

    # ---------- figure ----------
    ncell = len(ok_cells)
    fig, axes = plt.subplots(ncell, len(V_STEPS), figsize=(2.4 * len(V_STEPS), 1.8 * ncell),
                             squeeze=False)
    for i, cell in enumerate(ok_cells):
        for j, vv in enumerate(V_STEPS):
            ax = axes[i][j]
            r = next((x for x in rows if x["cell"] == cell and x["v"] == vv), None)
            if r is None:
                ax.axis("off"); continue
            ax.plot(np.array(r["t"]) * 1000, r["y"], ".", ms=1.5, color="gray")
            ax.plot(np.array(r["t"]) * 1000, r["fit"], "-", lw=1.0, color="crimson")
            ax.set_title(f"{cell[-4:]}@{vv} τ1={r['tau1'] * 1000:.0f} τ2={r['tau2'] * 1000:.0f}"
                         f"{'pass' if r['valid'] else 'drop'}", fontsize=6)
            ax.set_xscale("log")
            ax.tick_params(labelsize=5)
    fig.suptitle("steady-state activation rising edge, directly measured I=G*m(t)h(t)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fp1 = os.path.join(HERE, f"2026-09-13_α模型_激活时程{'_冒烟' if SMOKE else ''}.png")
    fig.savefig(fp1, dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for r in rows:
        if r["valid"]:
            ax2.plot(r["v"], r["tau1"] * 1000, "o", ms=5, color="navy", alpha=0.7)
            ax2.plot(r["v"], r["tau2"] * 1000, "s", ms=5, color="crimson", alpha=0.7)
    ax2.set_yscale("log"); ax2.set_xlabel("mV"); ax2.set_ylabel("τ (ms)")
    ax2.set_title("activation tau ladder: circle=tau1 square=tau2 (QC-valid only)")
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fp2 = os.path.join(HERE, f"2026-09-13_α模型_激活时程_阶梯{'_冒烟' if SMOKE else ''}.png")
    fig2.savefig(fp2, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    out = dict(summary=summ, final=final, sigma_dec=sigs, med_cell=med_cell,
               rows=[{k: v for k, v in r.items() if k not in ("t", "y", "fit")} for r in rows],
               figs=[fp1, fp2])
    fj = os.path.join(HERE, f"2026-09-13_α模型_激活时程{'_冒烟' if SMOKE else ''}_结果.json")
    json.dump(out, open(fj, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n  figure saved: {fp1}")
    print(f"  figure saved: {fp2}")
    print(f"  results saved: {fj}")


if __name__ == "__main__":
    main()
