# -*- coding: utf-8 -*-
# 2026-09-13_alpha model_activation kinetics_envelope constancy adjudication.py
# Purpose: extend the alpha model's second block - the modal structure of activation
#   kinetics and its cross-cell constancy.
#   Method (envelope method + alpha methodology: direct extraction, cross-cell CV):
#     activation_kinetics_1 (0 mV) / _2 (+40 mV): -120x50ms full-recovery pre-pulse
#     -> test voltage x variable duration T -> -120x2.5s rebound tail; rebound amplitude A(T) ~ m(T).
#     v3 (smoke finding: the envelope is not mono-exponential - initial lag plus multi-scale
#       rise across 30ms~1s): switch to a two-component rise A(T)=A_max - a1*e^(-T/tau1) - a2*e^(-T/tau2)
#       (tau2>=3*tau1, deterministic 2D grid + linear amplitudes), extracting the activation modal
#   +40 step: A_max pinned to the full-activation anchor (deactivation +50->-120 hook amplitude, self-extracted in-script);
#   0mV step: anchor unusable (m_ss(0)<m_ss(+50)); A_max free; if it hits the top edge, discard and register.
#   Excluded on record: readme states 16708016/16708060/16704007 activation_kinetics over-leak (6/9 entered the adjudication).
#   Verdict (pre-registered, fixed before run):
#     each voltage each tau: valid n>=4 and CV<0.3 -> constant; both taus constant => "activation mode = discrete constants";
#     only one => graded entry into table; neither => open a new card. tau_act(0) vs (+40) comparison is register-only.
#   QC: A_max>5*sigma and RMS<max(2*sigma, 3% of A_max) and tau not at grid edge and points>=5 and span >=1 decade.
#   Controls (noise self-calibrated, criteria unchanged): per-cell silent-residual block-bootstrap MC measures sigma_A (low-SNR end
#         A=0.6nA); control operating point = all-cell median sigma_A (old version hung on the first listed cell = arbitrary;
#         even earlier fixed sigma=0.04 was proven via CR to make the 15% criterion information-theoretically unreachable).
#         Noise-outlier cells (>2x median) printed on record, registered separately at verdict time.
#         C1 two-component (tau1=30ms, tau2=400ms equal weight) x20: both tau inversion error medians <15%;
#         C2 single-component (tau=100ms) must not hallucinate a second component (small |a|<8%A_max or at grid edge).
# Run: python this file (full six-cell set); SMOKE=1 for the 16713003 single-cell smoke test.
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
EXCLUDE = {"16708016", "16708060", "16704007"}          # readme over-leak on record
CELLS = ["16713003"] if SMOKE else [c for c in
        ["16704007", "16704047", "16707014", "16708016", "16708060",
         "16708118", "16713003", "16713110", "16715049"] if c not in EXCLUDE]
DT = 1e-4
SKIP_S = 0.0005
TAIL_WIN_S = 0.15
CV_PASS = 0.3
AMP_QC = 5.0
PROTOS = [("activation_kinetics_1_protocol.mat", "activation_kinetics_1", 0),
          ("activation_kinetics_2_protocol.mat", "activation_kinetics_2", 40)]

TR_GRID = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))
TD_GRID = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))
TAU_GRID = np.exp(np.linspace(np.log(0.003), np.log(3.000), 24))   # 3ms–3s


def load_mat(proto, cell, tag):
    V = sio.loadmat(f"{DATA}/data/protocols/{proto}")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/{tag}_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def quiet_sigma(V, I):
    edges = np.where(np.diff(V) != 0)[0] + 1
    sigs = []
    for s in np.split(np.arange(len(V)), edges):
        if len(s) * DT >= 0.15 and abs(float(V[s[0]]) + 80.0) < 2.0:
            seg = I[s[0]:s[-1] + 1].astype(float)
            t = np.arange(len(seg))
            tr = np.polyfit(t, seg, 1)
            sigs.append(float(np.std(seg - np.polyval(tr, t))))
    return float(np.median(sigs)) if sigs else np.nan


def doe_fit(t, y):
    if len(t) < 50:
        return None

    def scan(trg, tdg):
        best = None
        for tr in trg:
            td_ok = tdg[tdg >= 1.5 * tr]
            if not len(td_ok):
                continue
            er = np.exp(-t / tr)
            for td in td_ok:
                x = np.exp(-t / td) - er
                X = np.column_stack([np.ones(len(t)), x])
                sol, *_ = np.linalg.lstsq(X, y, rcond=None)
                sse = float(np.sum((y - X @ sol) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, tr, td, sol)
        return best

    b = scan(TR_GRID, TD_GRID)
    if b is None:
        return None
    _, tr0, td0, _ = b
    b = scan(tr0 * np.exp(np.linspace(-0.35, 0.35, 9)),
             td0 * np.exp(np.linspace(-0.35, 0.35, 9)))
    sse, tr, td, sol = b
    res = y - (sol[0] + sol[1] * (np.exp(-t / td) - np.exp(-t / tr)))
    return dict(tau_r=float(tr), tau_d=float(td), c=float(sol[0]), A=float(sol[1]),
                rms=float(np.sqrt(np.mean(res ** 2))),
                edge=bool(tr <= TR_GRID[0] * 1.02 or td >= TD_GRID[-1] * 0.98))


def quiet_residuals(V, I):
    """Silent-segment residual pool (-80/-120 steady-state segments, first 0.1s dropped, linearly detrended) - for noise calibration"""
    edges = np.where(np.diff(V) != 0)[0] + 1
    pool = []
    for s in np.split(np.arange(len(V)), edges):
        v = float(V[s[0]])
        if len(s) * DT >= 0.3 and (abs(v + 80.0) < 2.0 or abs(v + 120.0) < 2.0):
            seg = I[s[0] + int(0.1 / DT):s[-1] + 1].astype(float)
            if len(seg) < 2000:
                continue
            t = np.arange(len(seg))
            tr = np.polyfit(t, seg, 1)
            pool.append(seg - np.polyval(tr, t))
    return np.concatenate(pool) if pool else None


def calibrate_sigma_A(V, I, rng, nrep=40):
    """Self-calibration of envelope-point noise sigma_A: block-bootstrap reshuffling of real silent
    residuals (ACF preserved) measures the doe_fit amplitude-extraction noise at the low-SNR end
    (A=0.6 nA). The control noise uses this measured value, not a hand-waved constant."""
    res = quiet_residuals(V, I)
    if res is None or len(res) < 20000:
        return None
    n = int(TAIL_WIN_S / DT); t = np.arange(n) * DT
    A0, blk = 0.6, 2000
    errs = []
    for _ in range(nrep):
        noise = np.concatenate([res[i:i + blk]
                                for i in rng.integers(0, len(res) - blk, n // blk + 1)])[:n]
        r = doe_fit(t, A0 * (np.exp(-t / 0.030) - np.exp(-t / 0.003)) + noise)
        if r is not None:
            errs.append(r["A"] - A0)
    return float(np.std(errs)) if len(errs) >= nrep // 2 else None


def anchor_amp(cell):
    """Full-activation anchor: DoE amplitude of the first deactivation +50->-120 tail (inward flipped positive)"""
    V, I = load_mat("deactivation_protocol.mat", cell, "deactivation")
    if I is None:
        return None
    edges = np.where(np.diff(V) != 0)[0] + 1
    segs = np.split(np.arange(len(V)), edges)
    info = [(float(V[s[0]]), int(s[0]), len(s)) for s in segs]
    for k, (v, s0, n) in enumerate(info):
        if v < -100.0 and n * DT > 2.0 and k >= 1 and info[k - 1][0] > 30.0:
            y = I[s0 + int(SKIP_S / DT): s0 + int(0.3 / DT)].astype(float)
            t = np.arange(len(y)) * DT
            r = doe_fit(t, -y)
            if r is not None and not r["edge"] and r["A"] > 0:
                return r["A"]
    return None


def parse_pulses(V, vtest):
    edges = np.where(np.diff(V) != 0)[0] + 1
    segs = np.split(np.arange(len(V)), edges)
    info = [(float(V[s[0]]), int(s[0]), len(s)) for s in segs]
    out = []
    for k, (v, s0, n) in enumerate(info):
        if abs(v - vtest) < 2.0 and n * DT < 5.0 and k + 1 < len(info):
            nv, ns, nn = info[k + 1]
            if nv < -100.0 and nn * DT > 1.0:
                out.append((n * DT, ns))
    return out


def rebound_amp(I, tail_start, sigma):
    s0 = tail_start + int(SKIP_S / DT)
    n = int(TAIL_WIN_S / DT)
    y = I[s0:s0 + n].astype(float)
    t = np.arange(len(y)) * DT
    yy = -y
    r = doe_fit(t, yy)
    if r is not None and abs(r["A"]) > AMP_QC * sigma and not r["edge"]:
        return r["A"], r
    k = max(3, int(0.003 / DT) | 1)
    ys = np.convolve(yy, np.ones(k) / k, mode="same")
    a = float(ys.max())
    return (a if a > 3 * sigma else 0.0), r


def env2_fit(T, A, anchor=None):
    """A(T)=A_max - a1 e^{-T/tau1} - a2 e^{-T/tau2}, with the physical constraint a1+a2=A_max (m(0)=0:
    after the -120x50ms full-recovery pre-pulse all channels are closed, so the envelope must pass
    through the origin) => A=A_max(1-e2)-a1(e1-e2). This constraint kills the A_max<->(a2,tau2)
    degeneracy (tau2 does not saturate within the longest 1s pulse; A_max is determined mainly by
    the origin constraint). tau2>=3*tau1; with anchor, A_max is pinned (only a1 + two taus free).
    Variable projection: tau optimized continuously (grid multi-start + least_squares), amplitudes by linear lstsq."""
    T = np.asarray(T, float); A = np.asarray(A, float)
    lo, hi = float(np.log(TAU_GRID[0])), float(np.log(TAU_GRID[-1]))
    LG3 = float(np.log(3.0))

    def lin(t1, t2):
        e1 = np.exp(-T / t1); e2 = np.exp(-T / t2)
        d = e1 - e2                      # basis for a1: A = A_max(1-e2) - a1*d
        b = 1.0 - e2                     # basis for A_max
        if anchor is None:
            X = np.column_stack([b, -d]); Y = A
        else:
            X = (-d).reshape(-1, 1); Y = A - anchor * b
        sol, *_ = np.linalg.lstsq(X, Y, rcond=None)
        pred = (X @ sol) if anchor is None else (anchor * b + (X @ sol))
        return sol, A - pred

    scale = max(float(np.ptp(A)), 1e-6)

    def resid(p):
        u1, u2 = p
        pen = max(0.0, u1 + LG3 - u2)              # soft constraint tau2>=3*tau1
        _, r = lin(np.exp(u1), np.exp(u2))
        return np.concatenate([r, [20.0 * scale * pen]])

    cand = []
    for t1g in TAU_GRID:
        for t2g in TAU_GRID[TAU_GRID >= 3.0 * t1g]:
            _, r = lin(t1g, t2g)
            cand.append((float(r @ r), float(t1g), float(t2g)))
    if not cand:
        return None
    cand.sort()
    best = None
    for _, t10, t20 in cand[:5]:
        try:
            f = least_squares(resid, [np.log(t10), np.log(t20)],
                              bounds=([lo, lo], [hi, hi]),
                              xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
        except Exception:
            continue
        u1, u2 = float(f.x[0]), float(f.x[1])
        if u2 < u1 + LG3 - 1e-6:
            continue
        sol, r = lin(np.exp(u1), np.exp(u2))
        sse = float(r @ r)
        if best is None or sse < best[0]:
            best = (sse, float(np.exp(u1)), float(np.exp(u2)), sol, r)
    if best is None:
        # continuous optimization slid entirely into the penalty zone -> fall back to the best grid point satisfying tau2>=3*tau1 (guaranteed non-None)
        _, tg1, tg2 = cand[0]
        sol, r = lin(tg1, tg2)
        best = (float(r @ r), tg1, tg2, sol, r)
    sse, t1, t2, sol, res = best
    if anchor is None:
        a_max, a1 = float(sol[0]), float(sol[1])
    else:
        a_max, a1 = float(anchor), float(sol[0])
    a2 = a_max - a1
    return dict(tau1=float(t1), tau2=float(t2), A_max=a_max, a1=a1, a2=a2,
                rms=float(np.sqrt(np.mean(res ** 2))),
                edge=bool(t1 <= TAU_GRID[0] * 1.05 or t2 >= TAU_GRID[-1] * 0.95))


def main():
    rng = np.random.default_rng(11)
    print("=" * 78)
    print(" activation kinetics envelope constancy adjudication v3 (two-component rise)" + (" (smoke)" if SMOKE else " (full six-cell set)"))
    print(f" criteria: each voltage each tau n>=4 and CV<{CV_PASS} -> constant")
    print("=" * 78, flush=True)

    # ---------- noise self-calibration (all cells; control takes the median operating point, criteria unchanged) ----------
    print("\n[noise self-calibration]", flush=True)
    sig_map = {}
    for cell in CELLS:
        Vc, Ic = load_mat(PROTOS[0][0], cell, PROTOS[0][1])
        if Ic is None:
            print(f"  cell {cell}: file missing, skipping calibration", flush=True)
            continue
        sA = calibrate_sigma_A(Vc, Ic, rng)
        if sA is not None:
            sig_map[cell] = sA
            print(f"  cell {cell}: sigma_A = {sA:.4f} nA", flush=True)
    if not sig_map:
        print("  all calibrations failed -> controls have no noise anchor, halt.")
        return
    sigA = float(np.median(list(sig_map.values())))
    noisy = [c for c, s in sig_map.items() if s > 2.0 * sigA]
    print(f"  control operating point sigma_A(median) = {sigA:.4f} nA (n={len(sig_map)})"
          + (f"; noise outliers (>2x median) registered: {', '.join(noisy)}" if noisy else ""), flush=True)

    # ---------- controls ----------
    print("\n[controls]", flush=True)
    T_c = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    A0c = 4.0
    e1s, e2s = [], []
    for rep in range(20):
        A_c = A0c - (A0c / 2) * np.exp(-T_c / 0.030) - (A0c / 2) * np.exp(-T_c / 0.400) \
            + rng.normal(0, sigA, len(T_c))
        r1 = env2_fit(T_c, A_c)
        got = sorted([r1["tau1"], r1["tau2"]])
        e1s.append(abs(got[0] - 0.030) / 0.030)
        e2s.append(abs(got[1] - 0.400) / 0.400)
    print(f"  C1 two-component (30ms,400ms) x20: tau1 error median {np.median(e1s)*100:.1f}%  "
          f"tau2 {np.median(e2s)*100:.1f}% (criterion <15% each)", flush=True)
    collapse_ok = 0
    for rep in range(20):
        A_c2 = A0c - A0c * np.exp(-T_c / 0.100) + rng.normal(0, sigA, len(T_c))
        r2 = env2_fit(T_c, A_c2)
        small = min(abs(r2["a1"]), abs(r2["a2"]))
        if small < 0.08 * r2["A_max"] or r2["edge"]:
            collapse_ok += 1
    print(f"  C2 single-component (100ms) x20: correct collapse {collapse_ok}/20 (criterion >=16)", flush=True)
    if np.median(e1s) > 0.15 or np.median(e2s) > 0.15 or collapse_ok < 16:
        print("  controls not returned to baseline -> statistic voided, halt.")
        return
    print("  controls passed.", flush=True)

    # ---------- real data ----------
    rows = []
    tr_register = []
    for cell in CELLS:
        anch = anchor_amp(cell)
        print(f"  cell {cell}: full-activation anchor A_anchor={anch:.2f}nA" if anch else
              f"  cell {cell}: anchor extraction failed", flush=True)
        for proto, tag, vtest in PROTOS:
            V, I = load_mat(proto, cell, tag)
            if I is None:
                print(f"  cell {cell} {tag}: file missing")
                continue
            sigma = quiet_sigma(V, I)
            T, A, trs = [], [], []
            for dur, ts in parse_pulses(V, vtest):
                a, r = rebound_amp(I, ts, sigma)
                T.append(dur); A.append(a)
                if r is not None and not r["edge"]:
                    trs.append(r["tau_r"])
            order = np.argsort(T)
            T = [T[i] for i in order]; A = [A[i] for i in order]
            if len(T) < 5:
                print(f"  cell {cell} @{vtest}mV: insufficient pulse points ({len(T)})", flush=True)
                continue
            use_anchor = anch if vtest == 40 else None
            r = env2_fit(np.array(T), np.array(A), anchor=use_anchor)
            mode = "anchored" if use_anchor else "free"
            span = max(T) / max(min(T), 1e-9)
            valid = bool(abs(r["A_max"]) > AMP_QC * sigma
                         and r["rms"] < max(2 * sigma, 0.03 * abs(r["A_max"]))
                         and not r["edge"] and span >= 10)
            rows.append(dict(cell=cell, v=vtest, T=T, A=A, sigma=sigma, valid=valid,
                             mode=mode, **r))
            if trs:
                tr_register.append(dict(cell=cell, v=vtest, tau_r_med=float(np.median(trs))))
            print(f"  cell {cell} @{vtest}mV [{mode}]: tau1={r['tau1']*1000:.1f}ms "
                  f"τ2={r['tau2']*1000:.0f}ms A_max={r['A_max']:.2f} "
                  f"a1={r['a1']:.2f} a2={r['a2']:.2f} {'pass' if valid else 'drop'}", flush=True)

    # ---------- constancy ----------
    print("\n" + "-" * 78)
    print("[constancy] activation modes across cells (QC-valid only)")
    summ = {}
    verdicts = []
    for vv in (0, 40):
        rs = [r for r in rows if r["v"] == vv and r["valid"]]
        if len(rs) < 4:
            print(f"  {vv:>4}mV: valid n={len(rs)}<4, insufficient data")
            continue
        t1 = np.array([r["tau1"] for r in rs]); t2 = np.array([r["tau2"] for r in rs])
        cv1 = float(t1.std(ddof=1) / t1.mean()); cv2 = float(t2.std(ddof=1) / t2.mean())
        ok = cv1 < CV_PASS and cv2 < CV_PASS
        verdicts.append(ok)
        summ[vv] = dict(n=len(rs), t1_med=float(np.median(t1)), cv1=cv1,
                        t2_med=float(np.median(t2)), cv2=cv2, ok=ok)
        print(f"  {vv:>4}mV: n={len(rs)}  τ1 {np.median(t1)*1000:.1f}ms CV {cv1:.2f} | "
              f"tau2 {np.median(t2)*1000:.0f}ms CV {cv2:.2f}  {'constant' if ok else 'not constant'}")
    if tr_register:
        trv = np.array([r["tau_r_med"] for r in tr_register])
        print(f"[register] tail DoE tau_r(-120) median {np.median(trv)*1000:.2f}ms (hook value 3.04ms, cross-check)")

    # ---------- overall verdict ----------
    print("\n" + "=" * 78)
    print(" overall verdict:")
    print(f"  constant voltage steps {sum(verdicts)}/{len(summ)}")
    if len(summ) == 2 and all(verdicts):
        final = "activation mode = discrete constants (both taus pass at 0/+40) -> second block enters the table, proceed to the inactivation block"
    elif any(verdicts):
        final = "constant at some steps: constant steps enter the table, the rest registered (piecewise extension of the model is feasible)"
    else:
        final = "activation mode is not discrete constants (or insufficient data) -> second-block premise fails, open a new card"
    print("  " + final)
    print("=" * 78)

    # ---------- figure ----------
    ncell = len(CELLS)
    fig, axes = plt.subplots(ncell, 2, figsize=(11, 2.0 * ncell), squeeze=False)
    for i, cell in enumerate(CELLS):
        for j, vv in enumerate((0, 40)):
            ax = axes[i][j]
            r = next((x for x in rows if x["cell"] == cell and x["v"] == vv), None)
            if r is None:
                ax.axis("off"); continue
            T = np.array(r["T"]); A = np.array(r["A"])
            ax.plot(T * 1000, A, "o", ms=4, color="black", alpha=0.7)
            tt = np.exp(np.linspace(np.log(min(T) * 0.8), np.log(max(T) * 1.2), 200))
            ax.plot(tt * 1000, r["A_max"] - r["a1"] * np.exp(-tt / r["tau1"])
                    - r["a2"] * np.exp(-tt / r["tau2"]), "-", lw=1.3, color="crimson")
            if r["mode"] == "anchored":
                ax.axhline(r["A_max"], lw=0.8, ls="--", color="navy", alpha=0.6)
            ax.set_xscale("log")
            ax.set_title(f"{cell[-4:]} @{vv}mV [{r['mode']}] τ1={r['tau1']*1000:.0f} τ2={r['tau2']*1000:.0f}ms "
                         f"{'pass' if r['valid'] else 'drop'}", fontsize=8)
            ax.set_xlabel("pulse duration ms"); ax.grid(alpha=0.3, which="both")
    fig.suptitle("activation envelope two-component fit A(T)=A_max-a1e^{-T/tau1}-a2e^{-T/tau2}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fp = os.path.join(HERE, f"2026-09-13_α模型_激活包络{'_冒烟' if SMOKE else ''}.png")
    fig.savefig(fp, dpi=120, bbox_inches="tight")
    plt.close(fig)

    out = dict(summary=summ, final=final, sigma_A=sig_map, sigma_A_med=sigA,
               noisy_cells=noisy,
               rows=[{k: v for k, v in r.items()} for r in rows],
               tr_register=tr_register, fig=fp)
    fj = os.path.join(HERE, f"2026-09-13_α模型_激活包络{'_冒烟' if SMOKE else ''}_结果.json")
    json.dump(out, open(fj, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n  figure saved: {fp}")
    print(f"  results saved: {fj}")


if __name__ == "__main__":
    main()
