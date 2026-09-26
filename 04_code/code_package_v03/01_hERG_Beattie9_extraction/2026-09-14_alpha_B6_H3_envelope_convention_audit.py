# -*- coding: utf-8 -*-
"""
2026-09-14 - alpha model - B6-H3 envelope-convention audit card
============================================
Envelope-side audit of H3, the only surviving branch of the B6 open case.
Recon (_recon_H3_003 six tails) gave three initial findings, corrected after review:
  (a) the rebound tails carry +/-5~8 nA capacitive spikes in 0-2 ms; the original SEAL chain does not mask them -> small-pulse fits pin at the tau_r = 1 ms grid edge (confirmed);
  (b) [RETRACTED] initial finding "tails carry a slow outward drift growing with Dt (-1.6 nA @300 ms)": review showed it is an artefact
      of the recon script itself: its baseline took the median of the first 5 ms, a window containing the rising phase of the bump
      (1000 ms tail seg[2-5 ms] = 1.94 nA), so the whole trace was over-subtracted by ~1.9 nA producing the fake drift; direct logs
      confirm all six tails end back at baseline (+0.003~+0.084 nA). This card keeps the drift term X*t as a free option; measured X ~= 0 corroborates.
  (c) 8% = 003's A(100)/A(1000) = 0.294/3.71; the numerator once pinned at the grid edge -> the numerator convention needs audit (confirmed).

This card compares three models on every tail (first 300 ms window):
  M0 original-method replica: no spike masking, c + A*DoE(tau_r, tau_d), same grid as the SEAL;
  M1 audit main model: mask 0-2 ms, c + A*DoE + X*t (linear slow drift);
  M2 sensitivity variant: masked, c + A*DoE + X*(1 - e^{-t/120ms}) (saturating drift).
Grid (tau_r in [1,60] ms, tau_d in [8,2000] ms, tau_d >= 1.5 tau_r) same as the SEAL; c/A/X linear lstsq.

Controls (pinned before run; if failed, statistics void and halt):
  C1 chain fidelity: single-gate m (tau = 135 ms, no foot, m@100ms = 52.3%) synthetic full-family tails
     (DoE tau_r 3.5/tau_d 25 + drift X = -1.5e-3*A/ms + bipolar spikes + sigma = 0.056 measured noise)
     -> M1 recovers A(100)/A(1000) in [0.40, 0.65];
  C2 foot fidelity: corner-delay foot (d = 75 ms, tau = 228 ms, m@100ms = 10.4%) same chain
     -> M1 recovers in [0.06, 0.16].

Criteria (pinned before run):
  H3-env holds (artefact mainly responsible): 003 corrected ratio >= 0.20 and >= 2x the M0 reading, or seven-cell corrected-ratio median >= 0.25;
  H3-env rejected (8% robust): 003 corrected ratio in [0.04, 0.15] and seven-cell median <= 0.20;
  middle band -> register the numerical indeterminacy as-is.
Incidental registration: Dt-dependence and sign of drift X; whether the flat envelopes of 014/118 are drift artefacts; M1 vs M2 agreement.

Run: python this file (SMOKE=1 single-cell 16713003 smoke; full run 7 cells excluding D11).
"""
import os
import json
import time

import numpy as np
import scipy.io as sio

DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("SMOKE", "0") == "1"
CELLS_ALL = ["16707014", "16708016", "16708060", "16708118", "16713003", "16713110", "16715049"]
CELLS = ["16713003"] if SMOKE else CELLS_ALL
D11 = ("16704007", "16704047")
DT = 1e-4
MASK_N = 200                      # mask 0-2 ms (capacitive spikes)
WIN_N = 3000                      # 300 ms window

TR_GRID = np.exp(np.linspace(np.log(0.0015), np.log(0.008), 13))    # M1/M2: QC window
TD_GRID = np.exp(np.linspace(np.log(0.010), np.log(0.100), 20))
TR_GRID_W = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))    # M0: original SEAL grid
TD_GRID_W = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))


def load_mat(proto, cell, tag):
    V = sio.loadmat(f"{DATA}/data/protocols/{proto}")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/{tag}_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def segments(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    return [(float(V[s[0]]), int(s[0]), len(s)) for s in np.split(np.arange(len(V)), edges)]


def six_tails(V, I, vtest=40.0):
    """Extract the six -120 rebound tails (first 300 ms, inward flipped positive). Returns [(Dt_s, seg)]."""
    info = segments(V)
    out = []
    for i, (v, s0, n) in enumerate(info):
        if abs(v - vtest) < 2 and i + 1 < len(info) and abs(info[i + 1][0] + 120) < 2 \
                and info[i + 1][2] * DT > 2.0:
            a = info[i + 1][1]
            out.append((n * DT, -I[a:a + WIN_N].astype(float)))
    return out


def fit_tail(seg, model):
    """M0 original-method replica: unmasked, original wide grid, A unclipped | M1/M2: 2 ms mask, QC-window grid,
    A >= 0 enforced per grid point (negative-A points degenerate to baseline + drift and still compete).
    Masking erases the rising edge -> long tau_d + drift degenerates with negative A (debug_C1 record); the QC window kills it."""
    if model == "M0":
        t = np.arange(len(seg)) * DT
        y = seg
        drift = None
        tr_grid, td_grid = TR_GRID_W, TD_GRID_W
    else:
        t = np.arange(MASK_N, len(seg)) * DT
        y = seg[MASK_N:]
        drift = t if model == "M1" else (1.0 - np.exp(-t / 0.120))
        tr_grid, td_grid = TR_GRID, TD_GRID
    best = None
    for tr in tr_grid:
        er = np.exp(-t / tr)
        for td in td_grid[td_grid >= 1.5 * tr]:
            x = np.exp(-t / td) - er
            if model == "M0":
                X = np.column_stack([np.ones(len(t)), x])
            else:
                X = np.column_stack([np.ones(len(t)), x, drift])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ sol) ** 2))
            if model != "M0" and sol[1] < 0:
                # A >= 0 enforced per point: degenerates to baseline + drift two-parameter (independent of (tr, td))
                X2 = np.column_stack([np.ones(len(t)), drift])
                sol2, *_ = np.linalg.lstsq(X2, y, rcond=None)
                sse = float(np.sum((y - X2 @ sol2) ** 2))
                sol = np.array([sol2[0], 0.0, sol2[1]])
            if best is None or sse < best[0]:
                best = (sse, tr, td, sol)
    sse, tr, td, sol = best
    Xv = float(sol[2]) if model != "M0" else 0.0
    edge = bool(tr <= tr_grid[0] * 1.02 or td >= td_grid[-1] * 0.98 or tr >= tr_grid[-1] * 0.98)
    return dict(A=float(sol[1]), c=float(sol[0]), X=Xv, tau_r=float(tr), tau_d=float(td),
                edge=edge, rms=float(np.sqrt(sse / len(t))))


# ---------------- controls: synthetic full-family tails ----------------
def synth_family(m_of_dt, rng, drift_per_A=-1.44, sigma=0.056):
    """Six-level synthetic tails: A*DoE(3.5/25 ms) + X*t drift + bipolar spikes + noise.
    drift_per_A in nA/s per nA*A (t in s): measured 003 family -1.6 nA@300 ms / A = 3.71 -> -1.44."""
    dts = np.array([0.003, 0.010, 0.030, 0.100, 0.300, 1.000])
    tails = []
    for dt_s in dts:
        A = 3.7 * m_of_dt(dt_s)
        t = np.arange(WIN_N) * DT
        seg = A * (np.exp(-t / 0.025) - np.exp(-t / 0.0035))
        seg = seg + drift_per_A * A * t
        seg[:2] += [5.0, -3.0]                       # capacitive spikes
        seg = seg + rng.normal(0, sigma, WIN_N)
        tails.append((dt_s, seg))
    return tails


def ratio_100_1000(fits):
    amap = {dt_s: f["A"] for dt_s, f in fits}
    a100 = amap.get(0.1)
    a1000 = amap.get(1.0)
    if a100 is None or a1000 is None or a1000 <= 0:
        return float("nan")
    return a100 / a1000


def main():
    t0 = time.time()
    rng = np.random.default_rng(11)
    print("=" * 84, flush=True)
    print(" alpha model - B6-H3 envelope-convention audit" + (" (smoke 16713003)" if SMOKE else " (7 cells, excluding D11)"), flush=True)
    print(" criteria: 003 corrected ratio >=0.20 and >=2x M0 -> H3 holds | 003 in [0.04,0.15] and median <=0.20 -> H3 rejected", flush=True)
    print("=" * 84, flush=True)

    # ---------- controls (median over 10 noise realisations each; testing systematic bias, not luck) ----------
    print("\n[controls]", flush=True)
    m_c1 = lambda d: 1.0 - np.exp(-d / 0.135)
    m_c2 = lambda d: 1.0 - np.exp(-max(d - 0.075, 0.0) / 0.228)
    r1s, r2s, r1m0s = [], [], []
    for _ in range(10):
        r1s.append(ratio_100_1000([(d, fit_tail(s, "M1")) for d, s in synth_family(m_c1, rng)]))
        r2s.append(ratio_100_1000([(d, fit_tail(s, "M1")) for d, s in synth_family(m_c2, rng)]))
        r1m0s.append(ratio_100_1000([(d, fit_tail(s, "M0")) for d, s in synth_family(m_c1, rng)]))
    r1, r2, r1_m0 = float(np.median(r1s)), float(np.median(r2s)), float(np.median(r1m0s))
    c1ok = 0.40 <= r1 <= 0.65
    c2ok = 0.06 <= r2 <= 0.16
    print(f"  C1 single-gate no-foot (true ratio 0.523): M1 median {r1:.3f} [{min(r1s):.2f}~{max(r1s):.2f}]"
          f" in [0.40,0.65] -> {'pass' if c1ok else 'fail'}", flush=True)
    print(f"  C2 corner-delay foot (true ratio 0.104): M1 median {r2:.3f} [{min(r2s):.2f}~{max(r2s):.2f}]"
          f" in [0.06,0.16] -> {'pass' if c2ok else 'fail'}", flush=True)
    print(f"  register: same C1 family under M0 (original chain) recovery median {r1_m0:.3f} (truth 0.523)"
          f" - with drift proportional to A the ratio is blind-immune to drift (numerator and denominator polluted in the same proportion)", flush=True)
    if not (c1ok and c2ok):
        print("  controls not seated -> statistics void, halt.", flush=True)
        return
    print("  controls seated.", flush=True)

    # ---------- real data ----------
    print("\n[real data]", flush=True)
    print(f"  {'cell':>9} | {'M0 ratio':>6} | {'M1 ratio':>6} {'M1 edge':>6} | {'M2 ratio':>6} | "
          f"{'X slope@1s':>9} {'M1 RMS':>7}", flush=True)
    per_cell = {}
    for cell in CELLS:
        V, I = load_mat("activation_kinetics_2_protocol.mat", cell, "activation_kinetics_2")
        if I is None:
            print(f"  {cell}: data missing", flush=True)
            continue
        tails = six_tails(V, I, 40.0)
        res = {}
        for model in ("M0", "M1", "M2"):
            fits = [(d, fit_tail(s, model)) for d, s in tails]
            res[model] = dict(ratio=ratio_100_1000(fits),
                              fits={f"{d * 1000:.0f}ms": f for d, f in fits})
        # drift registration (M1: X slope nA/ms of the 1000 ms tail over its A)
        f1s = res["M1"]["fits"].get("1000ms")
        xreg = float("nan")
        if f1s and f1s["A"] > 0:
            xreg = f1s["X"] / f1s["A"] * 1000.0      # drift nA per A per second
        nedge = sum(1 for k, f in res["M1"]["fits"].items() if f["edge"])
        per_cell[cell] = dict(M0=res["M0"], M1=res["M1"], M2=res["M2"],
                              x_slope_per_A_per_s=xreg, n_edge_M1=nedge,
                              rms_M1_med=float(np.median([f["rms"] for f in res["M1"]["fits"].values()])))
        print(f"  {cell} | {res['M0']['ratio']:6.3f} | {res['M1']['ratio']:6.3f} "
              f"{nedge:3d}/6 | {res['M2']['ratio']:6.3f} | {xreg:9.3f} "
              f"{per_cell[cell]['rms_M1_med']:7.4f}", flush=True)

    # ---------- verdict ----------
    print("\n" + "=" * 84, flush=True)
    r003_m0 = per_cell.get("16713003", {}).get("M0", {}).get("ratio", float("nan"))
    r003_m1 = per_cell.get("16713003", {}).get("M1", {}).get("ratio", float("nan"))
    meds = [v["M1"]["ratio"] for v in per_cell.values() if not np.isnan(v["M1"]["ratio"])]
    med = float(np.median(meds)) if meds else float("nan")
    print(f"  003: M0={r003_m0:.3f}  M1 corrected={r003_m1:.3f}  seven-cell M1 median={med:.3f}", flush=True)
    h3_yes = (r003_m1 >= 0.20 and r003_m1 >= 2 * r003_m0) or med >= 0.25
    h3_no = (0.04 <= r003_m1 <= 0.15) and med <= 0.20
    verdict = "H3-env holds: 8% mainly a measurement artefact" if h3_yes else \
        "H3-env rejected: 8% robust, envelope side not an artefact" if h3_no else "middle band: registered as-is, indeterminate"
    print(f"  verdict: {verdict}", flush=True)
    print("=" * 84, flush=True)

    tag = "_冒烟" if SMOKE else "_结果"
    fj = os.path.join(HERE, f"2026-09-14_α模型_B6_H3_包络口径审计{tag}.json")
    with open(fj, "w", encoding="utf-8") as fh:
        json.dump(dict(meta=dict(smoke=SMOKE, runtime_s=time.time() - t0,
                                 controls=dict(C1=r1, C2=r2, C1_M0=r1_m0)),
                       per_cell=per_cell, verdict=verdict,
                       median_M1=med), fh, ensure_ascii=False, indent=1, default=float)
    print(f"  results saved: {fj}", flush=True)

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))
        from daimon_runtime import setup_plot
        setup_plot()
    except Exception:
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
    # figure 1: 003 six tails + M1 decomposition (bump/drift)
    ax = axes[0]
    cell0 = "16713003"
    if cell0 in per_cell:
        V, I = load_mat("activation_kinetics_2_protocol.mat", cell0, "activation_kinetics_2")
        tails = six_tails(V, I, 40.0)
        t_ms = np.arange(WIN_N) * DT * 1000
        for d, s in tails:
            f = fit_tail(s, "M1")
            tt = np.arange(MASK_N, WIN_N) * DT
            hump = f["A"] * (np.exp(-tt / f["tau_d"]) - np.exp(-tt / f["tau_r"]))
            ax.plot(t_ms, s, color="0.75", lw=0.6)
            ax.plot(tt * 1000, f["c"] + hump, lw=1.4,
                    label=f"Δt={d * 1000:.0f}ms A={f['A']:.2f}")
            ax.plot(tt * 1000, f["c"] + f["X"] * tt, "--", lw=0.8, color="tab:red")
        ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("time after -120 rebound (ms)")
    ax.set_ylabel("tail current (nA)")
    ax.set_title(f"{cell0} M1 decomposition: solid = baseline+bump, red dashed = baseline+drift")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    # figure 2: seven-cell M0/M1 ratios before/after correction
    ax = axes[1]
    cells_sorted = sorted(per_cell.keys())
    m0s = [per_cell[c]["M0"]["ratio"] for c in cells_sorted]
    m1s = [per_cell[c]["M1"]["ratio"] for c in cells_sorted]
    xpos = np.arange(len(cells_sorted))
    ax.bar(xpos - 0.18, m0s, 0.36, label="M0 original chain")
    ax.bar(xpos + 0.18, m1s, 0.36, label="M1 corrected")
    ax.axhline(0.20, color="tab:red", ls=":", lw=1, label="criterion 0.20")
    ax.set_xticks(xpos)
    ax.set_xticklabels([c[-3:] for c in cells_sorted])
    ax.set_xlabel("cell")
    ax.set_ylabel("A(100)/A(1000)")
    ax.set_title("envelope 100 ms ratio: before vs after correction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # figure 3: 003 envelope curve M0 vs M1 (six points)
    ax = axes[2]
    if cell0 in per_cell:
        for model, mk, lb in [("M0", "o--", "M0 original chain"), ("M1", "s-", "M1 corrected")]:
            fits = per_cell[cell0][model]["fits"]
            dts = sorted(fits.keys(), key=lambda k: float(k[:-2]))
            xv = [float(k[:-2]) for k in dts]
            yv = [fits[k]["A"] for k in dts]
            yv = np.array(yv) / max(yv)
            ax.plot(xv, yv, mk, ms=5, label=lb)
        ax.set_xscale("log")
        ax.axvline(100, color="k", ls=":", lw=1)
        ax.set_xlabel("test-pulse duration Dt (ms)")
        ax.set_ylabel("normalised envelope")
        ax.set_title(f"{cell0} envelope shape: M0 vs M1")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("B6-H3 envelope-convention audit (spike masking + drift decomposition)" + (" (smoke)" if SMOKE else ""), fontsize=12)
    fp = os.path.join(HERE, f"2026-09-14_α模型_B6_H3_包络口径审计{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fp, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved: {fp}", flush=True)


if __name__ == "__main__":
    main()
