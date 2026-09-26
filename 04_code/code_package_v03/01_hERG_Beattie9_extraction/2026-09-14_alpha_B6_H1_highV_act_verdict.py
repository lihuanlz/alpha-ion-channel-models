# -*- coding: utf-8 -*-
"""
2026-09-14 - alpha model - B6-H1 high-voltage activation verdict
============================================
B6 open case: the +40 envelope says activation is slow (m@100ms ~= 8% m_inf, tau_act 131-290 ms, delay 50-150 ms),
the AP short-peak phase requires m ~= 29% m_inf, a 3x contradiction. H1 reconciliation candidate: activation at
+50/+60/+70 is much faster than at +40 (envelope has only 0/+40 levels; no rebound data above +40, the envelope route cannot judge).

This card switches route: directly fit the slow-rising current at the onset of long depolarising steps. At V >= +40
h is pinned to h_ss within a few ms (tau_h ~= 1.5 ms); after the first 10 ms I(t) ~= G*h_ss*(V-E)*m(t), a direct portrait of m(t).
  data source: steady_activation protocol +40/+60 levels (5 s step, once per cell);
          deactivation protocol +50 level (2 s step x 9 repeats, averaged per cell).
  model: I(t) = a*(1 - exp(-max(0, t-d)/tau)) + b0 + b1*t; first 10 ms discarded (edge spike + h transient).

Criteria (pinned before run):
  H1 supported: population medians of tau(+50) and tau(+60) both <= tau(+40) median x 1/3, and per-cell direction concordance >= 7/9;
  H1 rejected: either level ratio > 1/2, or concordance < 6/9;
  middle ground: register "weak evidence", model untouched.
  controls (if failed, all statistics of this card void; register "insufficient data" as-is):
  C1 synthetic tau = 150 ms + measured noise -> recovered tau relative error < 30%;
  C2 synthetic tau = 50 ms vs 150 ms distinguishable under this noise floor (two-sample t test p < 0.05).

Discipline: this side only ast.parse + SMOKE=1 (16713003); the formal run (nine cells) is done by the user in Spyder:
  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-14_α模型_B6_H1_高压激活判决.py' --wdir
Output: _结果.json/.png next to this script (smoke carries the _冒烟 suffix).
"""

import os
import json
import time

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
DT = 1e-4
SMOKE = os.environ.get("SMOKE", "0") == "1"

CELLS = ["16713003"] if SMOKE else ["16704007", "16704047", "16707014", "16708016",
                                    "16708060", "16708118", "16713003", "16713110", "16715049"]
D11_DEAD = {"16704007", "16704047"}   # activation-protocol recording failure (D11 on record): results registered per cell, not excluded

SKIP_S = 0.010          # first 10 ms discarded (edge spike + h transient)
WIN_S = 0.500           # fit window 500 ms
T_GRID = np.arange(5.0, 401.0, 5.0) * 1e-3      # τ: 5–400ms
D_GRID = np.arange(0.0, 201.0, 5.0) * 1e-3      # delay d: 0-200 ms


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


def fit_onset(t, y):
    """Grid (tau, d), linear solve (a, b0, b1). Returns dict or None."""
    best = None
    for tau in T_GRID:
        for d in D_GRID:
            x = 1.0 - np.exp(-np.clip(t - d, 0.0, None) / tau)
            X = np.column_stack([x, np.ones(len(t)), t])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ sol) ** 2))
            if best is None or sse < best[0]:
                best = (sse, tau, d, sol)
    sse, tau, d, sol = best
    a, b0, b1 = float(sol[0]), float(sol[1]), float(sol[2])
    edge = bool(tau <= T_GRID[0] * 1.02 or tau >= T_GRID[-1] * 0.98 or d >= D_GRID[-1] * 0.98)
    return dict(tau_ms=tau * 1e3, d_ms=d * 1e3, a=a, b0=b0, b1=b1, edge=edge,
                sse=sse, n=len(y))


def onset_curve(I, s0, n):
    """Extract the fit window from the step segment (drop first 10 ms, take 500 ms); subtract the mean of the segment's first 10 ms as a coarse baseline correction."""
    w0 = s0 + int(SKIP_S / DT)
    w1 = min(s0 + n, w0 + int(WIN_S / DT))
    if (w1 - w0) * DT < 0.3:
        return None, None
    t = np.arange(w1 - w0) * DT
    y = I[w0:w1].astype(float)
    return t, y


def noise_sigma(I, info):
    """Per-cell noise floor: std of the longest -80 mV baseline segment (>= 0.5 s) after detrending."""
    best = None
    for v, s0, n in info:
        if abs(v + 80) < 2 and n * DT >= 0.5:
            if best is None or n > best[2]:
                best = (s0, n, n)
    if best is None:
        return None
    s0, n, _ = best
    seg = I[s0 + int(0.05 / DT): s0 + n].astype(float)
    t = np.arange(len(seg)) * DT
    A = np.vstack([t, np.ones(len(t))]).T
    sol, *_ = np.linalg.lstsq(A, seg, rcond=None)
    return float(np.std(seg - A @ sol))


def synth_cell_curve(sigma, n_rep=9, tau_true=0.150, d_true=0.080, a_true=0.03, rng=None):
    """Synthetic +50-level equivalent: onset curve after averaging 9 repeats (drift term zero-mean)."""
    rng = rng or np.random.default_rng(7)
    t = np.arange(int(WIN_S / DT)) * DT
    x = a_true * (1.0 - np.exp(-np.clip(t - d_true, 0.0, None) / tau_true))
    acc = np.zeros(len(t))
    for _ in range(n_rep):
        acc += x + rng.normal(0, sigma, len(t)) + rng.normal(0, 0.002) * t
    return t, acc / n_rep


def main():
    t_start = time.time()
    print("=" * 74, flush=True)
    print(" alpha model - B6-H1 high-voltage activation verdict" + (" (smoke 16713003)" if SMOKE else " (nine cells full)"), flush=True)
    print(" H1: activation at +50/+60 much faster than +40 - steady_act(+40/+60) + deact(+50 x9) onset direct fit", flush=True)
    print("=" * 74, flush=True)

    rng = np.random.default_rng(20260914)

    # ---------- per-cell three-level fits ----------
    cells_out = {}
    sigmas = {}
    for cell in CELLS:
        rec = {}
        # steady_activation: +40 / +60
        V, I = load_mat("steady_activation_protocol.mat", cell, "steady_activation")
        if I is not None:
            info = segments(V)
            sig = noise_sigma(I, info)
            sigmas[cell] = sig
            for vt in (40.0, 60.0):
                for v, s0, n in info:
                    if abs(v - vt) < 2 and n * DT > 2.0:
                        t, y = onset_curve(I, s0, n)
                        if t is not None:
                            r = fit_onset(t, y)
                            r["sigma"] = sig
                            r["snr_a_over_sigma"] = (abs(r["a"]) / sig) if sig else None
                            rec[f"steady_{vt:+.0f}"] = r
                        break
        # deactivation: +50 x9 average
        V, I = load_mat("deactivation_protocol.mat", cell, "deactivation")
        if I is not None:
            info = segments(V)
            if cell not in sigmas:
                sigmas[cell] = noise_sigma(I, info)
            reps = []
            for v, s0, n in info:
                if abs(v - 50.0) < 2 and n * DT > 1.5:
                    t, y = onset_curve(I, s0, n)
                    if t is not None and len(t) == int(WIN_S / DT):
                        reps.append(y)
            if reps:
                t = np.arange(int(WIN_S / DT)) * DT
                y = np.mean(reps, axis=0)
                r = fit_onset(t, y)
                r["n_rep"] = len(reps)
                r["sigma"] = sigmas.get(cell)
                rec["deact_+50"] = r
        cells_out[cell] = rec
        s40 = rec.get("steady_+40"); s50 = rec.get("deact_+50"); s60 = rec.get("steady_+60")
        line = f"  {cell}: "
        for tag, r in [("+40", s40), ("+50", s50), ("+60", s60)]:
            if r:
                line += (f"{tag}: τ={r['tau_ms']:7.1f}ms d={r['d_ms']:6.1f}ms"
                         f"{'[edge]' if r['edge'] else ''}  ")
            else:
                line += f"{tag}: no-data  "
        print(line, flush=True)

    # ---------- controls ----------
    print("\n[controls]", flush=True)
    sig_vals = [s for s in sigmas.values() if s]
    sig_ref = float(np.median(sig_vals)) if sig_vals else float("nan")
    if not np.isfinite(sig_ref):
        print("  [warning] no valid noise-floor segment, controls cannot be built -> whole card registered as insufficient data", flush=True)
        ctrl_ok = False
        c1 = c2 = False
        c1_err, c2_p = [float("nan")], float("nan")
    else:
        sig_avg = sig_ref / 3.0                   # equivalent noise after averaging 9 repeats
        c1_err = []
        t_syn, y_syn = synth_cell_curve(sig_avg, tau_true=0.150, d_true=0.080)
        for rep in range(20):
            _, y1 = synth_cell_curve(sig_avg, tau_true=0.150, d_true=0.080,
                                     rng=np.random.default_rng(1000 + rep))
            r1 = fit_onset(t_syn, y1)
            c1_err.append(abs(r1["tau_ms"] - 150.0) / 150.0)
        c1 = float(np.median(c1_err)) < 0.30
        print(f"  C1 synthetic tau=150ms x20: median recovery error {np.median(c1_err) * 100:.1f}% (<30%) -> "
              f"{'pass' if c1 else 'fail'}", flush=True)
        from scipy.stats import ttest_ind
        taus_fast, taus_slow = [], []
        for rep in range(12):
            _, yf = synth_cell_curve(sig_avg, tau_true=0.050, d_true=0.020,
                                     rng=np.random.default_rng(2000 + rep))
            _, ys2 = synth_cell_curve(sig_avg, tau_true=0.150, d_true=0.080,
                                      rng=np.random.default_rng(3000 + rep))
            taus_fast.append(fit_onset(t_syn, yf)["tau_ms"])
            taus_slow.append(fit_onset(t_syn, ys2)["tau_ms"])
        c2_p = float(ttest_ind(taus_fast, taus_slow).pvalue)
        c2 = c2_p < 0.05
        print(f"  C2 tau=50ms vs 150ms distinguishable: recovery medians {np.median(taus_fast):.0f} vs "
              f"{np.median(taus_slow):.0f} ms, t test p={c2_p:.2e} (<0.05) -> "
              f"{'pass' if c2 else 'fail'}", flush=True)
        ctrl_ok = c1 and c2

    # ---------- verdict ----------
    print("\n" + "=" * 74, flush=True)
    verdict = None
    if not ctrl_ok:
        print(" controls not seated -> statistics void, registered as-is: H1 not judgeable under this noise floor (insufficient data).", flush=True)
    else:
        rows = []
        for cell, rec in cells_out.items():
            r40, r50, r60 = rec.get("steady_+40"), rec.get("deact_+50"), rec.get("steady_+60")
            if r40 and r50 and r60 and not (r40["edge"] or r50["edge"] or r60["edge"]):
                rows.append(dict(cell=cell,
                                 r50=r50["tau_ms"] / r40["tau_ms"],
                                 r60=r60["tau_ms"] / r40["tau_ms"]))
        n = len(rows)
        agree = sum(1 for r in rows if r["r50"] <= 1 / 3 and r["r60"] <= 1 / 3)
        med50 = float(np.median([r["r50"] for r in rows])) if rows else float("nan")
        med60 = float(np.median([r["r60"] for r in rows])) if rows else float("nan")
        print(f" valid cells {n}/9 (edge-pinned excluded); tau(+50)/tau(+40) median={med50:.3f}  "
              f"tau(+60)/tau(+40) median={med60:.3f}; both-levels <=1/3 concordance {agree}/{n}", flush=True)
        if n >= 6:
            if med50 <= 1 / 3 and med60 <= 1 / 3 and agree >= 7 * n / 9:
                verdict = "H1 supported: +50/+60 activation markedly faster than +40 (>=3x); the B6 contradiction converges toward H1"
            elif med50 > 1 / 2 or med60 > 1 / 2 or agree < 6 * n / 9:
                verdict = "H1 rejected: +50/+60 activation not much faster than +40; the 3x contradiction does not resolve with voltage"
            else:
                verdict = "weak evidence: between the two criteria, registered, model untouched"
        else:
            verdict = f"insufficient valid cells ({n}<6) -> insufficient data, H1 registered undecided"
        print(f" verdict: {verdict}", flush=True)
        if not SMOKE:
            print(" (formal convention: verdict in force)", flush=True)
        else:
            print(" (smoke: criterion-path rehearsal, not a verdict)", flush=True)
    print("=" * 74, flush=True)

    # ---------- save ----------
    tag = "_冒烟" if SMOKE else "_结果"
    fjson = os.path.join(HERE, f"2026-09-14_α模型_B6_H1_高压激活判决{tag}.json")
    out = {"meta": {"smoke": SMOKE, "cells": CELLS, "sigma_ref": sig_ref,
                    "runtime_s": time.time() - t_start},
           "cells": cells_out,
           "controls": {"C1_tau_recovery_median_err": float(np.median(c1_err)),
                        "C1_pass": c1, "C2_p": c2_p, "C2_pass": c2},
           "verdict": verdict}
    with open(fjson, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    print(f"\n results saved: {fjson}", flush=True)

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

    ncell = len(CELLS)
    fig, axes = plt.subplots(ncell, 1, figsize=(11, 2.6 * ncell), squeeze=False)
    for i, cell in enumerate(CELLS):
        ax = axes[i, 0]
        rec = cells_out.get(cell, {})
        for key, col, lab in [("steady_+40", "tab:blue", "+40"), ("deact_+50", "tab:orange", "+50(x9 avg)"),
                              ("steady_+60", "tab:red", "+60")]:
            # redraw raw curve + fit
            pass
        # reload data directly to draw the three levels
        V, I = load_mat("steady_activation_protocol.mat", cell, "steady_activation")
        drawn = set()
        if I is not None:
            for v, s0, n in segments(V):
                for vt, col in ((40.0, "tab:blue"), (60.0, "tab:red")):
                    if abs(v - vt) < 2 and n * DT > 2.0 and vt not in drawn:
                        t, y = onset_curve(I, s0, n)
                        if t is not None:
                            ax.plot(t * 1e3, y, color=col, lw=0.8, alpha=0.85,
                                    label=f"{vt:+.0f}mV measured")
                            key = f"steady_{vt:+.0f}"
                            if key in rec:
                                r = rec[key]
                                tt = np.arange(len(y)) * DT
                                xf = r["a"] * (1 - np.exp(-np.clip(tt - r["d_ms"] * 1e-3, 0, None)
                                                          / (r["tau_ms"] * 1e-3))) + r["b0"] + r["b1"] * tt
                                ax.plot(tt * 1e3, xf, color=col, lw=1.6, ls="--",
                                        label=f"{vt:+.0f} fit tau={r['tau_ms']:.0f}ms d={r['d_ms']:.0f}ms")
                            drawn.add(vt)
        V, I = load_mat("deactivation_protocol.mat", cell, "deactivation")
        if I is not None:
            reps = []
            for v, s0, n in segments(V):
                if abs(v - 50.0) < 2 and n * DT > 1.5:
                    t, y = onset_curve(I, s0, n)
                    if t is not None and len(t) == int(WIN_S / DT):
                        reps.append(y)
            if reps:
                t = np.arange(int(WIN_S / DT)) * DT
                y = np.mean(reps, axis=0)
                ax.plot(t * 1e3, y, color="tab:orange", lw=0.8, alpha=0.85, label="+50mV measured (x9 avg)")
                if "deact_+50" in rec:
                    r = rec["deact_+50"]
                    xf = r["a"] * (1 - np.exp(-np.clip(t - r["d_ms"] * 1e-3, 0, None)
                                              / (r["tau_ms"] * 1e-3))) + r["b0"] + r["b1"] * t
                    ax.plot(t * 1e3, xf, color="tab:orange", lw=1.6, ls="--",
                            label=f"+50 fit tau={r['tau_ms']:.0f}ms d={r['d_ms']:.0f}ms")
        ax.set_title(f"cell {cell}" + (" (D11 recording failure, registered)" if cell in D11_DEAD else ""),
                     fontsize=9)
        ax.set_xlabel("t after step (ms)"); ax.set_ylabel("I (nA)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("B6-H1 high-voltage activation verdict: +40/+50/+60 onset m(t) direct fit"
                 + (" (smoke)" if SMOKE else ""), fontsize=12)
    fpng = os.path.join(HERE, f"2026-09-14_α模型_B6_H1_高压激活判决{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" figure saved: {fpng}", flush=True)

    if SMOKE:
        print("\n[smoke done] formal-run command (Spyder):\n"
              "  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/"
              "2026-09-14_α模型_B6_H1_高压激活判决.py' --wdir", flush=True)


if __name__ == "__main__":
    main()
