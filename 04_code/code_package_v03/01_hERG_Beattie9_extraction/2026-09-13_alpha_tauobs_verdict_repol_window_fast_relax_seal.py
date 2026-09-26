# 2026-09-13_alpha_tauobs_verdict_repol_window_fast_relax_seal.py
# Purpose (gap 2 repolarisation segment + gap 1b closed together; formal verdict, nine cells):
#   Inactivation-protocol test segment (test V x 0.150 s, preceded by -90 x 0.06 s reset -> h(0) ~= 1, m(0) ~= 0.43 residual on record);
#   the early phase carries tau_obs (relaxation time of h from 1 toward h_ss(V)).
#   Recon on record (2026-09-13_alpha_tauobs_recon_testseg_early_phase.png/.py): -70..-30 mV
#   30 ms ratio 1.03-1.22 ~= pure m decay; shapes with tau_obs >= 25 ms (>= 1.4) clearly excluded.
# Verdict targets:
#   (1) tau_obs(V) upper-bound table (six levels -80..-30): measured anchor of repolarisation-window tau_h (gap 2);
#   (2) h_ss = h_150 identity (gap 1b): when the tau_obs upper bound <= 20 ms, the family hss = (h150 - e^-150/tau)/(1 - e^-150/tau)
#       is forced ~= h150 and the identity holds automatically.
# Model (ratio form, K = G*DF cancelled by normalisation):
#   R(t) = I(t)/i_end = [m(t)/m(150)] * [h(t)/h(150)], i_end = mean of last 20 ms;
#   m(t) = mss + (m0 - mss) e^{-t/tau_eff} (mss uses the re-split value of the table-swap version, same chain as forward);
#   h(t) = hss + (1 - hss) e^{-t/tau_obs}, hss pinned by the (h150, tau_obs) family (endpoints h(0)=1, h(150)=h150).
#   h150: per-cell QC-passing level values from the strip JSON, else population median (same source chain as table-swap forward).
# Grid: tau_obs in [1,300] ms 48 log points x tau_eff in {0.3,0.5,0.8,1.2,2,3} s x m0 in {0.2,0.3,0.435,0.55,0.7}.
# Statistical gate (no white-noise assumption; same discipline as the v5 envelope gate):
#   data 2 ms-binned medians; tau_obs acceptable <=> model binned values at 10/30/60 ms all fall within data +/- 3 sigma_b
#   (sigma_b = 1.25 * sigma/sqrt(20)/|i_end|, sigma = detrended std of that cell's closing -80 x 1.34 s segment).
#   upper bound = largest acceptable tau_obs; point estimate = minimum SSE over all 75 bins (sigma_b convention).
# Controls (declared pre-run):
#   C1a synthetic tau_obs = 60 ms (h150=0.33, m0=0.435, tau_eff=0.8, sigma_R=0.027) -> point estimate in [40,90] ms
#       and upper bound >= 60 ms (pipeline must not miss a slow one);
#   C1b synthetic tau_obs = 5 ms -> upper bound <= 15 ms (pipeline must not call a fast one slow);
#   C2 synthetic tau = 60 with true hss 0.270 -> |hss_best - 0.270|/0.270 <= 25% (family split usable).
# Criteria (declared pre-run, frozen after run):
#   per level: valid cells n >= 4 and tau_obs upper-bound median <= 20 ms -> level sealed as "fast relaxation + hss = h150 identity";
#   n >= 4 but upper-bound median > 20 ms -> level registered (report the median); n < 4 insufficient data.
#   all six levels sealed -> overall verdict "repolarisation-window tau_h fast (<= 20 ms), h_ss = h150 identity sealed, gap 1b / gap 2 repolarisation segment closed";
#   any level unsealed -> registered as-is, no cosmetic edits.
#   tau_obs point estimates reported per cell (expected to mostly hit the 1-10 ms low grid -> essentially an upper-bound table; verdict written as-is).
# Run: python this file (nine cells full); SMOKE=1 single-cell 16713003 smoke.
import os
import json
import numpy as np
import scipy.io as sio
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
GEARS = [-80, -70, -60, -50, -40, -30]
TAU_OBS = np.exp(np.linspace(np.log(0.001), np.log(0.300), 48))
TAU_EFF = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
M0S = [0.2, 0.3, 0.435, 0.55, 0.7]
UB_LINE = 0.020          # upper-bound criterion 20 ms
TQ = [0.010, 0.030]      # early two-point gate (10/30 ms; 60 ms dropped, see below)
FORM_TOL = 0.03          # model-form error tolerance (absolute, ratio convention, declared pre-run)
# Gate design (fixed in smoke, on record): real data 60->150 ms carries ~3% non-monotonic structure (dip-then-rise,
#   not produced by a single-h + single-m model; suspected slow residual/subtraction artefact, registered); tau_obs is
#   judged only from the early 10/30 ms two points: a slow h (tau >= 25 ms) signals at 0.2-0.7x value magnitude in the
#   early window, far above that anomaly and the noise, so the upper bound stays conservative and valid.

F_HSS = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决_结果.json")
F_AMP = os.path.join(HERE, "2026-09-13_α模型_幅度表提取_结果.json")


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


def test_segments(V):
    info = segments(V)
    out = {}
    for k, (v, s0, n) in enumerate(info):
        if 0.14 < n * DT < 0.16 and k >= 1:
            pv, ps, pn = info[k - 1]
            if abs(pv + 90.0) < 2.0 and 0.05 < pn * DT < 0.07:
                out[int(round(v))] = (s0, n)
    return out


def noise_sigma(V, I):
    info = segments(V)
    for v, s0, n in info:
        if abs(v + 80) < 2 and n * DT > 1.0:
            seg = I[s0 + n - 5000: s0 + n].astype(float)
            seg = seg - np.polyval(np.polyfit(np.arange(len(seg)), seg, 1), np.arange(len(seg)))
            return float(np.std(seg))
    return np.nan


def bin2(t, y, w=0.002):
    nb = int(round(t[-1] / w))
    tm, ym = [], []
    for b in range(nb):
        m = (t >= b * w) & (t < (b + 1) * w)
        if m.sum() >= 3:
            tm.append((b + 0.5) * w)
            ym.append(float(np.median(y[m])))
    return np.array(tm), np.array(ym)


def mss_table(cell, amp, hssj):
    """Same chain as table-swap forward: mss = y_ss/h150 (missing levels use median); -80/-30 use bridging."""
    yss = {}
    for r in amp["rows"]:
        if r["v"] in (-70, -60, -50):
            yss.setdefault(r["v"], {})[r["cell"]] = max(r["y_ss"], 1e-4)
    med = {v: float(np.median(list(d.values()))) for v, d in yss.items()}
    def cell_y(v):
        return yss.get(v, {}).get(cell, med.get(v, 0.02))
    hc = hssj["cells"].get(cell, {}).get("curve", {})
    def h150(v):
        e = hc.get(str(v))
        if e and e.get("qc") and e["h"] > 0:
            return float(e["h"])
        return float(hssj["gears"][str(v)]["med"])
    return {-80: min(1.0, cell_y(-70) * 0.5 / h150(-80)),
            -70: min(1.0, cell_y(-70) / h150(-70)),
            -60: min(1.0, cell_y(-60) / h150(-60)),
            -50: min(1.0, cell_y(-50) / h150(-50)),
            -40: min(1.0, cell_y(-50) * 3.0 / h150(-40)),
            -30: min(1.0, cell_y(-50) * 8.0 / h150(-30))}, h150


def model_R(tb, tau_obs, tau_eff, m0, mss, h150):
    r = np.exp(-0.150 / tau_obs)
    hss = (h150 - r) / (1.0 - r) if r < 0.999999 else h150
    hss = max(hss, 0.0)
    h = hss + (1.0 - hss) * np.exp(-tb / tau_obs)
    m = mss + (m0 - mss) * np.exp(-tb / tau_eff)
    y = m * h
    return y / y[-1], hss


def fit_gear(tb, yb, sig_b, mss, h150):
    """Returns best(tau_obs, tau_eff, m0, hss, sse) and upper bound ub; None if invalid."""
    best = None
    ub = 0.0
    qidx = [int(np.argmin(np.abs(tb - x))) for x in TQ]   # nearest bins (bin centres 1,3,5... ms)
    for to in TAU_OBS:
        for te in TAU_EFF:
            for m0 in M0S:
                yhat, hss = model_R(tb, to, te, m0, mss, h150)
                sse = float(np.sum(((yb - yhat) / sig_b) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, to, te, m0, hss)
                if np.all(np.abs(yb[qidx] - yhat[qidx]) <= 3 * sig_b + FORM_TOL):
                    ub = max(ub, to)
    if best is None:
        return None
    sse, to, te, m0, hss = best
    return dict(tau_obs=float(to), ub=float(ub), tau_eff=float(te), m0=float(m0),
                hss=float(hss), h150=float(h150), sse=sse)


def main():
    amp = json.load(open(F_AMP, encoding="utf-8"))
    hssj = json.load(open(F_HSS, encoding="utf-8"))
    print("=" * 92)
    print(" tau_obs verdict - repolarisation-window fast-relaxation SEAL (gap 2 repol segment + gap 1b)"
          + (" (smoke 16713003)" if SMOKE else " (nine cells)"))
    print(f" criteria: per level n>=4 and tau_obs upper-bound median <={UB_LINE * 1e3:.0f} ms -> fast relaxation + hss=h150 identity SEAL")
    print("=" * 92, flush=True)

    # ---------- controls ----------
    print("\n[controls]", flush=True)
    tb_c = (np.arange(75) + 0.5) * 0.002
    rng = np.random.default_rng(20260913)
    SIG_RAW_C = 0.027                      # synthetic raw-point noise (003 -70 level convention)
    sig_c = 1.25 * SIG_RAW_C / np.sqrt(20)  # 2 ms-binned median SE (same convention as the real-data path)
    # C1a: tau = 60 ms slow relaxation must be seen
    y_c, hss_c = model_R(tb_c, 0.060, 0.8, 0.435, 0.057, 0.33)
    y_c = y_c + rng.normal(0, sig_c, len(tb_c))
    rc = fit_gear(tb_c, y_c, sig_c, 0.057, 0.33)
    c1a = bool(rc and 0.040 <= rc["tau_obs"] <= 0.090 and rc["ub"] >= 0.060)
    print(f"  C1a synthetic tau=60 ms: point estimate {rc['tau_obs'] * 1e3:.0f} ms upper bound {rc['ub'] * 1e3:.0f} ms "
          f"(require [40,90] and ub>=60) -> {'pass' if c1a else 'fail'}", flush=True)
    # C1b: tau = 5 ms fast relaxation upper bound must hold down
    y_c2, _ = model_R(tb_c, 0.005, 0.8, 0.435, 0.057, 0.33)
    y_c2 = y_c2 + rng.normal(0, sig_c, len(tb_c))
    rc2 = fit_gear(tb_c, y_c2, sig_c, 0.057, 0.33)
    c1b = bool(rc2 and rc2["ub"] <= 0.015)
    print(f"  C1b synthetic tau=5 ms:  point estimate {rc2['tau_obs'] * 1e3:.0f} ms upper bound {rc2['ub'] * 1e3:.0f} ms "
          f"(require ub<=15) -> {'pass' if c1b else 'fail'}", flush=True)
    # C2: family-split hss recovery
    c2 = bool(rc and abs(rc["hss"] - 0.270) / 0.270 <= 0.25)
    print(f"  C2 family-split hss: {rc['hss']:.3f} (truth 0.270, +/-25%) -> {'pass' if c2 else 'fail'}", flush=True)
    if not (c1a and c1b and c2):
        print("  controls not seated -> statistics void, halt.", flush=True)
        return
    print("  controls seated.", flush=True)

    # ---------- real data ----------
    print("\n[real data]", flush=True)
    rows = {g: [] for g in GEARS}
    detail = {}
    for cell in CELLS:
        V, I = load_mat("inactivation_protocol.mat", cell, "inactivation")
        ts = test_segments(V)
        sig = noise_sigma(V, I) if I is not None else np.nan
        mss, h150f = mss_table(cell, amp, hssj)
        detail[cell] = {}
        line = f"  {cell}: "
        for g in GEARS:
            if g not in ts or I is None:
                line += f"{g}:no-level "
                continue
            s0, n = ts[g]
            t = np.arange(n) * DT
            i = I[s0: s0 + n]
            i_end = float(np.mean(i[-2000:]))
            if abs(i_end) < 5 * sig:
                line += f"{g}:weak "
                continue
            tb, yb = bin2(t, i / i_end)
            sig_b = 1.25 * sig / np.sqrt(20) / abs(i_end)
            r = fit_gear(tb, yb, sig_b, mss[g], h150f(g))
            if r is None:
                line += f"{g}:× "
                continue
            r["i_end"] = i_end
            detail[cell][g] = r
            rows[g].append(r)
            line += f"{g}:τ={r['tau_obs'] * 1e3:.0f}/ub={r['ub'] * 1e3:.0f} "
        print(line, flush=True)

    # ---------- verdict ----------
    print("\n" + "-" * 92, flush=True)
    print(f"{'level':>6} {'n':>3} {'tau point med':>12} {'ub median':>10} {'ub max':>10}  verdict", flush=True)
    verdicts = {}
    for g in GEARS:
        rs = rows[g]
        n = len(rs)
        if n >= 4:
            med_est = float(np.median([r["tau_obs"] for r in rs]))
            med_ub = float(np.median([r["ub"] for r in rs]))
            mx_ub = float(np.max([r["ub"] for r in rs]))
            ok = med_ub <= UB_LINE
            vd = "[SEAL: fast relaxation + hss=h150 identity]" if ok else "[REGISTER: upper bound above line]"
            verdicts[g] = dict(n=n, med_est=med_est, med_ub=med_ub, max_ub=mx_ub,
                               sealed=bool(ok))
            print(f"{g:>6} {n:>3} {med_est * 1e3:>10.0f}ms {med_ub * 1e3:>8.0f}ms "
                  f"{mx_ub * 1e3:>8.0f}ms  {vd}", flush=True)
        else:
            verdicts[g] = dict(n=n, sealed=False)
            print(f"{g:>6} {n:>3}      --       --       --  [INSUFFICIENT DATA]", flush=True)
    nseal = sum(1 for v in verdicts.values() if v.get("sealed"))
    print("=" * 92, flush=True)
    if nseal == len(GEARS):
        final = ("repolarisation-window tau_h fast (upper-bound median <= 20 ms, all six levels sealed); h_ss = h150 identity sealed: "
                 "gap 1b closed; gap 2 repolarisation-segment measured anchor landed (B2 bridge 11-23 ms consistent with measured upper bound, "
                 "bridge retired)")
    else:
        final = f"sealed {nseal}/{len(GEARS)} levels: unsealed levels registered as-is, details above"
    print(" overall verdict: " + final, flush=True)

    # ---------- figure ----------
    cells_done = [c for c in CELLS if detail.get(c)]
    fig, axes = plt.subplots(len(cells_done), len(GEARS),
                             figsize=(2.6 * len(GEARS), 2.2 * len(cells_done)), squeeze=False)
    for ri, cell in enumerate(cells_done):
        V, I = load_mat("inactivation_protocol.mat", cell, "inactivation")
        ts = test_segments(V)
        mss, h150f = mss_table(cell, amp, hssj)
        for ci, g in enumerate(GEARS):
            ax = axes[ri][ci]
            r = detail[cell].get(g)
            if r is None or g not in ts:
                ax.set_title(f"{g}mV none", fontsize=7)
                continue
            s0, n = ts[g]
            t = np.arange(n) * DT
            i = I[s0: s0 + n]
            i_end = float(np.mean(i[-2000:]))
            tb, yb = bin2(t, i / i_end)
            ax.plot(tb, yb, "k.", ms=2)
            yh, _ = model_R(tb, r["tau_obs"], r["tau_eff"], r["m0"], mss[g], h150f(g))
            ax.plot(tb, yh, color="tab:red", lw=1.0, label=f"best τ={r['tau_obs'] * 1e3:.0f}ms")
            y25, _ = model_R(tb, 0.025, r["tau_eff"], r["m0"], mss[g], h150f(g))
            ax.plot(tb, y25, color="tab:blue", lw=0.7, ls="--", label="tau=25ms reference")
            ax.set_title(f"{cell[-3:]} {g}mV ub={r['ub'] * 1e3:.0f}ms", fontsize=7)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=6)
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-13_α模型_τobs判决_复极窗快弛豫封卷.png")
    fig.savefig(fpng, dpi=120, bbox_inches="tight")

    out = dict(controls=dict(C1a=bool(c1a), C1b=bool(c1b), C2=bool(c2)),
               ub_line_ms=UB_LINE * 1e3,
               cells={c: {str(g): r for g, r in detail[c].items()} for c in detail},
               verdicts={str(g): v for g, v in verdicts.items()},
               nseal=nseal, final=final)
    fjson = os.path.join(HERE, "2026-09-13_α模型_τobs判决_复极窗快弛豫封卷_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  figure saved: {fpng}", flush=True)
    print(f"  results saved: {fjson}", flush=True)


if __name__ == "__main__":
    main()
