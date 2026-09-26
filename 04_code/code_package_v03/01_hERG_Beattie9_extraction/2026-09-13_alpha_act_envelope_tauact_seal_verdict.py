# 2026-09-13_alpha model_activation envelope_tau_act sealed adjudication.py
# Purpose (gap 3 main target: seal tau_act(V) in place; the D7 workaround):
#   activation_kinetics_1/2 is a tail-current envelope protocol: -120x2.5s full recovery +
#   full deactivation reset (clean initial state m~0, h=1) -> -120x0.05 -> -80x0.2 ->
#   test pulse (k1=0mV / k2=+40mV, dt = 3/10/30/100/300/1000ms, six ascending steps) ->
#   -120x2.5s (rebound DoE: tau_r 3ms recovery, tau_d 30ms deactivation, sealed values) ->
#   rebound amplitude A ~ m(dt). The envelope A(dt) is the activation time course.
#   Advantage: bypasses D7 (no need to find m riding on the small +40 steady-state current);
#   the rebound is a large, fast signal. The DoE pipeline is seal-validated (16713003 smoke:
#
# [16713003 design smoke facts (basis for the criterion design)]
#   k2(+40) envelope ratios 0.012/0.011/0.015/0.079/0.637/1.0, main rise at 100-300ms
#     with a sigmoid foot (delay); A_max=3.71 vs G*DF prediction 4.15 (89%).
#   k1(0): first 5 steps all on the noise floor (<=0.13nA, DoE hits edge and is void);
#     only the 1000ms point 0.776 is solid -> 0mV activation is slow on the seconds scale,
#     consistent with the registry ladder (0mV: 2.0s); the shallow A(V) curve is the floor
#
# Criteria (declared before run):
#   QC1 (DoE gate): points entering the envelope fit must have tau_r in [1.5,8]ms and
#     tau_d in [12,80]ms (margin around sealed values 3.04/30.3) and A>=4*sigma_cell
#     (sigma_cell = std of this cell's -80x2.4s baseline segment); failing points are excluded and counted.
#   Fit form: A(dt)=A_inf*(1-exp(-max(dt-d,0)/tau)), grid d in {0,25,...,150}ms,
#     tau log-grid (k2: [0.02,1.5]s; k1: [0.2,8]s), A_inf by linear least squares.
#   QC2 (A_inf gate): A_inf_fit / (G*31.67) in [0.5,2.0] (G declared +/-30% + m_ss
#     declaration + h factor); out-of-range cells are flagged posthoc_ainf_out.
#   k1 additionally gets an anchored fit: A_inf fixed = G*31.67 (m_ss(0)=1 declaration),
#     single parameter tau_anch. Single-point anchor fallback (added 2026-09-13 before run):
#     when n_qc=1, tau_single = -dt/ln(1-A/(G*DF)), only if 0<A/(G*DF)<1; d=0 declared,
#     delay-degeneracy flag single_pt_d0 on record; A/(G*DF)>=1 means m_ss(0)<1 or G biased
#   Seal criteria (three tiers, fixed before run):
#     tau_act(+40): cells passing all QC >=7 and nine-cell tau CV<0.3 -> [SEAL];
#       QC<7 or CV>=0.3 -> [REGISTER] (report values as-is, no seal).
#     tau_act(0): best per-cell estimate = tau_anch (>=2 points) preferred, tau_single (1 point) fallback;
#       valid cells >=7 and CV<0.3 -> [HALF-SEAL] (anchored values enter the table, extrapolation
#       and single-point degeneracy declared); otherwise -> [REGISTER-INTERVAL] (report tau_free/tau_anch/tau_single).
# Controls (declared before run):
#   C1 DoE inversion of tau_r 3.5ms, error<15% (pipeline);
#   C2 synthetic envelope (d=50ms, tau=150ms, A_inf=3.7 + 16713003 measured baseline noise)
#     recovers tau with error<20% and d with error<50ms -> otherwise the statistic is voided, halt.
# Run: python this file (full nine-cell set); SMOKE=1 for the 16713003 single-cell smoke test.
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
E_REV = -88.33
DF_M120 = abs(-120.0 - E_REV)

F_HOOK = os.path.join(HERE, "2026-09-13_α模型_反弹hook_结果.json")
F_INACT = os.path.join(HERE, "2026-09-13_α模型_失活门_失活协议封卷判决_结果.json")

TR_GRID = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))
TD_GRID = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))
SEP_MIN = 1.5
D_GRID = np.arange(0, 0.1501, 0.025)          # delay grid, s
TAU_GRID_K2 = np.exp(np.linspace(np.log(0.02), np.log(1.5), 40))
TAU_GRID_K1 = np.exp(np.linspace(np.log(0.2), np.log(8.0), 40))
TR_OK = (0.0015, 0.008)
TD_OK = (0.012, 0.080)


def load_mat(proto, cell, tag):
    V = sio.loadmat(f"{DATA}/data/protocols/{proto}")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/{tag}_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def doe_fit(t, y):
    if len(t) < 50:
        return None
    best = None
    for tr in TR_GRID:
        for td in TD_GRID[TD_GRID >= SEP_MIN * tr]:
            x = np.exp(-t / td) - np.exp(-t / tr)
            X = np.column_stack([np.ones(len(t)), x])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ sol) ** 2))
            if best is None or sse < best[0]:
                best = (sse, tr, td, sol)
    sse, tr, td, sol = best
    return dict(tau_r=float(tr), tau_d=float(td), A=float(sol[1]),
                edge=bool(tr <= TR_GRID[0] * 1.02 or td >= TD_GRID[-1] * 0.98
                          or tr >= TR_GRID[-1] * 0.98))


def segments(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    return [(float(V[s[0]]), int(s[0]), len(s)) for s in np.split(np.arange(len(V)), edges)]


def envelope(cell, proto, vtest):
    """Extract the envelope: each period test pulse (vtest) -> -120 rebound DoE. Returns (dt list, fit points, sigma)."""
    V, I = load_mat(f"{proto}_protocol.mat", cell, proto)
    if I is None:
        return None
    info = segments(V)
    # sigma_cell: first -80x2.4s long baseline segment
    sigma = None
    for v, s0, n in info:
        if abs(v + 80) < 2 and n * DT > 2.0:
            sigma = float(np.std(I[s0 + 2000: s0 + n]))
            break
    pts = []
    for i, (v, s0, n) in enumerate(info):
        if abs(v - vtest) < 2 and i + 1 < len(info) and abs(info[i + 1][0] + 120) < 2 \
                and info[i + 1][2] * DT > 2.0:
            a = info[i + 1][1]
            w = a + 3000                          # 300ms before the rebound (DoE fast structure)
            t = np.arange(w - a) * DT
            r = doe_fit(t, -I[a:w])
            if r is None:
                continue
            ok = (not r["edge"]) and TR_OK[0] <= r["tau_r"] <= TR_OK[1] \
                and TD_OK[0] <= r["tau_d"] <= TD_OK[1] and r["A"] >= 4 * (sigma or 1)
            pts.append(dict(dt=n * DT, A=r["A"], tau_r=r["tau_r"], tau_d=r["tau_d"],
                            qc=bool(ok)))
    return dict(pts=pts, sigma=sigma)


def fit_env(dts, As, tau_grid, d_grid=D_GRID, ainf_fix=None):
    """Grid over (d, tau), A_inf by linear lstsq (or fixed). Returns best dict."""
    dts = np.asarray(dts, float)
    As = np.asarray(As, float)
    best = None
    for d in d_grid:
        rise = 1.0 - np.exp(-np.maximum(dts - d, 0.0)[:, None] / tau_grid[None, :])
        for j, tau in enumerate(tau_grid):
            x = rise[:, j]
            if ainf_fix is not None:
                ainf = ainf_fix
            else:
                ainf = float(As @ x / (x @ x)) if x @ x > 1e-12 else 0.0
            sse = float(np.sum((As - ainf * x) ** 2))
            if best is None or sse < best[0]:
                best = (sse, d, float(tau), ainf)
    sse, d, tau, ainf = best
    return dict(d=d, tau=tau, A_inf=ainf, sse=sse)


def g_anchor(cell, hook, inact):
    """G fallback chain (same convention as the sine smoke R1)."""
    for r in hook["A_rows"]:
        if r["cell"] == cell and r["v"] == -120 and r["valid"] and r["A"] > 0:
            return r["A"] / DF_M120, None
    cands = [r["A"] / DF_M120 for r in hook["A_rows"]
             if r["cell"] == cell and r["valid"] and r["A"] > 0]
    if cands:
        return float(np.median(cands)), None
    ci = inact["cells"].get(cell, {})
    if ci.get("g_hat") and ci["g_hat"] > 0:
        return float(ci["g_hat"]), "posthoc_G_inact"
    return None, "posthoc_G_fail"


def main():
    rng = np.random.default_rng(7)
    print("=" * 86)
    print(" alpha model activation envelope tau_act sealed adjudication" + (" (smoke 16713003)" if SMOKE else " (full nine-cell set)"))
    print(" criteria: +40 QC>=7 and CV<0.3 -> SEAL | 0mV free/anch consistent and CV<0.3 -> HALF-SEAL, else REGISTER")
    print("=" * 86, flush=True)

    hook = json.load(open(F_HOOK, encoding="utf-8"))
    inact = json.load(open(F_INACT, encoding="utf-8"))

    # ---------- controls ----------
    print("\n[controls]", flush=True)
    t_c = np.arange(int(0.3 / DT)) * DT
    y_c = 3.0 * (np.exp(-t_c / 0.030) - np.exp(-t_c / 0.0035)) + rng.normal(0, 0.02, len(t_c))
    rc = doe_fit(t_c, y_c)
    c1 = bool(rc and abs(rc["tau_r"] - 0.0035) / 0.0035 < 0.15)
    print(f"  C1 DoE inversion tau_r: {rc['tau_r'] * 1e3:.2f}ms (true 3.5) -> {'pass' if c1 else 'fail'}",
          flush=True)
    # C2 synthetic envelope recovery
    dts_syn = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    A_syn = 3.7 * (1 - np.exp(-np.maximum(dts_syn - 0.05, 0) / 0.15)) \
        + rng.normal(0, 0.04, len(dts_syn))
    f_syn = fit_env(dts_syn, A_syn, TAU_GRID_K2)
    c2 = bool(abs(f_syn["tau"] - 0.15) / 0.15 < 0.20 and abs(f_syn["d"] - 0.05) < 0.05)
    print(f"  C2 synthetic envelope recovery: d={f_syn['d'] * 1e3:.0f}ms (true 50) tau={f_syn['tau'] * 1e3:.0f}ms"
          f"(true 150) A_inf={f_syn['A_inf']:.2f} (true 3.7) -> {'pass' if c2 else 'fail'}", flush=True)
    if not (c1 and c2):
        print("  controls not returned to baseline -> statistic voided, halt.", flush=True)
        return
    print("  controls returned to baseline.", flush=True)

    # ---------- real data ----------
    print("\n[real data]", flush=True)
    res = {}
    for c in CELLS:
        G, gflag = g_anchor(c, hook, inact)
        e1 = envelope(c, "activation_kinetics_1", 0.0)
        e2 = envelope(c, "activation_kinetics_2", 40.0)
        out = dict(G=G, gflag=gflag)
        for tag, e, tg in (("k1_0mV", e1, TAU_GRID_K1), ("k2_+40", e2, TAU_GRID_K2)):
            if e is None or not e["pts"]:
                out[tag] = dict(n=0)
                continue
            good = [p for p in e["pts"] if p["qc"]]
            rec = dict(n=len(e["pts"]), n_qc=len(good), sigma=e["sigma"],
                       pts=[(round(p["dt"] * 1e3), round(p["A"], 4)) for p in e["pts"]])
            if len(good) >= 2:
                f = fit_env([p["dt"] for p in good], [p["A"] for p in good], tg)
                rec.update(tau=f["tau"], d=f["d"], A_inf=f["A_inf"])
                if G:
                    pred = G * DF_M120
                    rec["ainf_ratio"] = f["A_inf"] / pred
                    rec["qc2"] = bool(0.5 <= rec["ainf_ratio"] <= 2.0)
                    fa = fit_env([p["dt"] for p in good], [p["A"] for p in good], tg,
                                 ainf_fix=pred)
                    rec["tau_anch"] = fa["tau"]
                    rec["d_anch"] = fa["d"]
            elif len(good) == 1 and G:
                p = good[0]                               # single-point anchor fallback (added before run)
                r_ = p["A"] / (G * DF_M120)
                if 0 < r_ < 1:
                    rec["tau_single"] = float(-p["dt"] / np.log(1 - r_))
                    rec["sflag"] = "single_pt_d0"
                else:
                    rec["sflag"] = "mss_lt1_candidate"
            out[tag] = rec
        res[c] = out
        k1, k2 = out["k1_0mV"], out["k2_+40"]
        print(f"  {c}: G={G:.4f}" +
              (f" | +40: QC {k2.get('n_qc', 0)}/{k2.get('n', 0)} "
               f"τ={k2.get('tau', np.nan) * 1e3:.0f}ms d={k2.get('d', np.nan) * 1e3:.0f}ms "
               f"A_inf={k2.get('A_inf', np.nan):.2f}(ratio {k2.get('ainf_ratio', np.nan):.2f})"
               if k2.get("n") else " | +40: no data") +
              (f" | 0mV: QC {k1.get('n_qc', 0)}/{k1.get('n', 0)} "
               f"τf={k1.get('tau', np.nan):.2f}s τa={k1.get('tau_anch', np.nan):.2f}s "
               f"τs={k1.get('tau_single', np.nan):.2f}s"
               if k1.get("n") else " | 0mV: no data"), flush=True)

    # ---------- sealed adjudication ----------
    print("\n" + "-" * 86, flush=True)
    taus40 = [res[c]["k2_+40"]["tau"] for c in res
              if res[c]["k2_+40"].get("qc2") and res[c]["k2_+40"].get("n_qc", 0) >= 2]
    n40 = len(taus40)
    if n40 >= 2:
        cv40 = float(np.std(taus40) / np.mean(taus40))
    else:
        cv40 = np.nan
    seal40 = bool(n40 >= (1 if SMOKE else 7) and np.isfinite(cv40) and cv40 < 0.3)
    print(f"  tau_act(+40): QC2 passed {n40}/{len(res)}, tau median "
          f"{np.median(taus40) * 1e3 if taus40 else np.nan:.0f}ms，CV={cv40:.3f} -> "
          f"{'[SEAL]' if seal40 else '[REGISTER]'}", flush=True)

    taf, taa = [], []
    best0 = {}
    for c in res:
        k1 = res[c]["k1_0mV"]
        if k1.get("tau") and k1.get("tau_anch"):
            taf.append(k1["tau"])
            taa.append(k1["tau_anch"])
        est = k1.get("tau_anch", k1.get("tau_single"))   # tau_anch preferred, tau_single fallback
        if est:
            best0[c] = est
    n0 = len(best0)
    v0 = list(best0.values())
    if n0 >= 2:
        cv0 = float(np.std(v0) / np.mean(v0))
    else:
        cv0 = np.nan
    ratio01 = float(np.median(np.array(taf) / np.array(taa))) if len(taf) >= 2 else np.nan
    seal0 = bool(n0 >= (1 if SMOKE else 7) and np.isfinite(cv0) and cv0 < 0.3)
    print(f"  tau_act(0mV): valid estimates {n0}/{len(res)} (tau_anch preferred, tau_single fallback), "
          f"median {np.median(v0) if v0 else np.nan:.2f}s, CV={cv0:.3f}"
          f"(free/anch ratio {ratio01:.2f}, reference) -> "
          f"{'[HALF-SEAL]' if seal0 else '[REGISTER-INTERVAL]'}", flush=True)

    verdict = []
    verdict.append("tau_act(+40) " + ("SEAL" if seal40 else "REGISTER"))
    verdict.append("tau_act(0mV) " + ("HALF-SEAL" if seal0 else "REGISTER-INTERVAL"))
    print(" overall verdict:", "; ".join(verdict), flush=True)

    # ---------- figure ----------
    nfig = len(res)
    fig, axes = plt.subplots(nfig, 2, figsize=(13, 2.6 * nfig), squeeze=False)
    for row, c in enumerate(res):
        for col, (proto, vt, tag) in enumerate(
                [("activation_kinetics_1", 0.0, "k1_0mV"),
                 ("activation_kinetics_2", 40.0, "k2_+40")]):
            ax = axes[row][col]
            e = envelope(c, proto, vt)
            if e is None:
                continue
            dts = [p["dt"] * 1e3 for p in e["pts"]]
            As = [p["A"] for p in e["pts"]]
            qc = [p["qc"] for p in e["pts"]]
            ax.scatter([d for d, q in zip(dts, qc) if q],
                       [a for a, q in zip(As, qc) if q], c="tab:blue", s=40, label="QC pass")
            ax.scatter([d for d, q in zip(dts, qc) if not q],
                       [a for a, q in zip(As, qc) if not q], c="0.7", s=30, marker="x",
                       label="QC excluded")
            rec = res[c][tag]
            if rec.get("tau"):
                dd = np.linspace(0, 1.0, 200)
                ax.plot(dd * 1e3, rec["A_inf"] * (1 - np.exp(
                    -np.maximum(dd - rec["d"], 0) / rec["tau"])), "tab:red",
                    lw=1.2, label=f"fit tau={rec['tau'] * 1e3:.0f}ms d={rec['d'] * 1e3:.0f}ms")
            ax.set_xscale("log")
            ax.set_title(f"{c} {tag}", fontsize=9)
            ax.set_xlabel("Δt (ms)")
            ax.set_ylabel("rebound A (nA)")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-13_α模型_激活envelope_τact封卷判决.png")
    fig.savefig(fpng, dpi=120, bbox_inches="tight")

    fjson = os.path.join(HERE, "2026-09-13_α模型_激活envelope_τact封卷判决_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(dict(cells=res, seal40=seal40, cv40=cv40, tau40_med=float(
            np.median(taus40)) if taus40 else None, seal0=seal0, cv0=cv0,
            tau0_best=best0, tau0_med=float(np.median(v0)) if v0 else None,
            ratio_free_anch=ratio01), f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  figure saved: {fpng}", flush=True)
    print(f"  results saved: {fjson}", flush=True)


if __name__ == "__main__":
    main()
