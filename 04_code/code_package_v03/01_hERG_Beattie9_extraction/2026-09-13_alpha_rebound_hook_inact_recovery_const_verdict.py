# -*- coding: utf-8 -*-
# 2026-09-13_alpha_rebound_hook_inact_recovery_const_verdict.py
# Purpose: is inactivation-recovery kinetics also a set of discrete constants
#   (prerequisite of the extended alpha model)?
#   Method (alpha-model methodology: no assumed framework, direct extraction from
#     data, cross-cell constancy): hook/rebound current = rise (inactivation
#     relief, tau_rec) then decay (deactivation, tau_deact); fit form
#     I(t) = c + A*(e^(-t/tau_d) - e^(-t/tau_r)), tau_d >= 1.5*tau_r grid +
#     linear LS, deterministic.
#   Samples: arm A deactivation protocol, inward-tail hooks at -120/-110/-100
#     after +50 x 2 s (-90/-80 straddle E_rev -88.33 mV, signal ~ 0, not used);
#     arm B sine protocol, -120 rebound after the +40 segment (single-sample
#     down-step > 50 mV; one before and one after the chirp; the pre/post
#     comparison incidentally probes whether history changes time constants;
#     registered only, not entering the verdict).
#   Verdict (pre-registered): per voltage per tau, cross-cell CV < 0.3 constant;
#     median tau_d/tau_r > 2 separable; arm B vs arm A tau_rec(-120) per-cell
#     relative-difference median < 0.3 cross-protocol consistent; all three pass
#     => recovery = discrete constants, table-form extension prerequisite holds;
#     CV 0.3-0.5 weakly constant (register); > 0.5 or inseparable => prerequisite
#     fails, open a new card.
#   QC: |A| > 5 sigma and RMS < max(2 sigma, 2%|A|) and tau not pinned at grid
#     edge (ms-scale rise-edge sampling misalignment injects super-noise residuals,
#     so an absolute-sigma gate does not apply; this verdict targets tau stability,
#     not whitening).
#   Controls: C1 known DoE (tau_r=8 ms, tau_d=80 ms) recovery error > 15% voids
#     the run; C2 pure decay (no recovery) must pin tau_r at the lower grid edge
#     (no false finite recovery).
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
HOOK_GEARS = {-120: 0.30, -110: 0.50, -100: 0.80}   # voltage -> fit window length (s)
SKIP_S = 0.0005                                      # 0.5 ms let-pass after step (capacitive spike)
SEP_MIN = 1.5                                        # grid constraint tau_d >= 1.5 tau_r
SEP_PASS = 2.0                                       # separability criterion (median ratio)
CV_PASS = 0.3
AMP_QC = 5.0                                         # peak amplitude > 5 sigma

TR_GRID = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))     # 1–60 ms
TD_GRID = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))     # 8ms–2s


def load_mat(proto, cell, tag):
    V = sio.loadmat(f"{DATA}/data/protocols/{proto}")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/{tag}_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def find_tails(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    segs = np.split(np.arange(len(V)), edges)
    info = [(float(V[s[0]]), int(s[0]), len(s)) for s in segs]
    tails = []
    for k, (v, s0, n) in enumerate(info):
        if n * DT < 2.0 or k < 1:
            continue
        pv, ps, pn = info[k - 1]
        if pv > -10.0 and 0.2 < pn * DT < 5.0:
            tails.append(dict(v=int(round(v)), start=s0, n=n))
    return tails


def noise_sigma(I_segs):
    """Median detrended std of the silent segment (-80, full length) -> per-cell sigma (10 kHz full bandwidth)"""
    sigs = []
    for seg in I_segs:
        if len(seg) < 2000:
            continue
        t = np.arange(len(seg))
        tr = np.polyfit(t, seg, 1)
        sigs.append(float(np.std(seg - np.polyval(tr, t))))
    return float(np.median(sigs)) if sigs else np.nan


def quiet_segs(V, I):
    edges = np.where(np.diff(V) != 0)[0] + 1
    out = []
    for s in np.split(np.arange(len(V)), edges):
        if len(s) * DT >= 0.15 and abs(float(V[s[0]]) + 80.0) < 2.0:
            out.append(I[s[0]:s[-1] + 1].astype(float))
    return out


def doe_fit(t, y, tr_grid=TR_GRID, td_grid=TD_GRID):
    """Deterministic grid DoE fit + one local refinement. Returns dict or None (insufficient data)"""
    if len(t) < 50:
        return None

    def scan(trg, tdg):
        best = None
        for tr in trg:
            td_ok = tdg[tdg >= SEP_MIN * tr]
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

    b = scan(tr_grid, td_grid)
    if b is None:
        return None
    _, tr0, td0, _ = b
    trg = tr0 * np.exp(np.linspace(-0.35, 0.35, 9))
    tdg = td0 * np.exp(np.linspace(-0.35, 0.35, 9))
    b = scan(trg, tdg)
    sse, tr, td, sol = b
    res = y - (sol[0] + sol[1] * (np.exp(-t / td) - np.exp(-t / tr)))
    return dict(tau_r=float(tr), tau_d=float(td), c=float(sol[0]), A=float(sol[1]),
                rms=float(np.sqrt(np.mean(res ** 2))),
                edge=bool(tr <= TR_GRID[0] * 1.02 or tr >= TR_GRID[-1] * 0.98))


def qc_ok(r, sigma):
    return bool(abs(r["A"]) > AMP_QC * sigma
                and r["rms"] < max(2.0 * sigma, 0.02 * abs(r["A"]))
                and not r["edge"])


def fit_hook(I, s0, win_s, sigma):
    n = int(win_s / DT)
    y = I[s0 + int(SKIP_S / DT): s0 + n].astype(float)
    t = np.arange(len(y)) * DT
    half = y[:len(y) // 2]
    sgn = -1.0 if abs(half.min()) > abs(half.max()) else 1.0
    yy = sgn * y
    r = doe_fit(t, yy)
    if r is None:
        return None
    r["valid"] = qc_ok(r, sigma)
    r["sgn"] = sgn
    r["t"] = t
    r["y"] = yy
    return r


def sine_rebounds(V, I):
    """Edge with single-sample down-step > 50 mV landing below -100 mV (one before/after the chirp; continuous chirp does not trigger)"""
    dV = np.diff(V)
    idx = np.where((V[1:] < -100.0) & (V[:-1] > -40.0) & (dV < -50.0))[0]
    out = []
    for k, i in enumerate(idx):
        s0 = i + 1
        n = int(0.15 / DT)
        if s0 + n > len(I):
            continue
        out.append((k, s0, I[s0:s0 + n]))
    return out


def main():
    rng = np.random.default_rng(7)
    print("=" * 78)
    print(" rebound-hook inactivation-recovery constant verdict" + (" (smoke)" if SMOKE else " (nine cells full)"))
    print(f" criteria: CV<{CV_PASS} constant | median tau_d/tau_r>{SEP_PASS} separable | cross-protocol relative difference <0.3")
    print("=" * 78, flush=True)

    # ---------- controls ----------
    print("\n[controls]", flush=True)
    t_c = np.arange(int(0.3 / DT)) * DT
    y1 = 1.0 * (np.exp(-t_c / 0.08) - np.exp(-t_c / 0.008)) + 0.02 + rng.normal(0, 0.05, len(t_c))
    r1 = doe_fit(t_c, y1)
    e_r = abs(r1["tau_r"] - 0.008) / 0.008
    e_d = abs(r1["tau_d"] - 0.08) / 0.08
    print(f"  C1 DoE(8ms,80ms): τ_r={r1['tau_r']*1000:.1f}ms τ_d={r1['tau_d']*1000:.1f}ms "
          f"error {e_r*100:.1f}%/{e_d*100:.1f}%", flush=True)
    y2 = 1.0 * np.exp(-t_c / 0.08) + 0.02 + rng.normal(0, 0.05, len(t_c))
    r2 = doe_fit(t_c, y2)
    c2_ok = r2["tau_r"] <= 0.002 and abs(r2["tau_d"] - 0.08) / 0.08 < 0.15
    print(f"  C2 pure decay (80 ms): tau_r={r2['tau_r']*1000:.2f} ms (should pin at lower edge <=2 ms) tau_d={r2['tau_d']*1000:.1f} ms", flush=True)
    if max(e_r, e_d) > 0.15 or not c2_ok:
        print("  controls not seated -> statistics void, halt.")
        return
    print("  controls passed.", flush=True)

    # ---------- arm A: deactivation hook ----------
    print("\n[arm A] deactivation +50 then -120/-110/-100 tail hooks", flush=True)
    rows = []
    curves = {}
    for cell in CELLS:
        V, I = load_mat("deactivation_protocol.mat", cell, "deactivation")
        if I is None:
            print(f"  cell {cell}: file missing")
            continue
        sigma = noise_sigma(quiet_segs(V, I))
        for tl in find_tails(V):
            if tl["v"] not in HOOK_GEARS:
                continue
            r = fit_hook(I, tl["start"], HOOK_GEARS[tl["v"]], sigma)
            if r is None:
                continue
            rows.append(dict(cell=cell, v=tl["v"], tau_r=r["tau_r"], tau_d=r["tau_d"],
                             A=r["A"], c=r["c"], rms=r["rms"], valid=r["valid"], sigma=sigma))
            curves[(cell, tl["v"])] = r
        nv = sum(x["valid"] for x in rows if x["cell"] == cell)
        print(f"  cell {cell}: sigma={sigma*1000:.0f} pA  valid hooks {nv}/{len(HOOK_GEARS)}", flush=True)

    # ---------- arm B: sine rebound ----------
    print("\n[arm B] sine -120 rebound (one before/after the chirp)", flush=True)
    srows = []
    scurves = {}
    for cell in CELLS:
        V, I = load_mat("sine_wave_protocol.mat", cell, "sine_wave")
        if I is None:
            print(f"  cell {cell}: sine missing")
            continue
        sigma = noise_sigma(quiet_segs(V, I))
        for k, s0, seg in sine_rebounds(V, I):
            t = np.arange(len(seg)) * DT
            yy = -seg.astype(float)                     # inward -> flip positive
            r = doe_fit(t[int(SKIP_S / DT):], yy[int(SKIP_S / DT):])
            if r is None:
                continue
            valid = qc_ok(r, sigma)
            srows.append(dict(cell=cell, which=k, tau_r=r["tau_r"], tau_d=r["tau_d"],
                              A=r["A"], rms=r["rms"], valid=valid, sigma=sigma))
            scurves[(cell, k)] = (t, yy, r)
            tag = "pre-chirp" if k == 0 else f"post-chirp{k}"
            print(f"  cell {cell} {tag}: tau_r={r['tau_r']*1000:.2f} ms tau_d={r['tau_d']*1000:.1f} ms "
                  f"A={r['A']:.2f} nA {'pass' if valid else 'drop'}", flush=True)

    # ---------- constancy statistics ----------
    print("\n" + "-" * 78)
    print("[constancy] arm A cross-cell (QC-valid only; judged only when n>=3)")
    print(f"  {'V':>6} {'n':>3} | {'tau_rec med':>10} {'CV':>6} | {'tau_deact med':>10} {'CV':>6} | {'tau_d/tau_r':>8} verdict")
    summ = {}
    verdicts = []
    for vv in sorted(HOOK_GEARS):
        rs = [r for r in rows if r["v"] == vv and r["valid"]]
        if len(rs) < 3:
            print(f"  {vv:>6} {len(rs):>3} | insufficient valid")
            continue
        tr = np.array([r["tau_r"] for r in rs])
        td = np.array([r["tau_d"] for r in rs])
        ratio = float(np.median(td / tr))
        cv_r = float(tr.std(ddof=1) / tr.mean())
        cv_d = float(td.std(ddof=1) / td.mean())
        ok = cv_r < CV_PASS and cv_d < CV_PASS and ratio > SEP_PASS
        verdicts.append(ok)
        summ[vv] = dict(n=len(rs), tr_med=float(np.median(tr)), cv_r=cv_r,
                        td_med=float(np.median(td)), cv_d=cv_d, ratio=ratio, ok=ok)
        print(f"  {vv:>6} {len(rs):>3} | {np.median(tr)*1000:>8.1f}ms {cv_r:>6.2f} | "
              f"{np.median(td)*1000:>8.1f}ms {cv_d:>6.2f} | {ratio:>8.1f} {'constant' if ok else 'not constant'}")

    # cross-protocol consistency (-120: arm A vs arm B pre-chirp)
    cross = []
    for cell in CELLS:
        a = [r for r in rows if r["cell"] == cell and r["v"] == -120 and r["valid"]]
        b = [r for r in srows if r["cell"] == cell and r["which"] == 0 and r["valid"]]
        if a and b:
            cross.append(abs(a[0]["tau_r"] - b[0]["tau_r"]) / (0.5 * (a[0]["tau_r"] + b[0]["tau_r"])))
    med_cross = float(np.median(cross)) if cross else np.nan
    if cross:
        print(f"\n[cross-protocol] tau_rec(-120) arm A vs arm B per-cell relative-difference median: {med_cross:.2f} (n={len(cross)}, criterion <0.3)")
    else:
        print("\n[cross-protocol] insufficient comparable cells")

    # chirp post vs pre registration (not entering the verdict)
    pre = [r for r in srows if r["which"] == 0 and r["valid"]]
    post = [r for r in srows if r["which"] >= 1 and r["valid"]]
    if pre and post:
        print(f"[register] post-chirp vs pre: tau_r median {np.median([r['tau_r'] for r in pre])*1000:.2f}"
              f"->{np.median([r['tau_r'] for r in post])*1000:.2f}ms，"
              f"A median {np.median([r['A'] for r in pre]):.2f}->{np.median([r['A'] for r in post]):.2f} nA")

    # ---------- overall verdict ----------
    print("\n" + "=" * 78)
    print(" overall verdict:")
    n_ok = sum(verdicts)
    print(f"  arm A constant voltage levels {n_ok}/{len(summ)}; cross-protocol relative-difference median {med_cross:.2f}")
    if n_ok == len(summ) and len(summ) >= 2 and (not np.isnan(med_cross)) and med_cross < 0.3:
        final = "inactivation recovery = discrete constants and cross-protocol consistent -> table-form extension prerequisite holds, proceed (extended model covers sine/AP)"
    elif n_ok >= 1:
        final = "some levels constant: constant levels enter the table, the rest investigated separately (segmented extension feasible)"
    else:
        final = "inactivation recovery is not a discrete constant (or not separable) -> table-form extension prerequisite fails, open a new card"
    print("  " + final)
    print("=" * 78)

    # ---------- figure ----------
    # page 1: arm A hook overlay, 9 cells x 3 levels
    fig, axes = plt.subplots(len(CELLS), 3, figsize=(15, 1.9 * len(CELLS)), squeeze=False)
    for i, cell in enumerate(CELLS):
        for j, vv in enumerate(sorted(HOOK_GEARS)):
            ax = axes[i][j]
            r = curves.get((cell, vv))
            if r is None:
                ax.axis("off"); continue
            dn = max(1, int(0.0005 / DT))
            ax.plot(r["t"][::dn] * 1000, r["y"][::dn], ".", ms=1.5, color="black", alpha=0.5)
            yf = r["c"] + r["A"] * (np.exp(-r["t"] / r["tau_d"]) - np.exp(-r["t"] / r["tau_r"]))
            ax.plot(r["t"] * 1000, yf, "-", lw=1.3, color="crimson")
            ax.set_title(f"{cell[-4:]} @{vv}mV τ_r={r['tau_r']*1000:.1f}ms τ_d={r['tau_d']*1000:.0f}ms "
                         f"{'pass' if r['valid'] else 'drop'}", fontsize=8)
            ax.grid(alpha=0.3)
    fig.suptitle("arm A deactivation hook: raw vs DoE fit (rise = inactivation relief, decay = deactivation)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fp1 = os.path.join(HERE, f"2026-09-13_α模型_反弹hook_A路{'_冒烟' if SMOKE else ''}.png")
    fig.savefig(fp1, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # page 2: arm B rebound overlay + tau(V) summary
    fig, axes = plt.subplots(2, max(3, int(np.ceil(len(CELLS) / 2)) + 1), figsize=(17, 7))
    axs = axes.ravel()
    ax = axs[0]
    plotted = False
    for vv in sorted(HOOK_GEARS):
        if vv in summ:
            s = summ[vv]
            ax.errorbar([vv], [s["tr_med"] * 1000], yerr=s["tr_med"] * 1000 * s["cv_r"],
                        fmt="o", ms=7, capsize=4, color="crimson",
                        label="τ_rec" if not plotted else None)
            ax.errorbar([vv], [s["td_med"] * 1000], yerr=s["td_med"] * 1000 * s["cv_d"],
                        fmt="s", ms=7, capsize=4, color="navy",
                        label="τ_deact" if not plotted else None)
            plotted = True
    ax.set_yscale("log"); ax.set_xlabel("mV"); ax.set_ylabel("τ (ms)")
    ax.set_title("tau_rec / tau_deact(V) (median +/- CV*median)", fontweight="bold")
    if plotted:
        ax.legend()
    ax.grid(alpha=0.3, which="both")
    for i, cell in enumerate(CELLS):
        ax = axs[i + 1]
        got = False
        for k, lbl, clr in ((0, "pre-chirp", "crimson"), (1, "post-chirp", "darkorange")):
            key = (cell, k)
            if key not in scurves:
                continue
            t, yy, r = scurves[key]
            i0 = int(SKIP_S / DT)
            dn = 5
            ax.plot(t[i0::dn] * 1000, yy[i0::dn], ".", ms=1.5, color="black", alpha=0.4)
            yf = r["c"] + r["A"] * (np.exp(-t / r["tau_d"]) - np.exp(-t / r["tau_r"]))
            ax.plot(t[i0:] * 1000, yf[i0:], "-", lw=1.2, color=clr, label=lbl)
            got = True
        if got:
            ax.legend(fontsize=7)
        ax.set_title(f"{cell[-4:]} sine -120 rebound", fontsize=8)
        ax.grid(alpha=0.3)
    for k in range(len(CELLS) + 1, len(axs)):
        axs[k].axis("off")
    fig.suptitle("arm B sine -120 rebound: pre-chirp vs post-chirp (flipped positive)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp2 = os.path.join(HERE, f"2026-09-13_α模型_反弹hook_B路{'_冒烟' if SMOKE else ''}.png")
    fig.savefig(fp2, dpi=120, bbox_inches="tight")
    plt.close(fig)

    out = dict(summary=summ, med_cross=med_cross, final=final,
               A_rows=[{k: v for k, v in r.items() if k not in ("t", "y")} for r in rows],
               B_rows=srows, figA=fp1, figB=fp2)
    fj = os.path.join(HERE, f"2026-09-13_α模型_反弹hook{'_冒烟' if SMOKE else ''}_结果.json")
    json.dump(out, open(fj, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n  figure saved: {fp1}")
    print(f"           {fp2}")
    print(f"  results saved: {fj}")


if __name__ == "__main__":
    main()
