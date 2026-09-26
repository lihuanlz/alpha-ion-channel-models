# -*- coding: utf-8 -*-
# 2026-09-13_alpha model_forward validation_tails and sine.py
# Purpose: forward validation of the alpha model (tau(V) ladder + a(V) amplitudes, sealed 2026-09-13), two parts:
#   Part A tail reconstruction: sealed taus + per-cell linear amplitudes, reconstruct the deactivation
#           long tails, raw vs fitted overlay. (Same convention as the amplitude table extraction,
#   Part B sine prediction: wire the sealed tables into a relaxation dynamical system and make a
#           zero-fit prediction on the independent sine_wave protocol. Dynamics (current units, no G_Kr anchor):
#             dJ_i/dt = ( f_i(V)·g(V)·(V-E_rev) − J_i ) / τ_i(V)，I = Σ J_i
#             g(V)   = steady_activation per-cell measured I_ss(V)/(V-E_rev) (table lookup, interpolated)
#             f_i(V) = sealed w-table interpolation (-70..-40); extrapolation rules below
#             tau_late(V) = four-point ladder log-linear extrapolation (capped 500s); tau_f=0.234s, tau_m=1.009s constants
#                        (only -70/-60 measured; making them constant directly applies the "constant" seal verdict)
#           Null model for comparison: I_null = g(V)*(V-E_rev) (instantaneous steady state, no dynamics).
#           Criterion: the dynamic model RMS must be significantly smaller than the null model, otherwise the dynamics earns nothing.
#   Known extrapolation assumptions (written into the output as-is):
#     A1: tau_late extrapolated as frozen at V>-40 (>100s); fast activation/inactivation at positive
#     A2: f_i(V) extrapolated at V>-40 (w_s->0.95) is a trend extrapolation, no measured anchor;
#     A3: the -60/-40 points of g(V) are 5-6s steps not fully at steady state (tau_late 4.4s/24s), so negative-voltage g is underestimated;
#     A4: tau_f/tau_m constant across all voltages, measured only at -70/-60.
# Run: python this file        (full nine-cell set)
#       SMOKE=1 python this file (single-cell 16713003 smoke test)
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
CELLS_ALL = ["16704007", "16704047", "16707014", "16708016", "16708060",
             "16708118", "16713003", "16713110", "16715049"]
CELLS = ["16713003"] if SMOKE else CELLS_ALL
DT = 1e-4
DS = 10
SKIP_MS = 5.0
MASK_NA = 0.072
DEBOUNCE = 20
BASE_MS = 200.0
T0_S = 0.05
BIN_S = 0.02
WIN_MIN_S = 2.0
H_ACF = 20
VIOL_TOL = 2
ENV = None
E_REV = -80.0

# ---- sealed table values (2026-09-13 ladder table / amplitude table JSON, do not modify) ----
TAU_F = {-70: 0.19105, -60: 0.27645, -50: 0.6, -40: 0.6}      # -50/-40 shared refinement at grid edge (Part A reproduction only)
TAU_M = {-70: 0.81270, -60: 1.20549, -50: 1.8, -40: 0.8}
TAU_L = {-70: 2.05505, -60: 4.41849, -50: 10.14073, -40: 24.0539}
GEARS = [-70, -60, -50, -40]
# constant time scales for the sine dynamics (measured means at -70/-60)
TF_C = 0.5 * (TAU_F[-70] + TAU_F[-60])   # 0.234 s
TM_C = 0.5 * (TAU_M[-70] + TAU_M[-60])   # 1.009 s
# f_i(V) interpolation nodes (sealed w-table values; negative w_f at -40 clipped to 0 then renormalized; extrapolation capped from -30 up)
F_V = np.array([-70.0, -60.0, -50.0, -40.0, -30.0])
F_F = np.array([0.294, 0.144, 0.079, 0.000, 0.000])
F_M = np.array([0.486, 0.439, 0.379, 0.275, 0.050])
F_S = np.array([0.219, 0.418, 0.543, 0.725, 0.950])
# tau_late(V) log-linear extrapolation
_pl = np.polyfit(np.array(GEARS, float), np.log([TAU_L[v] for v in GEARS]), 1)


def tau_late(V):
    return np.clip(np.exp(_pl[0] * np.asarray(V, float) + _pl[1]), 0.01, 500.0)


# ================= Part A: tail reconstruction (same convention as the amplitude table extraction script) =================

def load_deact(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/deactivation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/deactivation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
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
        dur = n * DT
        if dur < 2.0 or k < 2:
            continue
        pv, ps, pn = info[k - 1]
        hv, hs, hn = info[k - 2]
        if pv > -10.0 and 0.2 < pn * DT < 5.0:
            b0 = hs + hn - int(BASE_MS / 1000.0 / DT)
            tails.append(dict(v=v, start=s0, n=n, base_idx=(b0, hs + hn)))
    return tails


def noise_segments(V, I):
    edges = np.where(np.diff(V) != 0)[0] + 1
    out = []
    for s in np.split(np.arange(len(V)), edges):
        if len(s) * DT >= 2.0 and abs(float(V[s[0]]) + 80.0) < 2.0:
            out.append(I[s[0]:s[-1] + 1:DS].astype(float))
    return out


def bin_means(y, bin_s, dt_s=1e-3):
    n_per = max(1, int(round(bin_s / dt_s)))
    nb = len(y) // n_per
    if nb < 8:
        return None
    return y[:nb * n_per].reshape(nb, n_per).mean(axis=1)


def acf(x, h):
    r = x - x.mean()
    d = float(np.sum(r ** 2))
    if d <= 0:
        return np.zeros(h)
    return np.array([float(np.sum(r[k:] * r[:-k])) / d for k in range(1, h + 1)])


def build_envelope(all_noise):
    global ENV
    acs = []
    for seg in all_noise:
        bm = bin_means(seg, BIN_S)
        if bm is None:
            continue
        t = np.arange(len(bm))
        tr = np.polyfit(t, bm, 1)
        acs.append(acf(bm - np.polyval(tr, t), H_ACF))
    A = np.abs(np.array(acs))
    env = np.sort(A, axis=0)[-2] if len(acs) >= 2 else A[0]
    ENV = np.maximum(env, 0.2)
    return len(acs)


def white_by_envelope(resid):
    a = np.abs(acf(resid, H_ACF))
    viol = int(np.sum(a > ENV[:len(a)]))
    return viol <= VIOL_TOL, viol


def tail_binned(I, tl):
    s0 = tl["start"] + int(SKIP_MS / 1000.0 / DT)
    y = I[s0: s0 + tl["n"]].astype(float)
    b0, b1 = tl["base_idx"]
    base_seg = I[b0:b1:DS].astype(float)
    base = float(np.median(base_seg))
    xs0 = y - base
    t_full = np.arange(len(xs0)) * DT
    t, x0 = t_full[::DS], xs0[::DS]
    early = x0[:int(0.2 / (t[1] - t[0]))]
    sgn = 1.0 if np.median(early) >= 0 else -1.0
    x = sgn * x0
    dt_s = t[1] - t[0]
    i0 = int(T0_S / dt_s)
    k = max(3, int(15.0 / 1000.0 / dt_s) | 1)
    ys = np.convolve(x, np.ones(k) / k, mode="same")
    if ys[i0] < 0.08:
        return None
    below = ys[i0:] < MASK_NA
    i1 = len(t) - 1
    if below.sum() >= DEBOUNCE:
        run = np.convolve(below.astype(int), np.ones(DEBOUNCE, int), mode="valid")
        hit = np.where(run >= DEBOUNCE)[0]
        if len(hit):
            i1 = i0 + int(hit[0])
    if (i1 - i0) * dt_s < WIN_MIN_S:
        return None
    tt, xx = t[i0:i1], x[i0:i1]
    nbin = max(8, int((tt[-1] - tt[0]) / BIN_S))
    bt, by = [], []
    for b in np.array_split(np.arange(len(tt)), nbin):
        bt.append(float(np.median(tt[b])))
        by.append(float(np.mean(xx[b])))
    return np.array(bt), np.array(by), sgn


def fit_tail(bt, by, vv):
    ef = np.exp(-bt / TAU_F[vv])
    em = np.exp(-bt / TAU_M[vv])
    es = np.exp(-bt / TAU_L[vv]) - 1.0
    X = np.column_stack([np.ones(len(bt)), ef, em, es])
    sol, *_ = np.linalg.lstsq(X, by, rcond=None)
    res = by - X @ sol
    white, viol = white_by_envelope(res)
    return dict(sol=sol, white=bool(white), viol=viol,
                rms=float(np.sqrt(np.mean(res ** 2))))


# ================= Part B: sine zero-fit prediction =================

def load_sine(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/sine_wave_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/sine_wave_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def g_of_V(cell):
    """steady_activation -> g(V)=I_ss/(V-E_rev) table; nodes [-80:0, -60..+60, held beyond +60]"""
    Vp = sio.loadmat(f"{DATA}/data/protocols/steady_activation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/steady_activation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return None, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(Vp))
    Vp, I = Vp[:n], I[:n]
    edges = np.where(np.diff(Vp) != 0)[0] + 1
    knots, gvals = [-80.0], [0.0]
    for s in np.split(np.arange(len(Vp)), edges):
        vv = float(Vp[s[0]])
        dur = len(s) * DT
        if dur >= 4.0 and vv > -70.0:          # test step (>=4s, not the -80 holding segment)
            m = s[int(0.8 * len(s)):]          # last 20% taken as steady state
            iss = float(np.mean(I[m]))
            knots.append(vv)
            gvals.append(iss / (vv - E_REV))
    order = np.argsort(knots)
    return np.array(knots)[order], np.array(gvals)[order]


def simulate_sine(Vs, knots, gvals):
    gV = np.interp(Vs, knots, gvals)
    drive = gV * (Vs - E_REV)                    # steady-state target current g(V)(V-Erev)
    f_f = np.interp(Vs, F_V, F_F)
    f_m = np.interp(Vs, F_V, F_M)
    f_s = np.interp(Vs, F_V, F_S)
    a_f = DT / TF_C
    a_m = DT / TM_C
    a_s = np.clip(DT / tau_late(Vs), 1e-9, 1.0)
    n = len(Vs)
    Jf = np.zeros(n); Jm = np.zeros(n); Js = np.zeros(n)
    jf = f_f[0] * drive[0]; jm = f_m[0] * drive[0]; js = f_s[0] * drive[0]
    Jf[0], Jm[0], Js[0] = jf, jm, js
    tf_t = f_f * drive; tm_t = f_m * drive; ts_t = f_s * drive
    for k in range(1, n):
        jf += (tf_t[k] - jf) * a_f
        jm += (tm_t[k] - jm) * a_m
        js += (ts_t[k] - js) * a_s[k]
        Jf[k], Jm[k], Js[k] = jf, jm, js
    return Jf + Jm + Js, drive                  # drive = null model (instantaneous steady state)


def main():
    print("=" * 78)
    print(" alpha model forward validation" + (" (smoke: single cell)" if SMOKE else " (full nine-cell set)"))
    print(" Part A tail reconstruction = sealed taus + per-cell linear amplitudes (reproduction check: whitening should be 28/33)")
    print(" Part B sine zero-fit prediction (extrapolation assumptions A1-A4 see file header)")
    print("=" * 78, flush=True)

    # ---------- noise envelope gate ----------
    all_noise = []
    for cell in CELLS_ALL:                        # the envelope gate always uses the full nine-cell set (same convention as the seal)
        V, I = load_deact(cell)
        if I is not None:
            all_noise.extend(noise_segments(V, I))
    nseg = build_envelope(all_noise)
    print(f"  noise envelope gate ready ({nseg} silent segments, full-set convention)", flush=True)

    # ---------- Part A ----------
    print("\n[Part A] tail reconstruction", flush=True)
    recs = []
    for cell in CELLS:
        V, I = load_deact(cell)
        if I is None:
            print(f"  cell {cell}: deactivation file missing")
            continue
        for tl in find_tails(V):
            vv = int(round(tl["v"]))
            if vv not in GEARS:
                continue
            tb = tail_binned(I, tl)
            if tb is None:
                continue
            bt, by, sgn = tb
            r = fit_tail(bt, by, vv)
            recs.append(dict(cell=cell, v=vv, bt=bt, by=by, **r))
        print(f"  cell {cell}: done", flush=True)
    n_white = sum(r["white"] for r in recs)
    n_tot = len(recs)
    rms_med = float(np.median([r["rms"] for r in recs])) if recs else np.nan
    print(f"  reproduced whitening {n_white}/{n_tot} (sealed value 28/33; mismatch = code-convention drift, halt and investigate)")
    print(f"  residual RMS median {rms_med*1000:.1f} pA")

    fig, axes = plt.subplots(len(CELLS), 4, figsize=(17, 2.1 * len(CELLS)), squeeze=False)
    for i, cell in enumerate(CELLS):
        for j, vv in enumerate(GEARS):
            ax = axes[i][j]
            r = next((r for r in recs if r["cell"] == cell and r["v"] == vv), None)
            if r is None:
                ax.axis("off")
                continue
            c, af, am, as_ = r["sol"]
            yrec = (c + af * np.exp(-r["bt"] / TAU_F[vv]) + am * np.exp(-r["bt"] / TAU_M[vv])
                    + as_ * (np.exp(-r["bt"] / TAU_L[vv]) - 1.0))
            ax.semilogy(r["bt"], r["by"], ".", ms=2.5, color="black", alpha=0.6, label="data")
            ax.semilogy(r["bt"], yrec, "-", lw=1.4, color="crimson", label="alpha model")
            ax.set_title(f"{cell[-4:]} @ {vv}mV  RMS={r['rms']*1000:.0f}pA  violations {r['viol']}",
                         fontsize=8)
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Part A tail reconstruction: raw vs alpha model (whitening {n_white}/{n_tot})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fpA = os.path.join(HERE, f"2026-09-13_α模型_前向验证_尾巴重建{'_冒烟' if SMOKE else ''}.png")
    fig.savefig(fpA, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---------- Part B ----------
    print("\n[Part B] sine zero-fit prediction", flush=True)
    print(f"  tau_f={TF_C:.3f}s tau_m={TM_C:.3f}s (constants) tau_late(V)=ladder log-linear extrapolation capped at 500s")
    srows = []
    traces = {}
    for cell in CELLS:
        Vs, Is = load_sine(cell)
        if Is is None:
            print(f"  cell {cell}: sine file missing")
            continue
        knots, gvals = g_of_V(cell)
        if knots is None:
            print(f"  cell {cell}: steady_activation missing")
            continue
        Idyn, Inull = simulate_sine(Vs, knots, gvals)
        res_d = Is - Idyn
        res_n = Is - Inull
        neg = Vs < -40.0
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
        row = dict(cell=cell,
                   rms_dyn=rms(res_d), rms_null=rms(res_n),
                   rms_dyn_neg=rms(res_d[neg]), rms_null_neg=rms(res_n[neg]),
                   rms_dyn_pos=rms(res_d[~neg]), rms_null_pos=rms(res_n[~neg]),
                   frac_neg=float(neg.mean()))
        row["ratio"] = row["rms_dyn"] / max(row["rms_null"], 1e-12)
        srows.append(row)
        traces[cell] = (Vs, Is, Idyn, Inull, knots, gvals)
        print(f"  cell {cell}: RMS dynamic {row['rms_dyn']:.3f} / null {row['rms_null']:.3f} nA "
              f"(ratio {row['ratio']:.2f})  negative-V segment {row['rms_dyn_neg']:.3f}/{row['rms_null_neg']:.3f}  "
              f"positive-V segment {row['rms_dyn_pos']:.3f}/{row['rms_null_pos']:.3f}", flush=True)

    if srows:
        fig, axes = plt.subplots(4, 3, figsize=(19, 13))
        axp = axes[0, 0]
        Vs0 = traces[srows[0]["cell"]][0]
        axp.plot(np.arange(len(Vs0)) * DT, Vs0, lw=0.6, color="darkgreen")
        axp.set_title("sine protocol V(t)", fontweight="bold")
        axp.set_ylabel("mV"); axp.grid(alpha=0.3)
        axg = axes[0, 1]
        for cell, (_, _, _, _, knots, gvals) in traces.items():
            axg.plot(knots, gvals, "o-", ms=3, lw=1, alpha=0.7, label=cell[-4:])
        axg.set_title("g(V)=I_ss/(V-E_rev) per-cell measured", fontweight="bold")
        axg.set_xlabel("mV"); axg.set_ylabel("µS")
        axg.legend(fontsize=6, ncol=3); axg.grid(alpha=0.3)
        axs = axes[0, 2]
        axs.axis("off")
        med_ratio = float(np.median([r["ratio"] for r in srows]))
        lines = ["Part B summary", "",
                 f"RMS dynamic/null median ratio: {med_ratio:.2f}",
                 "(<1 dynamics earns; >=1 dynamics wasted)", "",
                 "extrapolation assumptions A1-A4 see file header"]
        for k, L in enumerate(lines):
            axs.text(0.02, 0.92 - 0.11 * k, L, fontsize=10,
                     fontweight="bold" if k in (0, 2) else "normal")
        for idx, r in enumerate(srows):
            ax = axes[(idx + 3) // 3][(idx + 3) % 3]
            Vs, Is, Idyn, Inull, _, _ = traces[r["cell"]]
            t = np.arange(len(Vs)) * DT
            dn = 20
            ax.plot(t[::dn], Is[::dn], lw=0.5, color="black", alpha=0.75, label="data")
            ax.plot(t[::dn], Idyn[::dn], lw=0.8, color="crimson", label="alpha dynamic")
            ax.plot(t[::dn], Inull[::dn], lw=0.5, color="steelblue", alpha=0.6, ls="--", label="null model")
            ax.set_title(f"{r['cell'][-4:]}  RMS {r['rms_dyn']:.2f}/{r['rms_null']:.2f} nA "
                         f"(ratio {r['ratio']:.2f})", fontsize=8)
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)
        for k in range(len(srows) + 3, 12):
            axes[k // 3][k % 3].axis("off")
        fig.suptitle("Part B sine protocol: raw vs alpha dynamic vs null model (zero-fit prediction)", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fpB = os.path.join(HERE, f"2026-09-13_α模型_前向验证_sine{'_冒烟' if SMOKE else ''}.png")
        fig.savefig(fpB, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        fpB = None

    # ---------- verdict (fixed in advance) ----------
    print("\n" + "=" * 78)
    print(" overall verdict:")
    okA = (not SMOKE and n_white == 28 and n_tot == 33) or (SMOKE and n_tot > 0)
    print(f"  Part A: whitening reproduced {n_white}/{n_tot} -> {'consistent with the seal' if okA else 'inconsistent with the seal, check the code convention first'}")
    if srows:
        med_ratio = float(np.median([r["ratio"] for r in srows]))
        med_neg = float(np.median([r["rms_dyn_neg"] / max(r["rms_null_neg"], 1e-12) for r in srows]))
        med_pos = float(np.median([r["rms_dyn_pos"] / max(r["rms_null_pos"], 1e-12) for r in srows]))
        print(f"  Part B: RMS ratio (dynamic/null) full-trace median {med_ratio:.2f}, "
              f"negative-V segment {med_neg:.2f}, positive-V segment {med_pos:.2f}")
        if med_ratio < 0.8:
            vB = "sine full-trace prediction holds: dynamics significantly better than instantaneous steady state"
        elif med_neg < 0.8 <= med_pos:
            vB = "sine holds by voltage: dynamics earns in the negative-V segment, fails in the positive-V segment (within A1 expectation: fast activation/inactivation not in the model)"
        elif med_ratio < 1.0:
            vB = "sine weakly holds: dynamics slightly better than the null model; per-cell residual inspection needed"
        else:
            vB = "sine fails: the dynamic model is no better than the instantaneous steady-state lookup; alpha model extrapolation failed"
        print("  " + vB)
    print("=" * 78)

    out = dict(partA=dict(n_white=n_white, n_tot=n_tot, rms_med=rms_med,
                          expected="28/33 (non-smoke)"),
               partB=dict(rows=srows,
                          assumptions=["A1 tau_late frozen at V>-40", "A2 f_i trend extrapolation at V>-40",
                                       "A3 g(V) -60/-40 not fully at steady state", "A4 tau_f/tau_m constant"]),
               figA=fpA, figB=fpB)
    fj = os.path.join(HERE, f"2026-09-13_α模型_前向验证{'_冒烟' if SMOKE else ''}_结果.json")
    json.dump(out, open(fj, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n  figure A saved: {fpA}")
    if fpB:
        print(f"  figure B saved: {fpB}")
    print(f"  results saved: {fj}")


if __name__ == "__main__":
    main()
