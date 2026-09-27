# -*- coding: utf-8 -*-
# 2026-09-13_slow-layer_discrete-constant_verdict_v5_noise-envelope-gate.py
# v4 calibration hard fact: in real silent segments the noise LB rejection rates are 20ms=100%, 50ms=89%, 100ms=89%;
#   the noise is strongly correlated at all binning scales -- every white-noise gate of v1-v3 was skewed, the v3 "structure veto" is void.
# v5's gate: no white-noise assumption. Use the measured noise ACF envelope from 9 silent segments (per lag take the second-largest of 9,
#   floor 0.2); a tail-fit residual whose ACF violates the envelope <=2 times in 20 lags is judged white.
#   Residual correlation not exceeding the noise's own correlation = the model leaves no resolvable structure.
# Everything else unchanged: anchored triple exponential (fast[0.03,0.6] mid[0.4,3] slow[3,60]s), late-window slope tau_late,
#   constancy CV<0.30 and range ratio<2.0; control C1 must land back, C2 registered as-is.
# Run: python this_file (your machine, about 3-5 minutes)
import os
import json
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ["16704007", "16704047", "16707014", "16708016", "16708060",
         "16708118", "16713003", "16713110", "16715049"]
DT = 1e-4
DS = 10
SKIP_MS = 5.0
MASK_NA = 0.072
DEBOUNCE = 20
BASE_MS = 200.0
T0_S = 0.05
BIN_S = 0.02
WIN_MIN_S = 2.0
CV_TOL = 0.30
RANGE_TOL = 2.0
BANDS = [(0.03, 0.6), (0.4, 3.0), (3.0, 60.0)]
ANCHORS = [0.15, 1.0, 15.0]
SYN_SIGMA = 0.036
H_ACF = 20
VIOL_TOL = 2
ENV = None          # noise ACF envelope, set after calibration in main


def load(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/deactivation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/deactivation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None, "file missing"
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    diag = f"lenI={len(I)} lenV={len(V)} NaN={int(np.isnan(I).sum())}"
    n = min(len(I), len(V))
    return V[:n], I[:n], diag


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


def lb_pvalue(resid, h=20):
    n = len(resid)
    h = max(2, min(h, n // 4))
    r = resid - resid.mean()
    denom = float(np.sum(r ** 2))
    if denom <= 0:
        return 1.0
    rk = [float(np.sum(r[k:] * r[:-k]) / denom) for k in range(1, h + 1)]
    Q = n * (n + 2) * sum(rk[k - 1] ** 2 / (n - k) for k in range(1, h + 1))
    return float(1 - chi2.cdf(Q, h))


def build_envelope(all_noise, rng):
    """Silent-segment ACF envelope: per lag take the second-largest |acf| of the 9 segments, floor 0.2"""
    global ENV
    print("\n[calibration] silent-segment noise ACF envelope (20 ms binning, linear detrending)", flush=True)
    acs = []
    for seg in all_noise:
        bm = bin_means(seg, BIN_S)
        if bm is None:
            continue
        t = np.arange(len(bm))
        tr = np.polyfit(t, bm, 1)
        acs.append(acf(bm - np.polyval(tr, t), H_ACF))
    print(f"  silent segments {len(acs)}, about {len(bm)} bins each", flush=True)
    A = np.abs(np.array(acs))
    mean_acf = np.mean(np.array(acs), axis=0)
    print(f"  noise ACF mean r1={mean_acf[0]:.2f} r2={mean_acf[1]:.2f} r3={mean_acf[2]:.2f} "
          f"r5={mean_acf[4]:.2f} r10={mean_acf[9]:.2f} r20={mean_acf[19]:.2f}", flush=True)
    env = np.sort(A, axis=0)[-2] if len(acs) >= 2 else A[0]
    ENV = np.maximum(env, 0.2)
    print(f"  envelope (per-lag second-largest of 9, floor 0.2): E1={ENV[0]:.2f} E5={ENV[4]:.2f} "
          f"E10={ENV[9]:.2f} E20={ENV[19]:.2f}", flush=True)
    bm0 = bin_means(all_noise[0], BIN_S) if all_noise else None
    n_bm = len(bm0) if bm0 is not None else 0
    syn_viol = []
    for _ in range(10):
        w = rng.normal(0, 1, max(n_bm, 30))
        syn_viol.append(int(np.sum(np.abs(acf(w, H_ACF)) > ENV)))
    print(f"  synthetic white-noise violation count {syn_viol} (should be mostly <={VIOL_TOL}, proving the gate does not overkill)", flush=True)


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
    sig_raw = 1.4826 * float(np.median(np.abs(base_seg - base)))
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
    n_per_bin = max(1, int(round(BIN_S / dt_s)))
    nbin = max(8, int((tt[-1] - tt[0]) / BIN_S))
    bt, by = [], []
    for b in np.array_split(np.arange(len(tt)), nbin):
        bt.append(float(np.median(tt[b])))
        by.append(float(np.mean(xx[b])))
    return np.array(bt), np.array(by), tl["v"], sig_raw / np.sqrt(n_per_bin)


def fit_banded(bt, by, sig_bin):
    best = None
    amp0 = max(by[0], 0.05)
    for jit in (0.5, 2.0):
        x0 = [float(np.median(by[-10:])), amp0 / 3, amp0 / 3, amp0 / 3]
        for (lo, hi), a0 in zip(BANDS, ANCHORS):
            x0.append(min(max(a0 * jit, lo * 1.05), hi * 0.95))
        lb = [-1.0, -5.0, -5.0, -5.0] + [b[0] for b in BANDS]
        ub = [1.0, 5.0, 5.0, 5.0] + [b[1] for b in BANDS]
        def resid(th):
            return (th[0] + th[1] * np.exp(-bt / th[4])
                    + th[2] * np.exp(-bt / th[5]) + th[3] * np.exp(-bt / th[6])) - by
        try:
            r = least_squares(resid, x0, bounds=(lb, ub), max_nfev=800)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    if best is None:
        return None
    th = best.x
    taus = [float(th[4]), float(th[5]), float(th[6])]
    res = resid(th)
    white, viol = white_by_envelope(res)
    p_lb = lb_pvalue(res, 20)
    rms_ratio = float(np.sqrt(np.mean(res ** 2)) / max(sig_bin, 1e-12))
    edge = [bool(t <= lo * 1.02 or t >= hi * 0.98) for t, (lo, hi) in zip(taus, BANDS)]
    return dict(taus=taus, lb_p=p_lb, white=bool(white), viol=viol,
                rms_ratio=rms_ratio, edge=edge)


def tau_late(bt, by, sig_bin):
    n = len(bt)
    iL = int(n * 0.6)
    if n - iL < 8 or bt[-1] - bt[iL] < 1.0:
        return np.nan
    m = by[iL:] > 3 * sig_bin
    if m.sum() < 8:
        return np.nan
    slope = float(np.polyfit(bt[iL:][m], np.log(by[iL:][m]), 1)[0])
    if slope > -1e-4:
        return np.nan
    return -1.0 / slope


def bin_series(t, y):
    i0 = int(T0_S / (t[1] - t[0]))
    tt, xx = t[i0:], y[i0:]
    nbin = max(8, int((tt[-1] - tt[0]) / BIN_S))
    bt, by = [], []
    for b in np.array_split(np.arange(len(tt)), nbin):
        bt.append(float(np.median(tt[b])))
        by.append(float(np.mean(xx[b])))
    return np.array(bt), np.array(by)


def synth_check(rng):
    t = np.arange(0.05, 5.5, DT * DS)
    sig_bin = SYN_SIGMA / np.sqrt(BIN_S / (DT * DS))
    out = {}
    for name, y in (("C1_离散三指数", 0.3 * np.exp(-t / 0.15) + 0.5 * np.exp(-t / 1.0) + 0.4 * np.exp(-t / 15.0)),
                    ("C2_连续拉伸b0.5", 0.9 * np.exp(-(t / 2.0) ** 0.5))):
        yy = y + rng.normal(0, SYN_SIGMA, len(t))
        bt, by = bin_series(t, yy)
        f = fit_banded(bt, by, sig_bin)
        tl = tau_late(bt, by, sig_bin)
        out[name] = dict(fit=f, tau_late=tl)
    return out


def fmt(x, nd=2):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    rng = np.random.default_rng(23)
    print("=" * 76)
    print(" slow-layer discrete-constant verdict v5 · noise-envelope gate (no white-noise assumption)")
    print("=" * 76, flush=True)

    all_noise = []
    for cell in CELLS:
        V, I, _ = load(cell)
        if I is not None:
            all_noise.extend(noise_segments(V, I))
    build_envelope(all_noise, rng)

    print("\n[controls]", flush=True)
    cc = synth_check(rng)
    for name, r in cc.items():
        f = r["fit"]
        if f is None:
            print(f"  {name}: fit failed")
            continue
        print(f"  {name}: τ=({fmt(f['taus'][0],3)}, {fmt(f['taus'][1],3)}, {fmt(f['taus'][2],2)})s "
              f"tau_late={fmt(r['tau_late'],1)}s violations={f['viol']} LBp={f['lb_p']:.3f} edge={f['edge']}")
    c1 = cc.get("C1_离散三指数", {}).get("fit")
    c1_tl = cc.get("C1_离散三指数", {}).get("tau_late")
    c1ok = (c1 is not None and c1["white"]
            and 0.08 <= c1["taus"][0] <= 0.30
            and 0.5 <= c1["taus"][1] <= 2.0
            and c1_tl is not None and not np.isnan(c1_tl) and 8.0 <= c1_tl <= 30.0)
    if not c1ok:
        print("\n  control C1 did not land back -> statistics void, stop.")
        return
    print("  control C1 pass. C2 registered on record (discrete-vs-continuous separability limit written into the verdict as-is).")

    per_volt = {}
    ratios = []
    n_white = n_tot = 0
    edge_hits = [0, 0, 0]
    print("\n[real data]", flush=True)
    for cell in CELLS:
        V, I, diag = load(cell)
        if I is None:
            print(f"  cell {cell}: {diag}")
            continue
        marks = []
        for tl in find_tails(V):
            tb = tail_binned(I, tl)
            if tb is None:
                continue
            bt, by, vv, sig_bin = tb
            f = fit_banded(bt, by, sig_bin)
            if f is None:
                continue
            tl_late = tau_late(bt, by, sig_bin)
            n_tot += 1
            ratios.append(f["rms_ratio"])
            for bi in range(3):
                edge_hits[bi] += int(f["edge"][bi])
            marks.append(f"{vv:+.0f}:{'✓' if f['white'] else '×'}{f['viol']}")
            if not f["white"]:
                continue
            n_white += 1
            d = per_volt.setdefault(round(vv), dict(tau1=[], tau2=[], tau3=[], tau_late=[], cells=[]))
            d["tau1"].append(f["taus"][0])
            d["tau2"].append(f["taus"][1])
            d["tau3"].append(f["taus"][2])
            if not np.isnan(tl_late):
                d["tau_late"].append(tl_late)
            d["cells"].append(cell)
        print(f"  cell {cell}: {' '.join(marks)}  [{diag}]", flush=True)

    med_ratio = float(np.median(ratios)) if ratios else float("nan")
    print(f"\n  whitening rate {n_white}/{n_tot} (envelope gate); residual RMS/sigma median {med_ratio:.2f}; edge hits fast/mid/slow = {edge_hits}")

    print("\n" + "-" * 76)
    print(f"[constancy] cross-cell per voltage band (judged only at n>=4; CV<{CV_TOL} and max/min<{RANGE_TOL} = constant; tau_slow diagnostic only)")
    print(f"  {'V':>6} {'n':>3} | {'τ_f':>7} {'CV':>5} {'ratio':>5} | {'τ_m':>7} {'CV':>5} {'ratio':>5} | {'τ_s':>7} {'CV':>5} {'ratio':>5} | {'τ_late':>7} {'CV':>5}")
    gear_pass = gear_tot = 0
    tl40_cv = None
    table = []
    for v in sorted(per_volt):
        d = per_volt[v]
        line = f"  {v:>6} {len(d['cells']):>3} |"
        ok = tot = 0
        row = dict(v=v, n=len(d["cells"]))
        for key in ("tau1", "tau2", "tau3"):
            arr = np.array(d[key], float)
            if len(arr) >= 4:
                cv = float(arr.std(ddof=1) / arr.mean())
                rr = float(arr.max() / max(arr.min(), 1e-9))
                row[key] = dict(mean=float(arr.mean()), cv=cv, range_ratio=rr)
                good = cv < CV_TOL and rr < RANGE_TOL
                if key != "tau3":
                    ok += int(good)
                    tot += 1
                line += f" {arr.mean():>7.3f} {cv:>5.2f} {rr:>5.1f} |"
            else:
                line += f" {'--':>7} {'--':>5} {'--':>5} |"
        arr = np.array(d["tau_late"], float)
        arr = arr[~np.isnan(arr)]
        if len(arr) >= 4:
            cvl = float(arr.std(ddof=1) / arr.mean())
            row["tau_late"] = dict(mean=float(arr.mean()), cv=cvl)
            ok += int(cvl < CV_TOL)
            tot += 1
            if v == -40:
                tl40_cv = cvl
            line += f" {arr.mean():>7.1f} {cvl:>5.2f}"
        else:
            line += f" {'--':>7} {'--':>5}"
        passed = tot >= 2 and ok == tot
        gear_tot += 1
        gear_pass += int(passed)
        row["pass"] = passed
        table.append(row)
        print(line + ("  const" if passed else ""))

    frac = gear_pass / max(gear_tot, 1)
    c2 = cc.get("C2_连续拉伸b0.5", {}).get("fit")
    c2white = bool(c2 and c2["white"])
    print("\n" + "=" * 76)
    print(" overall verdict:")
    print(f"  whitening {n_white}/{n_tot} (envelope gate); constant bands {gear_pass}/{gear_tot}; -40 tau_late CV {fmt(tl40_cv,2)}")
    if n_white < n_tot * 0.5:
        final = ("结构否决（噪声相关已按真实基线包络排除，这次算数）：带锚三指数装不下长尾，"
                 "'离散常数'图景否掉；建模只剩逐细胞描述 + τ_rec 等已确证常数")
    elif frac >= 2 / 3 and tl40_cv is not None and tl40_cv < CV_TOL:
        final = ("常数图景成立（范围限定版）：每电压带内 τ + τ_late 跨细胞恒定 -> "
                 "建模 = 每电压 τ 表 + 幅度表直提")
        if c2white:
            final += "；限制：连续谱同样拟白（C2），离散vs连续不可分，恒定的是'有效时间尺度'"
    elif frac >= 2 / 3:
        final = ("半常数：窗内模态恒定，超慢层跨细胞不稳 -> 普适模型到 1s 尺度，超慢层逐细胞建表")
    else:
        final = ("常数图景不成立：带内 τ 跨细胞漂移 -> 只剩逐细胞描述")
    print("  " + final)
    print("=" * 76)

    vs = sorted(per_volt)
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    for ax, key, ttl in ((axes[0], "tau1", "τ_fast(V)"),
                         (axes[1], "tau2", "τ_mid(V)"),
                         (axes[2], "tau3", "tau_slow(V) within band (diagnostic)"),
                         (axes[3], "tau_late", "tau_late(V) late-window slope")):
        means = [np.nanmean(per_volt[v][key]) if per_volt[v][key] else np.nan for v in vs]
        stds = [np.nanstd(per_volt[v][key], ddof=1) if len(per_volt[v][key]) > 1 else np.nan for v in vs]
        ax.errorbar(vs, means, yerr=stds, fmt="o-", ms=5, lw=1.5, capsize=3, color="steelblue")
        ax.set_yscale("log")
        ax.set_title(ttl, fontweight="bold")
        ax.set_xlabel("tail voltage mV")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("τ (s)")
    ax = axes[4]
    ax.axis("off")
    lines = ["离散常数判决 v5（噪声包络门）", "",
             f"白化: {n_white}/{n_tot}  RMS/σ中位 {fmt(med_ratio,2)}",
             f"恒定档: {gear_pass}/{gear_tot}", f"-40 τ_late CV: {fmt(tl40_cv,2)}", "", final]
    y0 = 0.97
    for L in lines:
        ax.text(0.02, y0, L, fontsize=9.5,
                fontweight="bold" if (L.startswith("离散") or L == final) else "normal", wrap=True)
        y0 -= 0.09
    fpng = os.path.join(HERE, "2026-09-13_慢层离散常数判决v5.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")

    out = dict(envelope=ENV.tolist(),
               controls={k: dict(fit=v["fit"], tau_late=v["tau_late"]) for k, v in cc.items()},
               n_white=n_white, n_tot=n_tot, med_rms_ratio=med_ratio, edge_hits=edge_hits,
               table=table, gear_pass=gear_pass, gear_tot=gear_tot, tl40_cv=tl40_cv, final=final)
    fjson = os.path.join(HERE, "2026-09-13_slow_layer_discrete_const_verdictv5_results.json")
    json.dump(out, open(fjson, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n  figure saved: {fpng}")
    print(f"  results saved: {fjson}")


if __name__ == "__main__":
    main()
