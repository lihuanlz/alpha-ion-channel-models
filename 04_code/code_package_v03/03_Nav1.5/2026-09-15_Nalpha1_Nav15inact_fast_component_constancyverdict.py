# -*- coding: utf-8 -*-
"""
N-alpha-1: Nav1.5 inactivation fast component tau_h(V) cross-cell constancy verdict (2026-09-15)
Pre-registered item: results/pre-registration N-alpha-1 Nav1.5 inactivation fast-component constancy verdict 2026-09-15.md
(criteria pinned before run, frozen after; beta-arm implementation convention revised to v1.1:

Pipeline: noise self-calibration (C3, registered) -> controls C1/C2 (if failed, statistics void, halt)
 -> alpha arm (apparent direct readout, registered) / beta arm (full-chain forward convolution, verdict main arm) -> constancy verdict

Observation chain (sealed in prior work, silent battlefield sections A9/A11, same method as epc10z v1.26):
  EPC-10 cascade F1 6-pole Bessel 10 kHz + F2 4-pole Bessel 5 kHz (phase-normalised) + per-cell residual pole
  Rs(1-CP)*Cm first-order, whole chain ZOH-exact discrete at 25 kHz. Beta-arm model = A*[chain (x) exp(-t/tau)] + Ip
  (absolute time base); the chain contains that cell's residual pole - cell-specific elements enter the model,
  registered: the model assumes the true current step turns on instantly (finite activation ~0.1-0.3 ms not
  normalised; cross-cell common-mode direction registered); rising-edge information excluded from the window

Constancy pool: 80CP, 35C, 008/011/005; 314 batch1 listed separately. Criteria (pre-registration section 6):
  per voltage n = 3: CV < 0.3 and max/min < 2.0 -> constant; >= 5/7 levels -> H holds; 3-4 -> partial; <= 2 -> rejected.

Environment variables: SMOKE=1 smoke (single cell, 3 voltages, 10 control realisations); NREAL (control realisations, default 20).
Discipline: the judge side only ast.parse + SMOKE=1; the formal run is done by the user in Spyder:
  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-15_Nα1_Nav15失活快分量恒定性判决.py' --wdir
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

DT_MS = 0.04                       # ms（25 kHz）
DT_S = DT_MS * 1e-3
SKIP_MS = 0.15                     # post-peak head skip (convention pinned)
TAU_BND = (0.05, 3.0)              # ms, QC bounds
V_POOL = [-20, -10, 0, 10, 20, 30, 40]
V_SMOKE = [-20, 0, 20]
AMP_MIN_PA = 200.0                 # peak-amplitude floor (also requires >= 8 sigma_hold)

DATA = Path(__file__).resolve().parent.parent / "公开数据" / "Nav1.5_27193878" / "nav"
CELLS = {"008": "batch2/medium_res_data/220502_008_ch2",
         "011": "batch2/medium_res_data/220502_011_ch3",
         "005": "batch2/medium_res_data/220503_005_ch2"}
SUB_314 = "batch1/220314_001_ch1_csv"
CP = 80

rng = np.random.default_rng(20260915)


# ---------- observation chain ----------
def _cascade_modes(secs):
    """Modal expansion of the cascade H(s) = prod b_i/a_i (iterative partial fractions, distinct poles): returns (R, P), H = sum R/(s-p).
    Only low-order denominator roots per stage are needed, avoiding 11th-order polynomial roots / matrix exponentials; robust across scipy versions."""
    R = None; P = None
    for bi, ai in secs:
        c = float(np.atleast_1d(bi)[0]) / float(ai[0])     # pole-form constant b(0)/a_lead
        for p in np.roots(ai):
            if R is None:
                R = np.array([1.0 + 0j]); P = np.array([p])
            else:
                w = R / (P - p)                            # R_k/(q_k−p)
                R = np.concatenate([w, -w]); P = np.concatenate([P, np.full(len(P), p)])
        R = R * c
    return R, P


def chain_discrete(tau_res_s):
    """Full observation-chain ZOH step-invariant exact discretisation (modal expansion, same name and method as epc10z v1.26):
    residual first-order pole + F1(6p,10k) + F2(4p,5k) (phase-normalised) -> (FIR hd, [1]).
    h[0] = 0 (strictly proper + ZOH single-sample delay, consistent with the epc10z centroid convention); sum(hd) = 1 (DC) self-check."""
    secs = [([1.0], np.array([tau_res_s, 1.0]))]
    for N, fc in ((6, 1e4), (4, 5e3)):
        bb, aa = signal.bessel(N, 2 * np.pi * fc, analog=True, norm="phase")
        secs.append((bb, aa))
    R, P = _cascade_modes(secs)
    T = DT_S
    lam = np.exp(P * T)
    n = np.arange(1, 256)                                  # 10.24 ms, covering the chain memory
    hd = np.zeros(256)
    hd[1:] = np.sum((R / P)[:, None] * (lam - 1)[:, None] * lam[:, None] ** (n - 1)[None, :],
                    axis=0).real
    s = float(hd.sum())
    if abs(s - 1.0) > 1e-6:
        print(f"  [chain self-check] sum(hd)={s:.9f} deviates from 1, attribution needed", flush=True)
    hd /= s
    return hd, np.array([1.0])


def cell_meta(sub, temp=35):
    meta = pd.read_csv(DATA / sub / f"NaIV_{temp}C_{CP}CP_meta.csv")
    rs, cm = meta["rseries_Mohm"].iloc[0], meta["capacitance_pF"].iloc[0]
    return rs, cm, rs * (1 - CP / 100.0) * cm * 1e-6   # Rs(Mohm), Cm(pF), residual pole (s)


# ---------- data loading ----------
def load_naiv(sub, temp):
    df = pd.read_csv(DATA / sub / f"NaIV_{temp}C_{CP}CP.csv")
    volts = [float(c) for c in df.columns]
    n1, n2, _ = {975: (225, 500, 250), 1000: (250, 500, 250)}[len(df)]
    cur = {v: df[f"{v:.1f}"].to_numpy() * 1e3 for v in volts}   # nA→pA
    return volts, cur, n1, n2


def hold_noise(I, n1):
    """Holding segment (first n1 points): sigma = MAD; LB whiteness registered (hERG D8 lesson: calibrate first, then judge)."""
    from scipy.stats import chi2
    h = I[:n1]
    x = h - np.linspace(h[0], h[-1], len(h))
    sig = 1.4826 * np.median(np.abs(x - np.median(x)))
    n = len(x)
    r = np.correlate(x - x.mean(), x - x.mean(), "full")[n - 1:] / (np.sum((x - x.mean()) ** 2) + 1e-30)
    m = min(20, n // 4)
    Q = n * (n + 2) * np.sum((r[1:m + 1] ** 2) / (n - np.arange(1, m + 1)))
    return float(sig), float(chi2.sf(Q, m))


# ---------- estimator ----------
def fit_decay(seg_pa, ipk, bd_ad=None):
    """Step segment seg_pa (length n2), observed peak index ipk (within segment). Window = [ipk+SKIP, n2).
    alpha arm bd_ad=None: model A*exp(-(t-t_pk)/tau)+Ip (observed-peak local time base, apparent).
    beta arm bd_ad=(bd,ad): model A*[chain (x) exp(-t/tau)]+Ip (step absolute time base; chain includes that cell's residual pole)."""
    n2 = len(seg_pa)
    a = ipk + int(round(SKIP_MS / DT_MS))
    idx = np.arange(a, n2)
    y = -seg_pa[a:]
    tloc = (idx - ipk) * DT_MS
    A0 = max(float(y.max()), 1.0)
    if bd_ad is None:
        def mdl(tt, A, tau, Ip):
            return A * np.exp(-tt / tau) + Ip
        xd = tloc
    else:
        bd, ad = bd_ad
        def mdl(ii, A, tau, Ip):
            x = np.exp(-np.arange(n2) * DT_MS / tau)
            return A * signal.lfilter(bd, ad, x)[np.asarray(ii, dtype=int)] + Ip
        xd = idx.astype(float)
    best = None
    for t0 in (0.1, 0.25, 0.6, 1.5):
        try:
            p, _ = curve_fit(mdl, xd, y, p0=[A0, t0, 0.3 * A0],
                             bounds=([0, TAU_BND[0], -0.3 * A0], [100 * A0, TAU_BND[1], 2.0 * A0]),
                             maxfev=6000)
            ss = float(np.sum((y - mdl(xd, *p)) ** 2))
            if best is None or ss < best[1]:
                best = (ss, p)
        except Exception:
            pass
    if best is None:
        return None
    ss, (A, tau, Ip) = best
    return {"tau": float(tau), "A": float(A), "Ip": float(Ip),
            "rms": float(np.sqrt(ss / len(y))),
            "tau_edge": bool(abs(tau - TAU_BND[0]) < 1e-3 or abs(tau - TAU_BND[1]) < 1e-3)}


def analyze_cell(cell, sub, temp, volts_pool):
    volts, cur, n1, n2 = load_naiv(sub, temp)
    rs, cm, tau_res = cell_meta(sub, temp)
    ch = chain_discrete(tau_res)
    rows = []
    for v in volts:
        if v not in volts_pool:
            continue
        I = cur[v]
        seg = I[n1:n1 + n2]
        sig, lb_p = hold_noise(I, n1)
        sm = np.convolve(seg, [1 / 3, 1 / 3, 1 / 3], "same")
        ipk = int(sm.argmin())
        ipkA = -float(seg[ipk])
        if ipkA < max(AMP_MIN_PA, 8 * sig):
            rows.append({"cell": cell, "temp": temp, "V": v, "skipped": f"peak amp {ipkA:.0f}pA below threshold"})
            continue
        out = {"cell": cell, "temp": temp, "V": v, "skipped": "",
               "pk_pA": ipkA, "sigma_hold": sig, "lb_p": lb_p,
               "Verr_mV": float(ipkA / 1e3 * rs * (1 - CP / 100.0)), "tau_res_us": tau_res * 1e6}
        for tag, cc in (("a", None), ("b", ch)):
            f = fit_decay(seg, ipk, cc)
            if f:
                out.update({f"tau_{tag}": f["tau"], f"rms_{tag}": f["rms"], f"edge_{tag}": f["tau_edge"],
                            f"Ip_frac_{tag}": f["Ip"] / max(f["A"], 1e-9)})
            else:
                out.update({f"tau_{tag}": None, f"rms_{tag}": None, f"edge_{tag}": True, f"Ip_frac_{tag}": None})
        rows.append(out)
    return rows


# ---------- controls ----------
def synth_and_recover(tau_true_ms, tau_res_s, sig, bd, ad, n2):
    """Synthetics (instant-on exp + sustained, inward negative-sign convention) through the full chain + noise -> beta estimator recovery."""
    x = np.exp(-np.arange(n2) * DT_MS / tau_true_ms) + 0.05
    y = -signal.lfilter(bd, ad, x) * 3000.0 + rng.normal(0, sig, n2)
    ipk = int(np.argmin(y))
    f = fit_decay(y, ipk, (bd, ad))
    return None if f is None else f["tau"]


def controls(sig_med, n2):
    print("[controls]", flush=True)
    ok1 = True
    for tr in (7.7e-6, 10.1e-6, 18.1e-6):
        bd, ad = chain_discrete(tr)
        rec = [synth_and_recover(0.3, tr, sig_med, bd, ad, n2) for _ in range(NREAL)]
        rec = [r for r in rec if r is not None]
        err = float(np.median([abs(r - 0.3) / 0.3 for r in rec])) * 100
        ok = err <= 15.0
        ok1 &= ok
        print(f"  C1 tau=0.3ms residual pole {tr*1e6:5.1f}us: recovery median error {err:5.1f}% (n={len(rec)}, criterion <=15%) {'pass' if ok else '**FAIL**'}", flush=True)
    bd, ad = chain_discrete(10.1e-6)
    rec = [synth_and_recover(0.15, 10.1e-6, sig_med, bd, ad, n2) for _ in range(NREAL)]
    rec = [r for r in rec if r is not None]
    errb = float(np.median([abs(r - 0.15) / 0.15 for r in rec])) * 100
    print(f"  C1b tau=0.15ms (registered): recovery median error {errb:5.1f}%", flush=True)
    # C2: single component + sustained -> double-exponential gain collapse (criterion >= 16/20, smoke 8/10)
    def dbl_gain(ypos):
        t = np.arange(len(ypos)) * DT_MS
        try:
            p1, _ = curve_fit(lambda tt, A, tau, Ip: A * np.exp(-tt / tau) + Ip, t, ypos,
                              p0=[ypos[0], 0.3, 0.0], bounds=([0, 0.01, -ypos[0]], [1e6, 50, ypos[0]]), maxfev=6000)
            ss1 = np.sum((ypos - (p1[0] * np.exp(-t / p1[1]) + p1[2])) ** 2)
            p2, _ = curve_fit(lambda tt, A1, t1, A2, t2: A1 * np.exp(-tt / t1) + A2 * np.exp(-tt / t2), t, ypos,
                              p0=[ypos[0] * .7, 0.1, ypos[0] * .3, 1.0],
                              bounds=([0, 0.01, 0, 0.01], [1e6, 50, 1e6, 50]), maxfev=9000)
            ss2 = np.sum((ypos - (p2[0] * np.exp(-t / p2[1]) + p2[2] * np.exp(-t / p2[3]))) ** 2)
            return (ss1 - ss2) / ss1 if ss1 > 0 else 0.0
        except Exception:
            return 0.0
    collaps = 0
    for _ in range(NREAL):
        x = np.exp(-np.arange(n2) * DT_MS / 0.3) + 0.05
        y = -signal.lfilter(bd, ad, x) * 3000.0 + rng.normal(0, sig_med, n2)
        collaps += int(dbl_gain(-y) < 0.5)
    need = 16 if NREAL >= 20 else int(np.ceil(NREAL * 0.8))
    ok2 = collaps >= need
    print(f"  C2 single-component collapse: {collaps}/{NREAL} (criterion >={need}) {'pass' if ok2 else '**FAIL**'}", flush=True)
    return ok1 and ok2, {"C1b_err_pct": errb}


# ---------- constancy verdict ----------
def constancy_verdict(df, arm="b"):
    tab = {}
    for v in V_POOL:
        sub = df[(df["V"] == v) & (df["skipped"] == "")]
        taus = [t for t in sub[f"tau_{arm}"] if t is not None]
        if len(taus) < 3:
            tab[v] = {"n": len(taus), "verdict": "n<3 registered"}
            continue
        cv = float(np.std(taus, ddof=1) / np.mean(taus))
        mm = float(max(taus) / min(taus))
        ok = cv < 0.3 and mm < 2.0
        tab[v] = {"n": len(taus), "taus": [round(t, 4) for t in taus], "CV": round(cv, 4),
                  "maxmin": round(mm, 3), "verdict": "constant" if ok else "not constant"}
    n_ok = sum(1 for x in tab.values() if x.get("verdict") == "constant")
    n_all = sum(1 for x in tab.values() if "CV" in x)
    summ = ("H holds (constant picture)" if n_ok >= 5 else ("partially holds" if n_ok >= 3 else "H rejected")) if n_all == 7 \
        else f"judgeable levels {n_all}/7, constant {n_ok} (criterion counted on 7 levels, missing levels registered)"
    return tab, n_ok, n_all, summ


def main():
    print("=" * 74, flush=True)
    print(" N-alpha-1 Nav1.5 inactivation fast component tau_h(V) cross-cell constancy verdict" + (" (smoke)" if SMOKE else ""), flush=True)
    print("=" * 74, flush=True)

    pool = V_SMOKE if SMOKE else V_POOL
    cell_set = {"008": CELLS["008"]} if SMOKE else CELLS

    # ---------- C3 noise self-calibration (registered) ----------
    print("[C3 noise self-calibration] (registered)", flush=True)
    sigs = []
    n2_ref = 500
    for c, sub in cell_set.items():
        volts, cur, n1, n2 = load_naiv(sub, 35)
        n2_ref = n2
        I = cur[0.0] if 0.0 in cur else cur[volts[len(volts) // 2]]
        sig, lb_p = hold_noise(I, n1)
        sigs.append(sig)
        print(f"  {c}: sigma_hold={sig:.2f}pA  LB_p={lb_p:.4f} ({'white' if lb_p > 0.05 else 'non-white, registered'})", flush=True)
    sig_med = float(np.median(sigs))

    # ---------- controls ----------
    ok_ctrl, ctrl_extra = controls(sig_med, n2_ref)
    if not ok_ctrl:
        print("  controls not seated -> statistics void, halt.", flush=True)
        return

    # ---------- real data ----------
    print("\n[real data] 80CP 35C main pool + 314 separate listing", flush=True)
    rows = []
    for c, sub in cell_set.items():
        rows += analyze_cell(c, sub, 35, pool)
        print(f"  {c} done ({sum(1 for r in rows if r['cell']==c and r['skipped']=='')}/{len(pool)} levels judgeable)", flush=True)
    rows314 = []
    if not SMOKE:
        for temp in (25, 35):
            try:
                rows314 += analyze_cell("314", SUB_314, temp, pool)
                print(f"  314 {temp}C done", flush=True)
            except FileNotFoundError as e:
                print(f"  314 {temp}C missing levels: {e}", flush=True)

    df = pd.DataFrame(rows)
    tab_b, n_ok_b, n_all_b, summ_b = constancy_verdict(df.dropna(subset=["tau_b"]), "b")
    tab_a, n_ok_a, _, summ_a = constancy_verdict(df.dropna(subset=["tau_a"]), "a")

    print("\n[constancy verdict] beta arm (main)", flush=True)
    print(f"{'V':>6} | {'tau_008':>7} {'tau_011':>7} {'tau_005':>7} | {'CV':>5} {'ratio':>5} | verdict", flush=True)
    for v in V_POOL:
        x = tab_b.get(v, {})
        if "CV" not in x:
            print(f"{v:>6} | {'--':>7} {'--':>7} {'--':>7} | {'--':>5} {'--':>5} | {x.get('verdict','')}", flush=True)
            continue
        ts = x["taus"] + [None] * (3 - len(x["taus"]))
        print(f"{v:>6} | {ts[0]:7.3f} {ts[1]:7.3f} {ts[2]:7.3f} | {x['CV']:5.2f} {x['maxmin']:5.2f} | {x['verdict']}", flush=True)
    print(f"\n  beta arm: constant {n_ok_b}/{n_all_b} levels -> {summ_b}", flush=True)
    print(f"  alpha arm (registered): constant {n_ok_a} levels ({summ_a})", flush=True)
    if (n_ok_b >= 5) != (n_ok_a >= 5):
        print("  ** alpha/beta arm constancy DISAGREES - anomalous signal; attribute before re-judging (pre-registration section 4) **", flush=True)

    if rows314:
        df3 = pd.DataFrame(rows314)
        print("\n[314 separate listing] (batch1 same-chain assumption registered, not counted in the main ratio)", flush=True)
        for temp in (25, 35):
            d3 = df3[(df3["temp"] == temp) & (df3["skipped"] == "")]
            if len(d3) >= 2:
                ts = d3.set_index("V")["tau_b"]
                print(f"  {temp}C: " + "  ".join(f"{v}:{ts.get(v,float('nan')):.3f}" for v in V_POOL if v in ts.index), flush=True)

    # ---------- save ----------
    tag = "_冒烟" if SMOKE else "_结果"
    out = {"meta": {"smoke": SMOKE, "date": "2026-09-15", "card": "Nα-1",
                    "pool": "80CP 35C 008/011/005", "skip_ms": SKIP_MS, "tau_bnd": TAU_BND,
                    "chain": "beta-arm full-chain forward convolution (residual pole inside the chain); v1.1 convention",
                    "controls_C1C2": "pass", "C1b_err_pct": ctrl_extra["C1b_err_pct"],
                    "sig_hold_med_pA": sig_med},
           "verdict_beta": {"table": {str(v): tab_b.get(v, {}) for v in V_POOL},
                            "n_const": n_ok_b, "n_judge": n_all_b, "summary": summ_b},
           "verdict_alpha_register": {"n_const": n_ok_a, "summary": summ_a},
           "rows": rows, "rows314": rows314}
    base = Path(__file__).resolve().with_suffix("")
    fj = f"{base}{tag}.json"
    with open(fj, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    df_all = pd.concat([df] + ([pd.DataFrame(rows314)] if rows314 else []))
    fc = f"{base}{tag}.csv"
    df_all.to_csv(fc, index=False, encoding="utf-8-sig")
    print(f"\n results saved: {fj}\n per-level CSV: {fc}", flush=True)

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    for c, mk in (("008", "o"), ("011", "s"), ("005", "^")):
        d = df[(df["cell"] == c) & (df["skipped"] == "")]
        ax.plot(d["V"], d["tau_b"], mk + "-", label=c, ms=5)
    for v, x in tab_b.items():
        if "CV" in x:
            ax.annotate(f"{x['CV']:.2f}", (v, max(x["taus"]) * 1.05), ha="center", fontsize=8,
                        color="green" if x["verdict"] == "constant" else "red")
    ax.set_title("beta arm tau_h(V) three cells (annotation = CV)"); ax.set_xlabel("V (mV)"); ax.set_ylabel("tau_f (ms)")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    for c, mk in (("008", "o"), ("011", "s"), ("005", "^")):
        d = df[(df["cell"] == c) & (df["skipped"] == "")]
        ax.plot(d["V"], d["tau_a"], mk + "-", label=c, ms=5)
    ax.set_title("alpha arm (apparent, registered) tau_h(V)"); ax.set_xlabel("V (mV)"); ax.set_ylabel("tau_f (ms)")
    ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1, 0]
    for c in (["008"] if SMOKE else list(CELLS)):
        d = df[(df["cell"] == c) & (df["skipped"] == "")]
        if len(d):
            ax.plot(d["V"], d["rms_b"] / d["sigma_hold"], "o-", label=c, ms=5)
    ax.axhline(3, color="r", ls="--", lw=1)
    ax.set_title("fit residual RMS/sigma_hold (QC registration)"); ax.set_xlabel("V (mV)"); ax.legend(); ax.grid(alpha=0.3)
    ax = axes[1, 1]
    volts, cur, n1, n2 = load_naiv(CELLS["008"], 35)
    I = cur[0.0]
    seg = I[n1:n1 + n2]
    t_all = np.arange(n2) * DT_MS
    ax.plot(t_all, seg, "k-", lw=0.8, label="obs 0mV")
    sm = np.convolve(seg, [1 / 3, 1 / 3, 1 / 3], "same"); ipk = int(sm.argmin())
    f = fit_decay(seg, ipk, chain_discrete(cell_meta(CELLS["008"])[2]))
    if f:
        a = ipk + int(round(SKIP_MS / DT_MS))
        xf = np.exp(-np.arange(n2) * DT_MS / f["tau"])
        yf = -(f["A"] * signal.lfilter(*chain_discrete(cell_meta(CELLS["008"])[2]), xf) + f["Ip"])
        ax.plot(t_all[a:], yf[a:], "r--", lw=1.2, label=f"beta fit tau={f['tau']:.3f}ms")
    ax.set_title("008 @0mV fit example"); ax.set_xlabel("t (ms, within step)"); ax.set_ylabel("I (pA)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle(f"N-alpha-1 Nav1.5 tau_h constancy verdict {'(smoke)' if SMOKE else ''} - beta: {n_ok_b}/{n_all_b} constant")
    fig.tight_layout()
    fp = f"{base}{tag}.png"
    fig.savefig(fp, dpi=140, bbox_inches="tight")
    print(f" figure saved: {fp}", flush=True)
    if SMOKE:
        print("\n[smoke done] formal run (user Spyder):\n  %runfile "
              "'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-15_Nα1_Nav15失活快分量恒定性判决.py' --wdir", flush=True)


if __name__ == "__main__":
    main()
