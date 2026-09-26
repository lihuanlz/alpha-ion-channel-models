# -*- coding: utf-8 -*-
"""
Cα-1: CaV1.2 inactivation temperature law and charge-carrier verdict (preregistration v1.0, 2026-09-15)
Data: 公开数据/Cav12_Ren2022_g3msb/Temperature_{Ca2,Ba2}/*.abf (11 cells, unpaired)
Criteria: preregistration section 5 (Q10 constancy / tau37 constancy / CDI R>1.2 and CI>1.0 / carrier-effect constancy)
Usage: full run  %runfile this_file --wdir   (all 11 files)
      smoke    python this_file --smoke     (synthetic recovery S1/S2/S3)
"""
import os, sys, json, glob
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
DATA = os.path.join(ROOT, "公开数据", "Cav12_Ren2022_g3msb")
OUTJ = os.path.join(ROOT, "α模型", "2026-09-15_Cα1_CaV12失活温度律与电荷载子判决_结果.json")
OUTP = os.path.join(ROOT, "α模型", "2026-09-15_Cα1_CaV12失活温度律与电荷载子判决.png")
SMOKE = "--smoke" in sys.argv

DT = 1e-4            # 10 kHz
T47 = (0.42, 0.60)   # main measurement segment +47 mV
T17 = (0.38, 0.42)   # secondary segment +17 mV (registry only)
TRAMP = (0.61, 0.71) # ramp segment for E_rev

# QC gates (preregistration section 4)
G1_DV = 5.0      # read-back voltage |V-47| <= 5 mV
G2_PK = 100.0    # inward peak |peak| >= 100 pA
G3_RATIO = 0.9   # end/peak < 0.9
G4_R2 = 0.95     # fit R^2
MIN_SW_CELL = 20     # minimum valid sweeps for a cell to enter the pool
MIN_TSPAN = 3.0      # minimum temperature span for a cell to enter the pool
MIN_BINS = 4         # minimum non-empty bins for Q10 regression
MIN_PER_BIN = 3      # minimum sweeps per bin
AICC_MARGIN = 10.0   # AICc advantage required for the double exponential to win


# ---------- exponential models ----------
def exp1(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def exp2(t, A1, tau1, A2, tau2, C):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C

def aicc(n, k, rss):
    if rss <= 0 or n <= k + 1:
        return np.inf
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)

def fit_segment(t, y):
    """Fit single/double exponentials from just after the peak; AICc model selection. Returns dict or None (fit failed)."""
    i_pk = int(np.argmin(y))
    if i_pk >= len(y) - 30:
        return None
    tt = t[i_pk:] - t[i_pk]
    yy = y[i_pk:]
    pk = y[i_pk]
    tail = np.mean(yy[-200:])  # last 20 ms
    # ---- G3 pre-check: decay must exist
    if not (pk <= -G2_PK and abs(tail) < G3_RATIO * abs(pk)):
        return {"qc_fail": "G2G3", "peak": float(pk)}
    C0 = tail
    try:
        p1, _ = curve_fit(exp1, tt, yy, p0=[pk - C0, 0.03, C0],
                          bounds=([-2e4, 1e-3, -5e3], [0, 0.5, 5e3]), maxfev=20000)
        rss1 = float(np.sum((yy - exp1(tt, *p1)) ** 2))
    except Exception:
        return {"qc_fail": "fit1", "peak": float(pk)}
    r2_1 = 1 - rss1 / float(np.sum((yy - yy.mean()) ** 2))
    best = {"model": "1", "tau": float(p1[1]), "A": float(p1[0]), "C": float(p1[2]),
            "r2": float(r2_1), "peak": float(pk), "aicc1": aicc(len(yy), 3, rss1)}
    # double exponential
    try:
        p2, _ = curve_fit(exp2, tt, yy,
                          p0=[(pk - C0) * 0.6, 0.01, (pk - C0) * 0.4, 0.08, C0],
                          bounds=([-2e4, 1e-3, -2e4, 1e-3, -5e3], [0, 0.5, 0, 0.5, 5e3]),
                          maxfev=40000)
        rss2 = float(np.sum((yy - exp2(tt, *p2)) ** 2))
        r2_2 = 1 - rss2 / float(np.sum((yy - yy.mean()) ** 2))
        a2 = aicc(len(yy), 5, rss2)
        if a2 < best["aicc1"] - AICC_MARGIN:
            A1, t1, A2, t2, C = p2
            tw = (abs(A1) * t1 + abs(A2) * t2) / (abs(A1) + abs(A2))
            best = {"model": "2", "tau": float(tw), "tau1": float(t1), "tau2": float(t2),
                    "A1": float(A1), "A2": float(A2), "C": float(C),
                    "r2": float(r2_2), "peak": float(pk), "aicc1": best["aicc1"], "aicc2": float(a2)}
    except Exception:
        pass
    best["qc_fail"] = None if best["r2"] >= G4_R2 else "G4"
    return best


# ---------- real data ----------
def scan_file(path):
    import pyabf
    a = pyabf.ABF(path)
    recs = []
    for s in range(a.sweepCount):
        a.setSweep(s, channel=0); i = a.sweepY.astype(float); t = a.sweepX
        a.setSweep(s, channel=1); v = a.sweepY.astype(float)
        a.setSweep(s, channel=2); bt = a.sweepY.astype(float)
        T = float(bt.mean())
        m47 = (t >= T47[0]) & (t < T47[1])
        v47 = float(v[m47].mean())
        rec = {"sweep": s, "T": T, "V47": v47}
        if abs(v47 - 47.0) > G1_DV:
            rec["qc_fail"] = "G1"
            recs.append(rec); continue
        fr = fit_segment(t[m47], i[m47])
        if fr is None:
            rec["qc_fail"] = "peaklate"
        else:
            rec.update(fr)
        # E_rev (registry)
        mr = (t >= TRAMP[0]) & (t < TRAMP[1])
        vr, ir = v[mr], i[mr]
        k = np.argsort(vr); vr, ir = vr[k], ir[k]
        iz = np.where(np.diff(np.sign(ir)) != 0)[0]
        rec["E_rev"] = float(vr[iz[0]]) if len(iz) else None
        recs.append(rec)
    return recs


def cell_q10(recs):
    """Pooling + 1°C binning + log10(tau)~T regression. Returns dict."""
    ok = [r for r in recs if r.get("qc_fail") is None and "tau" in r]
    out = {"n_valid": len(ok)}
    if len(ok) < MIN_SW_CELL:
        out["pooled"] = False; out["why"] = "n<%d" % MIN_SW_CELL; return out
    Ts = np.array([r["T"] for r in ok]); taus = np.array([r["tau"] for r in ok])
    bins = np.floor(Ts)
    ub = sorted(set(bins))
    bx, by, bn = [], [], []
    for b in ub:
        m = bins == b
        if m.sum() >= MIN_PER_BIN:
            bx.append(b + 0.5); by.append(np.median(taus[m])); bn.append(int(m.sum()))
    out["bins"] = len(bx); out["Tspan"] = float(max(bx) - min(bx) + 1) if bx else 0.0
    if len(bx) < MIN_BINS or out["Tspan"] < MIN_TSPAN:
        out["pooled"] = False; out["why"] = "bins/span"; return out
    bx = np.array(bx); ly = np.log10(np.array(by) * 1e3)  # τ in ms
    sl, ic = np.polyfit(bx, ly, 1)
    pred = sl * bx + ic
    r2 = 1 - float(np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2))
    out.update({"pooled": True, "slope": float(sl), "Q10": float(10 ** (-10 * sl)),
                "tau37_ms": float(10 ** (ic + sl * 37)), "r2": r2,
                "bin_T": [float(x) for x in bx], "bin_tau_ms": [float(10 ** y) for y in ly],
                "bin_n": bn})
    return out


def bootstrap_R(ca, ba, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    ca = np.array(ca); ba = np.array(ba)
    Rs = []
    for _ in range(n):
        a = rng.choice(ca, size=len(ca), replace=True)
        b = rng.choice(ba, size=len(ba), replace=True)
        Rs.append(np.median(b) / np.median(a))
    Rs = np.array(Rs)
    return float(np.median(ba) / np.median(ca)), float(np.percentile(Rs, 5)), float(np.percentile(Rs, 95))


# ---------- smoke ----------
def bessel_lowpass(y, fc=2000.0, fs=10000.0, order=4):
    sos = signal.bessel(order, fc / (fs / 2), btype="low", output="sos", norm="mag")
    return signal.sosfilt(sos, y)

def smoke():
    print("=" * 72, flush=True)
    print(" [smoke] S1 tau recovery (incl. 2kHz Bessel bias) | S2 Q10 recovery | S3 model selection", flush=True)
    print("=" * 72, flush=True)
    rng = np.random.default_rng(7)
    # S1
    for tau_true in [0.010, 0.025, 0.050]:
        errs = []
        for _ in range(20):
            t = np.arange(0, 0.18, DT)
            y = -1000 * np.exp(-t / tau_true) - 50 + rng.normal(0, 20, len(t))
            y = bessel_lowpass(y)
            fr = fit_segment(t, y)
            if fr and fr.get("qc_fail") is None:
                errs.append(fr["tau"] / tau_true - 1)
        med = float(np.median(errs)) * 100 if errs else float("nan")
        print(f"  S1 tau={tau_true*1e3:.0f}ms: recovered median error {med:+.1f}% (n={len(errs)}, criterion |err|<=15% and bias<5%)",
              "pass" if errs and abs(med) <= 5 and np.percentile(np.abs(errs), 50) <= 0.15 else "fail", flush=True)
    # S2
    q10_true, tau37_true = 2.5, 0.030
    errs = []
    for _ in range(20):
        Ts = rng.uniform(30, 39, 120)
        taus = tau37_true * q10_true ** (-(Ts - 37) / 10)
        recs = []
        for T, ta in zip(Ts, taus):
            t = np.arange(0, 0.18, DT)
            y = -800 * np.exp(-t / ta) - 40 + rng.normal(0, 20, len(t))
            fr = fit_segment(t, y)
            if fr and fr.get("qc_fail") is None:
                recs.append({"T": float(T), "tau": fr["tau"], "qc_fail": None})
        q = cell_q10(recs)
        if q.get("pooled"):
            errs.append(q["Q10"] / q10_true - 1)
    med = float(np.median(errs)) * 100 if errs else float("nan")
    print(f"  S2 Q10={q10_true}: recovered median error {med:+.1f}% (n={len(errs)}, criterion <=10%)",
          "pass" if errs and abs(med) <= 10 else "fail", flush=True)
    # S3
    pick1 = 0
    for _ in range(20):
        t = np.arange(0, 0.18, DT)
        y = -900 * np.exp(-t / 0.025) - 60 + rng.normal(0, 25, len(t))
        fr = fit_segment(t, y)
        if fr and fr.get("model") == "1":
            pick1 += 1
    print(f"  S3 single exponential not stolen by double: {pick1}/20 (criterion >=16)", "pass" if pick1 >= 16 else "fail", flush=True)


# ---------- main flow ----------
def main():
    if SMOKE:
        smoke(); return
    print("=" * 72, flush=True)
    print(" Ca-1: CaV1.2 inactivation temperature law and charge-carrier verdict (preregistration v1.0 full run)", flush=True)
    print("=" * 72, flush=True)
    result = {"groups": {}, "verdict": {}}
    tau37 = {"Ca2": [], "Ba2": []}; q10s = {"Ca2": [], "Ba2": []}
    for grp in ["Ca2", "Ba2"]:
        files = sorted(glob.glob(os.path.join(DATA, "Temperature_" + grp, "*.abf")))
        result["groups"][grp] = {}
        print(f"\n[{grp} group] {len(files)} files", flush=True)
        for f in files:
            name = os.path.basename(f)
            recs = scan_file(f)
            q = cell_q10(recs)
            result["groups"][grp][name] = {"recs_summary": {
                "n_sweeps": len(recs),
                "n_valid": q.get("n_valid", 0),
                "fail_G1": sum(1 for r in recs if r.get("qc_fail") == "G1"),
                "fail_G2G3": sum(1 for r in recs if r.get("qc_fail") == "G2G3"),
                "fail_G4": sum(1 for r in recs if r.get("qc_fail") == "G4"),
                "model2": sum(1 for r in recs if r.get("model") == "2"),
            }, "q10": q}
            if q.get("pooled"):
                tau37[grp].append(q["tau37_ms"]); q10s[grp].append(q["Q10"])
                print(f"  {name}: valid {q['n_valid']} sweeps | Q10={q['Q10']:.2f} tau37={q['tau37_ms']:.1f}ms R2={q['r2']:.3f} (span {q['Tspan']:.0f}°C)", flush=True)
            else:
                print(f"  {name}: valid {q.get('n_valid',0)} sweeps | not pooled ({q.get('why')})", flush=True)
    # ---- verdicts
    v = {}
    for grp in ["Ca2", "Ba2"]:
        a = np.array(q10s[grp]); b = np.array(tau37[grp])
        v[grp] = {"n_pooled": len(a)}
        if len(a) >= 3:
            v[grp]["Q10"] = {"median": float(np.median(a)), "CV": float(np.std(a) / np.mean(a)),
                             "maxmin": float(a.max() / a.min())}
            v[grp]["tau37"] = {"median": float(np.median(b)), "CV": float(np.std(b) / np.mean(b)),
                               "maxmin": float(b.max() / b.min())}
    allq = np.array(q10s["Ca2"] + q10s["Ba2"])
    if len(allq) >= 3:
        v["Q10_all"] = {"n": len(allq), "median": float(np.median(allq)),
                        "CV": float(np.std(allq) / np.mean(allq)), "maxmin": float(allq.max() / allq.min()),
                        "pass": bool(np.std(allq) / np.mean(allq) < 0.3 and allq.max() / allq.min() < 2.0)}
    if len(tau37["Ca2"]) >= 3 and len(tau37["Ba2"]) >= 3:
        R, lo, hi = bootstrap_R(tau37["Ca2"], tau37["Ba2"])
        v["carrier"] = {"R": R, "CI90": [lo, hi],
                        "CDI": bool(R > 1.2 and lo > 1.0)}
    result["verdict"] = v
    print("\n" + "=" * 72, flush=True)
    print(" verdict components", flush=True)
    print(json.dumps(v, ensure_ascii=False, indent=2), flush=True)
    with open(OUTJ, "w", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, indent=1)
    print("\n results saved:", OUTJ, flush=True)
    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for f_ in font_manager.findSystemFonts():
            if "msyh" in f_.lower() or "simhei" in f_.lower():
                font_manager.fontManager.addfont(f_)
        plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for grp, c in [("Ca2", "tab:red"), ("Ba2", "tab:blue")]:
            for name, d in result["groups"][grp].items():
                q = d["q10"]
                if q.get("pooled"):
                    axes[0].scatter(q["bin_T"], q["bin_tau_ms"], c=c, s=18, alpha=0.7)
        axes[0].set_xlabel("bath temperature °C"); axes[0].set_ylabel("τ_inact (ms)")
        axes[0].set_title("tau(T) binned (red Ca2+ / blue Ba2+)"); axes[0].set_yscale("log")
        labels, vals = [], []
        for grp in ["Ca2", "Ba2"]:
            for name, d in result["groups"][grp].items():
                q = d["q10"]
                if q.get("pooled"):
                    labels.append(name.replace(".abf", "") + ("*" if grp == "Ba2" else ""))
                    vals.append(q["Q10"])
        axes[1].bar(range(len(vals)), vals, color=["tab:blue" if L.endswith("*") else "tab:red" for L in labels])
        axes[1].set_xticks(range(len(vals))); axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[1].set_ylabel("Q10"); axes[1].set_title("Q10 per cell (* = Ba2+)")
        labels2, vals2 = [], []
        for grp in ["Ca2", "Ba2"]:
            for name, d in result["groups"][grp].items():
                q = d["q10"]
                if q.get("pooled"):
                    labels2.append(name.replace(".abf", "") + ("*" if grp == "Ba2" else ""))
                    vals2.append(q["tau37_ms"])
        axes[2].bar(range(len(vals2)), vals2, color=["tab:blue" if L.endswith("*") else "tab:red" for L in labels2])
        axes[2].set_xticks(range(len(vals2))); axes[2].set_xticklabels(labels2, rotation=45, ha="right", fontsize=7)
        axes[2].set_ylabel("τ37 (ms)"); axes[2].set_title("37 °C reference tau per cell")
        fig.tight_layout(); fig.savefig(OUTP, dpi=140, bbox_inches="tight")
        print(" figure saved:", OUTP, flush=True)
    except Exception as e:
        print(" plotting failed (verdicts unaffected):", e, flush=True)


if __name__ == "__main__":
    main()
