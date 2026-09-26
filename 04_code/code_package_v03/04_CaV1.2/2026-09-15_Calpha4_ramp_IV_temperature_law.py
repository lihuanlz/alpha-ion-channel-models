# -*- coding: utf-8 -*-
"""
Cα-4: CaV1.2 ramp I-V temperature law and reversal-potential carrier verdict (preregistration v1.2, 2026-09-15)
Extraction: per-sweep chord conductance G=(I(+45)-I(+30))/15 (QC: V_pk<=+25 and G>0), E_chord=45-I(+45)/G
      Ba2+ zero-crossing inside the chord window -> interpolation grade; Ca2+ -> strict lower bound (GHK underestimates, calibrated in smoke S4b)
Criteria: preregistration section 3 (crit-1 Q10(G) constancy / crit-2 E_chord carrier difference / crit-3 E_chord within-group constancy)
Usage: full run  %runfile this_file --wdir
      smoke    python this_file --smoke
"""
import os, sys, json, glob
import numpy as np

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
DATA = os.path.join(ROOT, "公开数据", "Cav12_Ren2022_g3msb")
OUTJ = os.path.join(ROOT, "α模型", "2026-09-15_Cα4_斜坡IV温度律_结果.json")
OUTP = os.path.join(ROOT, "α模型", "2026-09-15_Cα4_斜坡IV温度律.png")
SMOKE = "--smoke" in sys.argv

TRAMP = (0.605, 0.715)     # ramp window (seconds)
VPK_LO, VPK_HI = -10.0, 48.0   # V_pk search range (file mV)
VPK_GATE = 25.0            # QC1: peak must lie below the chord-window floor
CHORD_V = (30.0, 45.0)     # chord window (file mV)
MIN_SW = 20
MIN_BINS = 4
MIN_TSPAN = 3.0
MIN_PER_BIN = 3
EXCLUDE_PRIMARY = {"2021_06_30_0016.abf"}   # clamp anomaly registry exclusion (carried over from the Ca-1 series)

def sweep_chord(V0, I0):
    """Single sweep: V_pk localization + chord conductance. Returns dict or None."""
    k = np.argsort(V0); Vs, Is = V0[k], I0[k]
    Ism = np.convolve(Is, np.ones(5) / 5, mode="same")
    vpk = float(Vs[2:-2][np.argmin(Ism[2:-2])])
    if vpk > VPK_GATE:
        return None
    def iat(vq):
        mm = (V0 >= vq - 1) & (V0 <= vq + 1)
        return float(np.median(I0[mm])) if mm.sum() >= 10 else None
    i30, i45 = iat(CHORD_V[0]), iat(CHORD_V[1])
    if i30 is None or i45 is None:
        return None
    G = (i45 - i30) / (CHORD_V[1] - CHORD_V[0])
    if G <= 0:
        return None
    return {"G": G, "Ec": CHORD_V[1] - i45 / G, "vpk": vpk}

def scan_file(path):
    import pyabf
    a = pyabf.ABF(path)
    recs = []
    for s in range(a.sweepCount):
        a.setSweep(s, channel=0); i = a.sweepY.astype(float); t = a.sweepX
        a.setSweep(s, channel=1); v = a.sweepY.astype(float)
        a.setSweep(s, channel=2); bt = a.sweepY.astype(float)
        m = (t >= TRAMP[0]) & (t < TRAMP[1]) & (v >= VPK_LO) & (v <= VPK_HI)
        if m.sum() < 300:
            continue
        r = sweep_chord(v[m], i[m])
        if r is None:
            continue
        r["T"] = float(bt.mean()); r["sweep"] = s
        recs.append(r)
    return recs

def cell_q10(recs):
    out = {"n_valid": len(recs)}
    if len(recs) < MIN_SW:
        out["pooled"] = False; out["why"] = "n<%d" % MIN_SW; return out
    Ts = np.array([r["T"] for r in recs]); Gs = np.array([r["G"] for r in recs])
    Es = np.array([r["Ec"] for r in recs])
    out["E_chord_med"] = float(np.median(Es))
    out["E_chord_iqr"] = float(np.percentile(Es, 75) - np.percentile(Es, 25))
    out["G_med"] = float(np.median(Gs))
    bins = np.floor(Ts); bx, by = [], []
    for b in sorted(set(bins)):
        mm = bins == b
        if mm.sum() >= MIN_PER_BIN:
            bx.append(b + 0.5); by.append(float(np.median(Gs[mm])))
    out["bins"] = len(bx); out["Tspan"] = float(max(bx) - min(bx) + 1) if bx else 0.0
    if len(bx) < MIN_BINS or out["Tspan"] < MIN_TSPAN:
        out["pooled"] = False; out["why"] = "bins/span"; return out
    bx = np.array(bx); ly = np.log10(np.array(by))
    sl, ic = np.polyfit(bx, ly, 1)
    pred = sl * bx + ic
    r2 = 1 - float(np.sum((ly - pred) ** 2) / np.sum((ly - ly.mean()) ** 2))
    out.update({"pooled": True, "Q10": float(10 ** (10 * sl)), "q10_r2": r2,
                "bin_T": [float(x) for x in bx], "bin_G": [float(y) for y in by]})
    return out

def bootstrap_dE(ca, ba, n=10000, seed=1):
    rng = np.random.default_rng(seed)
    ca = np.array(ca); ba = np.array(ba)
    ds = []
    for _ in range(n):
        a = rng.choice(ca, size=len(ca), replace=True)
        b = rng.choice(ba, size=len(ba), replace=True)
        ds.append(np.median(a) - np.median(b))
    ds = np.array(ds)
    return float(np.median(ca) - np.median(ba)), float(np.percentile(ds, 5)), float(np.percentile(ds, 95))

# ---------------------------------------------------------------- smoke
def synth_iv(V, Gmax, E_rev, vhalf, k, ghk_s=None):
    """m-inf(V)·G·(V-E_rev), optional GHK flattening near E_rev (ghk_s = flattening scale in mV)."""
    m = 1.0 / (1.0 + np.exp(-(V - vhalf) / k))
    g = Gmax * m
    if ghk_s is not None:
        g = g * (1.0 - np.exp(-(E_rev - V) / ghk_s))
    return g * (V - E_rev)

def smoke():
    print("=" * 72, flush=True)
    print(" [smoke Ca-4 v1.2] S1 chord recovery | S2 V_pk gate | S3 zero Q10 | S4 E_chord boundary behaviour", flush=True)
    rng = np.random.default_rng(7)
    ok_all = True

    # S1: linear I-V (E_rev=+42), chord G and E_chord recovery
    recs = []
    for T in np.linspace(30, 39, 60):
        V = np.linspace(-10, 48, 1160)
        G_true = 40 * 1.4 ** ((T - 37) / 10)
        I = G_true * (V - 42.0) + rng.normal(0, 8, len(V))
        # push the peak below +3 to mimic the Ba shape
        m = 1.0 / (1.0 + np.exp(-(V - (-15)) / 8))
        r = sweep_chord(V, I * m)
        if r: r["T"] = T; recs.append(r)
    q = cell_q10(recs)
    e_bias = np.median([r["Ec"] for r in recs]) - 42.0
    g_rec = np.median([r["G"] for r in recs if abs(r["T"] - 37) < 0.6]) / 40.0 - 1
    ok = q.get("pooled") and abs(q["Q10"] / 1.4 - 1) <= 0.15 and abs(e_bias) <= 2 and abs(g_rec) <= 0.10
    ok_all &= ok
    print(f"  S1: Q10 recovery {q.get('Q10', float('nan')):.2f} (true 1.4, ±15%) E_chord bias {e_bias:+.2f}mV (<=2) G error {g_rec*100:+.1f}% (<=10%)", "pass" if ok else "fail", flush=True)

    # S2: V_pk gate -- peak at +19 passes, peak at +30 is rejected
    V = np.linspace(-10, 48, 1160)
    n_pass, n_block = 0, 0
    for _ in range(20):
        I1 = synth_iv(V, 40, 63, -5, 6) + rng.normal(0, 8, len(V))   # peak ~+19
        I2 = synth_iv(V, 40, 63, 22, 6) + rng.normal(0, 8, len(V))   # peak ~+30
        if sweep_chord(V, I1): n_pass += 1
        if sweep_chord(V, I2) is None: n_block += 1
    ok = n_pass >= 19 and n_block >= 19
    ok_all &= ok
    print(f"  S2: peak+19 passes {n_pass}/20 (>=19) peak+30 rejected {n_block}/20 (>=19)", "pass" if ok else "fail", flush=True)

    # S3: zero Q10
    recs = []
    for T in np.linspace(30, 39, 60):
        V = np.linspace(-10, 48, 1160)
        I = 40.0 * (V - 42.0) / (1.0 + np.exp(-(V - (-15)) / 8)) + rng.normal(0, 8, len(V))
        r = sweep_chord(V, I)
        if r: r["T"] = T; recs.append(r)
    q3 = cell_q10(recs)
    ok = q3.get("pooled") and 0.9 <= q3["Q10"] <= 1.1
    ok_all &= ok
    print(f"  S3: zero-Q10 recovery {q3.get('Q10', float('nan')):.2f} (in [0.9,1.1])", "pass" if ok else "fail", flush=True)

    # S4a: Ba-like (E_rev=+43, activation half-voltage -8 -> peak ~+3, m-inf saturated and near-linear in the chord window) E_chord vs upper-limb line-fit intercept
    diffs = []
    for _ in range(20):
        V = np.linspace(-10, 48, 1160)
        I = synth_iv(V, 40, 43, -8, 7) + rng.normal(0, 8, len(V))
        r = sweep_chord(V, I)
        if r is None: continue
        mm = (V >= r["vpk"] + 5) & (V <= 45)
        A = np.column_stack([np.ones(mm.sum()), V[mm]])
        (a0, b0), *_ = np.linalg.lstsq(A, I[mm], rcond=None)
        e_limb = -a0 / b0
        diffs.append(abs(r["Ec"] - e_limb))
    ok_a = len(diffs) >= 19 and float(np.median(diffs)) <= 3.0
    # S4b: Ca-like (E_rev=+63, activation still rising inside the chord window vhalf=+10 + GHK flattening above the window top) E_chord empirical lower bound + underestimation magnitude
    unders = []
    for _ in range(20):
        V = np.linspace(-10, 48, 1160)
        I = synth_iv(V, 40, 63, 10, 8, ghk_s=15.0) + rng.normal(0, 8, len(V))
        r = sweep_chord(V, I)
        if r: unders.append(63.0 - r["Ec"])
    unders = np.array(unders)
    ok_b = len(unders) >= 19 and float(np.median(unders)) > 2 and float(np.median(unders)) < 25
    ok = ok_a and ok_b
    ok_all &= ok
    print(f"  S4a: E_chord vs body-fit difference median {np.median(diffs):.2f}mV (<=3)", "pass" if ok_a else "fail", flush=True)
    print(f"  S4b: Ca-like E_chord underestimation median {np.median(unders):.1f}mV (2<x<25, empirical lower bound holds)", "pass" if ok_b else "fail", flush=True)

    print(" smoke overall:", "all pass -> full run allowed" if ok_all else "not all pass -> full run forbidden", flush=True)
    return ok_all

# ---------------------------------------------------------------- full run
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {"Ca2": sorted(glob.glob(os.path.join(DATA, "Temperature_Ca2", "*.abf"))),
              "Ba2": sorted(glob.glob(os.path.join(DATA, "Temperature_Ba2", "*.abf")))}
    result = {"groups": {}, "verdict": {}}
    q10s, erevs, r2s = {"Ca2": [], "Ba2": []}, {"Ca2": [], "Ba2": []}, {"Ca2": [], "Ba2": []}
    print("=" * 72, flush=True)
    print(" Ca-4: CaV1.2 ramp chord-conductance temperature law and E_chord carrier verdict (preregistration v1.2 full run)", flush=True)
    print("=" * 72, flush=True)
    for grp, files in groups.items():
        print(f"\n[{grp} group] {len(files)} files", flush=True)
        result["groups"][grp] = {}
        for p in files:
            name = os.path.basename(p)
            recs = scan_file(p)
            q = cell_q10(recs)
            result["groups"][grp][name] = q
            primary = name not in EXCLUDE_PRIMARY
            if q.get("pooled"):
                if primary:
                    q10s[grp].append(q["Q10"]); erevs[grp].append(q["E_chord_med"]); r2s[grp].append(q["q10_r2"])
                print(f"  {name}: valid {q['n_valid']} | Q10(G)={q['Q10']:.2f} R2={q['q10_r2']:.2f} E_chord={q['E_chord_med']:+.1f}mV"
                      + ("" if primary else " (registry-excluded)"), flush=True)
            else:
                print(f"  {name}: valid {q['n_valid']} | not pooled ({q.get('why','?')})", flush=True)

    vd = {}
    for grp in ("Ca2", "Ba2"):
        arr = np.array(q10s[grp]); er = np.array(erevs[grp])
        vd[grp] = {"n_pooled": len(arr),
                   "Q10_med": float(np.median(arr)) if len(arr) else None,
                   "Q10_cv": float(np.std(arr) / np.mean(arr)) if len(arr) > 1 else None,
                   "Q10_maxmin": float(np.max(arr) / np.min(arr)) if len(arr) > 1 else None,
                   "Q10_r2_ge05": float(np.mean([x >= 0.5 for x in r2s[grp]])) if r2s[grp] else None,
                   "E_chord_med": float(np.median(er)) if len(er) else None,
                   "E_chord_sd": float(np.std(er)) if len(er) > 1 else None,
                   "E_chord_range": float(np.max(er) - np.min(er)) if len(er) > 1 else None}
    # crit-1
    j1 = {}
    for grp in ("Ca2", "Ba2"):
        v = vd[grp]
        j1[grp] = bool(v["n_pooled"] >= 3 and v["Q10_cv"] is not None and v["Q10_cv"] < 0.3
                       and v["Q10_maxmin"] < 2.0 and v["Q10_r2_ge05"] == 1.0)
    # crit-2
    if len(erevs["Ca2"]) >= 2 and len(erevs["Ba2"]) >= 2:
        d, lo, hi = bootstrap_dE(erevs["Ca2"], erevs["Ba2"])
        vd["判2"] = {"dE_med": d, "ci90": [lo, hi], "pass": bool(d > 0 and lo > 0)}
    else:
        vd["判2"] = {"pass": False, "why": "too few cells"}
    # crit-3
    vd["判3"] = {grp: bool(vd[grp]["E_chord_sd"] is not None and vd[grp]["E_chord_sd"] < 5
                           and vd[grp]["E_chord_range"] < 12) for grp in ("Ca2", "Ba2")}
    vd["判1"] = j1
    result["verdict"] = vd

    print("\n" + "=" * 72, flush=True)
    print(" verdict components", flush=True)
    print(json.dumps(vd, ensure_ascii=False, indent=2), flush=True)

    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for grp, c in (("Ca2", "tab:red"), ("Ba2", "tab:blue")):
        for name, q in result["groups"][grp].items():
            if q.get("pooled"):
                axes[0].plot(q["bin_T"], q["bin_G"], "o-", alpha=0.7, color=c,
                             label=f"{grp} {name}" if grp == "Ca2" else None)
    axes[0].set_xlabel("bath T (°C)"); axes[0].set_ylabel("chord G (pA/mV)")
    axes[0].set_title("chord conductance x bath temperature (red=Ca2+ blue=Ba2+)")
    for i, grp in enumerate(("Ca2", "Ba2")):
        axes[1].plot(np.full(len(q10s[grp]), i) + np.linspace(-0.15, 0.15, max(1, len(q10s[grp]))),
                     q10s[grp], "o", color=("tab:red", "tab:blue")[i])
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Ca²⁺", "Ba²⁺"])
    axes[1].set_ylabel("Q10(G)"); axes[1].set_title("conductance temperature law per cell")
    for i, grp in enumerate(("Ca2", "Ba2")):
        axes[2].plot(np.full(len(erevs[grp]), i) + np.linspace(-0.15, 0.15, max(1, len(erevs[grp]))),
                     erevs[grp], "o", color=("tab:red", "tab:blue")[i])
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["Ca2+ (lower bound)", "Ba2+ (interpolation)"])
    axes[2].set_ylabel("E_chord (mV, file coordinates)"); axes[2].set_title("reversal potential per cell")
    fig.tight_layout(); fig.savefig(OUTP, dpi=140, bbox_inches="tight")
    print(f"\n results saved: {OUTJ}\n figure saved: {OUTP}", flush=True)

if __name__ == "__main__":
    if SMOKE:
        smoke()
    else:
        main()
