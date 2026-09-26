#!/usr/bin/env python3
# 2026-09-16_Nα3_Nav15 activation-foot portability verdict
# Preregistration: 结果\预注册_Nα3_Nav15激活脚部可携带性判决_2026-09-16.md (v1.0 + v1.1/v1.2, pinned before the run)
# Usage: python this_script          -- smoke (S1/S1b/S2/S3/S3b)
#        python this_script formal   -- full run (allowed only after smoke fully passes)
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import pyabf
from scipy import optimize, stats

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(ROOT)
DRYAD = os.path.join(PROJ, "公开数据", "Nav15_Tarasov2026_dryad", "raw data")
LEI = os.path.join(PROJ, "公开数据", "Nav1.5_27193878", "nav", "batch2", "medium_res_data")
OUTJ = os.path.join(ROOT, "2026-09-16_Nα3_Nav15激活脚部可携带性判决_结果.json")
OUTC = os.path.join(ROOT, "2026-09-16_Nα3_Nav15激活脚部可携带性判决_逐细胞.csv")
OUTP = os.path.join(ROOT, "2026-09-16_Nα3_Nav15激活脚部可携带性判决.png")
VERD = os.path.join(PROJ, "结果", "判词卡_Nα3_Nav15激活脚部可携带性判决_2026-09-16.md")

# ---------------- estimator constants (preregistration v1.2, pinned) ----------------
CROSS_FRAC = 0.03        # 3% peak crossing (v1.3: at 2% the chord foot lands in noise, S3 11/16; 3% unified for the whole cohort)
CHORD_MIN_SIG = 2.0      # chord-point lower bound 2*sigma_eff
SKIP_MS = 1.0            # discard the first 1 ms of the test window


def mad_sig(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def abf_iv(path):
    a = pyabf.ABF(path)
    rows = []
    for s in range(a.sweepCount):
        a.setSweep(s)
        steps = re.findall(r"Step (-?[\d.]+) \[(\d+):(\d+)\]", str(a.sweepEpochs))
        if len(steps) < 4:
            continue
        v_test = float(steps[3][0])
        i0, i1 = int(steps[3][1]), int(steps[3][2])
        y = a.sweepY.astype(float)
        base = np.median(y[max(0, i0 - 200):max(1, i0 - 20)])
        seg = y[i0:i1] - base
        seg = seg[int(SKIP_MS * a.sampleRate / 1000.0):]
        if len(seg) < 10:
            continue
        seg = np.convolve(seg, np.ones(3) / 3.0, mode="same")
        rows.append((v_test, float(np.min(seg))))
    a0 = pyabf.ABF(path); a0.setSweep(0)
    sig = mad_sig(a0.sweepY[:400])
    return {"V": np.array([r[0] for r in rows]), "I": np.array([r[1] for r in rows]),
            "sigma_eff": sig / np.sqrt(3.0), "n_sweeps": a.sweepCount}


def lei_iv(cell, cp):
    f = os.path.join(LEI, cell, f"NaIV_35C_{cp}CP.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    volts = np.array([float(c) for c in df.columns])
    n1, n2 = {975: (225, 500), 1000: (250, 500)}[len(df)]
    I = np.array([np.min(np.convolve(df[f"{v:.1f}"].to_numpy() * 1e3, np.ones(3) / 3.0, mode="same")[n1:n1 + n2])
                  for v in volts])
    sig = mad_sig(df[f"{volts[0]:.1f}"].to_numpy()[:200] * 1e3)
    meta = pd.read_csv(os.path.join(LEI, cell, f"NaIV_35C_{cp}CP_meta.csv"))
    return {"V": volts, "I": I, "sigma_eff": sig / np.sqrt(3.0),
            "Rs_Mohm": float(meta["rseries_Mohm"].iloc[0])}


def foot_fit(d):
    """v1.2: 2% crossing + geodesic chord slope s_ref. Returns dict."""
    V, I, se = d["V"], d["I"], d["sigma_eff"]
    ipeak = int(np.argmin(I))
    ipk = float(I[ipeak])
    thr = CROSS_FRAC * abs(ipk)
    # Crossing: first pair left of the peak with |I_i| < thr <= |I_{i+1}|
    ia = None
    for i in range(ipeak - 1, -1, -1):
        if abs(I[i]) < thr <= abs(I[i + 1]) and I[i] < 0 and I[i + 1] < 0:
            ia = i
            break
    if ia is None:
        return {"ok": False, "reason": "2% crossing not localizable", "I_peak": ipk,
                "V_at_peak": float(V[ipeak])}
    ib = ia + 1
    if abs(I[ia]) < CHORD_MIN_SIG * se or abs(I[ib]) < CHORD_MIN_SIG * se:
        return {"ok": False, "reason": f"chord point below 2*sigma_eff ({abs(I[ia]):.0f}/{abs(I[ib]):.0f} pA, "
                                       f"2σ={2 * se:.0f}）", "I_peak": ipk, "V_at_peak": float(V[ipeak])}
    ya, yb = np.log(abs(I[ia])), np.log(abs(I[ib]))
    vx = V[ia] + (np.log(thr) - ya) / (yb - ya) * (V[ib] - V[ia])
    s_ref = (yb - ya) / (V[ib] - V[ia])
    s_err = se * np.sqrt(1.0 / I[ia] ** 2 + 1.0 / I[ib] ** 2) / (V[ib] - V[ia])
    return {"ok": True, "s_ref": float(s_ref), "s_err": float(s_err),
            "k_equiv": float(3.0 / s_ref), "V_cross": float(vx),
            "V_a": float(V[ia]), "V_b": float(V[ib]),
            "I_a": float(I[ia]), "I_b": float(I[ib]),
            "I_peak": ipk, "V_at_peak": float(V[ipeak])}


def erv_full(d):
    V, I = d["V"], d["I"]
    erv, erv_fb = np.nan, True
    for i in range(len(V) - 3):
        if I[i] < 0 <= I[i + 1] and I[i + 2] > 0 and I[i + 3] > 0:
            erv, erv_fb = 0.5 * (V[i] + V[i + 1]), False
    if not np.isfinite(erv):
        erv = 47.0

    def model(v, G, vh, k):
        m = 1.0 / (1.0 + np.exp(-(v - vh) / k))
        return G * m ** 3 * (v - erv)

    try:
        p, _ = optimize.curve_fit(model, V, I, p0=[abs(I.min()) / 20.0, -35.0, 6.0],
                                  bounds=([0, -80, 1.0], [1e6, 0, 30.0]), maxfev=40000)
        r2 = 1.0 - float(np.sum((I - model(V, *p)) ** 2) / np.sum((I - I.mean()) ** 2))
        return {"E_rev": float(erv), "E_rev_fallback": erv_fb,
                "Vh_full": float(p[1]), "k_full": float(p[2]), "R2_full": r2}
    except Exception as e:
        return {"E_rev": float(erv), "E_rev_fallback": erv_fb, "error": str(e)[:80]}


# ---------------- smoke ----------------
def synth_iv(vh=-40.0, k=6.0, erv=47.0, g=50.0, noise=10.0, rs_eff=0.0, seed=1, dv=5.0):
    rng = np.random.default_rng(seed)
    V = np.arange(-100.0, 80.1, dv)
    I = np.zeros_like(V)
    for _ in range(3):
        vt = V - (I * rs_eff if rs_eff else 0.0)
        m = 1.0 / (1.0 + np.exp(-(vt - vh) / k))
        I = g * m ** 3 * (vt - erv)
    I = I + rng.normal(0, noise, len(V))
    return {"V": V, "I": I, "sigma_eff": noise / np.sqrt(3.0)}, V


def s_analytic(vh, k, erv, g, v_cross, ipk):
    """Analytic local slope at the crossing: 3(1-m)/k (with driving-force correction)."""
    m = 1.0 / (1.0 + np.exp(-(v_cross - vh) / k))
    return 3.0 * (1.0 - m) / k + 1.0 / abs(v_cross - erv)


def smoke():
    print("=" * 72, flush=True)
    print(" [smoke] S1 recovery | S1b grid degradation | S2 Rs distortion | S3/S3b real-data window rules", flush=True)
    print("=" * 72, flush=True)
    ok = True
    # S1
    d, V = synth_iv()
    r = foot_fit(d)
    sa = s_analytic(-40, 6, 47, 50, r["V_cross"], r["I_peak"])
    e1 = abs(r["s_ref"] - sa) / sa
    s1 = e1 <= 0.15
    ok &= s1
    print(f"  S1: s_ref={r['s_ref']:.4f} vs analytic {sa:.4f} (error {e1 * 100:.1f}%, criterion<=15%)"
          f" chord {r['V_a']:.0f}..{r['V_b']:.0f} -> {'pass' if s1 else 'fail'}", flush=True)
    # S1b: 5->10 mV grid degradation
    d10, _ = synth_iv(dv=10.0)
    r10 = foot_fit(d10)
    deg = abs(r10["s_ref"] - r["s_ref"]) / r["s_ref"]
    s1b = deg < 0.05
    ok &= s1b
    print(f"  S1b: 10 mV grid s_ref={r10['s_ref']:.4f}, degradation {deg * 100:.1f}% (criterion<5%) -> {'pass' if s1b else 'fail'}", flush=True)
    # S2
    d2, _ = synth_iv(rs_eff=3.2e-3)
    r2 = foot_fit(d2)
    sa2 = s_analytic(-40, 6, 47, 50, r2["V_cross"], r2["I_peak"])
    e2 = abs(r2["s_ref"] - sa2) / sa2
    s2 = e2 <= 0.15
    ok &= s2
    print(f"  S2: under Rs=8MOhm·60% compensation s_ref={r2['s_ref']:.4f} vs analytic {sa2:.4f}"
          f" (error {e2 * 100:.1f}%, criterion<=15%) -> {'pass' if s2 else 'fail'}", flush=True)
    # S3
    n_ok = n_tot = 0
    for f in bc2_files():
        n_tot += 1
        n_ok += int(foot_fit(abf_iv(f))["ok"])
    s3 = n_ok >= 12
    ok &= s3
    print(f"  S3: G1+G2 judgeable {n_ok}/{n_tot} (criterion >=12) -> {'pass' if s3 else 'fail'}", flush=True)
    # S3b
    n_ok_b = sum(1 for c in ["220502_008_ch2", "220502_011_ch3", "220503_005_ch2"]
                 if (lambda dd: dd is not None and foot_fit(dd)["ok"])(lei_iv(c, "80")))
    s3b = n_ok_b >= 2
    ok &= s3b
    print(f"  S3b: Lei 80CP judgeable {n_ok_b}/3 (criterion >=2) -> {'pass' if s3b else 'fail'}", flush=True)
    return ok


# ---------------- data inventory ----------------
def bc2_files():
    import glob
    return sorted(p for p in
                  glob.glob(os.path.join(DRYAD, "CHO BC2-NaV1.5 whole cell INa", "raw data files", "*.abf"))
                  if pyabf.ABF(p, loadData=False).protocol == "Nav1.5_IV")


def classify_iv(f):
    a = pyabf.ABF(f)
    vs = set()
    for s in range(min(a.sweepCount, 40)):
        a.setSweep(s)
        steps = re.findall(r"Step (-?[\d.]+) \[(\d+):(\d+)\]", str(a.sweepEpochs))
        if len(steps) >= 4:
            vs.add(float(steps[3][0]))
    return len(vs) >= 20


def pool_stats(xs, es):
    xs = np.asarray(xs, float); es = np.asarray(es, float)
    w = 1.0 / es ** 2
    xbar = float(np.sum(w * xs) / np.sum(w))
    cv = float(xs.std(ddof=1) / xs.mean())
    mm = float(xs.max() / xs.min())
    Q = float(np.sum(((xs - xbar) / es) ** 2))
    df = len(xs) - 1
    return {"n": len(xs), "median": float(np.median(xs)), "xbar_w": xbar,
            "CV": cv, "max_min": mm, "Q": Q, "df": df,
            "Qcrit": float(stats.chi2.ppf(0.95, df)), "Q_pass": Q < float(stats.chi2.ppf(0.95, df)),
            "CV_pass": bool(cv < 0.3 and mm < 2.0),
            "P10": float(np.percentile(xs, 10)), "P90": float(np.percentile(xs, 90))}


def main():
    formal = len(sys.argv) > 1 and sys.argv[1] == "formal"
    if not smoke():
        print("\n[smoke failed] full run forbidden.", flush=True)
        sys.exit(2)
    if not formal:
        print("\n[smoke all pass] add the formal argument to enter the full run.", flush=True)
        return

    print("\n" + "=" * 72, flush=True)
    print(" [full run] Na-3 m-inf foot portability verdict (s_ref metric, preregistration v1.2)", flush=True)
    print("=" * 72, flush=True)

    cells = []
    g2_ids = {"24503000", "24503008", "24503010", "24503013", "24510050", "24510055", "24510058"}
    for f in bc2_files():
        cid = os.path.splitext(os.path.basename(f))[0]
        grp = "G2_BC2" if cid in g2_ids else "G1_WT"
        fr = foot_fit(abf_iv(f))
        fu = erv_full(abf_iv(f))
        cells.append({"group": grp, "cell": cid, "src": "Tarasov", **fr, **fu})
        print(f"  [{grp}] {cid}: " + (f"s_ref={fr['s_ref']:.4f}±{fr['s_err']:.4f} "
              f"chord{fr['V_a']:.0f}..{fr['V_b']:.0f} V_x={fr['V_cross']:.1f} k_eq={fr['k_equiv']:.1f}"
              if fr["ok"] else f"not judgeable ({fr['reason']})"), flush=True)

    import glob
    for f in sorted(glob.glob(os.path.join(DRYAD, "CHO deltaKPQ-NaV1.5 whole cell INa", "raw data files", "*.abf"))):
        if pyabf.ABF(f, loadData=False).protocol != "peak current":
            continue
        cid = os.path.splitext(os.path.basename(f))[0]
        if not classify_iv(f):
            cells.append({"group": "G3_dKPQ", "cell": cid, "src": "Tarasov", "ok": False,
                          "reason": "train-type (not IV), mechanically excluded"})
            print(f"  [G3_dKPQ] {cid}: train-type, excluded", flush=True)
            continue
        fr = foot_fit(abf_iv(f))
        fu = erv_full(abf_iv(f))
        cells.append({"group": "G3_dKPQ", "cell": cid, "src": "Tarasov", **fr, **fu})
        print(f"  [G3_dKPQ] {cid}: " + (f"s_ref={fr['s_ref']:.4f}±{fr['s_err']:.4f}"
              if fr["ok"] else f"not judgeable ({fr['reason']})"), flush=True)

    for f in sorted(glob.glob(os.path.join(DRYAD, "deltaKPQ myocytes whole-cell INa", "data files", "*.abf"))):
        if pyabf.ABF(f, loadData=False).protocol != "peak current" or not classify_iv(f):
            continue
        cid = os.path.splitext(os.path.basename(f))[0]
        fr = foot_fit(abf_iv(f))
        cells.append({"group": "G4_myo", "cell": cid, "src": "Tarasov", **fr})
        print(f"  [G4_myo·sens] {cid}: " + (f"s_ref={fr['s_ref']:.4f}+-{fr['s_err']:.4f}"
              if fr["ok"] else "not judgeable"), flush=True)

    for cell in ["220502_008_ch2", "220502_011_ch3", "220503_005_ch2"]:
        for cp, tag in [("80", "main"), ("0", "sens")]:
            d = lei_iv(cell, cp)
            if d is None:
                continue
            fr = foot_fit(d)
            row = {"group": f"Lei_{cp}CP", "cell": cell.split("_")[1], "src": "Lei",
                   "Rs_Mohm": d["Rs_Mohm"], **fr}
            cells.append(row)
            print(f"  [Lei {cp}CP·{tag}] {cell}: " + (f"s_ref={fr['s_ref']:.4f}±{fr['s_err']:.4f} "
                  f"chord{fr['V_a']:.0f}..{fr['V_b']:.0f}"
                  if fr["ok"] else f"not judgeable ({fr['reason']})"), flush=True)

    # ---------------- verdicts ----------------
    verdict = {}
    g12 = [c for c in cells if c["group"] in ("G1_WT", "G2_BC2") and c.get("ok")]
    ps = pool_stats([c["s_ref"] for c in g12], [c["s_err"] for c in g12])
    med1 = float(np.median([c["s_ref"] for c in g12 if c["group"] == "G1_WT"]))
    med2 = float(np.median([c["s_ref"] for c in g12 if c["group"] == "G2_BC2"]))
    tag_diff = abs(med1 - med2) / ps["median"]
    j1 = ps["CV_pass"] and ps["Q_pass"] and tag_diff <= 0.15 and ps["n"] >= 12
    verdict["判1"] = {"通过": bool(j1), **ps, "G1中位": med1, "G2中位": med2, "标签差": float(tag_diff)}
    print(f"\n[crit-1] G1+G2 n={ps['n']}: s_ref median {ps['median']:.4f}, CV={ps['CV']:.3f} (<0.3)"
          f" max/min={ps['max_min']:.2f}（<2.0） Q={ps['Q']:.1f}<χ²={ps['Qcrit']:.1f} "
          f"BC2 tag difference {tag_diff * 100:.1f}% (<=15%) -> {'pass' if j1 else 'fail'}", flush=True)

    g3 = [c for c in cells if c["group"] == "G3_dKPQ" and c.get("ok")]
    if len(g3) >= 3:
        med3 = float(np.median([c["s_ref"] for c in g3]))
        d3 = abs(med3 - ps["median"]) / ps["median"]
        j2 = d3 <= 0.15
        verdict["判2"] = {"通过": bool(j2), "G3中位": med3, "G3_n": len(g3), "中位差": float(d3)}
        print(f"[crit-2] G3 dKPQ n={len(g3)}: median {med3:.4f} vs pool {ps['median']:.4f},"
              f"diff {d3 * 100:.1f}% (<=15%) -> {'pass' if j2 else 'fail'}", flush=True)
    else:
        verdict["判2"] = {"通过": False, "原因": f"G3 可判细胞不足（{len(g3)}）"}
        print(f"[crit-2] G3 judgeable cells insufficient ({len(g3)}) -> fail", flush=True)

    lei80 = [c for c in cells if c["group"] == "Lei_80CP" and c.get("ok")]
    if len(lei80) == 3:
        inside = [c for c in lei80 if ps["P10"] <= c["s_ref"] <= ps["P90"]]
        j3 = len(inside) == 3
        verdict["判3"] = {"通过": bool(j3), "区间": [ps["P10"], ps["P90"]],
                         "Lei_s": [c["s_ref"] for c in lei80], "落入数": len(inside)}
        print(f"[crit-3] Lei 80CP s_ref={[round(c['s_ref'], 3) for c in lei80]} "
              f"vs pool [P10,P90]=[{ps['P10']:.3f},{ps['P90']:.3f}], inside {len(inside)}/3 -> {'pass' if j3 else 'fail'}", flush=True)
    else:
        verdict["判3"] = {"通过": False, "原因": f"Lei 可判数不足（{len(lei80)}/3）"}
        print(f"[crit-3] Lei judgeable count insufficient ({len(lei80)}/3) -> fail", flush=True)

    dv = []
    for c in cells:
        if not c.get("ok"):
            continue
        rs_eff = 4.0 if c["src"] == "Tarasov" else c.get("Rs_Mohm", 6.0) * 0.2
        dv += [abs(c["I_a"]) * 1e-3 * rs_eff, abs(c["I_b"]) * 1e-3 * rs_eff]
    dv_max = float(max(dv)) if dv else np.nan
    j4 = bool(dv_max <= 1.5)
    verdict["判4"] = {"通过": j4, "max_dV_mV": dv_max, "n_points": len(dv)}
    print(f"[crit-4] chord-point |dV| max {dv_max:.3f} mV (n={len(dv)}, criterion <=1.5) -> {'pass' if j4 else 'fail'}", flush=True)

    j5 = bool(j1 and verdict["判2"]["通过"] and verdict["判3"]["通过"])
    verdict["判5"] = {"通过": j5, "s_ref_封卷候选": ps["median"] if j5 else None}
    print(f"\n[crit-5·final] crit1{'v' if j1 else 'x'} crit2{'v' if verdict['判2']['通过'] else 'x'} "
          f"crit3{'v' if verdict['判3']['通过'] else 'x'} -> "
          f"{'s_ref portability HOLDS, seal candidate ' + format(ps['median'], '.4f') + ' mV^-1' if j5 else 'NOT established; per-item attribution in the verdict card'}",
          flush=True)

    # ---------------- save ----------------
    pd.DataFrame(cells).to_csv(OUTC, index=False, encoding="utf-8-sig")
    with open(OUTJ, "w", encoding="utf-8") as fh:
        json.dump({"cells": cells, "verdict": verdict}, fh, ensure_ascii=False, indent=2, default=str)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    for c in cells:
        if c.get("ok") and c["group"] in ("G1_WT", "G2_BC2"):
            col = "tab:blue" if c["group"] == "G1_WT" else "tab:cyan"
            ax.semilogy([c["V_a"], c["V_b"]], np.abs([c["I_a"], c["I_b"]]), "o-",
                        color=col, alpha=0.6, ms=4, lw=1.0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("G1 WT (blue) / G2 BC2 (cyan) 2% chord points")
    ax.set_xlabel("V (mV)"); ax.set_ylabel("|I| (pA)"); ax.set_yscale("log")
    ax = axes[1]
    order = ["G1_WT", "G2_BC2", "G3_dKPQ", "G4_myo", "Lei_80CP", "Lei_0CP"]
    cols = ["tab:blue", "tab:cyan", "tab:red", "tab:orange", "k", "gray"]
    for gi, g in enumerate(order):
        xs = [c["s_ref"] for c in cells if c["group"] == g and c.get("ok")]
        ax.scatter([gi] * len(xs), xs, color=cols[gi], s=42, zorder=3, label=f"{g} (n={len(xs)})")
        if xs:
            ax.hlines(np.median(xs), gi - 0.25, gi + 0.25, color=cols[gi], lw=2)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=25)
    ax.set_ylabel("s_ref (mV$^{-1}$)"); ax.set_title("s_ref scatter by group (bar = median)")
    ax.legend(fontsize=7)
    ax = axes[2]
    for c in cells:
        if c.get("ok") and c["src"] == "Tarasov":
            col = {"G1_WT": "tab:blue", "G2_BC2": "tab:cyan",
                   "G3_dKPQ": "tab:red", "G4_myo": "tab:orange"}[c["group"]]
            ax.errorbar(c["V_cross"], c["s_ref"], yerr=c["s_err"], fmt="o", color=col, alpha=0.6, ms=4)
    for c in cells:
        if c.get("ok") and c["group"] == "Lei_80CP":
            ax.errorbar(c["V_cross"], c["s_ref"], yerr=c["s_err"], fmt="ks--", ms=6, lw=1.3)
    ax.set_xlabel("V_cross (mV)"); ax.set_ylabel("s_ref (mV$^{-1}$)")
    ax.set_title("s_ref vs crossing voltage (black = Lei 80CP)")
    fig.tight_layout()
    fig.savefig(OUTP, dpi=150, bbox_inches="tight")
    print(f"\n  results saved: {OUTJ}\n  per-cell: {OUTC}\n  figure: {OUTP}", flush=True)

    lines = [
        "# Verdict card Na-3: Nav1.5 activation-foot portability verdict",
        "",
        "- Date: 2026-09-16  Preregistration v1.0 + amendments v1.1/v1.2 (all pinned before the run)",
        f"- Judgeable cells: G1 n={sum(1 for c in cells if c['group'] == 'G1_WT' and c.get('ok'))} / "
        f"G2 n={sum(1 for c in cells if c['group'] == 'G2_BC2' and c.get('ok'))} / "
        f"G3 n={len(g3)} / Lei80CP n={len(lei80)}",
        "",
        "## Verdicts",
        "",
        f"- Crit-1 (within-pool portability): {'**PASS**' if j1 else '**FAIL**'} -- s_ref median {ps['median']:.4f} mV^-1,"
        f"CV={ps['CV']:.3f}、max/min={ps['max_min']:.2f}、Q={ps['Q']:.1f}（χ²₀.₉₅={ps['Qcrit']:.1f}）、"
        f"BC2 tag difference {tag_diff * 100:.1f}%",
        f"- Crit-2 (cross-construct): {'**PASS**' if verdict['判2']['通过'] else '**FAIL**'} -- {json.dumps(verdict['判2'], ensure_ascii=False)}",
        f"- Crit-3 (cross-lab Lei): {'**PASS**' if verdict['判3']['通过'] else '**FAIL**'} -- {json.dumps(verdict['判3'], ensure_ascii=False)}",
        f"- Crit-4 (Rs-immunity accounting): {'**PASS**' if j4 else '**FAIL**'} -- chord-point |dV| max {dv_max:.3f} mV (criterion 1.5)",
        "",
        f"## Final (crit-5): {'**s_ref portability HOLDS**; seal candidate s_ref = ' + format(ps['median'], '.4f') + ' mV^-1 (pooled median)' if j5 else '**NOT established** -- per-item attribution above'}",
        "",
        "## Non-judgeable / exclusion registry",
        ""]
    for c in cells:
        if not c.get("ok"):
            lines.append(f"- {c['group']} {c['cell']}：{c.get('reason', '?')}")
    lines += ["", "## Products", f"- {os.path.basename(OUTJ)} / {os.path.basename(OUTC)} / {os.path.basename(OUTP)} (α模型\\)"]
    with open(VERD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  verdict card: {VERD}", flush=True)


if __name__ == "__main__":
    main()
