# -*- coding: utf-8 -*-
"""
Cα-5: CaV1.2 drug shape-modulation verdict (preregistration v1.0, 2026-09-15)
Pure block (scales G only) vs gating modulation (changes inactivation shape / CDI fast component / state preference)
crit-1 eps_x>max(Null-B p90,0.03) rejects pure block | crit-2 kappa*<0.10 and eps_x1/eps_x>0.5 amplitude type | crit-3 CDI special verdict | crit-4 external cross-check
Usage: full run  %runfile this_file --wdir
      smoke    python this_file --smoke
"""
import os, sys, json, glob
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from scipy.stats import binomtest

ROOT = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4"
DATA = os.path.join(ROOT, "公开数据", "Cav12_Ren2022_g3msb", "Drugs")
OUTJ = os.path.join(ROOT, "α模型", "2026-09-15_Cα5_CaV12药物形状调制判决_结果.json")
OUTP = os.path.join(ROOT, "α模型", "2026-09-15_Cα5_CaV12药物形状调制判决.png")
SMOKE = "--smoke" in sys.argv

# ---- protocol epochs (seconds, 10 kHz) ----
I73_WIN = (0.250, 0.275)   # end of the -73 mV segment (leak point 1)
I63_WIN = (0.350, 0.375)   # end of the -63 mV segment (leak point 2)
STEP = (0.3781, 0.4181)    # +17 mV/40 ms (nominal 0 mV)
RAMP = (0.6181, 0.7181)    # ramp +47 -> -63
FS = 10000.0
TMS40 = np.arange(400) * 0.1   # ms axis of the step segment

GROUPS = ["Verapamil_Ca2+_PT", "Verapamil_Ca2+_RT", "Methadone_Ca2+_PT",
          "Diltiazem_Ba2+_PT", "Buprenorphine_Ca2+_PT",
          "Norbuprenorphine_Ca2+_PT", "Norbuprenorphine_Ba2+_PT", "Naloxone_Ba2+_PT"]
INACT_POOL = {"Verapamil_Ca2+_PT", "Verapamil_Ca2+_RT", "Diltiazem_Ba2+_PT"}   # paper-evidenced inactivated-state preference
OPEN_POOL = {"Norbuprenorphine_Ca2+_PT"}                                       # paper-evidenced open-state preference (Ca2+)
# crit-4 five rows: row name -> (source, expectation, paper basis)
J4_ROWS = {
    "vp-Ca-PT":   ("Verapamil_Ca2+_PT drugwin",  ">1.2",  "step0.9/ramp0.4=2.25×"),
    "vp-Ca-RT":   ("Verapamil_Ca2+_RT drugwin",  "eq",     "step1.5/ramp1.6≈0.94"),
    "vp-Ba-PT":   ("Ba2+ group verapamil tail (100uM)", ">1.2",  "step1.5/ramp0.3=5x; measured tail residual step 20-32%/ramp 2-4%"),
    "norbu-Ca-PT": ("Norbuprenorphine_Ca2+_PT drugwin", "eq", "paper: no step/ramp difference"),
    "dilt-Ba-PT": ("Diltiazem_Ba2+_PT drugwin", ">1.2",  "paper: Ba2+ ramp>step"),
}
EPS_ABS = 0.03     # hERG card-5 absolute line
KAP_LINE = 0.10    # hERG card-6 time/amplitude boundary
TAF_GRID = (4.0, 6.0, 8.0, 12.0, 16.0)   # as in Ca-3
AICC_MARGIN = 10.0

# ================= waveform level =================
def subtract_ipassive(t, i, v):
    """Paper-method Ipassive: linear extrapolation from two points (-73,-63), subtracted pointwise."""
    m73 = (t >= I73_WIN[0]) & (t < I73_WIN[1]); m63 = (t >= I63_WIN[0]) & (t < I63_WIN[1])
    if m73.sum() < 10 or m63.sum() < 10:
        return None
    i73, i63 = float(i[m73].mean()), float(i[m63].mean())
    v73, v63 = float(v[m73].mean()), float(v[m63].mean())
    if abs(v63 - v73) < 3:
        return None
    slope = (i63 - i73) / (v63 - v73)
    return i - (i73 + slope * (v - v73))

def f_inact_seg(seg):
    """As in Ca-1b: f = 1 - (last-10ms mean)/peak."""
    pk = float(seg.min())
    if pk > -50:
        return None
    return 1.0 - float(np.mean(seg[-100:])) / pk

def h0(t, C, A, tau): return C + A * np.exp(-t / tau)
def h1(t, C, A, tau, w, tf): return C + A * ((1 - w) * np.exp(-t / tau) + w * np.exp(-t / tf))
def aicc(n, k, rss):
    if rss <= 0 or n <= k + 1: return np.inf
    return n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)

def fit_grid(t, y):
    """As in Ca-3: H0 vs H1 (tau_f grid joint selection)."""
    out = {"H0": None, "H1": None, "pick": None}
    try:
        p0, _ = curve_fit(h0, t, y, p0=[0.5, 0.5, 30.0],
                          bounds=([-0.2, 0.0, 0.5], [1.2, 1.5, 300.0]), maxfev=20000)
        rss0 = float(np.sum((y - h0(t, *p0)) ** 2))
        out["H0"] = {"aicc": aicc(len(y), 3, rss0)}
    except Exception:
        pass
    best = None
    for tf in TAF_GRID:
        try:
            p1, _ = curve_fit(lambda tt, C, A, tau, w: h1(tt, C, A, tau, w, tf), t, y,
                              p0=[0.4, 0.6, 40.0, 0.3],
                              bounds=([-0.2, 0.0, 0.5, 0.0], [1.2, 1.5, 300.0, 1.0]), maxfev=40000)
            rss1 = float(np.sum((y - h1(t, *p1, tf)) ** 2))
            a1 = aicc(len(y), 4, rss1)
            if best is None or a1 < best[0]:
                best = (a1, tf, p1)
        except Exception:
            pass
    if best is not None:
        out["H1"] = {"w": float(best[2][3]), "tauf": float(best[1]), "tau": float(best[2][2]), "aicc": float(best[0])}
    if out["H0"] and out["H1"]:
        out["pick"] = "H1" if out["H1"]["aicc"] < out["H0"]["aicc"] - AICC_MARGIN else "H0"
    return out

def norm_wf(seg_mean):
    pk = float(seg_mean.min())
    if pk > -30:
        return None
    return seg_mean / abs(pk)

def eps_kappa(wc, wd):
    """eps_x (normalized RMS shape difference) + kappa* (optimal-stretch warp) + eps_x1 (post-warp residual). Comparison domain [0,26]ms."""
    dom = TMS40 <= 26.0
    t = TMS40[dom]; a = wc[dom]; b = wd[dom]
    ex = float(np.sqrt(np.mean((a - b) ** 2)))
    best = (ex, 0.0)
    for kap in np.arange(-0.5, 0.5001, 0.005):
        tw = t * (1.0 + kap)
        bw = np.interp(t, tw, wd[dom])
        r = float(np.sqrt(np.mean((a - bw) ** 2)))
        if r < best[0]:
            best = (r, float(kap))
    return {"eps_x": ex, "kappa": best[1], "eps_x1": best[0]}

# ================= file level =================
def load_file(path):
    import pyabf
    a = pyabf.ABF(path)
    sw = []
    for s in range(a.sweepCount):
        a.setSweep(s, channel=0); i = a.sweepY.astype(float); t = a.sweepX
        a.setSweep(s, channel=1); v = a.sweepY.astype(float)
        a.setSweep(s, channel=2); bt = a.sweepY.astype(float)
        isub = subtract_ipassive(t, i, v)
        if isub is None:
            continue
        ms = (t >= STEP[0]) & (t < STEP[1]); mr = (t >= RAMP[0]) & (t < RAMP[1])
        if ms.sum() != 400 or mr.sum() < 900:
            continue
        seg = isub[ms]; rseg = isub[mr]
        sw.append({"s": s, "T": float(bt.mean()), "step_seg": seg,
                   "step_pk": float(seg.min()), "ramp_pk": float(rseg.min())})
    return sw

def _logexp(s, A, tau, C):
    return np.log(np.maximum(A * np.exp(-s / tau) + C, 1.0))

def segment_cell(x):
    """x = ramp_pk trajectory (negative). dev method: anchor [8,45] exponential+plateau rundown model extrapolated; sustained negative deviation = inhibition / positive = facilitation."""
    N = len(x)
    if N < 70:
        return {"flag": "N<70"}
    x = np.asarray(x, float)
    c0 = float(np.median(x[10:30]))
    if c0 > -80:
        return {"flag": "small_or_outward"}
    thr = max(0.15 * abs(c0), 60.0)
    t0 = N
    while t0 > 0 and abs(x[t0 - 1]) < thr:
        t0 -= 1
    tail = (int(t0), N) if N - t0 >= 5 else None
    bend = tail[0] if tail else N
    xs = medfilt(np.abs(x[:bend]), 5)
    lx = np.log(np.maximum(xs, 1.0))
    sa = np.arange(8, min(45, bend))
    if len(sa) < 12:
        return {"flag": "anchor_short"}
    base = None
    try:  # exponential+plateau rundown model (free offset, avoids misreading decelerating bend as facilitation)
        x8 = float(np.abs(x[8]))
        p_, _ = curve_fit(_logexp, sa, lx[sa], p0=[max(x8 - 0.3 * x8, 50), 60.0, max(0.3 * x8, 20)],
                          bounds=([1.0, 5.0, 1.0], [1e6, 2000.0, x8]), maxfev=20000)
        base = _logexp(np.arange(bend), *p_)
    except Exception:
        try:
            sl_, ic_ = np.polyfit(sa, lx[sa], 1)
            base = sl_ * np.arange(bend) + ic_
        except Exception:
            return {"flag": "anchor_fit"}
    dev = lx - base
    onset, facil = None, False
    for s in range(30, bend - 4):
        e = min(s + 8, bend)
        if e - s < 4:
            continue
        if np.all(dev[s:e] < -0.15):
            onset = s; break
        if np.all(dev[s:e] > 0.12):
            onset = s; facil = True; break
    if onset is None:
        pc = (15, 35) if bend >= 50 else None
        return {"flag": "no_drug_region", "tail": tail, "pseudo_ctrl": pc}
    if onset < 48:
        return {"flag": "ctrl_short", "onset": onset}
    ctrl = (onset - 30, onset - 10)         # control window 20 (leaving a 10-sweep onset buffer)
    drug = (bend - 12, bend - 2)            # drug window = last 10 (before the tail)
    if drug[1] - drug[0] < 8 or drug[0] <= ctrl[1]:
        return {"flag": "drug_short", "onset": onset}
    dw = np.abs(x[drug[0]:drug[1]])
    dsl = np.median(np.diff(medfilt(dw, 5)))
    if abs(dsl) > max(10.0, 0.05 * float(np.median(dw))):
        return {"flag": "drug_not_plateau", "onset": onset}
    return {"ctrl": ctrl, "drug": drug, "tail": tail, "facil": facil, "onset": onset}

def cell_metrics(sw, seg):
    """All metrics for one cell."""
    x = np.array([r["ramp_pk"] for r in sw])
    out = {}
    c0, c1 = seg["ctrl"]; d0, d1 = seg["drug"]
    ctrl_sw = sw[c0:c1]; drug_sw = sw[d0:d1]
    if len(ctrl_sw) < 15 or len(drug_sw) < 8:
        return {"flag": "window_sparse"}
    sc = np.array([r["step_pk"] for r in ctrl_sw]); sd = np.array([r["step_pk"] for r in drug_sw])
    rc = np.array([r["ramp_pk"] for r in ctrl_sw]); rd = np.array([r["ramp_pk"] for r in drug_sw])
    out["b_step"] = 1.0 - abs(np.median(sd)) / abs(np.median(sc))
    out["b_ramp"] = 1.0 - abs(np.median(rd)) / abs(np.median(rc))
    if not seg.get("facil") and out["b_ramp"] < 0.15:
        return {"flag": "weak_effect", **out}
    if seg.get("facil") and out["b_ramp"] > -0.10:
        return {"flag": "weak_effect", **out}
    # shape (mean waveform of the step segment)
    Wc = norm_wf(np.mean([r["step_seg"] for r in ctrl_sw], axis=0))
    Wd = norm_wf(np.mean([r["step_seg"] for r in drug_sw], axis=0))
    if Wc is None or Wd is None:
        return {"flag": "wf_bad", **out}
    out.update(eps_kappa(Wc, Wd))
    # f_inact
    fc = [f_inact_seg(r["step_seg"]) for r in ctrl_sw]; fd = [f_inact_seg(r["step_seg"]) for r in drug_sw]
    fc = [f for f in fc if f is not None]; fd = [f for f in fd if f is not None]
    if len(fc) >= 10 and len(fd) >= 6:
        out["f_ctrl"] = float(np.median(fc)); out["f_drug"] = float(np.median(fd))
        out["df"] = out["f_drug"] - out["f_ctrl"]
    # w_fast (Ca-3 fitter, registry)
    gc = fit_grid(TMS40, Wc); gd = fit_grid(TMS40, Wd)
    if gc.get("H1"): out["w_ctrl"] = gc["H1"]["w"]; out["pick_ctrl"] = gc["pick"]
    if gd.get("H1"): out["w_drug"] = gd["H1"]["w"]; out["pick_drug"] = gd["pick"]
    if "w_ctrl" in out and "w_drug" in out:
        out["dw"] = out["w_drug"] - out["w_ctrl"]
    # Null-A: control window split in halves (odd/even)
    e1 = [r["step_seg"] for k, r in enumerate(ctrl_sw) if k % 2 == 0]
    e2 = [r["step_seg"] for k, r in enumerate(ctrl_sw) if k % 2 == 1]
    if len(e1) >= 6 and len(e2) >= 6:
        Wa = norm_wf(np.mean(e1, axis=0)); Wb = norm_wf(np.mean(e2, axis=0))
        if Wa is not None and Wb is not None:
            out["nullA_eps"] = eps_kappa(Wa, Wb)["eps_x"]
    # Null-B: first 10 vs last 10 of the control segment (gap >=10)
    on = seg["onset"]
    if on - 40 >= 6:
        eb = sw[on - 40:on - 30]; lb = sw[on - 20:on - 10]
        We = norm_wf(np.mean([r["step_seg"] for r in eb], axis=0))
        Wl = norm_wf(np.mean([r["step_seg"] for r in lb], axis=0))
        if We is not None and Wl is not None:
            out["nullB_eps"] = eps_kappa(We, Wl)["eps_x"]
        fe = [f_inact_seg(r["step_seg"]) for r in eb]; fl = [f_inact_seg(r["step_seg"]) for r in lb]
        fe = [f for f in fe if f is not None]; fl = [f for f in fl if f is not None]
        if len(fe) >= 6 and len(fl) >= 6:
            out["nullB_df"] = float(np.median(fl) - np.median(fe))
    # temperature
    out["T_med"] = float(np.median([r["T"] for r in sw]))
    out["n_sweeps"] = len(sw)
    return out

def tail_metrics(sw, seg):
    """Inhibition of the verapamil tail (100uM) relative to the control window (drug-free cells use pseudo_ctrl=(15,35)); crit-4 row vp-Ba-PT."""
    if not seg.get("tail"):
        return None
    ctrl = seg.get("ctrl") or seg.get("pseudo_ctrl")
    if ctrl is None:
        return None
    c0, c1 = ctrl; t0, t1 = seg["tail"]
    if t1 - t0 < 5 or c1 - c0 < 15:
        return None
    cs = sw[c0:c1]; ts = sw[t0:t1]
    b_step = 1.0 - abs(np.median([r["step_pk"] for r in ts])) / abs(np.median([r["step_pk"] for r in cs]))
    b_ramp = 1.0 - abs(np.median([r["ramp_pk"] for r in ts])) / abs(np.median([r["ramp_pk"] for r in cs]))
    return {"b_step": float(b_step), "b_ramp": float(b_ramp)}

# ================= smoke =================
def synth_step(shape_w=0.30, tf=8.0, ts=50.0, C=0.40, warp=0.0, morph=0.0, noise=0.01, rng=None):
    rng = rng or np.random.default_rng(0)
    t = TMS40 * (1.0 + warp)
    y = C + (1 - C) * ((1 - shape_w) * np.exp(-t / ts) + shape_w * np.exp(-t / tf))
    if morph > 0:  # blend toward the single-exponential slow morphology (amplitude-type warp)
        y2 = C + (1 - C) * np.exp(-t / ts)
        y = (1 - morph) * y + morph * y2
    y = y + rng.normal(0, noise, len(y))
    return -y * 1000.0  # negative current, pA scale (norm_wf gate -30)

def smoke():
    print("=" * 72, flush=True)
    print(" [smoke Ca-5] S1 eps_x/kappa* recovery | S2 segmenter | S3 real-file pipeline", flush=True)
    rng = np.random.default_rng(3)
    ok_all = True
    # S1: eps_x truth recovery (noiseless truth vs noise-corrupted estimate, ±25% / zero band <=0.03); kappa* injected 0.10 recovery
    print(" S1 shape-estimator recovery:", flush=True)
    ok1 = True
    for morph, name in ((0.0, "zero"), (0.35, "mid"), (1.0, "high")):
        true_ex = eps_kappa(norm_wf(synth_step(morph=0.0, noise=0.0, rng=rng)),
                            norm_wf(synth_step(morph=morph, noise=0.0, rng=rng)))["eps_x"]
        ests = []
        for _ in range(12):
            wc = norm_wf(synth_step(noise=0.008, rng=rng))
            wd = norm_wf(synth_step(morph=morph, noise=0.008, rng=rng))
            ests.append(eps_kappa(wc, wd)["eps_x"])
        med = float(np.median(ests))
        ok = (med <= 0.03) if true_ex < 0.02 else (0.75 * true_ex <= med <= 1.33 * true_ex)
        ok1 &= ok
        print(f"   eps_x {name}: truth {true_ex:.3f} estimate median {med:.3f} (±25%/zero band<=0.03)", "pass" if ok else "fail", flush=True)
    krs = []
    for _ in range(12):
        wc = norm_wf(synth_step(noise=0.008, rng=rng))
        wd = norm_wf(synth_step(warp=0.10, noise=0.008, rng=rng))
        krs.append(eps_kappa(wc, wd)["kappa"])
    okk = abs(float(np.median(krs)) - 0.10) <= 0.02
    ok1 &= okk
    print(f"   kappa* injected 0.10: recovered median {np.median(krs):+.3f} (|d|<=0.02)", "pass" if okk else "fail", flush=True)
    ok_all &= ok1
    # S2: segmenter synthesis (rundown + two drug plateaus + tail collapse)
    N = 200
    x = -1500 * np.exp(-np.arange(N) / 60.0) - 600 + rng.normal(0, 12, N)
    tr0, tr1 = 90, 150
    x[tr0:tr0 + 15] += np.linspace(0, 250, 15)     # drug 1 onset (|x| drops by 250)
    x[tr0 + 15:tr1] += 250
    x[tr1:tr1 + 15] += np.linspace(0, 150, 15)     # drug 2 onset
    x[tr1 + 15:] += 150
    x[188:] = -40 + rng.normal(0, 6, 12)           # verapamil tail
    seg = segment_cell(x)
    ok2 = ("onset" in seg and tr0 <= seg["onset"] <= tr0 + 15 and seg.get("tail") == (188, N)
           and "drug" in seg and seg["drug"][1] in (186, 187))
    print(f" S2 segmentation: onset={seg.get('onset')} (true {tr0}±3) tail={seg.get('tail')} drug={seg.get('drug')}",
          "pass" if ok2 else f"fail({seg})", flush=True)
    x2 = -1500 * np.exp(-np.arange(N) / 60.0) - 600 + rng.normal(0, 12, N)  # drug-free pure rundown
    x2[188:] = -40 + rng.normal(0, 6, 12)
    seg2 = segment_cell(x2)
    ok2b = seg2.get("flag") == "no_drug_region"
    print(f" S2b drug-free cell: flag={seg2.get('flag')} (expected no_drug_region)", "pass" if ok2b else "fail", flush=True)
    ok_all &= ok2 and ok2b
    # S3: three real files run through (no verdict)
    try:
        base = os.path.join(DATA, "Verapamil_Ca2+_PT")
        for fn in ["18821006.abf", "PT_18430001.abf", "ALR_19322000.abf"]:
            sw = load_file(os.path.join(base, fn))
            xx = np.array([r["ramp_pk"] for r in sw])
            sg = segment_cell(xx)
            info = f"flag={sg.get('flag')}" if "ctrl" not in sg else f"ctrl={sg['ctrl']} drug={sg['drug']} tail={sg.get('tail')}"
            print(f" S3 {fn}: n={len(sw)} {info}", flush=True)
        print(" S3 pipeline pass", flush=True)
    except Exception as e:
        ok_all = False
        print(" S3 fail:", e, flush=True)
    print(" smoke overall:", "all pass -> full run allowed" if ok_all else "not all pass -> full run forbidden", flush=True)
    return ok_all

# ================= full run =================
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("=" * 78, flush=True)
    print(" Ca-5: CaV1.2 drug shape-modulation verdict (preregistration v1.0 full run)", flush=True)
    print("=" * 78, flush=True)
    cells = []   # one entry per valid drug cell
    flags = {}   # exclusion registry
    tails_ba = []  # Ba2+ group verapamil tails (crit-4 row vp-Ba-PT)
    for g in GROUPS:
        files = sorted(glob.glob(os.path.join(DATA, g, "*.abf")))
        n_ok = 0
        for p in files:
            fn = os.path.basename(p)
            try:
                sw = load_file(p)
            except Exception as e:
                flags.setdefault("load_err", []).append(f"{g}/{fn}:{e}"); continue
            if len(sw) < 70:
                flags.setdefault("N<70", []).append(f"{g}/{fn}"); continue
            x = np.array([r["ramp_pk"] for r in sw])
            seg = segment_cell(x)
            if "ctrl" not in seg:
                flags.setdefault(seg.get("flag", "?"), []).append(f"{g}/{fn}")
                # drug-free cells are also checked for a tail (still usable for the verapamil-tail row)
                tm = tail_metrics(sw, seg) if seg.get("tail") else None
                if tm and "Ba2+" in g:
                    tm2 = dict(tm); tm2["group"] = g; tm2["file"] = fn; tails_ba.append(tm2)
                continue
            m = cell_metrics(sw, seg)
            if "flag" in m:
                flags.setdefault(m["flag"], []).append(f"{g}/{fn}")
                tm = tail_metrics(sw, seg)
                if tm and "Ba2+" in g:
                    tm2 = dict(tm); tm2["group"] = g; tm2["file"] = fn; tails_ba.append(tm2)
                continue
            m["group"] = g; m["file"] = fn; m["facil"] = bool(seg.get("facil"))
            cells.append(m); n_ok += 1
            tm = tail_metrics(sw, seg)
            if tm and "Ba2+" in g:
                tm2 = dict(tm); tm2["group"] = g; tm2["file"] = fn; tails_ba.append(tm2)
        print(f"  [{g}] files {len(files)} valid drug cells {n_ok}", flush=True)

    print("\n[exclusion registry]", flush=True)
    for k, v in sorted(flags.items()):
        print(f"  {k}: {len(v)}", flush=True)

    # ---------- crit-1 ----------
    ex = np.array([c["eps_x"] for c in cells])
    nB = np.array([c["nullB_eps"] for c in cells if "nullB_eps" in c])
    nA = np.array([c["nullA_eps"] for c in cells if "nullA_eps" in c])
    p90B = float(np.percentile(nB, 90)) if len(nB) else float("nan")
    line1 = max(p90B, EPS_ABS)
    j1 = bool(np.median(ex) > line1)
    # ---------- crit-2 ----------
    kap = np.array([c["kappa"] for c in cells])
    ratio = np.array([c["eps_x1"] / c["eps_x"] for c in cells if c["eps_x"] > 1e-9])
    j2_amp = bool(np.median(kap) < KAP_LINE and np.median(ratio) > 0.5)
    # ---------- crit-3 ----------
    df_in = np.array([c["df"] for c in cells if c["group"] in INACT_POOL and "df" in c])
    df_op = np.array([c["df"] for c in cells if c["group"] in OPEN_POOL and "df" in c])
    ndf = np.abs(np.array([c["nullB_df"] for c in cells if "nullB_df" in c]))
    p90df = float(np.percentile(ndf, 90)) if len(ndf) else float("nan")
    if len(df_in) >= 8:
        bt = binomtest(int(np.sum(df_in > 0)), len(df_in), 0.5, alternative="greater")
        j3_in = {"n": len(df_in), "pos": int(np.sum(df_in > 0)), "p": float(bt.pvalue),
                 "med": float(np.median(df_in))}
    else:
        j3_in = {"n": len(df_in)}
    j3_op = {"n": len(df_op), "med": float(np.median(df_op)) if len(df_op) else None,
             "abs_med": float(np.median(np.abs(df_op))) if len(df_op) else None}
    j3 = bool(j3_in.get("p", 1) < 0.05 and j3_in.get("med", 0) > 0
              and (j3_op["abs_med"] is None or j3_op["abs_med"] < p90df))
    # ---------- crit-4 ----------
    rows = {}
    def row_ratio(ms):
        ms = list(ms)
        if len(ms) < 3:
            return None
        r = np.array([m["b_ramp"] / m["b_step"] for m in ms if m["b_step"] > 0.05])
        return {"n": len(r), "med": float(np.median(r))} if len(r) >= 3 else None
    rows["vp-Ca-PT"] = row_ratio([c for c in cells if c["group"] == "Verapamil_Ca2+_PT"])
    rows["vp-Ca-RT"] = row_ratio([c for c in cells if c["group"] == "Verapamil_Ca2+_RT"])
    rows["vp-Ba-PT"] = row_ratio(tails_ba)
    rows["norbu-Ca-PT"] = row_ratio([c for c in cells if c["group"] == "Norbuprenorphine_Ca2+_PT"])
    rows["dilt-Ba-PT"] = row_ratio([c for c in cells if c["group"] == "Diltiazem_Ba2+_PT"])
    n_pass4 = 0; detail4 = {}
    for k, exp in (("vp-Ca-PT", ">"), ("vp-Ca-RT", "eq"), ("vp-Ba-PT", ">"),
                   ("norbu-Ca-PT", "eq"), ("dilt-Ba-PT", ">")):
        r = rows.get(k)
        if r is None:
            detail4[k] = {"row": r, "pass": None}; continue
        ok = (r["med"] > 1.2) if exp == ">" else (0.8 <= r["med"] <= 1.25)
        n_pass4 += int(ok)
        detail4[k] = {"row": r, "expect": J4_ROWS[k][1], "pass": ok}
    j4 = n_pass4 >= 4

    verdict = {
        "n_cells": len(cells),
        "判1": {"eps_med": float(np.median(ex)), "nullB_p90": p90B, "nullA_med": float(np.median(nA)) if len(nA) else None,
                "line": line1, "pass": j1},
        "判2": {"kappa_med": float(np.median(kap)), "eps_ratio_med": float(np.median(ratio)), "pass_amp_type": j2_amp},
        "判3": {"inact": j3_in, "open": j3_op, "nullB_df_p90": p90df, "pass": j3},
        "判4": {"rows": detail4, "n_pass": n_pass4, "pass": j4},
        "per_group": {g: {"n": len([c for c in cells if c["group"] == g]),
                          "eps_med": float(np.median([c["eps_x"] for c in cells if c["group"] == g])) if any(c["group"] == g for c in cells) else None,
                          "df_med": float(np.median([c["df"] for c in cells if c["group"] == g and "df" in c])) if any(c["group"] == g and "df" in c for c in cells) else None}
                      for g in GROUPS},
    }
    print("\n" + "=" * 78, flush=True)
    print(" verdict components", flush=True)
    print(json.dumps(verdict, ensure_ascii=False, indent=2, default=str), flush=True)

    result = {"cells": [{k: v for k, v in c.items()} for c in cells],
              "flags": {k: v for k, v in flags.items()},
              "tails_ba": tails_ba, "verdict": verdict}
    with open(OUTJ, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=str)

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    ax = axes[0]
    for g, mk in (("Verapamil_Ca2+_PT", "18821006.abf"),):
        p = os.path.join(DATA, g, mk)
        sw = load_file(p); xx = np.array([r["ramp_pk"] for r in sw])
        sg = segment_cell(xx)
        ax.plot(xx, lw=1)
        if "ctrl" in sg:
            ax.axvspan(sg["ctrl"][0], sg["ctrl"][1], color="g", alpha=0.2, label="control window")
            ax.axvspan(sg["drug"][0], sg["drug"][1], color="r", alpha=0.2, label="drug window")
            if sg.get("tail"): ax.axvspan(sg["tail"][0], sg["tail"][1], color="k", alpha=0.2, label="verapamil tail")
        ax.legend(fontsize=8); ax.set_title("segmentation example Verapamil/18821006")
    ax = axes[1]
    ax.hist(ex, bins=30, alpha=0.6, label=f"drug eps_x median {np.median(ex):.3f}")
    if len(nB): ax.hist(nB, bins=30, alpha=0.6, label=f"Null-B p90={p90B:.3f}")
    ax.axvline(EPS_ABS, color="k", ls="--", label="0.03 absolute line")
    ax.legend(fontsize=8); ax.set_title("crit-1 shape warp vs dual null distributions")
    ax = axes[2]
    for arr, lab, c in ((df_in, f"inactivated pool n={len(df_in)}", "tab:red"),
                        (df_op, f"open pool n={len(df_op)}", "tab:blue")):
        if len(arr): ax.hist(arr, bins=20, alpha=0.6, label=lab, color=c)
    ax.axvline(0, color="k", lw=0.8)
    ax.legend(fontsize=8); ax.set_title("crit-3 Df_inact (drug - control)")
    ax = axes[3]
    names = list(detail4.keys()); meds = [detail4[k]["row"]["med"] if detail4[k]["row"] else np.nan for k in names]
    cols = ["tab:green" if detail4[k]["pass"] else ("tab:red" if detail4[k]["pass"] is False else "gray") for k in names]
    ax.bar(range(len(names)), meds, color=cols)
    ax.axhline(1.2, color="k", ls="--", lw=0.8); ax.axhline(0.8, color="k", ls=":", lw=0.8)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_title(f"crit-4 b_ramp/b_step cross-check {n_pass4}/5")
    fig.tight_layout(); fig.savefig(OUTP, dpi=140, bbox_inches="tight")
    print(f"\n results saved: {OUTJ}\n figure saved: {OUTP}", flush=True)

if __name__ == "__main__":
    if SMOKE:
        smoke()
    else:
        main()
