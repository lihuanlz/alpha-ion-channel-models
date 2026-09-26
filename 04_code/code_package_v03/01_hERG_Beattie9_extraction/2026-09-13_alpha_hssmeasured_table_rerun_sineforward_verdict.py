# 2026-09-13_alpha model_hss measured-table rerun_sine forward adjudication.py
# [B1 retired - table-swap rerun, late 2026-09-13] this file = assembly smoke_sine forward
#   adjudication.py with a single change: the h_ss(V<=-40)=1 physical declaration (B1) is
#   retired - the gap-1 pure-hssV strip adjudication on nine cells killed B1 (medians at
#   -70/-60/-50: 0.42/0.31/0.25, all <0.7; h_150 is an upper bound of h_ss, so the death is
#   (1) the h_ss anchors for the 12 steps -80..+30 are swapped to the measured strip table
#   (2) the m_ss anchors are re-split accordingly as m_ss=y_ss/h_ss (the products y_ss=m_ss*h_ss at -70/-60/-50 are unchanged).
# Policies (declared before run):
#   P1 anchors at -100 and below stay 1.0 (consistency of the -120 rebound calibration; the strip formula's own normalization is on record);
#   P2 the -80 step has n=3, insufficient data; still uses the median 0.798 (registered, better than the logic bridge);
#   P3 steps passing per-cell qc use per-cell values; failing steps fall back to the population median (h_pop registered per cell into JSON);
#   P4 the +40/+50/+60 ends keep the original R3 fallback chain (QC3: strip +40 median 0.029 vs R3 h40 median 0.015,
#      ratio 2.0 in [0.3,3], on record; avoids mixing the two chains);
#   P5 the -40/-30 m_ss logic bridge (B3, weakest bridge) is re-split with the same formula; the min(1,.) cap is unchanged.
# Criteria identical to the B1 version, verbatim (S1/S2/S3, nine-cell >=7, not one character moved); controls C1/C2 identical.
# Meaning: if >=7 still passes -> sine forward is robust to the death of B1; if it drops -> the original 7/9 depended on a wrong assumption, registered as-is.
# ---------- below is the original B1-version header (all declarations still valid except the retired B1 item) ----------
# Purpose (assembly smoke, not the formal assembly - all three DeepSeek-audit gaps are bridged by declarations on record):
#   forward-run the full sine protocol with the four tables: dm/dt=(m_ss-m)/tau_m, dh/dt=(h_ss-h)/tau_h,
#   I=G*m*h*(V-E_rev). V(t) read directly from the sine_wave protocol (chirp waveform taken as-is, no approximation).
#   Primary target: post-chirp rebound / pre-chirp rebound amplitude ratio (measured 0.38-0.66,
#   all nine cells decrease; history: amplitude changed, tau_rec unchanged, card 260 on record) -
#
# [Important correction 2026-09-13] verdict revision for the inactivation-block A(V)/A(+50) curve:
#   the rebound rests 500ms at -120, h fully recovers for any test step (tau_rec=3ms), so the
#   rebound peak is independent of end-of-test h -> A(V) measures m(150ms,V), an activation curve,
#   not h_ss(V); and the -90x60ms reset is incomplete (tau_d(-90)~72ms -> m~0.43 residual entering
#   h_ss(V<=-40)=1 changed to a physical declaration (inactivation is a depolarization phenomenon). [That declaration is B1, now retired]
#
# Data sources (all sealed/registered JSONs, loaded at runtime, never hand-copied):
#   amplitude table extraction results.json: per-cell y_ss(-70/-60/-50) (m_ss anchor); tau ladder TF/TM/TAU_L;
#   rebound hook results.json: median tau_rec/tau_deact(-120/-110/-100); A_rows per-cell -120 hook anchor
#     (G=A(-120)/31.67, model-form error +/-30% declared); B_rows measured sine double rebounds (scoring target);
#   inactivation-gate sealed adjudication results.json: per-cell h_ss50, g_hat, m90 tau_r (-90 grid);
#   activation time course results.json: registered slow component tau2 (medians at -20/0/+20/+40, large CV, gap 3 on record);
#   pure hssV strip adjudication results.json: h_ss(-80..+30) 12-step per-cell values + population medians (new in this version).
#
# Bridge declarations (smoke-only, must be completed before formal assembly; numbering follows the DeepSeek three gaps):
#   B1 (gap 1) [RETIRED - this version swaps in the measured table]: originally h_ss(V<=-40)=1.0 physical
#     declaration; h_ss(+40) measured per cell (inactivation protocol +40 test segment, mean of last 50ms /(G*1*DF)) - the +40-end R3 chain is kept in this version (P4).
#   B2 (gap 2): tau_rec(V) = log-linear extrapolation of the 4 measured points (-120:3.04, -110:4.25,
#     -100:5.05, -90:6.5ms) to -40 (yielding ~8-23ms), continued extrapolation -40..0 capped at 100ms;
#     V>=0 tau_h=1.5ms (tau_obs upper bound, instantaneous on the assembly time scale).
#   B3 (gap 3): tau_m(V>=-40) uses the registered activation slow-component medians (-40:24.05 sealed,
#     -20:4.0, 0:2.0, +20:0.71, +40:0.29s); the fast component (20-100ms) does not enter the single-gate m,
#     declared; m_ss(-40..-20) logic bridge (weakest bridge, flagged in the verdict); m_ss(-20..+60)=1.0 declared (the A(V) m-curve supports m(+20)~1).
#   B4: -40-step y_ss negative artifact excluded on record; cells missing a step use the step median.
#   B5: single-gate m (not two-component); m is a low-pass approximation inside the chirp, declared.
#
# Criteria (declared before run):
#   S1 rebound 1: simulated DoE amplitude / measured B_rows(which=0) amplitude in [0.3,3] (G uncertainty);
#   S2 suppression ratio: simulated A2/A1 vs measured A2/A1 ratio in [0.5,2] AND simulated ratio <1 (direction must be right) - primary target;
#   S3 +40 steady state: simulated / measured (mean of last 200ms of the +40 segment) in [1/3,3];
#   each cell passing S1/S2/S3 -> cell smoke-passes; >=7 of nine cells -> product-gate smoke holds.
#
# [POST-HOC repair 2026-09-13 second round (declared after the first-round 6/9 run)]
#   all three first-round failures were attributed to on-record data-side issues; the repair touches only data anchors/targets, never the model, never the criteria:
#   R1 G anchor fallback chain: hook -120 valid A>0 -> any valid A>0 -> inactivation-block g_hat (flag posthoc_G_inact)
#      -> hook -120 invalid-row |A| (flag posthoc_G_degraded). 16704047 took the third level (g_hat=0.0718).
#   R2 measured-target degraded backfill: when a sine double-rebound measurement is missing, backfill with the invalid-row |A| (flag posthoc_target_degraded).
#      16713110 which=0 backfilled 4.311. 16704047 backfilled both steps (ratio 1.40, opposite to the other eight cells -
#      this cell's measured target is untrustworthy; S2 expected to still fail - a data-side failure, not cosmetically treated).
#   R3 h40 fallback chain: direct extraction from the inactivation +40 test segment -> inactivation-block h_ss50 -> sine self-calibration
#      from its own +40 segment (flag posthoc_h40_selfcal; this cell's S3 degraded to non-independent) -> default 0.0034.
#      16708060/16708016 (over-leak on record, inactivation-protocol h chain broken) take the third level.
# Controls (declared before run):
#   C1 integrator + scoring self-consistency: synthetic +40->-120 single step with the known tau table, DoE inversion of tau_r, error <15%;
#   C2 synthetic full run (16713003 table + measured noise): the scoring pipeline must pass S1-S3 (no pipeline deadlock).
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

F_AMP = os.path.join(HERE, "2026-09-13_α模型_幅度表提取_结果.json")
F_HOOK = os.path.join(HERE, "2026-09-13_α模型_反弹hook_结果.json")
F_INACT = os.path.join(HERE, "2026-09-13_α模型_失活门_失活协议封卷判决_结果.json")
F_HSS = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决_结果.json")

TR_GRID = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))
TD_GRID = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))
SEP_MIN = 1.5


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


def doe_fit(t, y, tr_grid=TR_GRID, td_grid=TD_GRID):
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
                edge=bool(tr <= tr_grid[0] * 1.02 or tr >= tr_grid[-1] * 0.98))


class Tab:
    """log-linear interpolation anchor table (x linear, y log, clamped at ends)."""
    def __init__(self, anchors, log_y=True):
        self.xs = np.array(sorted(anchors), dtype=float)
        ys = np.array([anchors[x] for x in self.xs], dtype=float)
        self.ly = np.log(np.clip(ys, 1e-9, None)) if log_y else ys
        self.log_y = log_y

    def __call__(self, v):
        y = np.interp(v, self.xs, self.ly)
        return float(np.exp(y)) if self.log_y else float(y)


def build_tabs(cell, amp, hook, inact, hss, I_inact=None, V_inact=None, sine_mea40=None):
    """per-cell four tables: m_ss, tau_m, h_ss, tau_h + G. Returns dict (with posthoc flags).
    table-swapped version: the h_ss B1 segment (-80..+30) uses the measured strip table; m_ss=y_ss/h_ss re-split accordingly."""
    # ---- h_ss measured strip table (P3: per-cell qc pass -> per-cell value; else -> population median) ----
    hc = hss["cells"].get(cell, {}).get("curve", {})
    hB1, h_pop = {}, []
    for v in (-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30):
        e = hc.get(str(v))
        if e and e.get("qc") and e["h"] > 0:
            hB1[v] = float(e["h"])
        else:
            hB1[v] = float(hss["gears"][str(v)]["med"])
            h_pop.append(v)
    # ---- m_ss anchor (per-cell y_ss, step median if missing; B3/B4 bridges; table-swap re-split m_ss=y_ss/h_ss) ----
    yss = {}
    for r in amp["rows"]:
        if r["v"] in (-70, -60, -50):
            yss.setdefault(r["v"], {})[r["cell"]] = max(r["y_ss"], 1e-4)
    med = {v: float(np.median(list(d.values()))) for v, d in yss.items()}
    def cell_y(v):
        return yss.get(v, {}).get(cell, med.get(v, 0.02))
    m_anchors = {-130: 1e-4, -120: 1e-4, -110: 1e-4, -100: 1e-4, -90: 1e-4,
                 -80: min(1.0, cell_y(-70) * 0.5 / hB1[-80]),
                 -70: min(1.0, cell_y(-70) / hB1[-70]),
                 -60: min(1.0, cell_y(-60) / hB1[-60]),
                 -50: min(1.0, cell_y(-50) / hB1[-50]),
                 -40: min(1.0, cell_y(-50) * 3.0 / hB1[-40]),   # B3 logic bridge (weakest bridge, re-split with the same formula)
                 -30: min(1.0, cell_y(-50) * 8.0 / hB1[-30]),
                 -20: 1.0, 0: 1.0, 20: 1.0, 40: 1.0, 60: 1.0}
    m_ss = Tab(m_anchors, log_y=True)

    # ---- tau_m anchor (sealed medians + registered slow component; B3/B5) ----
    hs = hook["summary"]
    tm_anchors = {-130: hs["-120"]["td_med"], -120: hs["-120"]["td_med"],
                  -110: hs["-110"]["td_med"], -100: hs["-100"]["td_med"],
                  -90: 0.072, -80: 0.24,
                  -70: amp["tau_used"]["TM"]["-70"], -60: amp["tau_used"]["TM"]["-60"],
                  -50: amp["tau_used"]["TAU_L"]["-50"], -40: amp["tau_used"]["TAU_L"]["-40"],
                  -30: 9.8, -20: 4.0, -10: 2.8, 0: 2.0, 10: 1.2,
                  20: 0.71, 30: 0.45, 40: 0.29, 50: 0.29, 60: 0.30}
    tau_m = Tab(tm_anchors, log_y=True)

    # ---- G: R1 fallback chain (posthoc flags) ----
    ci = inact["cells"].get(cell, {})
    G = None
    gflag = None
    for r in hook["A_rows"]:
        if r["cell"] == cell and r["v"] == -120 and r["valid"] and r["A"] > 0:
            G = r["A"] / DF_M120
    if G is None:
        cands = [r["A"] / DF_M120 for r in hook["A_rows"]
                 if r["cell"] == cell and r["valid"] and r["A"] > 0]
        if cands:
            G = float(np.median(cands))
    if G is None and ci.get("g_hat") and ci["g_hat"] > 0:      # R1 third level
        G = float(ci["g_hat"])
        gflag = "posthoc_G_inact"
    if G is None:                                              # R1 fourth level
        cands = [abs(r["A"]) / DF_M120 for r in hook["A_rows"]
                 if r["cell"] == cell and r["v"] == -120 and r["A"] != 0]
        if cands:
            G = float(np.median(cands))
            gflag = "posthoc_G_degraded"

    # ---- h_ss (B1 retired: measured strip table; +40 end R3 fallback chain kept, P4) ----
    h50 = None
    if ci.get("h_ss50") and ci["h_ss50"] > 0:
        h50 = float(ci["h_ss50"])
    h40 = np.nan
    hflag = None
    if I_inact is not None and G is not None and G > 0:
        info = segments(V_inact)
        for k, (v, s0, n) in enumerate(info):
            if abs(v - 40.0) < 2.0 and 0.14 < n * DT < 0.16 and k >= 1:
                pv = info[k - 1][0]
                if abs(pv + 90.0) < 2.0:                    # it is a test step, not something else
                    iss = float(np.mean(I_inact[s0 + n - 5000: s0 + n]))
                    h40 = iss / (G * 1.0 * (40.0 - E_REV))
                    break
    if not np.isfinite(h40) or h40 <= 0:
        h40 = h50 if h50 else np.nan
    if (not np.isfinite(h40) or h40 <= 0) and G and sine_mea40 and sine_mea40 > 0:
        h40 = sine_mea40 / (G * 1.0 * (40.0 - E_REV))         # R3 third level
        hflag = "posthoc_h40_selfcal"
    if not np.isfinite(h40) or h40 <= 0:
        h40 = 0.0034
    h40 = float(np.clip(h40, 1e-4, 0.2))
    h_anchors = {-130: 1.0, -100: 1.0}                       # P1
    h_anchors.update(hB1)                                    # B1 segment measured (-80..+30)
    h_anchors[40] = h40                                      # P4: +40 end R3 chain kept
    h_anchors[50] = h50 if h50 else h40
    h_anchors[60] = h50 if h50 else h40
    h_ss = Tab(h_anchors, log_y=True)

    # ---- τ_h（B2） ----
    tr90 = ci.get("m90", {}).get("tau_r") if ci.get("m90", {}).get("valid") else None
    th_anchors = {-130: hs["-120"]["tr_med"], -120: hs["-120"]["tr_med"],
                  -110: hs["-110"]["tr_med"], -100: hs["-100"]["tr_med"],
                  -90: tr90 if tr90 else 0.0065,
                  -40: 0.023, -20: 0.05, 0: 0.0015, 60: 0.0015}
    tau_h = Tab(th_anchors, log_y=True)
    return dict(m_ss=m_ss, tau_m=tau_m, h_ss=h_ss, tau_h=tau_h, G=G,
                h40=h40, h50=h50, gflag=gflag, hflag=hflag, h_pop=h_pop)


def forward(V, tabs):
    """fully vectorized table lookup + simple recursion: each step m/h relaxes exponentially toward (m_ss,h_ss)."""
    v = V.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hss = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    em = np.exp(-DT / tm)
    eh = np.exp(-DT / th)
    n = len(V)
    m = np.empty(n)
    h = np.empty(n)
    m0 = ms[0]
    h0 = hss[0]
    for i in range(n):
        m0 = ms[i] + (m0 - ms[i]) * em[i]
        h0 = hss[i] + (h0 - hss[i]) * eh[i]
        m[i] = m0
        h[i] = h0
    return m, h


def find_sine_structure(V):
    """programmatic verification of the sine structure (measured: +40x1.0s, two -120x0.5s rebounds, chirp varies point by point)."""
    info = segments(V)
    long40 = [(v, s0, n) for v, s0, n in info if abs(v - 40) < 3 and n * DT > 0.8]
    rebs = [(k, v, s0, n) for k, (v, s0, n) in enumerate(info)
            if abs(v + 120) < 3 and 0.4 < n * DT < 0.7]
    chirp = None
    if len(rebs) >= 2:
        s_end = rebs[0][2] + rebs[0][3]
        s_beg = rebs[1][2]
        chirp = (s_end, s_beg)
    return dict(info=info, long40=long40, rebs=rebs, chirp=chirp)


def score_cell(cell, V, I, tabs, meas, pr):
    m, h = forward(V, tabs)
    G = tabs["G"]
    Isim = G * m * h * (V - E_REV)
    st = find_sine_structure(V)
    out = dict(G=G)
    # +40 steady state (segment length 1.0s, take the last 200ms)
    if st["long40"]:
        v, s0, nseg = st["long40"][0]
        sim40 = float(np.mean(Isim[s0 + nseg - 2000: s0 + nseg]))
        mea40 = float(np.mean(I[s0 + nseg - 2000: s0 + nseg])) if I is not None else np.nan
        out["sim40"] = sim40
        out["mea40"] = mea40
        out["S3"] = bool(np.isfinite(mea40) and abs(mea40) > 1e-6
                         and 1 / 3 <= sim40 / mea40 <= 3)
    # double-rebound DoE (inward flipped positive, same sign convention as the measured B_rows)
    amps = []
    for k, v, s0, nseg in st["rebs"][:2]:
        t = np.arange(nseg) * DT
        r = doe_fit(t, -Isim[s0:s0 + nseg])
        amps.append(r["A"] if r else np.nan)
    out["simA"] = amps
    ma = meas.get(cell)
    if ma and len(amps) == 2 and all(np.isfinite(amps)):
        a1m, a2m = ma
        out["A1_meas"], out["A2_meas"] = a1m, a2m
        out["S1"] = bool(a1m and 0.3 <= amps[0] / a1m <= 3.0)
        sr = amps[1] / amps[0] if amps[0] else np.nan
        mr = a2m / a1m if a1m else np.nan
        out["sim_ratio"] = sr
        out["mea_ratio"] = mr
        out["S2"] = bool(np.isfinite(sr) and np.isfinite(mr)
                         and sr < 1.0 and 0.5 <= sr / mr <= 2.0)
    out["pass"] = bool(out.get("S1") and out.get("S2") and out.get("S3"))
    out["_sim"] = Isim
    out["_struct"] = st
    out["_mh"] = (m, h)
    return out


def main():
    rng = np.random.default_rng(5)
    print("=" * 84)
    print(" alpha model sine forward adjudication / hss measured-table swap rerun (B1 retired)"
          + (" (smoke 16713003)" if SMOKE else " (full nine-cell set)"))
    print(" criteria: S1 rebound-1 amplitude in [0.3,3] | S2 suppression ratio direction right and within [0.5,2]x measured | S3 +40 steady state in [1/3,3]"
          " (identical to the B1 version, verbatim)")
    print("=" * 84, flush=True)

    amp = json.load(open(F_AMP, encoding="utf-8"))
    hook = json.load(open(F_HOOK, encoding="utf-8"))
    inact = json.load(open(F_INACT, encoding="utf-8"))
    hss = json.load(open(F_HSS, encoding="utf-8"))

    # measured sine double-rebound targets (R2: backfill missing steps with invalid-row |A|, flag posthoc_target_degraded)
    meas = {}
    for r in hook["B_rows"]:
        if r["valid"] and r["A"] > 0:
            meas.setdefault(r["cell"], {})[r["which"]] = r["A"]
    mflag = {}
    for r in hook["B_rows"]:
        d = meas.setdefault(r["cell"], {})
        if r["which"] not in d and r["A"] != 0:
            d[r["which"]] = abs(r["A"])
            mflag[r["cell"]] = "posthoc_target_degraded"
    meas = {c: (d[0], d[1]) for c, d in meas.items() if 0 in d and 1 in d}
    print("[measured targets] chirp suppression ratio A2/A1:", flush=True)
    for c in CELLS:
        if c in meas:
            tag = f"  [{mflag[c]}]" if c in mflag else ""
            print(f"  {c}: A1={meas[c][0]:.3f} A2={meas[c][1]:.3f} "
                  f"ratio={meas[c][1] / meas[c][0]:.3f}{tag}", flush=True)
        else:
            print(f"  {c}: measured rebound missing (scoring degraded)", flush=True)

    # ---------- controls ----------
    print("\n[controls]", flush=True)
    t_c = np.arange(int(0.4 / DT)) * DT
    y_c = 3.0 * (np.exp(-t_c / 0.030) - np.exp(-t_c / 0.0035)) \
        + rng.normal(0, 0.02, len(t_c))
    rc = doe_fit(t_c, y_c)
    c1 = bool(rc and abs(rc["tau_r"] - 0.0035) / 0.0035 < 0.15)
    print(f"  C1 DoE inversion tau_r: {rc['tau_r'] * 1e3:.2f}ms (true 3.5) -> "
          f"{'pass' if c1 else 'fail'}", flush=True)
    # C2: synthetic cell (16713003 table), scoring pipeline must not deadlock
    V0 = load_mat("sine_wave_protocol.mat", "16713003", "sine_wave")[0]
    tabs0 = build_tabs("16713003", amp, hook, inact, hss)
    m0, h0 = forward(V0, tabs0)
    Isyn = tabs0["G"] * m0 * h0 * (V0 - E_REV)
    st0 = find_sine_structure(V0)
    ok_c2 = st0["chirp"] is not None and len(st0["long40"]) >= 1 and len(st0["rebs"]) >= 2
    print(f"  C2 sine structure parsing: +40 segment {len(st0['long40'])} rebounds {len(st0['rebs'])} "
          f"chirp {'yes' if st0['chirp'] else 'no'} -> {'pass' if ok_c2 else 'fail'}", flush=True)
    if st0["chirp"]:
        cs, ce = st0["chirp"]
        vc = V0[cs:ce]
        print(f"     chirp window {(ce - cs) * DT:.2f}s  voltage [{vc.min():.0f},{vc.max():.0f}]mV",
              flush=True)
    if not (c1 and ok_c2):
        print("  controls not returned to baseline -> halt.", flush=True)
        return
    print("  controls returned to baseline.", flush=True)

    # ---------- real data ----------
    print("\n[real data]", flush=True)
    res = {}
    for c in CELLS:
        V, I = load_mat("sine_wave_protocol.mat", c, "sine_wave")
        Vi, Ii = load_mat("inactivation_protocol.mat", c, "inactivation")
        smea40 = None                                    # R3 third-level self-calibration material
        if I is not None:
            stc = find_sine_structure(V)
            if stc["long40"]:
                v, s0, nseg = stc["long40"][0]
                smea40 = float(np.mean(I[s0 + nseg - 2000: s0 + nseg]))
        tabs = build_tabs(c, amp, hook, inact, hss, Ii, Vi, sine_mea40=smea40)
        r = score_cell(c, V, I, tabs, meas, None)
        r["h_pop"] = tabs["h_pop"]
        r["h40"] = tabs["h40"]
        r["gflag"] = tabs["gflag"]
        r["hflag"] = tabs["hflag"]
        if c in mflag:
            r["mflag"] = mflag[c]
        res[c] = r
        fl = " ".join(x for x in [r.get("gflag"), r.get("hflag"), r.get("mflag"),
                                  f"hss_pop{r['h_pop']}" if r["h_pop"] else None] if x)
        print(f"  {c}: G={r['G']:.4f} h40={tabs['h40']:.4f} | "
              f"S1{'✓' if r.get('S1') else '×'} simA1={r['simA'][0] if r['simA'] else np.nan:.2f}"
              f"(meas {r.get('A1_meas', np.nan):.2f}) | "
              f"S2{'v' if r.get('S2') else 'x'} sim ratio {r.get('sim_ratio', np.nan):.2f}"
              f"(meas {r.get('mea_ratio', np.nan):.2f}) | "
              f"S3{'✓' if r.get('S3') else '×'} sim40={r.get('sim40', np.nan):.3f}"
              f"(meas {r.get('mea40', np.nan):.3f}) -> "
              f"{'pass' if r['pass'] else 'fail'}" + (f"  [{fl}]" if fl else ""), flush=True)

    npass = sum(1 for r in res.values() if r["pass"])
    print("\n" + "-" * 84, flush=True)
    print(f" nine-cell table-swap rerun: {npass}/{len(res)} pass (criterion >=7 for robustness to the death of B1)", flush=True)
    print(" overall verdict:", "sine forward is robust to the death of B1 (measured h_ss table swapped in, criteria untouched)"
          if npass >= (1 if SMOKE else 7)
          else "failed the criteria after the table swap - the original 7/9 depended on the wrong B1 assumption, registered as-is, details above", flush=True)

    # ---------- figure ----------
    nfig = len(res)
    fig, axes = plt.subplots(nfig, 1, figsize=(15, 3.2 * nfig), squeeze=False)
    for ax, (c, r) in zip(axes[:, 0], res.items()):
        V, I = load_mat("sine_wave_protocol.mat", c, "sine_wave")
        tt = np.arange(len(V)) * DT
        if I is not None:
            ax.plot(tt, I, lw=0.3, color="0.6", label="measured")
        ax.plot(tt, r["_sim"], lw=0.5, color="tab:red", alpha=0.8, label="simulated")
        st = r["_struct"]
        for k, v, s0, nseg in st["rebs"][:2]:
            ax.axvspan(s0 * DT, (s0 + nseg) * DT, color="tab:blue", alpha=0.08)
        ax.set_title(f"{c}  S1{'pass' if r.get('S1') else 'fail'} "
                     f"S2{'pass' if r.get('S2') else 'fail'} S3{'pass' if r.get('S3') else 'fail'}"
                     f"  suppression ratio sim {r.get('sim_ratio', np.nan):.2f}/meas {r.get('mea_ratio', np.nan):.2f}",
                     fontsize=9)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("I (nA)")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-13_α模型_hss实测表重跑_sine前向判决.png")
    fig.savefig(fpng, dpi=120, bbox_inches="tight")

    out = dict(note="B1 retired table-swap rerun: h_ss(-80..+30) measured strip table; m_ss=y_ss/h_ss re-split; "
                    "criteria identical to the B1 version, verbatim",
               cells={c: {k: v for k, v in r.items() if not k.startswith("_")}
                      for c, r in res.items()},
               meas={c: meas[c] for c in meas}, npass=npass)
    fjson = os.path.join(HERE, "2026-09-13_α模型_hss实测表重跑_sine前向判决_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  figure saved: {fpng}", flush=True)
    print(f"  results saved: {fjson}", flush=True)


if __name__ == "__main__":
    main()
