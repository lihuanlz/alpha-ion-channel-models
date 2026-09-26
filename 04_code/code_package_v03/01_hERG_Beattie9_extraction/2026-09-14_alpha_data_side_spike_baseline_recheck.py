# 2026-09-14_alpha_data_side_spike_baseline_recheck.py
# Purpose (closing-document section 9 todo item 4; both data-side items done at once; model and criteria untouched):
#   item A: 16707014 AP spike re-evaluation (D4 on record) + population-wide measured-peak audit:
#     014's AP recording carries +/-8-9 nA single-sample impulse spikes (step-edge capacitive + ringing, D4 family);
#     repolarisation-window scoring takes max(I) -> the measured denominator may be inflated -> A1 peak ratio 0.21
#     [Smoke-fixing 2026-09-14] the initial "trace de-spiking" was self-mutilating: the true repolarisation-peak onset
#       edge (fast rise + overshoot) was misjudged as an artefact, moving 003's peak ratio 1.50->2.27 (51%). Smoke also
#       revealed a population-wide issue: the measured peaks of 003 and other cells themselves contain spike components
#       (single-sample overshoots of 0.4-1.3 nA riding on true peaks), so the raw-max scoring convention systematically
#       inflates denominators. Hence the switch to [scoring-side robust peak estimation]: measured repolarisation peak =
#       max(med5(I)) (5-point centred rolling median): single-sample impulses vanish under the median, sustained peaks
#     [Criteria, pinned before run]
#       A-main: 014's A1 peak ratio (robust convention) enters [0.3,3] and A2/A3 hold -> the 014 spike explanation
#         is confirmed;
#       A-audit (population): per-cell report raw ratio -> robust ratio; cells with |shift| > 15% registered as
#         "measured peak contains spike component"; cells whose pass state flips registered separately (SEAL-convention
#       A-reject: 014's peak-ratio shift < 15% -> spikes not the main cause; 014's failure is model-side, registered as-is.
#     reference column (no verdict): robust peak ratio under the k3 rectified-cascade forward (relevant if B6 seals).
#   item B: 16708060/16708016 baseline-offset recheck ("over-subtraction leak" on record):
#     nine cells x five protocols; collect all -80/-90 mV constant segments (>= 0.2 s, skipping the first 0.1 s)
#     and jointly estimate DC offset and sigma; protocols without a qualifying segment are marked "no coverage".
#     [Criterion, pinned before run] if any protocol of 060 or 016 has |offset| > 3x population MAD -> that cell's
#       baseline offset in that protocol is confirmed (named, magnitude registered as-is, re-subtraction deferred to
#       after formal assembly); otherwise the "over-subtraction leak" is invisible in silent segments, registration stands.
# Controls (declared pre-run): C1 DoE inversion tau_r error < 15%; C2 AP structure parsing 17 peaks + hook;
#   C3 robust-peak self-consistency: synthetic sustained peak (5 ms wide) + single-sample spikes +/-8 nA -> sustained
#     peak recovered with error < 5% and spike contribution < 10%.
# Run: python this file (nine cells full); SMOKE=1 forward on 16713003 only (baseline check still full).
import os
import json
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("SMOKE", "0") == "1"
CELLS_FWD = ["16713003"] if SMOKE else ["16704007", "16704047", "16707014", "16708016",
                                         "16708060", "16708118", "16713003", "16713110", "16715049"]
CELLS_ALL = ["16704007", "16704047", "16707014", "16708016", "16708060",
             "16708118", "16713003", "16713110", "16715049"]
DT = 1e-4
E_REV = -88.33
DF_M120 = abs(-120.0 - E_REV)
NOISE3 = 3 * 0.0069

F_AMP = os.path.join(HERE, "2026-09-13_α模型_幅度表提取_结果.json")
F_HOOK = os.path.join(HERE, "2026-09-13_α模型_反弹hook_结果.json")
F_INACT = os.path.join(HERE, "2026-09-13_α模型_失活门_失活协议封卷判决_结果.json")
F_HSS = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决_结果.json")
F_B6 = os.path.join(HERE, "2026-09-14_α模型_B6激活足_k3级联判决_双协议换形重跑_结果.json")

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


def med5(I):
    """5-point centred rolling median (edge-padded endpoints). Raw material for robust peak estimation."""
    return np.median(sliding_window_view(np.pad(I, 2, mode="edge"), 5), axis=1)


# ================= forward pipeline (hss-measured-table rerun version, copied verbatim) =================

class Tab:
    def __init__(self, anchors, log_y=True):
        self.xs = np.array(sorted(anchors), dtype=float)
        ys = np.array([anchors[x] for x in self.xs], dtype=float)
        self.ly = np.log(np.clip(ys, 1e-9, None)) if log_y else ys
        self.log_y = log_y

    def __call__(self, v):
        y = np.interp(v, self.xs, self.ly)
        return float(np.exp(y)) if self.log_y else float(y)


def build_tabs(cell, amp, hook, inact, hss, I_inact=None, V_inact=None,
               tauc40=None):
    hc = hss["cells"].get(cell, {}).get("curve", {})
    hB1, h_pop = {}, []
    for v in (-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30):
        e = hc.get(str(v))
        if e and e.get("qc") and e["h"] > 0:
            hB1[v] = float(e["h"])
        else:
            hB1[v] = float(hss["gears"][str(v)]["med"])
            h_pop.append(v)
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
                 -40: min(1.0, cell_y(-50) * 3.0 / hB1[-40]),
                 -30: min(1.0, cell_y(-50) * 8.0 / hB1[-30]),
                 -20: 1.0, 0: 1.0, 20: 1.0, 40: 1.0, 60: 1.0}
    m_ss = Tab(m_anchors, log_y=True)
    hs = hook["summary"]
    tm_anchors = {-130: hs["-120"]["td_med"], -120: hs["-120"]["td_med"],
                  -110: hs["-110"]["td_med"], -100: hs["-100"]["td_med"],
                  -90: 0.072, -80: 0.24,
                  -70: amp["tau_used"]["TM"]["-70"], -60: amp["tau_used"]["TM"]["-60"],
                  -50: amp["tau_used"]["TAU_L"]["-50"], -40: amp["tau_used"]["TAU_L"]["-40"],
                  -30: 9.8, -20: 4.0, -10: 2.8, 0: 2.0, 10: 1.2,
                  20: 0.71, 30: 0.45, 40: 0.29, 50: 0.29, 60: 0.30}
    tau_m = Tab(tm_anchors, log_y=True)
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
    if G is None and ci.get("g_hat") and ci["g_hat"] > 0:
        G = float(ci["g_hat"])
        gflag = "posthoc_G_inact"
    if G is None:
        cands = [abs(r["A"]) / DF_M120 for r in hook["A_rows"]
                 if r["cell"] == cell and r["v"] == -120 and r["A"] != 0]
        if cands:
            G = float(np.median(cands))
            gflag = "posthoc_G_degraded"
    h50 = None
    if ci.get("h_ss50") and ci["h_ss50"] > 0:
        h50 = float(ci["h_ss50"])
    h40 = np.nan
    if I_inact is not None and G is not None and G > 0:
        info = segments(V_inact)
        for k, (v, s0, n) in enumerate(info):
            if abs(v - 40.0) < 2.0 and 0.14 < n * DT < 0.16 and k >= 1:
                pv = info[k - 1][0]
                if abs(pv + 90.0) < 2.0:
                    iss = float(np.mean(I_inact[s0 + n - 5000: s0 + n]))
                    h40 = iss / (G * 1.0 * (40.0 - E_REV))
                    break
    if not np.isfinite(h40) or h40 <= 0:
        h40 = h50 if h50 else np.nan
    if not np.isfinite(h40) or h40 <= 0:
        h40 = 0.0034
    h40 = float(np.clip(h40, 1e-4, 0.2))
    h_anchors = {-130: 1.0, -100: 1.0}
    h_anchors.update(hB1)
    h_anchors[40] = h40
    h_anchors[50] = h50 if h50 else h40
    h_anchors[60] = h50 if h50 else h40
    h_ss = Tab(h_anchors, log_y=True)
    tr90 = ci.get("m90", {}).get("tau_r") if ci.get("m90", {}).get("valid") else None
    th_anchors = {-130: hs["-120"]["tr_med"], -120: hs["-120"]["tr_med"],
                  -110: hs["-110"]["tr_med"], -100: hs["-100"]["tr_med"],
                  -90: tr90 if tr90 else 0.0065,
                  -40: 0.023, -20: 0.05, 0: 0.0015, 60: 0.0015}
    tau_h = Tab(th_anchors, log_y=True)
    out = dict(m_ss=m_ss, tau_m=tau_m, h_ss=h_ss, tau_h=tau_h, G=G,
               h40=h40, h50=h50, gflag=gflag, h_pop=h_pop)
    if tauc40 is not None:
        tc_anchors = {v: (a if v <= -30 else a / 3.0) for v, a in tm_anchors.items()}
        tc_anchors[40] = tauc40
        tc_anchors[50] = 0.29 / 3.0
        tc_anchors[60] = 0.30 / 3.0
        out["tau_c"] = Tab(tc_anchors, log_y=True)
    return out


def forward_single(V, tabs):
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


def forward_k3r(V, tabs):
    """B6 rectified three-stage cascade (reference column, same copy as the B6 verdict script)."""
    v = V.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tc = np.exp(np.interp(v, tabs["tau_c"].xs, tabs["tau_c"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hss = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    ec = np.exp(-DT / tc)
    em = np.exp(-DT / tm)
    eh = np.exp(-DT / th)
    n = len(V)
    m3 = np.empty(n)
    h = np.empty(n)
    m1 = m2 = m3_ = ms[0]
    h0 = hss[0]
    for i in range(n):
        if ms[i] >= m3_:
            m1 = ms[i] + (m1 - ms[i]) * ec[i]
            m2 = m1 + (m2 - m1) * ec[i]
            m3_ = m2 + (m3_ - m2) * ec[i]
        else:
            m3_ = ms[i] + (m3_ - ms[i]) * em[i]
            m1 = m3_
            m2 = m3_
        h0 = hss[i] + (h0 - hss[i]) * eh[i]
        m3[i] = m3_
        h[i] = h0
    return m3, h


def find_ap_structure(V):
    n = len(V)
    idx = np.where(V > 0)[0]
    grp = np.split(idx, np.where(np.diff(idx) > 100)[0] + 1) if len(idx) else []
    cycles = []
    for g in grp:
        pk = g[np.argmax(V[g])]
        after = np.where(V[pk:] < -40)[0]
        w_end = pk + int(after[0]) if len(after) else min(pk + 2000, n - 1)
        plat = (max(0, pk - 100), min(pk + 100, n))
        cycles.append(dict(pk=int(pk), repol=(int(pk), int(w_end)),
                           plat=plat, vmax=float(V[pk])))
    inters = []
    for k in range(len(grp) - 1):
        a, b = int(grp[k][-1]), int(grp[k + 1][0])
        if b - a > 500:
            inters.append((a, b))
    h120 = np.where(V < -119.5)[0]
    gh = np.split(h120, np.where(np.diff(h120) > 100)[0] + 1) if len(h120) else []
    hook = None
    for g in gh:
        if len(g) * DT > 0.3:
            hook = (int(g[0]), int(g[-1]) + 1)
    return dict(cycles=cycles, inters=inters, hook=hook)


def score_ap(V, I_mea, Isim, st, robust=False):
    """Verbatim to the sealed version; when robust=True the measured peak uses max(med5(I)) (the only change, declared pre-run)."""
    Ipk = med5(I_mea) if robust else I_mea
    out = {}
    ratios, npeak = [], 0
    for cyc in st["cycles"]:
        a, b = cyc["repol"]
        if b - a < 20:
            continue
        p0, p1 = cyc["plat"]
        s_pk = float(np.max(Isim[a:b]))
        s_pl = float(np.mean(Isim[p0:p1]))
        npeak += int(s_pk > s_pl + NOISE3)
        m_pk = float(np.max(Ipk[a:b]))
        if m_pk > NOISE3:
            ratios.append(s_pk / m_pk)
    ncy = len(st["cycles"])
    out["repol_npeak"] = npeak
    out["repol_ncy"] = ncy
    out["repol_ratio_med"] = float(np.median(ratios)) if ratios else np.nan
    out["A1"] = bool(ncy and npeak >= 0.8 * ncy
                     and np.isfinite(out["repol_ratio_med"])
                     and 0.3 <= out["repol_ratio_med"] <= 3.0)
    if st["hook"]:
        a, b = st["hook"]
        t = np.arange(b - a) * DT
        rs = doe_fit(t, -Isim[a:b])
        out["hook_simA"] = rs["A"] if rs else np.nan
        rm = doe_fit(t, -I_mea[a:b])
        out["hook_meaA"] = rm["A"] if rm else np.nan
        out["A2"] = bool(rs and rm and rm["A"] > 0
                         and 0.3 <= rs["A"] / rm["A"] <= 3.0)
    if len(st["inters"]) >= 3:
        k = max(1, len(st["inters"]) // 3)
        first = np.concatenate([Isim[a:b] for a, b in st["inters"][:k]])
        last = np.concatenate([Isim[a:b] for a, b in st["inters"][-k:]])
        out["inter_first"] = float(np.mean(first))
        out["inter_last"] = float(np.mean(last))
        out["A3"] = bool(out["inter_last"] > out["inter_first"])
    out["pass"] = bool(out.get("A1") and out.get("A2") and out.get("A3"))
    return out


def main():
    rng = np.random.default_rng(23)
    print("=" * 88)
    print(" alpha model data-side leftovers recheck: 014 spike robust-peak re-evaluation + 060/016 baseline offset"
          + (" (smoke 16713003)" if SMOKE else " (nine cells full)"))
    print(" criteria: A-main 014 robust peak ratio in [0.3,3] with A2/A3 holding | A-audit per-cell raw->robust | "
          "B offset > 3x population MAD")
    print("=" * 88, flush=True)

    # ---------- controls ----------
    print("\n[controls]", flush=True)
    t_c = np.arange(int(0.4 / DT)) * DT
    y_c = 3.0 * (np.exp(-t_c / 0.030) - np.exp(-t_c / 0.0035)) \
        + rng.normal(0, 0.02, len(t_c))
    rc = doe_fit(t_c, y_c)
    c1 = bool(rc and abs(rc["tau_r"] - 0.0035) / 0.0035 < 0.15)
    print(f"  C1 DoE inversion tau_r: {rc['tau_r'] * 1e3:.2f} ms (truth 3.5) -> "
          f"{'pass' if c1 else 'fail'}", flush=True)
    V0a = load_mat("ap_protocol.mat", "16713003", "ap")[0]
    st0a = find_ap_structure(V0a)
    c2 = bool(len(st0a["cycles"]) >= 14 and st0a["hook"] is not None)
    print(f"  C2 AP structure: peaks {len(st0a['cycles'])} hook {'yes' if st0a['hook'] else 'no'} -> "
          f"{'pass' if c2 else 'fail'}", flush=True)
    # C3 robust-peak self-consistency: sustained peak (5 ms wide, height 1.0) + single-sample spikes +/-8
    tt = np.arange(2000) * DT
    peak = np.exp(-((tt - 0.100) / 0.0025) ** 2)          # sustained peak 5 ms wide
    Isyn = peak + rng.normal(0, 0.01, len(tt))
    Isyn[600] += 8.0
    Isyn[1200] -= 8.0
    est = float(np.max(med5(Isyn)))
    raw_est = float(np.max(Isyn))
    c3 = bool(abs(est - 1.0) / 1.0 < 0.05 and abs(raw_est - 8.0) < 1.0)
    print(f"  C3 robust peak: sustained-peak truth 1.0, estimate {est:.3f} (error <5%), "
          f"raw max={raw_est:.2f} (dominated by spikes, proving the robust convention is needed) -> "
          f"{'pass' if c3 else 'fail'}", flush=True)
    if not (c1 and c2 and c3):
        print("  controls not seated -> halt.", flush=True)
        return
    print("  controls seated.", flush=True)

    # ---------- item A: robust-peak re-evaluation ----------
    print("\n[item A AP robust-peak re-evaluation] (single-gate pipeline verbatim; k3 reference column)", flush=True)
    amp = json.load(open(F_AMP, encoding="utf-8"))
    hook = json.load(open(F_HOOK, encoding="utf-8"))
    inact = json.load(open(F_INACT, encoding="utf-8"))
    hss = json.load(open(F_HSS, encoding="utf-8"))
    tauc_tab = {}
    if os.path.exists(F_B6):
        b6 = json.load(open(F_B6, encoding="utf-8"))
        tauc_tab = b6.get("tauc40", {}) or {}
    res = {}
    for c in CELLS_FWD:
        V, I = load_mat("ap_protocol.mat", c, "ap")
        Vi, Ii = load_mat("inactivation_protocol.mat", c, "inactivation")
        st = find_ap_structure(V)
        tabs = build_tabs(c, amp, hook, inact, hss, Ii, Vi,
                          tauc40=tauc_tab.get(c) if tauc_tab else None)
        m_s, h_s = forward_single(V, tabs)
        Isim = tabs["G"] * m_s * h_s * (V - E_REV)
        raw = score_ap(V, I, Isim, st, robust=False)
        rob = score_ap(V, I, Isim, st, robust=True)
        row = dict(G=tabs["G"], raw=raw, robust=rob)
        if "tau_c" in tabs:
            m3, h3 = forward_k3r(V, tabs)
            Isim3 = tabs["G"] * m3 * h3 * (V - E_REV)
            row["k3_robust"] = score_ap(V, I, Isim3, st, robust=True)
        res[c] = row
        k3txt = ""
        if "k3_robust" in row:
            k3txt = f" | k3 reference robust ratio {row['k3_robust']['repol_ratio_med']:.2f}"
        print(f"  {c}: peak ratio raw {raw['repol_ratio_med']:.2f} -> robust "
              f"{rob['repol_ratio_med']:.2f} | A1 {'✓' if rob['A1'] else '×'}"
              f" A2 {'✓' if rob.get('A2') else '×'} A3 {'✓' if rob.get('A3') else '×'}"
              f" -> {'pass' if rob['pass'] else 'fail'}{k3txt}", flush=True)

    print("\n" + "-" * 88, flush=True)
    a_verdict = []
    if "16707014" in res:
        r14 = res["16707014"]
        move = abs(r14["robust"]["repol_ratio_med"] - r14["raw"]["repol_ratio_med"]) \
            / max(abs(r14["raw"]["repol_ratio_med"]), 1e-9)
        main_ok = bool(r14["robust"]["A1"] and r14["robust"].get("A2")
                       and r14["robust"].get("A3"))
        a_verdict.append(
            f"A-main: 014 peak ratio raw {r14['raw']['repol_ratio_med']:.2f} -> robust "
            f"{r14['robust']['repol_ratio_med']:.2f} (shift {move * 100:.0f}%)"
            f" A1/A2/A3 {'all pass -> spike explanation confirmed' if main_ok else 'not all pass'}")
        if not main_ok:
            a_verdict.append("A-reject: 014 still fails under the robust convention -> spikes not the (sole) main cause; "
                             "model-side gap registered as-is")
        for c, r in res.items():
            mv = abs(r["robust"]["repol_ratio_med"] - r["raw"]["repol_ratio_med"]) \
                / max(abs(r["raw"]["repol_ratio_med"]), 1e-9)
            flip = r["robust"]["pass"] != r["raw"]["pass"]
            if mv > 0.15 or flip:
                a_verdict.append(
                    f"A-audit: {c} peak ratio {r['raw']['repol_ratio_med']:.2f} -> "
                    f"{r['robust']['repol_ratio_med']:.2f}（{mv * 100:.0f}%"
                    f"{', pass flipped' if flip else ''}) registered: measured peak contains spike component")
    npass_rob = sum(1 for r in res.values() if r["robust"]["pass"])
    a_verdict.append(f"AP (robust-peak convention, single gate): {npass_rob}/{len(res)} pass")
    for line in a_verdict:
        print(" ", line, flush=True)

    # ---------- item B: baseline-offset recheck ----------
    print("\n[item B 060/016 baseline-offset recheck] (-80/-90 constant segments >=0.2 s, skipping first 0.1 s)", flush=True)
    protos = [("sine_wave", "sine_wave"), ("ap", "ap"), ("inactivation", "inactivation"),
              ("activation_kinetics_1", "activation_kinetics_1"),
              ("activation_kinetics_2", "activation_kinetics_2")]
    btab = {}
    for c in CELLS_ALL:
        row = {}
        for proto, tag in protos:
            V, I = load_mat(f"{proto}_protocol.mat", c, tag)
            if I is None:
                continue
            pool = []
            for v, s0, nseg in segments(V):
                if abs(v + 80) < 2 or abs(v + 90) < 2:
                    if nseg * DT >= 0.2:
                        pool.append(I[s0 + 1000: s0 + nseg])
            if pool:
                d = np.concatenate(pool)
                row[proto] = dict(offset=float(np.mean(d)), sigma=float(np.std(d)),
                                  n=int(len(d)))
            else:
                row[proto] = dict(offset=None, note="no coverage")
        btab[c] = row
    offs = [abs(d["offset"]) for c in btab.values() for d in c.values()
            if d.get("offset") is not None]
    mad_pop = float(np.median(np.abs(np.array(offs) - np.median(offs)))) * 1.4826
    print(f"  population |offset| median {np.median(offs):.4f} nA, MAD {mad_pop:.4f} nA"
          f" (criterion 3xMAD={3 * mad_pop:.4f})", flush=True)
    b_flag = {}
    for c in CELLS_ALL:
        parts, fl = [], []
        for p, d in btab[c].items():
            if d.get("offset") is None:
                parts.append(f"{p}:no-coverage")
            else:
                outl = abs(d["offset"]) > 3 * mad_pop
                parts.append(f"{p}:{d['offset']:+.4f}{'*' if outl else ''}")
                if outl:
                    fl.append(p)
        b_flag[c] = fl
        mark = "  <== outlier" if fl and c in ("16708060", "16708016") else \
               ("  (outlier)" if fl else "")
        print(f"  {c}: {' '.join(parts)}{mark}", flush=True)
    tgt = {c: b_flag[c] for c in ("16708060", "16708016")}
    if any(tgt.values()):
        print(f"  item-B verdict: " + "; ".join(f"{c} at {v} offset outlier confirmed" for c, v in tgt.items() if v)
              + " (subtraction-parameter issue, magnitudes above; re-subtraction deferred to after formal assembly)", flush=True)
    else:
        print("  item-B verdict: no >3xMAD silent-segment offset seen in 060/016 - over-subtraction leak invisible in the baseline convention, registration stands",
              flush=True)

    # ---------- figure ----------
    cells_fig = CELLS_FWD if SMOKE else ["16707014", "16704047", "16713003", "16708016"]
    nfig = len(cells_fig)
    fig, axes = plt.subplots(nfig, 1, figsize=(15, 3.0 * nfig), squeeze=False)
    for ax, c in zip(axes[:, 0], cells_fig):
        V, I = load_mat("ap_protocol.mat", c, "ap")
        tt = np.arange(len(V)) * DT
        ax.plot(tt, I, lw=0.3, color="0.6", label="measured raw")
        ax.plot(tt, med5(I), lw=0.4, color="tab:red", alpha=0.8, label="med5 (robust-peak raw material)")
        ax.set_title(f"{c}  AP robust-peak convention", fontsize=9)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("I (nA)")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-14_α模型_数据侧遗留_尖刺剔除与基线偏移复核.png")
    fig.savefig(fpng, dpi=120, bbox_inches="tight")

    out = dict(note="section-9 todo-4 data-side recheck: scoring-side robust peak max(med5(I)); single-gate pipeline verbatim; k3 reference column",
               ap={c: r for c, r in res.items()},
               ap_verdict=a_verdict, npass_robust=npass_rob,
               baseline={c: btab[c] for c in CELLS_ALL},
               baseline_mad=mad_pop, baseline_flag=b_flag)
    fjson = os.path.join(HERE, "2026-09-14_α模型_数据侧遗留_尖刺剔除与基线偏移复核_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  figure saved: {fpng}", flush=True)
    print(f"  results saved: {fjson}", flush=True)


if __name__ == "__main__":
    main()
