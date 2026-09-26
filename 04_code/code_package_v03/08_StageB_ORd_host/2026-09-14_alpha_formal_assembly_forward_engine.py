# 2026-09-14_α模型_正式组装_前向引擎.py
# ============================================================================
# alpha model · formal assembly · the single formal forward engine (finalized 2026-09-14)
#
# model equations (single gate m, product gate):
#   dm/dt = (m_ss(V) - m) / tau_m(V)
#   dh/dt = (h_ss(V) - h) / tau_h(V)
#   I(t)  = G * m * h * (V - E_rev)          E_rev = -88.33 mV
#
# This file inherits the model part of 2026-09-13_α模型_hss实测表重跑_AP前向判决.py
#   (Tab / forward verbatim; build_tabs changes only th_anchors to the new anchor - the 2026-09-14 tau_obs re-anchoring,
#    consistent with the tau_obs re-anchored judge script; the scoring pieces doe_fit/find_ap_structure/score_cell
#    belong to the judge pipeline and are not part of the engine). The four measured tables are loaded at runtime
#    from the sealed JSONs, never hand-copied:
#     幅度表提取_结果.json        y_ss(-70/-60/-50) per cell, tau ladder TF/TM/TAU_L
#     反弹hook_结果.json          tau_rec/tau_deact(-120/-110/-100) medians, G anchor A(-120)
#     失活门_失活协议封卷判决_结果.json  h_ss50, g_hat, m90 tau_r(-90)
#     纯hssV剥离判决_结果.json    h_ss(-80..+30) 12 levels, per-cell values + population median
#
# sealed track record (produced by this engine as-is; criteria pinned before the runs):
#   AP forward 8/9 (robust-peak convention; sole failure 16707014, model-side registry: h table population median + D4)
#   sine forward 7/9 (failures 047 data-side posthoc_target_degraded, 060 data-side D1)
#   deactivation-ladder whitening 28/33; h_ss 12-level measured stripping (4 levels sealed / 8 registered);
#   tau_obs repolarization-window fast relaxation sealed 5/6 levels (h_ss = h_150 identity sealed);
#   tau_obs re-anchored rerun (B2 retired 2026-09-14): sine 7/9, AP 8/9, same sets and same attributions; criteria unmoved.
#
# registered limitations (attached; item-by-item in model card sections 7-9):
#   B2 retired (2026-09-14 re-anchored rerun passed: sine 7/9, AP 8/9, same sets and same attributions; criteria unmoved):
#     tau_h(-70/-60/-50/-40/-30) = measured upper-bound medians of tau_obs: 7.0/7.0/8.9/14.4/14.4 ms;
#     -20: no measured coverage at 50 ms; bridge anchor kept as registry (the sole remaining bridge anchor in the tau_h chain);
#   B3 tau_m(V>=-40) registry band (activation slow-component median, CV 0.36, not sealed);
#   B5 single-gate-m declaration - B6 verdict (2026-09-14): the activation foot (delay 50-150 ms) at the +40 step
#     is sealed, but globally applying the rectification cascade was rejected by both protocols (sine 1/9, AP 2/9) -> the single gate is kept as
#     the assembly form; registered cost: m overestimated ~3x in the first 100-200 ms of depolarization (one mechanism behind the AP A1 quantitative spread);
#     the 3x contradiction (envelope m~0.08*m_inf @100 ms vs AP requiring m~0.29*m_inf) reconciliation candidates H1/H2/H3
#     all registered unresolved.
#
# [REJECTED · registry] k3 rectification cascade (B6b re-shape, rejected, not in the model, kept for reference):
#   activation direction (m_ss>=m3) three-stage equal-tau cascade (tau_c(+40) per cell 48-108 ms, median 86 ms),
#   deactivation direction single-gate relaxation (ladder untouched); controls C3 6.7e-4 / C4 4.6e-14 restored.
#   rejection reason: m over-suppressed 2x during chirp (sine S2 collapse), m depressed ~3x during the short AP peak (A1 collapse).
#   the symmetric cascade was rejected earlier (fat-tail artifact: 003 AP inter-interval accumulation 0.10->0.54 out of window).
#
# Run: this file is a module; running it directly prints the 16713003 anchor tables as a self-check (not a judgement, no verdict).
#   The formal judge pipeline is *hss实测表重跑_*前向判决.py in the same directory (criteria pinned before the run).
# ============================================================================
import os
import json
import numpy as np
import scipy.io as sio

DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
HERE = os.path.dirname(os.path.abspath(__file__))
DT = 1e-4
E_REV = -88.33
DF_M120 = abs(-120.0 - E_REV)

F_AMP = os.path.join(HERE, "2026-09-13_α模型_幅度表提取_结果.json")
F_HOOK = os.path.join(HERE, "2026-09-13_α模型_反弹hook_结果.json")
F_INACT = os.path.join(HERE, "2026-09-13_α模型_失活门_失活协议封卷判决_结果.json")
F_HSS = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决_结果.json")


def load_mat(proto, cell, tag):
    V = sio.loadmat(f"{DATA}/data/protocols/{proto}")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/{tag}_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


class Tab:
    """Log-linear interpolated anchor table (linear in x, log in y, endpoints beyond the range). Same copy as the sine smoke."""
    def __init__(self, anchors, log_y=True):
        self.xs = np.array(sorted(anchors), dtype=float)
        ys = np.array([anchors[x] for x in self.xs], dtype=float)
        self.ly = np.log(np.clip(ys, 1e-9, None)) if log_y else ys
        self.log_y = log_y

    def __call__(self, v):
        y = np.interp(v, self.xs, self.ly)
        return float(np.exp(y)) if self.log_y else float(y)


def build_tabs(cell, amp, hook, inact, hss, I_inact=None, V_inact=None):
    """The same four tables as the sine table-swapped version (no raw material at sine_mea40 level; the rest verbatim). Returns dict.
    Table-swapped version: the h_ss B1 segment (-80..+30) uses the measured stripped table; m_ss=y_ss/h_ss re-split accordingly."""
    # ---- h_ss measured stripped table (P3: per-cell qc pass -> per-cell value; else population median) ----
    hc = hss["cells"].get(cell, {}).get("curve", {})
    hB1, h_pop = {}, []
    for v in (-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30):
        e = hc.get(str(v))
        if e and e.get("qc") and e["h"] > 0:
            hB1[v] = float(e["h"])
        else:
            hB1[v] = float(hss["gears"][str(v)]["med"])
            h_pop.append(v)
    # ---- m_ss anchors (per-cell y_ss, missing levels use the median; table-swap re-splits m_ss=y_ss/h_ss) ----
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

    # ---- h_ss (B1 retired: measured stripped table; +40 end R3 fallback chain kept, P4) ----
    h50 = None
    if ci.get("h_ss50") and ci["h_ss50"] > 0:
        h50 = float(ci["h_ss50"])
    h40 = np.nan
    if I_inact is not None and G is not None and G > 0:
        edges = np.where(np.diff(V_inact) != 0)[0] + 1
        info = [(float(V_inact[s[0]]), int(s[0]), len(s))
                for s in np.split(np.arange(len(V_inact)), edges)]
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
    h_anchors = {-130: 1.0, -100: 1.0}                       # P1
    h_anchors.update(hB1)                                    # B1 segment measured (-80..+30)
    h_anchors[40] = h40                                      # P4: +40 end R3 chain kept
    h_anchors[50] = h50 if h50 else h40
    h_anchors[60] = h50 if h50 else h40
    h_ss = Tab(h_anchors, log_y=True)

    # ---- tau_h (B2 retired 2026-09-14: tau_obs sealed measured upper-bound median table; -20 bridge anchor kept as registry) ----
    tr90 = ci.get("m90", {}).get("tau_r") if ci.get("m90", {}).get("valid") else None
    th_anchors = {-130: hs["-120"]["tr_med"], -120: hs["-120"]["tr_med"],
                  -110: hs["-110"]["tr_med"], -100: hs["-100"]["tr_med"],
                  -90: tr90 if tr90 else 0.0065,
                  -70: 0.0070, -60: 0.0070, -50: 0.0089, -40: 0.0144, -30: 0.0144,
                  -20: 0.05, 0: 0.0015, 60: 0.0015}
    tau_h = Tab(th_anchors, log_y=True)
    return dict(m_ss=m_ss, tau_m=tau_m, h_ss=h_ss, tau_h=tau_h, G=G,
                h40=h40, h50=h50, gflag=gflag, h_pop=h_pop)


def forward(V, tabs):
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


# ----------------------------------------------------------------------------
# [REJECTED · registry] k3 rectification cascade re-shape (B6b, verdict 2026-09-14: sine 1/9, AP 2/9, rejected)
# Kept for reference, not in the model. Reference implementation (from the B6 judge script, commented out):
#
#   def forward_k3_rectified(V, tabs, tauc_tab):
#       # activation direction (m_ss >= m3): three-stage equal-tau cascade (foot)
#       #   dm1/dt=(m_ss-m1)/τc, dm2/dt=(m1-m2)/τc, dm3/dt=(m2-m3)/τc
#       # deactivation direction (m_ss < m3): m3 relaxes with the sealed tau_m single gate (ladder verbatim untouched)
#       ... (see 2026-09-14_α模型_B6激活足_k3级联判决_双协议换形重跑.py)
#   tau_c(+40) per-cell anchors 48-108 ms (median 86 ms; D11 two cells 007/047 take the median).
#   rejection mechanism: m over-suppressed ~2x during chirp (S2 collapse); m depressed ~3x during the short AP peak (A1 collapse).
# ----------------------------------------------------------------------------


def _selfcheck():
    """Anchor-table print self-check (not a judgement, no verdict): 16713003 four-table anchor values + G."""
    amp = json.load(open(F_AMP, encoding="utf-8"))
    hook = json.load(open(F_HOOK, encoding="utf-8"))
    inact = json.load(open(F_INACT, encoding="utf-8"))
    hss = json.load(open(F_HSS, encoding="utf-8"))
    Vi, Ii = load_mat("inactivation_protocol.mat", "16713003", "inactivation")
    tabs = build_tabs("16713003", amp, hook, inact, hss, Ii, Vi)
    print("alpha-model formal engine self-check (16713003, not a judgement)")
    print(f"  G = {tabs['G']:.4f} nA/mV  h40 = {tabs['h40']:.4f}  gflag = {tabs['gflag']}")
    print(f"  h_pop (population-median fallback level) = {tabs['h_pop']}")
    for v in (-120, -90, -70, -50, -40, -30, 0, 20, 40):
        print(f"  V={v:+5.0f}mV: m_ss={tabs['m_ss'](v):.4f}  tau_m={tabs['tau_m'](v)*1e3:9.2f}ms"
              f"  h_ss={tabs['h_ss'](v):.4f}  tau_h={tabs['tau_h'](v)*1e3:7.2f}ms")


if __name__ == "__main__":
    _selfcheck()
