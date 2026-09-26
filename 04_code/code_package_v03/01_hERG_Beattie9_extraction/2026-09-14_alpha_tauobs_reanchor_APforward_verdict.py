# 2026-09-14_alpha_tauobs_reanchor_APforward_verdict.py
# [tau_obs re-anchoring, B2 retirement 2026-09-14] This file = hss-measured-table-rerun AP-forward verdict with one change only:
#   tau_h repolarisation-segment B2 bridge anchors retired, replaced by the tau_obs sealed measured upper-bound median table
#   (2026-09-13_alpha_tauobs_verdict_repol_window_fast_relax_seal_结果.json, sealed 5/6 levels, section 13): new explicit
#   anchors -70: 7.0 ms, -60: 7.0 ms, -50: 8.9 ms, -40: 14.4 ms, -30: 14.4 ms; the old bridge anchor -40: 23 ms yields;
#   -20: 50 ms has no measured coverage, bridge anchor kept (registered); 0/+60: 1.5 ms unchanged. Everything else
# Criteria (pinned before run, verbatim identical to the hss-measured-table rerun, not one character moved):
#   A1 repolarisation rebound >= 80% of cycles and peak-ratio median in [0.3,3]; A2 hook DoE in [0.3,3]; A3 inter-beat
#   accumulation direction; per cell A1/A2/A3 all pass -> pass; nine cells >= 7 -> AP forward robust to B2 retirement.
# Meaning: still >= 7 -> the B2 bridge retires without damage, tau_h repolarisation segment closes on measurement;
# Run: python this file (nine cells full); SMOKE=1 single-cell 16713003 smoke.
# ---------- below: header of the hss-measured-table rerun version (historical declarations; this note governs the change) ----------
# 2026-09-13_alpha model_hss measured-table rerun_AP forward adjudication.py
# [B1 retirement, table-swap rerun, late 2026-09-13] This file = AP-forward verdict with one change only:
#   h_ss B1 segment (-80..+30, 12 levels) anchors swapped to the measured strip table (loaded at runtime from
#   2026-09-13_alpha_pure_hssV_strip_verdict_结果.json), m_ss = y_ss/h_ss re-split in sync; policies P1-P5 verbatim
# Criteria verbatim identical to the B1 version (A1/A2/A3, nine cells >= 7, not one character moved); controls C1/C2 same.
# Meaning: still >= 7 -> AP forward robust to the death of B1; a drop -> the original 7/9 relied on a wrong assumption,
# ---------- below: original B1-version header (the B1 clause is void; all other declarations still in force) ----------
# Purpose (AP-clamp decisive battle, not parameter tuning: the same four tables, same G, same E_rev = -88.33 as the
#   sine smoke; zero-tuning forward against the AP protocol): dm/dt = (m_ss - m)/tau_m, dh/dt = (h_ss - h)/tau_h,
#   I = G*m*h*(V - E_rev). V(t) read directly from ap_protocol.mat (digitised AP-clamp waveform with 17 depolarising
#
# [Protocol structure (programmatically parsed 2026-09-13, on record)]
#   lead-in -80 x0.25 s -> -120 x0.05 s prepulse -> -80; 17 depolarising peaks from 0.57 s
#   (peaks +8.8~+71.8mV, inter-peak returns to -80); 7.325s tail -120x0.5s rebound hook; final -80x1.0s.
#   total length 8.82s, 88245 points @DT=1e-4.
#
# [Measured-shape basis (16713003 sampling, basis for criterion design)]
#   repolarisation-window peaks 0.09~2.05 nA >> plateau window -0.001~0.335 (the hERG repolarisation-rebound signature);
#   inter-beat -80 window current 0.020 -> 0.189 nA cross-cycle accumulation (~8x);
#   -120 hook flipped amplitude ~1.2 nA; noise std 0.0069 nA (lead-in -80 segment).
#
# Criteria (declared pre-run):
#   A1 spontaneous repolarisation rebound (top target): in >= 80% of cycles (>= 14 of 17) the simulated repolarisation
#      window (after the peak until V < -40) shows a current peak (max > plateau-window mean + 0.021 = 3x noise),
#      and across those cycles the median ratio simulated/measured repolarisation peak is in [0.3,3] (G uncertainty);
#   A2 -120 hook rebound amplitude: DoE inversion simulated/measured in [0.3,3] (same convention as sine S1);
#   A3 inter-beat accumulation direction: simulated inter-beat -80 window current, last-third mean > first-third mean
#      (direction judgement; measured 0.020 -> 0.189, about 8x);
#   per cell A1/A2/A3 all pass -> that cell passes; nine cells >= 7 pass -> alpha-model AP forward holds.
#
# Bridge declarations (same set as the sine smoke, not one character moved):
#   B1 [VOID, this version swaps in the measured table] h_ss(V <= -40) = 1.0 physical claim; h40 fallback chain
#      (inactivation-protocol direct extraction -> h_ss50 -> default 0.0034; the +40-end R3 chain kept in this version
#   B2 tau_rec(V) 4-point log extrapolation; B3 tau_m anchors/logic bridge; B4 -40-level y_ss artefact removal;
#   B5 single-gate m claim. R1 G fallback chain same as sine (16704047 takes posthoc_G_inact = 0.0718).
# Controls (declared pre-run):
#   C1 DoE inversion tau_r error < 15%;
#   C2 synthetic full run (16713003 tables): structure parsing finds 17 peaks + hook and A1/A2/A3 computable (pipeline
# Run: python this file (nine cells full); SMOKE=1 single-cell 16713003 smoke.
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
NOISE3 = 3 * 0.0069          # A1 peak threshold (16713003 lead-in segment noise, same convention for nine cells)

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
    """Log-linear interpolation anchor table (x linear, y log, endpoints out of range). Same copy as the sine smoke."""
    def __init__(self, anchors, log_y=True):
        self.xs = np.array(sorted(anchors), dtype=float)
        ys = np.array([anchors[x] for x in self.xs], dtype=float)
        self.ly = np.log(np.clip(ys, 1e-9, None)) if log_y else ys
        self.log_y = log_y

    def __call__(self, v):
        y = np.interp(v, self.xs, self.ly)
        return float(np.exp(y)) if self.log_y else float(y)


def build_tabs(cell, amp, hook, inact, hss, I_inact=None, V_inact=None):
    """Same four tables as the sine table-swap version (no raw material at sine_mea40 level; the rest verbatim). Returns dict.
    Table-swap version: h_ss B1 segment (-80..+30) uses the measured strip table; m_ss = y_ss/h_ss re-split in sync."""
    # ---- h_ss measured strip table (P3: per-cell qc pass -> per-cell value; else population median) ----
    hc = hss["cells"].get(cell, {}).get("curve", {})
    hB1, h_pop = {}, []
    for v in (-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30):
        e = hc.get(str(v))
        if e and e.get("qc") and e["h"] > 0:
            hB1[v] = float(e["h"])
        else:
            hB1[v] = float(hss["gears"][str(v)]["med"])
            h_pop.append(v)
    # ---- m_ss anchors (per-cell y_ss, missing levels use median; table-swap re-split m_ss = y_ss/h_ss) ----
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

    # ---- h_ss (B1 retired: measured strip table; +40-end R3 fallback chain kept, P4) ----
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
    h_anchors.update(hB1)                                    # B1-segment measured (-80..+30)
    h_anchors[40] = h40                                      # P4: +40-end R3 chain kept
    h_anchors[50] = h50 if h50 else h40
    h_anchors[60] = h50 if h50 else h40
    h_ss = Tab(h_anchors, log_y=True)

    # ---- tau_h (B2 retired: tau_obs sealed measured upper-bound median table; -20 bridge anchor kept, registered) ----
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


def find_ap_structure(V):
    """Programmatically parse AP structure: >0 mV depolarised groups, repolarisation windows, plateau windows, inter-beat windows, -120 hook."""
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
        if len(g) * DT > 0.3:                      # 0.5 s tail hook (not the 0.05 s prepulse)
            hook = (int(g[0]), int(g[-1]) + 1)
    return dict(cycles=cycles, inters=inters, hook=hook)


def score_cell(cell, V, I, tabs):
    m, h = forward(V, tabs)
    G = tabs["G"]
    Isim = G * m * h * (V - E_REV)
    st = find_ap_structure(V)
    out = dict(G=G, h40=tabs["h40"], gflag=tabs["gflag"])
    # ---- A1 repolarization rebound ----
    ratios, npeak = [], 0
    for cyc in st["cycles"]:
        a, b = cyc["repol"]
        if b - a < 20:
            continue
        p0, p1 = cyc["plat"]
        s_pk = float(np.max(Isim[a:b]))
        s_pl = float(np.mean(Isim[p0:p1]))
        has = s_pk > s_pl + NOISE3
        npeak += int(has)
        if I is not None:
            m_pk = float(np.max(I[a:b]))
            if m_pk > NOISE3:
                ratios.append(s_pk / m_pk)
    ncy = len(st["cycles"])
    out["repol_npeak"] = npeak
    out["repol_ncy"] = ncy
    out["repol_ratio_med"] = float(np.median(ratios)) if ratios else np.nan
    out["A1"] = bool(ncy and npeak >= 0.8 * ncy
                     and np.isfinite(out["repol_ratio_med"])
                     and 0.3 <= out["repol_ratio_med"] <= 3.0)
    # ---- A2 -120 hook DoE ----
    if st["hook"]:
        a, b = st["hook"]
        t = np.arange(b - a) * DT
        rs = doe_fit(t, -Isim[a:b])
        out["hook_simA"] = rs["A"] if rs else np.nan
        if I is not None:
            rm = doe_fit(t, -I[a:b])
            out["hook_meaA"] = rm["A"] if rm else np.nan
            out["A2"] = bool(rs and rm and rm["A"] > 0
                             and 0.3 <= rs["A"] / rm["A"] <= 3.0)
    # ---- A3 inter-beat accumulation direction ----
    if len(st["inters"]) >= 3:
        k = max(1, len(st["inters"]) // 3)
        first = np.concatenate([Isim[a:b] for a, b in st["inters"][:k]])
        last = np.concatenate([Isim[a:b] for a, b in st["inters"][-k:]])
        out["inter_first"] = float(np.mean(first))
        out["inter_last"] = float(np.mean(last))
        out["A3"] = bool(out["inter_last"] > out["inter_first"])
    out["pass"] = bool(out.get("A1") and out.get("A2") and out.get("A3"))
    out["_sim"] = Isim
    out["_struct"] = st
    return out


def main():
    rng = np.random.default_rng(5)
    print("=" * 84)
    print(" alpha model AP forward verdict - tau_obs re-anchoring rerun (B2 retired)"
          + (" (smoke 16713003)" if SMOKE else " (nine cells full)"))
    print(" criteria: A1 repolarisation rebound >=80% cycles and peak-ratio median in [0.3,3] | A2 hook DoE in [0.3,3] | A3 accumulation direction"
          " (verbatim identical to the hss-measured-table rerun)")
    print(" change: only tau_h repolarisation segment -70..-30 swapped to tau_obs measured upper-bound medians 7.0/7.0/8.9/14.4/14.4 ms")
    print(" discipline: same four tables, same G, same E_rev as the sine table-swap version; zero tuning")
    print("=" * 84, flush=True)

    amp = json.load(open(F_AMP, encoding="utf-8"))
    hook = json.load(open(F_HOOK, encoding="utf-8"))
    inact = json.load(open(F_INACT, encoding="utf-8"))
    hss = json.load(open(F_HSS, encoding="utf-8"))

    # ---------- controls ----------
    print("\n[controls]", flush=True)
    t_c = np.arange(int(0.4 / DT)) * DT
    y_c = 3.0 * (np.exp(-t_c / 0.030) - np.exp(-t_c / 0.0035)) \
        + rng.normal(0, 0.02, len(t_c))
    rc = doe_fit(t_c, y_c)
    c1 = bool(rc and abs(rc["tau_r"] - 0.0035) / 0.0035 < 0.15)
    print(f"  C1 DoE inversion tau_r: {rc['tau_r'] * 1e3:.2f} ms (truth 3.5) -> "
          f"{'pass' if c1 else 'fail'}", flush=True)
    V0 = load_mat("ap_protocol.mat", "16713003", "ap")[0]
    st0 = find_ap_structure(V0)
    ok_c2 = len(st0["cycles"]) >= 14 and st0["hook"] is not None
    print(f"  C2 AP structure parsing: depolarising peaks {len(st0['cycles'])}, inter-beat windows {len(st0['inters'])}, "
          f"hook {'yes' if st0['hook'] else 'no'} -> {'pass' if ok_c2 else 'fail'}", flush=True)
    if not (c1 and ok_c2):
        print("  controls not seated -> halt.", flush=True)
        return
    print("  controls seated.", flush=True)

    # ---------- real data ----------
    print("\n[real data]", flush=True)
    res = {}
    for c in CELLS:
        V, I = load_mat("ap_protocol.mat", c, "ap")
        Vi, Ii = load_mat("inactivation_protocol.mat", c, "inactivation")
        tabs = build_tabs(c, amp, hook, inact, hss, Ii, Vi)
        r = score_cell(c, V, I, tabs)
        r["h_pop"] = tabs["h_pop"]
        res[c] = r
        fl = (f"  [{r['gflag']}]" if r.get("gflag") else "") + \
             (f"  [hss_pop{r['h_pop']}]" if r["h_pop"] else "")
        print(f"  {c}: G={r['G']:.4f} h40={r['h40']:.4f} | "
              f"A1{'v' if r.get('A1') else 'x'} peaks {r.get('repol_npeak')}/{r.get('repol_ncy')}"
              f" ratio med {r.get('repol_ratio_med', np.nan):.2f} | "
              f"A2{'v' if r.get('A2') else 'x'} hook sim {r.get('hook_simA', np.nan):.2f}"
              f"/meas {r.get('hook_meaA', np.nan):.2f} | "
              f"A3{'v' if r.get('A3') else 'x'} inter {r.get('inter_first', np.nan):.4f}"
              f"->{r.get('inter_last', np.nan):.4f} -> "
              f"{'pass' if r['pass'] else 'fail'}{fl}", flush=True)

    npass = sum(1 for r in res.values() if r["pass"])
    print("\n" + "-" * 84, flush=True)
    print(f" nine-cell AP tau_obs re-anchoring rerun: {npass}/{len(res)} pass (criterion >=7, robust to B2 retirement)", flush=True)
    print(" overall verdict:", "AP forward robust to B2 retirement (tau_obs measured anchors, criteria unmoved) - B2 bridge retirement closed"
          if npass >= (1 if SMOKE else 7)
          else "criteria failed after re-anchoring - the B2 bridge value carried load; registered as-is with rollback on file, details above", flush=True)

    # ---------- figure ----------
    nfig = len(res)
    fig, axes = plt.subplots(nfig, 1, figsize=(15, 3.0 * nfig), squeeze=False)
    for ax, (c, r) in zip(axes[:, 0], res.items()):
        V, I = load_mat("ap_protocol.mat", c, "ap")
        tt = np.arange(len(V)) * DT
        if I is not None:
            ax.plot(tt, I, lw=0.3, color="0.6", label="measured")
        ax.plot(tt, r["_sim"], lw=0.5, color="tab:red", alpha=0.8, label="simulated")
        if r["_struct"]["hook"]:
            a, b = r["_struct"]["hook"]
            ax.axvspan(a * DT, b * DT, color="tab:blue", alpha=0.08)
        ax.set_title(f"{c}  A1{'pass' if r.get('A1') else 'fail'} A2{'pass' if r.get('A2') else 'fail'} "
                     f"A3{'pass' if r.get('A3') else 'fail'}  repol peak-ratio median "
                     f"{r.get('repol_ratio_med', np.nan):.2f}", fontsize=9)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("I (nA)")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-14_α模型_τobs换锚_AP前向判决.png")
    fig.savefig(fpng, dpi=120, bbox_inches="tight")

    out = dict(note="B2 retirement tau_obs re-anchoring rerun: tau_h(-70/-60/-50/-40/-30) = measured upper-bound medians "
                    "7.0/7.0/8.9/14.4/14.4 ms; -20 bridge anchor 50 ms kept, registered; "
                    "criteria verbatim identical to the hss-measured-table rerun",
               cells={c: {k: v for k, v in r.items() if not k.startswith("_")}
                      for c, r in res.items()}, npass=npass)
    fjson = os.path.join(HERE, "2026-09-14_α模型_τobs换锚_AP前向判决_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  figure saved: {fpng}", flush=True)
    print(f"  results saved: {fjson}", flush=True)


if __name__ == "__main__":
    main()
