# -*- coding: utf-8 -*-
"""
2026-09-14 - alpha model - B6-H2 quantitative prediction card (pure simulation, pre-registered)
==========================================================
B6 open case: isolated-beat envelope m@100ms ~= 8% m_inf vs AP repolarisation peak requiring m ~= 29% m_inf (3x).
The falsifiable hard prediction of H2 (mixed-state / inter-beat accumulation, the only candidate with directional support):
**accumulation must vary monotonically with cycle length (CL)**: short CL, deactivation does not finish during rest, pedestal high;
long CL, pedestal vanishes, m falls back to the isolated-beat convention.

This card uses the formal alpha forward engine on the digitised AP waveform to sweep CL and pin three prediction curves,
as a pre-registration: when identical-waveform train data arrive (grandi_2hz class; locally 0/9 on record),
matching the curves closes B6; a flat measured R(CL) -> H2 rejected, B6 passes to H3.

Method: take one representative beat from ap_protocol (median-vmax beat), concatenate with -80 rest into a regular-CL train
(20 beats, initial m = m_ss(-80)), record per beat:
  m_onset(n)   beat-onset m (pedestal);
  m@100ms(n)   m at 100 ms into the beat (envelope convention);
  J(n)         repolarisation-window normalised peak current = max I/(v - E_rev);
  R = median of last 5 beats / median of first 2 beats (same convention as the H2 verdict card).
Isolated +40 step reference: m@100ms (envelope replica).

Pre-registered criteria (executed once data arrive):
  P1 model predicts R(CL=500) >= 3 and R(CL=2000) <= 1.5;
  P2 measured R(CL) matches the predicted curve within log error <= 0.35 -> B6 closed under H2 (context difference);
  P3 measured R(CL) within [0.8, 1.3] at all CL -> H2 rejected, B6 passes to H3.

Run: run directly on this side (lightweight pure simulation). Output _结果.json/.png.

Revision log: v2 (2026-09-14) three fixes before data matching:
 (1) repolarisation-window alignment bug: the original window I_beat[500:2500] falls in the rest segment for CL > 500 (rest_n > 500),
   and v_w wrongly used a V_beat slice for normalisation, distorting J absolute values and R(CL>=750) (spurious 14x cross-CL difference);
   changed to I_beat[rest_n+500:rest_n+2500] against V_beat[500:2500].
 (2) m_onset convention changed from "rest entry" to "the instant of entering the beat" (true pedestal).
 (3) isolated-step expectation note corrected: model tau_m(+40) anchor 0.29 s -> m@100ms ~= 30%;
   the difference from the envelope measured ~8% is the B6 body itself; the model does not reproduce the envelope side.
"""

import os
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("SMOKE", "0") == "1"
CELLS = ["16713003"] if SMOKE else ["16704007", "16704047", "16707014", "16708016",
                                    "16708060", "16708118", "16713003", "16713110", "16715049"]
DT = 1e-4
E_REV = -88.33
CL_LIST_MS = [500, 750, 1000, 1500, 2000, 4000]
N_BEATS = 20


def load_engine():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "alpha_engine", os.path.join(HERE, "2026-09-14_α模型_正式组装_前向引擎.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def representative_beat(V):
    """Take the median-vmax beat: from 50 ms before the peak to 400 ms after, baseline -80."""
    idx = np.where(V > 0)[0]
    grp = np.split(idx, np.where(np.diff(idx) > 100)[0] + 1)
    pks = [(g[np.argmax(V[g])], float(V[g[np.argmax(V[g])]])) for g in grp]
    pks.sort(key=lambda x: x[1])
    pk = pks[len(pks) // 2][0]
    a, b = int(pk) - 500, int(pk) + 4000
    return V[a:b].copy()


def simulate_cell(eng, tabs, V_beat):
    """Returns {CL: dict(m_onset, m100, R, J trajectory)} + isolated-step reference."""
    ms_tab, tm_tab = tabs["m_ss"], tabs["tau_m"]
    hs_tab, th_tab = tabs["h_ss"], tabs["tau_h"]
    G = tabs["G"]

    def run_train(CL_ms):
        rest_n = int(round(CL_ms / 1000.0 / DT)) - len(V_beat)
        if rest_n < 0:
            return None
        V = np.concatenate([np.full(rest_n, -80.0), V_beat])
        per = len(V)
        Vt = np.tile(V, N_BEATS)
        # per-sample relaxation (same formula as eng.forward, but full trajectory kept for per-beat slicing)
        ms = np.array([ms_tab(v) for v in Vt])
        tm = np.array([tm_tab(v) for v in Vt])
        hs = np.array([hs_tab(v) for v in Vt])
        th = np.array([th_tab(v) for v in Vt])
        em = np.exp(-DT / tm)
        eh = np.exp(-DT / th)
        m0, h0 = ms[0], hs[0]
        m_on, m100, J = [], [], []
        beat_len = per
        i_up = rest_n + 500                     # peak position (within beat)
        for b in range(N_BEATS):
            seg0 = b * beat_len
            m_beat = np.empty(beat_len)
            I_beat = np.empty(beat_len)
            for i in range(beat_len):
                k = seg0 + i
                m0 = ms[k] + (m0 - ms[k]) * em[k]
                h0 = hs[k] + (h0 - hs[k]) * eh[k]
                m_beat[i] = m0
                I_beat[i] = G * m0 * h0 * (Vt[k] - E_REV)
            m_on.append(m_beat[rest_n])          # beat onset = instant of entering the beat from rest (true pedestal)
            m100.append(m_beat[min(rest_n + 1000, beat_len - 1)])
            w = I_beat[rest_n + 500:rest_n + 2500]  # repolarisation window (peak to ~200 ms after, aligned to beat start)
            v_w = V_beat[500:2500]
            k = int(np.argmax(w))
            J.append(float(w[k] / (v_w[k] - E_REV)))
        R = float(np.median(J[-5:]) / np.median(J[:2])) if np.median(J[:2]) > 0 else float("nan")
        return dict(CL_ms=CL_ms, m_onset_last=float(m_on[-1]), m100_last=float(m100[-1]),
                    R=R, J=J, m_onset=m_on, m100=m100)

    out = {}
    for CL in CL_LIST_MS:
        r = run_train(CL)
        if r:
            out[str(CL)] = r

    # isolated +40 step (envelope-convention replica): -80 100 ms -> +40 400 ms
    Vs = np.concatenate([np.full(1000, -80.0), np.full(4000, 40.0)])
    m0 = float(ms_tab(-80.0))
    for i, v in enumerate(Vs):
        m0 = ms_tab(v) + (m0 - ms_tab(v)) * np.exp(-DT / tm_tab(v))
        if i == 1999:                            # 100 ms after the step
            m_step100 = float(m0)
    return out, m_step100


def main():
    t0 = time.time()
    print("=" * 74, flush=True)
    print(" alpha model - B6-H2 quantitative prediction card (pure-simulation pre-registration)" + (" (smoke 16713003)" if SMOKE else " (nine cells)"), flush=True)
    print(" prediction: R(CL=500)>=3 and R(CL=2000)<=1.5; measured curve log error <=0.35 -> B6 closes under H2", flush=True)
    print("=" * 74, flush=True)

    eng = load_engine()
    amp = json.load(open(eng.F_AMP, encoding="utf-8"))
    hook = json.load(open(eng.F_HOOK, encoding="utf-8"))
    inact = json.load(open(eng.F_INACT, encoding="utf-8"))
    hss = json.load(open(eng.F_HSS, encoding="utf-8"))
    V_ap, _ = eng.load_mat("ap_protocol.mat", "16713003", "ap")
    V_beat = representative_beat(V_ap)
    print(f" representative beat: duration {len(V_beat) * DT * 1000:.0f} ms, vmax={V_beat.max():.1f} mV", flush=True)

    per_cell = {}
    for cell in CELLS:
        Vi, Ii = eng.load_mat("inactivation_protocol.mat", cell, "inactivation")
        tabs = eng.build_tabs(cell, amp, hook, inact, hss, Ii, Vi)
        out, m_step100 = simulate_cell(eng, tabs, V_beat)
        per_cell[cell] = {"cl": out, "m_step100_isolated": m_step100}
        line = f"  {cell}: isolated +40 m@100ms={m_step100 * 100:5.1f}%  | "
        line += "  ".join(f"CL{c}: R={out[c]['R']:.2f} pedestal={out[c]['m_onset_last'] * 100:.0f}%"
                          for c in ("500", "1000", "2000") if c in out)
        print(line, flush=True)

    # population curves
    print("\n[population predicted curves (median)]", flush=True)
    curve = {}
    for CL in CL_LIST_MS:
        rs = [per_cell[c]["cl"][str(CL)]["R"] for c in per_cell if str(CL) in per_cell[c]["cl"]]
        ons = [per_cell[c]["cl"][str(CL)]["m_onset_last"] for c in per_cell if str(CL) in per_cell[c]["cl"]]
        m100s = [per_cell[c]["cl"][str(CL)]["m100_last"] for c in per_cell if str(CL) in per_cell[c]["cl"]]
        curve[str(CL)] = dict(R_med=float(np.median(rs)), R_lo=float(np.min(rs)), R_hi=float(np.max(rs)),
                              onset_med=float(np.median(ons)), m100_med=float(np.median(m100s)))
        print(f"  CL={CL:5d}ms: R median={curve[str(CL)]['R_med']:.2f}"
              f" [{curve[str(CL)]['R_lo']:.2f}~{curve[str(CL)]['R_hi']:.2f}]"
              f"  pedestal m_onset={curve[str(CL)]['onset_med'] * 100:.1f}%"
              f"  m@100ms={curve[str(CL)]['m100_med'] * 100:.1f}%", flush=True)
    iso = [per_cell[c]["m_step100_isolated"] for c in per_cell]
    print(f"  isolated +40 step m@100ms median = {np.median(iso) * 100:.1f}%"
          f" (model tau_m(+40) anchor 0.29 s convention, should be ~=30%; the difference from envelope measured ~8% is the B6 body)", flush=True)
    p1 = curve["500"]["R_med"] >= 3.0 and curve["2000"]["R_med"] <= 1.5
    print(f"\n pre-registered P1 (R(500)>=3 and R(2000)<=1.5) -> model prediction {'satisfied' if p1 else 'NOT satisfied: H2 mechanism magnitude insufficient, registered'}", flush=True)

    tag = "_冒烟" if SMOKE else "_结果"
    fjson = os.path.join(HERE, f"2026-09-14_α模型_B6_H2_定量预测卡{tag}.json")
    out = {"meta": {"smoke": SMOKE, "cl_list_ms": CL_LIST_MS, "n_beats": N_BEATS,
                    "runtime_s": time.time() - t0},
           "per_cell": per_cell, "curve": curve,
           "isolated_step_m100_median": float(np.median(iso)), "P1_pass": p1}
    with open(fjson, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    print(f"\n results saved: {fjson}", flush=True)

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))
        from daimon_runtime import setup_plot
        setup_plot()
    except Exception:
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    cls = [int(c) for c in CL_LIST_MS]
    Rmed = [curve[str(c)]["R_med"] for c in cls]
    Rlo = [curve[str(c)]["R_lo"] for c in cls]
    Rhi = [curve[str(c)]["R_hi"] for c in cls]
    ax.fill_between(cls, Rlo, Rhi, alpha=0.2, color="tab:blue")
    ax.plot(cls, Rmed, "o-", color="tab:blue", label="R(CL) population median")
    ax.axhline(3.0, color="tab:green", ls=":", lw=1, label="pre-registered line R(500)>=3")
    ax.axhline(1.5, color="tab:red", ls=":", lw=1, label="pre-registered line R(2000)<=1.5")
    ax.axhline(3.6, color="k", ls="--", lw=1, label="B6 gap 3.6x")
    ax.set_xscale("log"); ax.set_xticks(cls); ax.set_xticklabels(cls)
    ax.set_xlabel("cycle length CL (ms)"); ax.set_ylabel("accumulation ratio R = last5/first2 beats")
    ax.set_title("H2 prediction: accumulation ratio vs cycle length"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    onmed = [curve[str(c)]["onset_med"] * 100 for c in cls]
    m1med = [curve[str(c)]["m100_med"] * 100 for c in cls]
    ax.plot(cls, onmed, "o-", label="pedestal m_onset (last beat)")
    ax.plot(cls, m1med, "s--", label="in-beat m@100ms (last beat)")
    ax.axhline(float(np.median(iso)) * 100, color="k", ls="--", lw=1,
               label=f"isolated beat m@100ms = {np.median(iso) * 100:.0f}%")
    ax.axhline(29, color="tab:red", ls=":", lw=1, label="AP requirement 29%")
    ax.set_xscale("log"); ax.set_xticks(cls); ax.set_xticklabels(cls)
    ax.set_xlabel("CL (ms)"); ax.set_ylabel("m (% m∞)")
    ax.set_title("pedestal and in-beat m vs cycle length"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    c0 = CELLS[0]
    for CLs, col in [("500", "tab:red"), ("1000", "tab:blue"), ("4000", "tab:green")]:
        if CLs in per_cell[c0]["cl"]:
            J = per_cell[c0]["cl"][CLs]["J"]
            ax.plot(np.arange(1, len(J) + 1), J, "o-", ms=3, color=col, label=f"CL={CLs}ms")
    ax.set_xlabel("beat index n"); ax.set_ylabel("J(n)")
    ax.set_title(f"per-beat normalised peak current ({c0})"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.suptitle("B6-H2 quantitative prediction card (pre-registered; B6 to be closed with identical-waveform train data)"
                 + (" (smoke)" if SMOKE else ""), fontsize=12)
    fpng = os.path.join(HERE, f"2026-09-14_α模型_B6_H2_定量预测卡{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" figure saved: {fpng}", flush=True)


if __name__ == "__main__":
    main()
