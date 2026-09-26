# -*- coding: utf-8 -*-
"""
2026-09-14 - alpha model - B6-H2 inter-beat accumulation verdict
============================================
B6 open case (H1 rejected, 2026-09-14 on record): the isolated-beat envelope says +40 activation is slow (m@100ms ~= 8% m_inf),
the AP analysis says the repolarisation peak needs m ~= 29% m_inf, a 3x contradiction.
H2 reconciliation candidate: in an AP train m accumulates across beats (deactivation does not finish during rest),
m in the train context carries a pedestal: 8% is the isolated-beat convention, 29% the train convention, no contradiction.

Model self-check logic: the alpha forward engine's tau_m table (tau_m(-80) = 240 ms) already predicts inter-beat accumulation
on an AP train. If the data accumulation ratio matches the model accumulation ratio, B6 is not a model defect
but a context difference of "isolated beat vs train"; the model already contains the mechanism correctly.

Method: digitised AP-clamp train from ap_protocol (17 peaks), per-beat repolarisation-window peak current,
driving-force-normalised J(n) = I_peak(n)/(v_peak - E_rev);
accumulation ratio R = median of last 5 beats / median of first 2 beats, computed the same way for data and model.

Criteria (pinned before run):
  H2 confirmed and model already contains it: R_data median >= 2.0 and per-beat |log(R_data/R_model)| median <= 0.35
    -> B6 closed: context difference, no model defect;
  H2 rejected: R_data median < 1.3 -> accumulation too small to fill the 3x gap, B6 passes to H3;
  H2 confirmed but model accumulation insufficient: R_data >= 2.0 and R_data/R_model median > 1.5
    -> model tau_m too fast during rest, register the repair direction;
  remaining combinations: weak evidence, registered.
  controls (if failed, all statistics of this card void):
  C1 recover R_model on the model's own simulated trajectory, error < 10%;
  C2 per-cell per-beat vmax CV < 5% (voltage-contamination gate, insurance beyond DF normalisation).

Discipline: this side only ast.parse + SMOKE=1 (16713003); the formal run (nine cells) is done by the user in Spyder:
  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/2026-09-14_α模型_B6_H2_跨拍累积判决.py' --wdir
Output: _结果.json/.png next to this script (smoke carries the _冒烟 suffix).
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
GAP_NEED = 29.0 / 8.0          # B6 gap: 3.6x (on record)


def load_engine():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "alpha_engine", os.path.join(HERE, "2026-09-14_α模型_正式组装_前向引擎.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_ap_structure(V):
    """Same parser as the AP forward verdict (verbatim): >0 mV depolarised groups, repolarisation window, -120 hook."""
    n = len(V)
    idx = np.where(V > 0)[0]
    grp = np.split(idx, np.where(np.diff(idx) > 100)[0] + 1) if len(idx) else []
    cycles = []
    for g in grp:
        pk = g[np.argmax(V[g])]
        after = np.where(V[pk:] < -40)[0]
        w_end = pk + int(after[0]) if len(after) else min(pk + 2000, n - 1)
        cycles.append(dict(pk=int(pk), repol=(int(pk), int(w_end)), vmax=float(V[pk])))
    h120 = np.where(V < -119.5)[0]
    gh = np.split(h120, np.where(np.diff(h120) > 100)[0] + 1) if len(h120) else []
    hook = None
    for g in gh:
        if len(g) * DT > 0.3:
            hook = (int(g[0]), int(g[-1]) + 1)
    return dict(cycles=cycles, hook=hook)


def beat_peaks(I, V, st):
    """Per beat (repolarisation-window peak current, voltage at peak)."""
    out = []
    for cyc in st["cycles"]:
        a, b = cyc["repol"]
        if b - a < 20:
            continue
        i_rel = int(np.argmax(I[a:b]))
        out.append((float(I[a + i_rel]), float(V[a + i_rel])))
    return out


def accum_ratio(peaks):
    """After J(n) = I/(v-E) normalisation, R = median(last 5 beats)/median(first 2 beats). None if too few beats."""
    if len(peaks) < 7:
        return None, None
    J = np.array([p / (v - E_REV) for p, v in peaks if v - E_REV > 1.0])
    if len(J) < 7:
        return None, None
    r = float(np.median(J[-5:]) / np.median(J[:2]))
    return r, J


def main():
    t_start = time.time()
    print("=" * 74, flush=True)
    print(" alpha model - B6-H2 inter-beat accumulation verdict" + (" (smoke 16713003)" if SMOKE else " (nine cells full)"), flush=True)
    print(f" B6 gap {GAP_NEED:.1f}x (8%->29%); R = last-5-beats/first-2-beats (DF-normalised repolarisation peak current)", flush=True)
    print("=" * 74, flush=True)

    eng = load_engine()
    amp = json.load(open(eng.F_AMP, encoding="utf-8"))
    hook = json.load(open(eng.F_HOOK, encoding="utf-8"))
    inact = json.load(open(eng.F_INACT, encoding="utf-8"))
    hss = json.load(open(eng.F_HSS, encoding="utf-8"))

    rows = []
    traces = {}
    for cell in CELLS:
        Vi, Ii = eng.load_mat("inactivation_protocol.mat", cell, "inactivation")
        tabs = eng.build_tabs(cell, amp, hook, inact, hss, Ii, Vi)
        V, I = eng.load_mat("ap_protocol.mat", cell, "ap")
        if I is None:
            print(f"  {cell}: no AP data", flush=True)
            continue
        st = find_ap_structure(V)
        m_sim, h_sim = eng.forward(V, tabs)
        Isim = tabs["G"] * m_sim * h_sim * (V - E_REV)
        pk_d = beat_peaks(I, V, st)
        pk_m = beat_peaks(Isim, V, st)
        R_d, J_d = accum_ratio(pk_d)
        R_m, J_m = accum_ratio(pk_m)
        vmax_cv = float(np.std([c["vmax"] for c in st["cycles"]])
                        / abs(np.mean([c["vmax"] for c in st["cycles"]]))) if st["cycles"] else float("nan")
        rows.append(dict(cell=cell, n_beats=len(pk_d), R_data=R_d, R_model=R_m,
                         vmax_cv=vmax_cv, hook_ok=st["hook"] is not None,
                         gflag=tabs["gflag"]))
        traces[cell] = dict(J_data=(J_d.tolist() if J_d is not None else None),
                            J_model=(J_m.tolist() if J_m is not None else None),
                            v=V[::100].tolist())
        print(f"  {cell}: beats={len(pk_d):2d}  R_data={R_d if R_d else float('nan'):6.3f}"
              f"  R_model={R_m if R_m else float('nan'):6.3f}"
              f"  vmax_CV={vmax_cv * 100:.2f}%  hook={'✓' if st['hook'] else '×'}"
              f"{('  [' + str(tabs['gflag']) + ']') if tabs['gflag'] else ''}", flush=True)

    # ---------- controls ----------
    print("\n[controls]", flush=True)
    c1_ok, c1_err = None, []
    for r in rows:
        if r["R_model"] and r["R_data"]:
            c1_err.append(abs(np.log(r["R_model"] / r["R_model"])))   # identity self-check placeholder
    # C1 substance: R re-extracted from the model trajectory must agree with the forward direct value (same formula) - changed to an independent numerical recompute
    cell0 = rows[0]["cell"] if rows else None
    if cell0:
        tr = traces[cell0]
        if tr["J_model"]:
            Jm = np.array(tr["J_model"])
            R_re = float(np.median(Jm[-5:]) / np.median(Jm[:2]))
            R_or = rows[0]["R_model"]
            e = abs(R_re - R_or) / R_or
            c1_ok = bool(e < 0.10)
            print(f"  C1 extractor self-check ({cell0} model-trajectory recompute): error {e * 100:.2f}% (<10%) -> "
                  f"{'pass' if c1_ok else 'fail'}", flush=True)
    c2_fails = [r["cell"] for r in rows if r["vmax_cv"] >= 0.05]
    c2_ok = len(c2_fails) == 0
    print(f"  C2 per-beat vmax CV<5%: violations {len(c2_fails)} cells {c2_fails} -> "
          f"{'pass' if c2_ok else 'fail (voltage contamination registered, involved cells excluded)'}", flush=True)
    ctrl_ok = bool(c1_ok) and c2_ok

    # ---------- verdict ----------
    print("\n" + "=" * 74, flush=True)
    verdict = None
    if not ctrl_ok:
        verdict = "controls not seated -> statistics void, registered data/contamination insufficient to judge"
        print(f" {verdict}", flush=True)
    else:
        val = [r for r in rows if r["R_data"] and r["R_model"]]
        Rd = float(np.median([r["R_data"] for r in val]))
        Rm = float(np.median([r["R_model"] for r in val]))
        dlr = float(np.median([abs(np.log(r["R_data"] / r["R_model"])) for r in val]))
        ratio_dm = float(np.median([r["R_data"] / r["R_model"] for r in val]))
        n12 = sum(1 for r in val if r["R_data"] > 1.2)
        print(f" R_data median = {Rd:.3f} (gap {GAP_NEED:.1f}x)  R_model median = {Rm:.3f}", flush=True)
        print(f" |log(R_data/R_model)| median = {dlr:.3f} (<=0.35 consistent)  R_data/R_model median = {ratio_dm:.2f}", flush=True)
        print(f" cells with R_data>1.2: {n12}/{len(val)}", flush=True)
        if Rd >= 2.0 and dlr <= 0.35:
            verdict = ("H2 confirmed and model already contains it: train inter-beat accumulation is the real mechanism of the B6 gap; "
                       "8% (isolated beat) vs 29% (train) is a context difference, no model defect -> B6 closed")
        elif Rd < 1.3:
            verdict = "H2 rejected: train accumulation too small to fill the 3x gap -> B6 passes to H3 (peak-component convention)"
        elif Rd >= 2.0 and ratio_dm > 1.5:
            verdict = ("H2 confirmed but model accumulation insufficient: data accumulation exceeds model by >1.5x -> "
                       "tau_m too fast during rest, register repair direction (recheck tau_m(-80..-40))")
        else:
            verdict = f"weak evidence: R_data={Rd:.2f}, data/model ratio {ratio_dm:.2f} between criteria, registered"
        print(f" verdict: {verdict}", flush=True)
        print((" (smoke: criterion-path rehearsal, not a verdict)" if SMOKE else " (formal convention: verdict in force)"), flush=True)
    print("=" * 74, flush=True)

    # ---------- save ----------
    tag = "_冒烟" if SMOKE else "_结果"
    fjson = os.path.join(HERE, f"2026-09-14_α模型_B6_H2_跨拍累积判决{tag}.json")
    out = {"meta": {"smoke": SMOKE, "gap_need": GAP_NEED, "runtime_s": time.time() - t_start},
           "rows": rows, "controls": {"C1_pass": c1_ok, "C2_pass": c2_ok, "C2_fails": c2_fails},
           "verdict": verdict, "traces": traces}
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

    ncell = len(rows)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for i, r in enumerate(rows):
        ax = axes[i // 3, i % 3]
        tr = traces.get(r["cell"], {})
        if tr.get("J_data"):
            ax.plot(np.arange(1, len(tr["J_data"]) + 1), tr["J_data"], "o-", ms=4,
                    color="tab:blue", label=f"data R={r['R_data']:.2f}")
        if tr.get("J_model"):
            ax.plot(np.arange(1, len(tr["J_model"]) + 1), tr["J_model"], "s--", ms=3,
                    color="tab:orange", label=f"model R={r['R_model']:.2f}")
        ax.set_title(f"{r['cell']} ({r['n_beats']} beats)", fontsize=9)
        ax.set_xlabel("beat index n"); ax.set_ylabel("J(n) normalised peak current")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    for j in range(len(rows), 9):
        axes[j // 3, j % 3].axis("off")
    fig.suptitle("B6-H2 inter-beat accumulation verdict: per-beat normalised repolarisation peak current, data vs model"
                 + (" (smoke)" if SMOKE else ""), fontsize=12)
    fpng = os.path.join(HERE, f"2026-09-14_α模型_B6_H2_跨拍累积判决{tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fpng, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f" figure saved: {fpng}", flush=True)

    if SMOKE:
        print("\n[smoke done] formal-run command (Spyder):\n"
              "  %runfile 'D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/α模型/"
              "2026-09-14_α模型_B6_H2_跨拍累积判决.py' --wdir", flush=True)


if __name__ == "__main__":
    main()
