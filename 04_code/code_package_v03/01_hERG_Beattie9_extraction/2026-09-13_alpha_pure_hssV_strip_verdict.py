# 2026-09-13_alpha_pure_hssV_strip_verdict.py
# Purpose (gap-1 main attack: pure h_ss(V) curve, m contamination stripped):
#   Inactivation protocol per cycle: +50 x 0.6 s full activation + full inactivation
#   -> -90 x 0.06 s reset (h fully recovered, m residual ~0.43) -> test level
#   V x 0.150 s (family -100..+50) -> -120 x 0.5 s rebound DoE.
#   Strip derivation (zero tau dependence, G cancels exactly):
#     end-of-test current I_test = G * m_T * h_T * (V - E_rev) (for levels with
#       tau_h(V) << 150 ms, h_T ~= h_ss(V); for slower tau_h levels h_T is an
#       upper bound of h_ss, declared);
#     rebound DoE amplitude A = G * DF120 * m_T (I_reb = G*m_T*DF*[e^{-t/tau_d} -
#       (1-h_T) e^{-t/tau_r}]; DoE-fitted A ~= G*DF*m_T for any h_T, derived).
#
# [Protocol structure (programmatically parsed 2026-09-13, on record)]
#   total length 46.41 s; cycle = [-120x0.05, -80x0.2, +50x0.6, -90x0.06,
#   test V x 0.150, -120x0.5, -80x1.34]; test family -100..+50 (acurve 16 levels,
#
# Criteria (declared pre-run):
#   QC1 (physics gate): h_ss(V) in [-0.1, 1.3] and |I_test| >= 4 sigma (sigma =
#     baseline std of that cell's -80 x 1.34 s segment); else the cell-level point
#   QC2 (B1 life-or-death): median h_ss of valid levels -100..-50 in [0.7, 1.3]
#     -> B1 (h_ss = 1 physical claim) supported; any level median < 0.7 -> B1
#   QC3 (positive-level consistency): +40 stripped h_ss vs section 3.2 h40
#     nine-cell values (0.0023-0.0296), median ratio in [0.3, 3].
#   SEAL criterion (per level, independent): QC1-passing cells >= 5 and CV < 0.3
#     -> [SEAL]; >= 5 but CV >= 0.3 -> [REGISTER]; < 5 -> [INSUFFICIENT DATA].
# Controls (declared pre-run):
#   C1 DoE inversion of tau_r 3.5 ms, error < 15%;
#   C2 synthetic cycle (m_T=0.5, h_T=0.3, G=0.1, V=-20, noise = measured sigma
#     of 16713003): strip formula recovers h_T with error < 10% -> if failed,
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
DF120 = abs(-120.0 - E_REV)

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


def doe_fit(t, y):
    if len(t) < 50:
        return None
    best = None
    for tr in TR_GRID:
        for td in TD_GRID[TD_GRID >= SEP_MIN * tr]:
            x = np.exp(-t / td) - np.exp(-t / tr)
            X = np.column_stack([np.ones(len(t)), x])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ sol) ** 2))
            if best is None or sse < best[0]:
                best = (sse, tr, td, sol)
    sse, tr, td, sol = best
    return dict(tau_r=float(tr), tau_d=float(td), A=float(sol[1]),
                edge=bool(tr <= TR_GRID[0] * 1.02 or td >= TD_GRID[-1] * 0.98
                          or tr >= TR_GRID[-1] * 0.98))


def strip_cell(V, I):
    """Per cell: strip h_ss at each test level. Returns {gear: dict(I_test, A, h, qc, flag)}, sigma."""
    info = segments(V)
    sig = []
    for v, s0, n in info:
        if abs(v + 80) < 2 and n * DT > 1.0:
            sig.append(np.std(I[s0 + 2000: s0 + n]))
    sigma = float(np.median(sig)) if sig else np.nan
    out = {}
    for i, (v, s0, n) in enumerate(info):
        # test level: 0.14-0.16 s, followed by -120 x 0.5 s; -90 reset within prior two segments
        if not (0.14 < n * DT < 0.16):
            continue
        if i + 1 >= len(info) or not (abs(info[i + 1][0] + 120) < 2
                                      and info[i + 1][2] * DT > 0.4):
            continue
        prev90 = any(abs(info[j][0] + 90) < 2 for j in range(max(0, i - 2), i))
        if not prev90 or abs(v + 90) < 2:                    # -90 level DF ~= 0, dropped
            continue
        i_test = float(np.mean(I[s0 + n - 2000: s0 + n]))  # last 20 ms of test segment
        a, b = info[i + 1][1], info[i + 1][1] + 3000       # 300 ms before rebound
        r = doe_fit(np.arange(3000) * DT, -I[a:b])
        if r is None:
            continue
        df = v - E_REV
        h = i_test * DF120 / (r["A"] * df) if r["A"] != 0 and abs(df) > 1e-6 else np.nan
        qc = bool(np.isfinite(h) and -0.1 <= h <= 1.3
                  and abs(i_test) >= 4 * (sigma or 0) and not r["edge"])
        out[round(v)] = dict(I_test=i_test, A=r["A"], h=float(h), qc=qc,
                             tau_r=r["tau_r"], tau_d=r["tau_d"],
                             flag=None if qc else "qc1_fail")
    return out, sigma


def main():
    rng = np.random.default_rng(7)
    print("=" * 88)
    print(" alpha-model pure h_ss(V) strip verdict" + (" (smoke 16713003)" if SMOKE else " (nine cells full)"))
    print(" criteria: per level QC1>=5 and CV<0.3 SEAL | QC2 B1 life-or-death (-100..-50 median in [0.7,1.3]) | QC3 +40 consistency")
    print("=" * 88, flush=True)

    # ---------- controls ----------
    print("\n[controls]", flush=True)
    t_c = np.arange(int(0.3 / DT)) * DT
    y_c = 3.0 * (np.exp(-t_c / 0.030) - np.exp(-t_c / 0.0035)) + rng.normal(0, 0.02, len(t_c))
    rc = doe_fit(t_c, y_c)
    c1 = bool(rc and abs(rc["tau_r"] - 0.0035) / 0.0035 < 0.15)
    print(f"  C1 DoE inversion tau_r: {rc['tau_r'] * 1e3:.2f} ms (truth 3.5) -> {'pass' if c1 else 'fail'}",
          flush=True)
    # C2 synthetic cycle: m_T=0.5 h_T=0.3 G=0.1 V=-20
    G0, mT, hT, vt = 0.1, 0.5, 0.3, -20.0
    i_syn = G0 * mT * hT * (vt - E_REV) + rng.normal(0, 0.02)
    reb = G0 * DF120 * mT * (np.exp(-t_c / 0.030) - (1 - hT) * np.exp(-t_c / 0.003)) \
        + rng.normal(0, 0.02, len(t_c))
    rs = doe_fit(t_c, reb)
    h_rec = i_syn * DF120 / (rs["A"] * (vt - E_REV))
    c2 = bool(rs and abs(h_rec - hT) / hT < 0.10)
    print(f"  C2 synthetic strip: h_rec={h_rec:.3f} (truth 0.300, A={rs['A']:.3f}) -> "
          f"{'pass' if c2 else 'fail'}", flush=True)
    if not (c1 and c2):
        print("  controls not seated -> statistics void, halt.", flush=True)
        return
    print("  controls seated.", flush=True)

    # ---------- real data ----------
    print("\n[real data]", flush=True)
    res = {}
    all_gears = set()
    for c in CELLS:
        V, I = load_mat("inactivation_protocol.mat", c, "inactivation")
        if I is None:
            print(f"  {c}: no data", flush=True)
            continue
        cur, sigma = strip_cell(V, I)
        res[c] = dict(curve=cur, sigma=sigma)
        all_gears.update(cur.keys())
        line = " ".join(f"{g}:{cur[g]['h']:.2f}{'' if cur[g]['qc'] else '×'}"
                        for g in sorted(cur))
        print(f"  {c}: σ={sigma:.4f} | {line}", flush=True)

    gears = sorted(all_gears)
    print("\n" + "-" * 88, flush=True)
    print("[per-level verdict]", flush=True)
    gear_verdict = {}
    for g in gears:
        vals = [res[c]["curve"][g]["h"] for c in res
                if g in res[c]["curve"] and res[c]["curve"][g]["qc"]]
        n = len(vals)
        if n >= 2:
            med = float(np.median(vals))
            cv = float(np.std(vals) / abs(np.mean(vals))) if abs(np.mean(vals)) > 1e-9 \
                else np.inf
        else:
            med, cv = (float(vals[0]), np.inf) if n == 1 else (np.nan, np.inf)
        if n >= 5 and cv < 0.3:
            v = "[SEAL]"
        elif n >= 5:
            v = "[REGISTER]"
        else:
            v = "[INSUFFICIENT DATA]"
        gear_verdict[g] = dict(n=n, med=med, cv=cv, verdict=v)
        print(f"  {g:+5d}mV: n={n} median={med:.3f} CV={cv:.3f} {v}", flush=True)

    # ---------- QC2 B1 life-or-death ----------
    b1_gears = [g for g in gears if -100 <= g <= -50 and gear_verdict[g]["n"] >= 5]
    b1_dead = []
    if not b1_gears:
        b1_ok = None
        print("\n[QC2] B1: insufficient valid levels (n<5), not judged", flush=True)
    else:
        b1_dead = [g for g in b1_gears if gear_verdict[g]["med"] < 0.7]
        b1_ok = not b1_dead
        print(f"\n[QC2] B1 (h_ss(-100..-50)=1 physical claim): "
              f"{'supported (all level medians in [0.7,1.3])' if b1_ok else f'dead levels {b1_dead}'}", flush=True)

    # ---------- QC3 +40 consistency ----------
    h40_ref = {c: v for c, v in
               [("16704007", 0.0216), ("16704047", 0.0197), ("16707014", 0.0143),
                ("16708016", 0.0042), ("16708060", 0.0022), ("16708118", 0.0296),
                ("16713003", 0.0023), ("16713110", 0.0164), ("16715049", 0.0148)]}
    qc3 = None
    if 40 in gear_verdict and gear_verdict[40]["n"] >= 3:
        ref = float(np.median(list(h40_ref.values())))
        ratio = gear_verdict[40]["med"] / ref if ref else np.nan
        qc3 = bool(0.3 <= ratio <= 3.0)
        print(f"[QC3] +40 stripped median={gear_verdict[40]['med']:.4f} vs h40 reference median"
              f"={ref:.4f} ratio={ratio:.2f} -> {'consistent' if qc3 else 'inconsistent'}", flush=True)
    else:
        print("[QC3] +40 level valid cells insufficient, not judged", flush=True)

    n_seal = sum(1 for d in gear_verdict.values() if d["verdict"] == "[SEAL]")
    n_reg = sum(1 for d in gear_verdict.values() if d["verdict"] == "[REGISTER]")
    b1_txt = "supported" if b1_ok else ("dead (see above)" if b1_ok is False else "not judged")
    print(f"\n overall verdict: SEAL {n_seal} levels, REGISTER {n_reg} levels, "
          f"insufficient data {sum(1 for d in gear_verdict.values() if d['verdict'] == '[INSUFFICIENT DATA]')} levels; "
          f"B1 {b1_txt}；QC3 "
          f"{'consistent' if qc3 else ('inconsistent' if qc3 is False else 'not judged')}", flush=True)

    # ---------- figure ----------
    fig, ax = plt.subplots(figsize=(9, 6))
    for c in res:
        gs = sorted(res[c]["curve"])
        hs = [res[c]["curve"][g]["h"] for g in gs]
        qc = [res[c]["curve"][g]["qc"] for g in gs]
        ax.plot([g for g, q in zip(gs, qc) if q], [h for h, q in zip(hs, qc) if q],
                "o-", ms=4, lw=0.8, alpha=0.55, label=c)
        ax.plot([g for g, q in zip(gs, qc) if not q],
                [h for h, q in zip(hs, qc) if not q], "x", ms=5, alpha=0.4)
    meds = [gear_verdict[g]["med"] for g in gears]
    ax.plot(gears, meds, "k-s", lw=2.2, ms=7, label="cross-cell median")
    ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
    ax.axhspan(0.7, 1.3, color="tab:green", alpha=0.06, label="B1 criterion band")
    ax.set_xlabel("V (mV)")
    ax.set_ylabel("h_ss(V) stripped value")
    ax.set_title("pure h_ss(V) strip (x = QC1-rejected points)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")

    fjson = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(dict(cells=res, gears=gear_verdict, b1_ok=b1_ok, b1_dead=b1_dead,
                       qc3=qc3), f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  figure saved: {fpng}", flush=True)
    print(f"  results saved: {fjson}", flush=True)


if __name__ == "__main__":
    main()
