# judge182.py — code-182 model62 identifiability spectrum card, judge (preregistration 2026-09-11)
# Independent recomputation: this file is self-contained expm simulation (rewritten from model card section 3; imports no function from judge179/de182).
# Accepts only: canonical parameters, raw data (de174.load174 pipeline), and the execution side's final saved artifacts.
# Usage: python judge182.py            -> full A1-A5 recomputation + verdict md saved
import json, math, os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from de174 import load174, PROTOS
from de173 import DT
from de179 import EREV   # EREV is a data-pipeline constant (Nernst, not fitted); the judge keeps the same data metric
import scipy.linalg

OUT = os.path.join(HERE, "model62_identifiability")
GX_K = 10.0
P6X = [0.180967466, -46.1749861, 22.4481177, 3.97337168, 0.00763900352, 0.0747609133]


def sim_judge(V, x16):
    """Judge-written expm channel (model card section 3 implemented line by line: 4-state ring with equal opposite-edge rates + k43 double exponential + X slow component)."""
    P = 10.0 ** np.asarray(x16[:8], float)
    gkr = 10.0 ** x16[8]
    pf1, pf2 = 10.0 ** x16[9], x16[10]
    amp, tsc = 10.0 ** x16[11], 10.0 ** x16[12]
    gmid, VH, KX = x16[13], x16[14], 10.0 ** x16[15]
    P0, P1, P2, P3, P4, P5, P6, P7 = P
    Vv = np.asarray(V, float)
    out = np.empty(len(Vv))
    y = np.array([0., 0., 0., 1.])
    dt_ms = DT * 1000.0
    for i, v in enumerate(Vv):
        k32 = P4 * np.exp(P5 * v);  k23 = P6 * np.exp(-P7 * v)
        k43 = P0 * np.exp(P1 * v) + pf1 * np.exp(pf2 * v);  k34 = P2 * np.exp(-P3 * v)
        M = np.array([[-(k43 + k23), k34, 0., k32], [k43, -(k34 + k23), k32, 0.],
                      [0., k23, -(k32 + k34), k43], [k23, 0., k34, -(k43 + k32)]])
        y = expm(M * dt_ms) @ y
        out[i] = gkr * y[2] * (v - EREV)
    # X slow component: x_s exponential relaxation + negative-potential gating (model card section 3.3; stepping semantics aligned verbatim with de169.evolve_x)
    XMAX, TAU0, Z, ALPHA = P6X[0], P6X[3], P6X[4], P6X[5]
    TAU0S = TAU0 * tsc
    xs = np.empty(len(Vv))
    X = XMAX / (1.0 + np.exp(-(Vv[0] - VH) / KX))
    for i in range(len(Vv)):
        xi = XMAX / (1.0 + np.exp(-(Vv[i] - VH) / KX))
        tv = TAU0S * math.exp(Z * Vv[i])
        if tv < 0.05:
            xs[i] = np.nan
            continue
        X = xi + (X - xi) * math.exp(-DT / tv)
        xs[i] = X
    gx = 1.0 / (1.0 + np.exp((Vv - gmid) / GX_K))
    return out + amp * ALPHA * xs * gx * (Vv - EREV)


def _job(job):
    key, xl, v = job
    return key, sim_judge(v, xl)


def step_h(x16, j):
    return 1e-3 * abs(x16[j]) if j == 10 else 1e-3


def main():
    t0 = time.time()
    diffs = []
    x16v = json.load(open(os.path.join(HERE, "counter_181_model62_判官复核件.json"), encoding='utf-8'))
    x16 = np.asarray(x16v["x16"], float)
    data2, keep = load174()
    base = {pr: sim_judge(data2[pr][0], x16) for pr in PROTOS}
    # -- A0 prerequisite: court opens only if the judge baseline is digit-identical to the canonical judge181 block --
    for pr in PROTOS:
        c_k = np.asarray(data2[pr][1], float)[keep[pr]]
        r2 = 1.0 - float(((base[pr] - np.asarray(data2[pr][1], float))[keep[pr]] ** 2).sum()) / float(c_k @ c_k)
        d = abs(r2 - x16v["judge181"]["r2"][pr])
        print(f"[judge182-A0] {pr} baseline R2 delta={d:.2e}", flush=True)
        if d > 1e-6:
            diffs.append(f"A0 {pr}: judge baseline differs from canonical by {d:.2e} -- court adjourned")
    if diffs:
        print("[judge182] baseline incorrect, court adjourned", flush=True)
        sys.exit(1)
    # -- A1: independent recomputation of the 13x13 Fisher --
    sig2 = {}
    for pr in PROTOS:
        c_ = np.asarray(data2[pr][1], float)
        km = keep[pr]
        res = base[pr] - c_
        segs, i, n = [], 0, len(km)
        while i < n:
            if km[i]:
                j = i
                while j < n and km[j]:
                    j += 1
                segs.append((i, j)); i = j
            else:
                i += 1
        pooled = float((res[km] ** 2).mean())
        s2 = np.full(n, pooled)
        for (a, b) in segs:
            s2[a:b] = max(float((res[a:b] ** 2).mean()), pooled * 1e-2)
        sig2[pr] = s2
    jobs = []
    for j in range(13):
        h = step_h(x16, j)
        for sgn in (+1, -1):
            xj = x16.copy(); xj[j] += sgn * h
            for pr in PROTOS:
                jobs.append(((j, sgn, pr), list(xj), data2[pr][0]))
    with Pool(24) as p:
        res = dict(p.map(_job, jobs))
    F = np.zeros((13, 13))
    for pr in PROTOS:
        km = keep[pr]
        w = 1.0 / sig2[pr][km]
        Jm = np.zeros((int(km.sum()), 13))
        for j in range(13):
            h = step_h(x16, j)
            Jm[:, j] = (res[(j, 1, pr)][km] - res[(j, -1, pr)][km]) / (2.0 * h)
        F += (Jm * w[:, None]).T @ Jm
    eF = json.load(open(os.path.join(OUT, "fisher_matrix_13x13.json"), encoding='utf-8'))["matrix"]
    eF = np.array(eF, float)
    scale = np.abs(F).max()
    a1 = float(np.abs(F - eF).max() / scale)
    print(f"[judge182-A1] Fisher max relative diff={a1:.2e} (criterion <1e-6)", flush=True)
    # -- A2: independent diagonalization (scipy.linalg.eigh, a different call path from the execution side's np.linalg.eigh) --
    lam_j, _ = scipy.linalg.eigh(F)
    eigs = json.load(open(os.path.join(OUT, "fisher_eigen.json"), encoding='utf-8'))
    lam_e = np.array(eigs["特征值_升序"], float)
    a2 = float(np.max(np.abs(lam_j - lam_e) / np.maximum(np.abs(lam_e), 1e-30)))
    cond_j = float(lam_j[-1] / lam_j[0]) if lam_j[0] > 0 else float("inf")
    a2c = abs(cond_j - eigs["条件数χ"]) / abs(eigs["条件数χ"])
    print(f"[judge182-A2] eigenvalue max relative diff={a2:.2e}, chi relative diff={a2c:.2e} (criterion <1e-6)", flush=True)
    # -- A3: independent recomputation of Delta_s (s in 1.01/1.1/1.5) --
    a3 = 0.0
    jobs = []
    for s in [1.01, 1.1, 1.5]:
        xs = x16.copy()
        ls = math.log10(s)
        xs[0:8] += ls; xs[9] += ls
        xs[8] -= ls; xs[11] -= ls; xs[12] -= ls
        for pr in PROTOS:
            jobs.append(((s, pr), list(xs), data2[pr][0]))
    with Pool(12) as p:
        res3 = dict(p.map(_job, jobs))
    for s in [1.01, 1.1, 1.5]:
        for pr in PROTOS:
            km = keep[pr]
            d = res3[(s, pr)][km] - base[pr][km]
            dj = float(np.sqrt(d @ d) / np.sqrt(base[pr][km] @ base[pr][km]))
            eD = json.load(open(os.path.join(OUT, f"scale_group_A_{pr}.json"), encoding='utf-8'))["读数"][str(s)]["Delta_s"]
            a3 = max(a3, abs(dj - eD) / max(abs(eD), 1e-30))
    print(f"[judge182-A3] Delta_s max relative diff={a3:.2e} (criterion <1e-4)", flush=True)
    # -- A4: independent check of steady-state ratio R_ss over nine cells (read judgeA r40/r60 from counters, compare with the execution side's report) --
    eDD = json.load(open(os.path.join(OUT, "degenerate_directions.json"), encoding='utf-8'))
    a4 = 0.0
    cells9 = ["16713003", "16715049", "16708016", "16708060", "16713110",
              "16708118", "16704007", "16704047", "16707014"]
    n9 = 0
    for c in cells9:
        fp = "counter_181_model62_判官复核件.json" if c == "16713003" else f"counter_A_model62_{c}.json"
        blk = json.load(open(os.path.join(HERE, fp), encoding='utf-8'))
        ja = blk.get("judgeA", blk)
        r40, r60 = ja["r40"], ja["r60"]
        ok = (0.05 <= r40 <= 0.30) and (r60 <= 0.8 * r40)
        n9 += ok
        er = eDD["九细胞检验_引自judgeA块"][c]
        a4 = max(a4, abs(r40 - er["r40"]) / max(abs(er["r40"]), 1e-30),
                 abs(r60 - er["r60"]) / max(abs(er["r60"]), 1e-30))
    print(f"[judge182-A4] R_ss relative diff={a4:.2e} (criterion <1e-3), nine-cell within limits={n9}/9 (execution side={eDD['九细胞过线数']})", flush=True)
    # -- A5: boundary-slot check --
    a5_ok = True
    esc_report = eDD.get("跨细胞贴界登记")
    if esc_report:
        for c in cells9[1:]:
            blk = json.load(open(os.path.join(HERE, f"counter_A_model62_{c}.json"), encoding='utf-8'))
            real = set(blk["judgeA"]["escape"])
            if set(esc_report.get(c, [])) != real:
                a5_ok = False
                print(f"[judge182-A5] {c} boundary registry inconsistent", flush=True)
    print(f"[judge182-A5] boundary registry check={'consistent' if a5_ok else 'inconsistent'}", flush=True)
    verdict = {"A0基线": "过", "A1_Fisher": a1, "A2_特征谱": max(a2, a2c), "A3_Δs": a3,
               "A4_Rss": {"相对差": a4, "过线": f"{n9}/9"}, "A5_贴界": a5_ok,
               "用时s": round(time.time() - t0, 1)}
    json.dump(verdict, open(os.path.join(HERE, "judge182_verdict.json"), "w", encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print("[judge182] recomputation complete, verdict artifact judge182_verdict.json", flush=True)


if __name__ == "__main__":
    main()
