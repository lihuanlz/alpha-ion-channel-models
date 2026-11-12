# de178.py — 代码178 model61 机制补全卡 @16713003（预注册 2026-09-10）
# model61 = 官方环(k43 双指数化 +Pf1·e^{Pf2·v}) + X 件(GX_MID 解冻) —— 14 参。
# 用法：python de178.py [anchor|fit|resume] [workers]   （Windows 用 python 不用 python3）
import json, math, os, sys, time
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import de173
import de174
import de176
import de177
from de173 import DT, find_step, tail_fit_kernel, TAIL_WIN, KCT_T0NOM
from de174 import load174, windows174, PROTOS
from de176 import EREV, HRS_DATA, M4_AMIN, M4_TAU_WIN, TAIL_SSE_MAX, R2_BARS
from de177 import proto_powers

# ---- model61 冻结常数与起点（预注册 §1/§2）----
from de176 import F11, GKR0
PF0 = (1.22e-4, 0.08)        # Pf1, Pf2（探针证据带）
A0 = (0.5, 0.44)             # α_amp, α_τ
GXMID0 = -60.0
LAM_A, LAM_T = 1.0, 1.0
A_DATA, TAU_DATA = 0.0579, 0.704
NPOP78, DE_MAXITER78 = 140, 60
NM_MAXFEV78 = 2500
DE_SEED78 = 20260910
DECIM178 = {pr: 5 for pr in PROTOS}
WALL_BUDGET = 3600.0
WALL_STOP = WALL_BUDGET

# ---- 锚（预注册 §7：沙箱注册后回填；None=拒跑）----
J0_ANCHOR_178 = 2.022490
SSE0_ANCHOR_178 = 43340.177192


@njit(cache=True)
def _hh61_cn(V, dt_ms, P0, P1, P2, P3, P4, P5, P6, P7, pf1, pf2, vr, gkr):
    """model61 核：k43/k12 双指数（+pf1·e^{pf2·v}）；CN 步进（验收路径沿用 de176 同式）。"""
    n = len(V); out = np.empty(n)
    y1, y2, y3 = 0., 0., 0.
    for i in range(n):
        v = V[i]
        k32 = P4 * np.exp(P5 * v);  k23 = P6 * np.exp(-P7 * v)
        k43 = P0 * np.exp(P1 * v) + pf1 * np.exp(pf2 * v);  k34 = P2 * np.exp(-P3 * v)
        k12, k21, k41, k14 = k43, k34, k32, k23
        A11 = -(k12 + k14) - k41; A12 = k21 - k41; A13 = -k41;      b1 = k41
        A21 = k12;            A22 = -(k21 + k23); A23 = k32;        b2 = 0.0
        A31 = -k43;           A32 = k23 - k43;  A33 = -(k32 + k34) - k43; b3 = k43
        h = dt_ms / 2.0
        c11 = 1 - h * A11; c12 = -h * A12;  c13 = -h * A13
        c21 = -h * A21;  c22 = 1 - h * A22; c23 = -h * A23
        c31 = -h * A31;  c32 = -h * A32;  c33 = 1 - h * A33
        r1 = (1 + h * A11) * y1 + h * A12 * y2 + h * A13 * y3 + 2 * h * b1
        r2 = h * A21 * y1 + (1 + h * A22) * y2 + h * A23 * y3 + 2 * h * b2
        r3 = h * A31 * y1 + h * A32 * y2 + (1 + h * A33) * y3 + 2 * h * b3
        det = c11 * (c22 * c33 - c23 * c32) - c12 * (c21 * c33 - c23 * c31) \
            + c13 * (c21 * c32 - c22 * c31)
        x1 = (r1 * (c22 * c33 - c23 * c32) - c12 * (r2 * c33 - c23 * r3)
              + c13 * (r2 * c32 - c22 * r3)) / det
        x2 = (c11 * (r2 * c33 - c23 * r3) - r1 * (c21 * c33 - c23 * c31)
              + c13 * (c21 * r3 - r2 * c31)) / det
        x3 = (c11 * (c22 * r3 - c32 * r2) - c12 * (c21 * r3 - r2 * c31)
              + r1 * (c21 * c32 - c22 * c31)) / det
        y1, y2, y3 = x1, x2, x3
        out[i] = gkr * y3 * (v - vr)
    return out


def sim61(V, x14):
    """x14 = [log10 P1..P8, log10 GKr, log10 Pf1, Pf2, log10 α_amp, log10 α_τ, GX_MID]。"""
    P = 10.0 ** np.asarray(x14[:8], float)
    gkr = 10.0 ** x14[8]
    pf1 = 10.0 ** x14[9]
    pf2 = x14[10]
    amp, tsc = 10.0 ** x14[11], 10.0 ** x14[12]
    gmid = x14[13]
    I = _hh61_cn(np.asarray(V, float), DT * 1000.0, *P, pf1, pf2, EREV, gkr)
    Vv = np.asarray(V, float)
    xs = de173.evolve_x(Vv, DT, de173.P6X[0], de173.P6X[1], de173.P6X[2],
                        de173.P6X[3] * tsc, de173.P6X[4])
    gx = 1.0 / (1.0 + np.exp((Vv - gmid) / de173.GX_K))
    return I + amp * de173.P6X[5] * xs * gx * (Vv - EREV)


X_START61 = np.array([math.log10(v) for v in F11] + [math.log10(GKR0),
                     math.log10(PF0[0]), PF0[1], math.log10(A0[0]), math.log10(A0[1]),
                     GXMID0])
BOUNDS78 = ([(x - 0.5, x + 0.5) for x in X_START61[:8]]
            + [(X_START61[8] - 0.5, X_START61[8] + 0.5),
               (X_START61[9] - 0.7, X_START61[9] + 0.7), (0.05, 0.13),
               (X_START61[11] - 0.4, X_START61[11] + 0.7),
               (X_START61[12] - 0.65, X_START61[12] + 0.65),
               (-90.0, -45.0)])


def resid61(x14, data, keep, decim):
    rd, raw = {}, None
    for proto, (v, c) in data.items():
        try:
            m = sim61(np.asarray(v, float), x14)
        except (ZeroDivisionError, OverflowError, FloatingPointError):
            return None
        if not np.all(np.isfinite(m)):
            return None
        r = m - np.asarray(c, float)
        if keep is not None and proto in keep:
            r = r[keep[proto]]
        r = r[np.isfinite(r)]
        if proto == 'deactivation':
            raw = r
        if decim is not None and proto in decim:
            ds = int(decim[proto])
            r = r[::ds] * np.sqrt(ds)
        rd[proto] = r
    return rd, raw


def sse61(x14, data, keep):
    out = resid61(x14, data, keep, None)
    if out is None:
        return 1e30
    rd, _ = out
    tot = 0.0
    for pr, r in rd.items():
        tot += float(r @ r)
        if not np.isfinite(tot):
            return 1e30
    return tot


def obj78_factory(wins, PPR):
    IWT = wins['IWT']

    def obj(x14, data, keep, decim):
        out = resid61(x14, data, keep, decim)
        if out is None:
            return 1e12
        rd, raw = out
        tot = 0.0
        for pr, r in rd.items():
            tot += float(r @ r) / PPR[pr]
        if not np.isfinite(tot):
            return 1e12
        vd_, _ = data['deactivation']
        n1 = int(9.1612 / DT)
        m1 = sim61(np.asarray(vd_[:n1], float), x14)
        if not np.all(np.isfinite(m1)):
            return 1e12
        a_, t_ = tail_fit_kernel(m1, IWT[0], IWT[1], DT)
        if a_ is None:
            return 1e12
        return tot + LAM_A * ((abs(a_) - A_DATA) / A_DATA) ** 2 \
                  + LAM_T * ((t_ - TAU_DATA) / TAU_DATA) ** 2
    return obj


# ---- L1-L8 判线块（de 落盘与 judge178 同式唯一事实源）----
def llines61(x14, data2, keep, wins):
    vd, cd = data2["deactivation"]
    vsa, csa = data2["steady_activation"]
    ISH, IPK, IHD, IWT = wins['ISH'], wins['IPK'], wins['IHD'], wins['IWT']
    n1 = int(9.1612 / DT)
    m1 = sim61(np.asarray(vd[:n1], float), x14)
    c1 = np.asarray(cd[:n1], float)
    finite1 = bool(np.all(np.isfinite(m1)))
    pk = float(np.min(m1[IPK[0]:IPK[1]])) if finite1 else np.nan
    sh = float(np.min(m1[ISH[0]:ISH[1]])) if finite1 else np.nan
    hold = float(np.mean(m1[IHD[0]:IHD[1]])) if finite1 else np.nan
    seg = m1[IWT[0]:IWT[1]] if finite1 else np.full(IWT[1] - IWT[0], np.nan)
    sse_tail = float(np.sum((seg - c1[IWT[0]:IWT[1]]) ** 2)) if finite1 else np.inf
    a_fit, tau_fit = tail_fit_kernel(m1, IWT[0], IWT[1], DT) if finite1 else (None, None)
    L1 = bool(finite1 and abs(pk - wins['data_pk']) / abs(wins['data_pk']) <= 0.10)
    L2 = bool(finite1 and 0.5 <= sh / wins['data_sh'] <= 1.5)
    L3 = bool(finite1 and -0.015 <= hold <= 0.005)
    L4 = bool(finite1 and sse_tail <= TAIL_SSE_MAX and tau_fit is not None
              and M4_TAU_WIN[0] <= tau_fit <= M4_TAU_WIN[1] and abs(a_fit) >= M4_AMIN)
    msa = sim61(np.asarray(vsa, float), x14)
    NOM = [17.15, 25.40, 33.66, 41.92]; SKIP = [0.6, 0.6, 0.05, 0.05]
    hrs = [de173.halfrise(msa, find_step(np.asarray(vsa, float), t), s) for t, s in zip(NOM, SKIP)]
    crs = [de173.creep(msa, find_step(np.asarray(vsa, float), t)) for t in [17.15, 25.40]]
    trise = de173.tail_rise(msa, find_step(np.asarray(vsa, float), 41.92))
    r40, r60 = de173.steady_ratios(msa)[:2]
    L5 = bool(np.all(np.isfinite(hrs)) and
              all(0.4 * d <= h <= 2.5 * d for h, d in zip(hrs, HRS_DATA)))
    r2 = {}
    for pr in PROTOS:
        v_, c_ = data2[pr]
        m_ = sim61(np.asarray(v_, float), x14)
        km_ = keep[pr]
        pw_ = float(np.asarray(c_, float)[km_] @ np.asarray(c_, float)[km_])
        r2[pr] = 1 - float(((m_ - np.asarray(c_, float))[km_] ** 2).sum()) / pw_
    L7 = bool(all(r2[pr] >= R2_BARS[pr] for pr in PROTOS))
    return dict(pk=pk, sh=sh, hold=hold, sse_tail=sse_tail, a_fit=a_fit, tau_fit=tau_fit,
                L1=L1, L2=L2, L3=L3, L4=L4, L5=L5, L7=L7, r2=r2, msa=msa,
                hrs=hrs, crs=crs, trise=trise, r40=r40, r60=r60)


def fit178(mode, workers):
    from concurrent.futures import ThreadPoolExecutor
    from scipy.optimize import minimize, differential_evolution

    data2, keep = load174()
    vd, cd = data2["deactivation"]
    wins = windows174(vd, cd)
    PPR = proto_powers(data2, keep)
    obj = obj78_factory(wins, PPR)
    print(f"[178] model61 机制补全卡启动", flush=True)

    tt = time.time()
    J0 = obj(X_START61, data2, keep, None)
    sse0 = sse61(X_START61, data2, keep)
    print(f"[178] 锚：sse0={sse0:.6f}（锚 {SSE0_ANCHOR_178}）J0={J0:.6f}（锚 {J0_ANCHOR_178}）"
          f" 单评估实测 {time.time()-tt:.2f}s", flush=True)
    if mode == "anchor":
        rec = {"date": "2026-09-10", "card": "job_178_model61", "model": "model61",
               "cell": "16713003", "basis": "F11+Pf+X(0.5,0.44)+GXMID −60 起点 @178 口径沙箱注册",
               "EREV": EREV, "SSE0_ANCHOR_178": sse0, "J0_ANCHOR_178": J0}
        json.dump(rec, open(os.path.join(HERE, "anchor178_注册锚_2026-09-10.json"), "w",
                            encoding='utf-8'), ensure_ascii=False, indent=1)
        print("[178] 锚注册落盘 anchor178_注册锚_2026-09-10.json——回填常量后方可 fit", flush=True)
        return
    if J0_ANCHOR_178 is None or SSE0_ANCHOR_178 is None:
        print("[178] 锚未注册，拒跑——先 python de178.py anchor", flush=True)
        return
    if abs(sse0 - SSE0_ANCHOR_178) > 1e-6 * SSE0_ANCHOR_178 or abs(J0 - J0_ANCHOR_178) > 1e-6 * abs(J0_ANCHOR_178):
        print("[178] 锚未中，拒跑", flush=True)
        return

    rng = np.random.default_rng(DE_SEED78)
    lo = np.array([b[0] for b in BOUNDS78]); hi = np.array([b[1] for b in BOUNDS78])
    pop = np.empty((NPOP78, 14))
    if mode == "resume" and os.path.exists(os.path.join(HERE, "ckpt178_de.json")):
        ck = json.load(open(os.path.join(HERE, "ckpt178_de.json"), encoding='utf-8'))
        xc = np.asarray(ck["x_best"], float); done = int(ck["gen"])
        print(f"[178] 续跑：自 ckpt gen={done}（RNG 不续，双录）", flush=True)
    else:
        xc, done = X_START61, 0
    pop[0] = xc
    for j in range(1, NPOP78):
        x = xc + rng.normal(0, 0.1, 14)
        x[10] = xc[10] + rng.normal(0, 0.005)   # Pf2 线性槽小抖动
        x[13] = xc[13] + rng.normal(0, 2.0)     # GX_MID 线性槽 ±2mV 抖动
        pop[j] = np.clip(x, lo, hi)

    tt0 = time.time()
    nfev_de = [0]
    tp = ThreadPoolExecutor(workers)

    def f_de(x):
        nfev_de[0] += 1
        return obj(x, data2, keep, DECIM178)

    state = {"gen": [done]}

    def cb_de(xk, convergence=0):
        state["gen"][0] += 1
        el = time.time() - tt0
        fk = float(obj(xk, data2, keep, DECIM178))
        json.dump({"gen": state["gen"][0], "x_best": list(map(float, xk)), "J_best_dec5": fk,
                   "wall_s": el},
                  open(os.path.join(HERE, "ckpt178_de.json"), "w", encoding='utf-8'))
        print(f"  [178] DE gen {state['gen'][0]}/{DE_MAXITER78} J(dec5)={fk:.4f} ({el:.0f}s)",
              flush=True)
        if el > WALL_BUDGET:
            print("[178] wall 超 1h 硬墙（双录）", flush=True)

    rde = differential_evolution(
        f_de, BOUNDS78, maxiter=DE_MAXITER78 - done, tol=1e-6, strategy="rand1bin",
        mutation=0.7, recombination=0.9, seed=DE_SEED78, polish=False,
        init=pop, workers=tp.map, updating="deferred", callback=cb_de)
    tp.shutdown()
    de_rec = {"maxiter": DE_MAXITER78, "npop": NPOP78, "seed": DE_SEED78, "data": "dec5",
              "nfev": int(rde.nfev), "J": float(rde.fun), "message": str(rde.message),
              "resumed_from_gen": done}
    print(f"[178] DE 完：J={rde.fun:.4f} nfev={rde.nfev} ({time.time()-tt0:.0f}s)", flush=True)
    if rde.fun >= 1e11:
        print("[178] DE 全墙——早停条款判死记录", flush=True)
        dump178(np.asarray(rde.x, float), data2, keep, wins, nfev_de[0], de_rec, None,
                "de_allwall", tt0, stage="de_allwall")
        return

    nfev_nm = [0]
    best = {"f": None, "x": np.asarray(rde.x, float)}

    def f_pw(x):
        nfev_nm[0] += 1
        v = obj(x, data2, keep, None)
        if best["f"] is None or v < best["f"]:
            best["f"] = v; best["x"] = np.array(x).copy()
        if nfev_nm[0] % 100 == 0:
            json.dump({"nfev": nfev_nm[0], "J_best": best["f"],
                       "x_best": list(map(float, best["x"])), "wall_s": time.time() - tt0},
                      open(os.path.join(HERE, "ckpt178_nm.json"), "w", encoding='utf-8'))
        return v

    try:
        rpw = minimize(f_pw, np.asarray(rde.x, float), method="Powell", bounds=BOUNDS78,
                       options={"maxfev": NM_MAXFEV78, "xtol": 1e-7, "ftol": 1e-7})
        x_stage, msg = np.asarray(rpw.x, float), str(rpw.message)
    except Exception as e:
        x_stage, msg = best["x"], f"polish_exception:{e}"
    nm_rec = {"method": "Powell_bounded", "maxfev": NM_MAXFEV78, "nfev": nfev_nm[0],
              "J_best": best["f"], "message": msg}
    print(f"[178] polish 完：bestJ={best['f']:.4f} nfev={nfev_nm[0]} ({time.time()-tt0:.0f}s) {msg}",
          flush=True)
    dump178(np.asarray(x_stage, float), data2, keep, wins, nfev_de[0] + nfev_nm[0],
            de_rec, nm_rec, msg, tt0, stage="final")


def dump178(x14, data2, keep, wins, nfev, de_rec, nm_rec, msg, tt0, stage="final"):
    quick = (stage != "final")
    ll = llines61(x14, data2, keep, wins)
    sse_full = sse61(x14, data2, keep)
    esc = []
    for j, (lo_, hi_) in enumerate(BOUNDS78):
        if x14[j] <= lo_ + 0.001 * (hi_ - lo_) or x14[j] >= hi_ - 0.001 * (hi_ - lo_):
            esc.append(f"槽{j} x={x14[j]:.4g} 贴/越界[{lo_},{hi_}]")
    drift8 = float(np.max(np.abs(x14[:8] - X_START61[:8])))
    if quick:
        improve, flat, L6 = None, None, None
        L8 = None; branch = stage
    else:
        sse0c = sse61(X_START61, data2, keep)
        improve = (sse0c - sse_full) / sse0c if sse0c > 0 else np.nan
        rel = np.abs(10.0 ** x14[[0,1,2,3,4,5,6,7,8,9,11,12]] - 10.0 ** X_START61[[0,1,2,3,4,5,6,7,8,9,11,12]]) \
              / np.maximum(np.abs(10.0 ** X_START61[[0,1,2,3,4,5,6,7,8,9,11,12]]), 1e-12)
        flat = bool(rel.max() > 0.5 and improve < 0.05)
        L6 = bool(len(esc) == 0 and not flat)
        L8 = bool(drift8 <= 0.5)
        Ls = [ll['L1'], ll['L2'], ll['L3'], ll['L4'], ll['L5'], L6, ll['L7'], L8]
        branch = "全过" if all(Ls) else "未全过"
    n_pass = None if quick else int(sum([ll['L1'], ll['L2'], ll['L3'], ll['L4'], ll['L5'],
                                         L6, ll['L7'], L8]))
    verdict = {
        "tag": "178", "model": "model61", "stage": stage, "cell": "16713003",
        "architecture": "官方HH4态环(k43双指数)+X慢件(α,GX_MID)（14参）",
        "x14": list(map(float, x14)), "bounds": [list(b) for b in BOUNDS78],
        "P1_P8": list(map(float, 10.0 ** x14[:8])), "GKr": float(10.0 ** x14[8]),
        "Pf1": float(10.0 ** x14[9]), "Pf2": float(x14[10]),
        "alpha_amp": float(10.0 ** x14[11]), "alpha_tau": float(10.0 ** x14[12]),
        "GX_MID": float(x14[13]), "EREV": EREV,
        "sse_full": sse_full, "sse_tail": ll["sse_tail"],
        "J_obj_final": None if nm_rec is None else nm_rec.get("J_best"),
        "r2": ll["r2"],
        "L1_peak_nA": ll["pk"], "L2_step_nA": ll["sh"], "L3_hold_nA": ll["hold"],
        "L4_tau_s": ll["tau_fit"], "L4_a_nA": ll["a_fit"],
        "data_pk": wins["data_pk"], "data_sh": wins["data_sh"],
        "L1": ll["L1"], "L2": ll["L2"], "L3": ll["L3"], "L4": ll["L4"],
        "L5": ll["L5"], "L6": L6, "L7": ll["L7"], "L8": L8,
        "hrs": ll["hrs"], "hrs_data_smooth": HRS_DATA, "creep": ll["crs"],
        "trise": ll["trise"], "r40": ll["r40"], "r60": ll["r60"],
        "escape": esc, "flat": flat, "drift8_dex_max": drift8,
        "improve_vs_start": improve,
        "de": de_rec, "nm": nm_rec, "nfev": nfev, "message": msg,
        "wall_s": time.time() - tt0, "branch": branch, "n_pass": n_pass,
    }
    out = os.path.join(HERE, "counter_178_model61.json")
    json.dump(verdict, open(out, "w", encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[178] 落盘 {out}（stage={stage}）", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    workers = int(sys.argv[2] if len(sys.argv) > 2 else 3
    )
    assert mode in ("anchor", "fit", "resume"), "用法：de178.py [anchor|fit|resume] [workers]"
    fit178(mode, workers)
