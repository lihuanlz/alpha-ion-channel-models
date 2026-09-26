# de176.py — 代码176 model60 全合成卡 @16713003（预注册 2026-09-10）
# model60 = 官方 HH 4 态环（MexHH.c 逐字）+ X 慢件两旋钮；11 自由槽。
# 数值：numba CN 步进（精度验收过：对 expm 判段最大偏差 1.34e-4nA）；判官 expm 独立路径。
# 用法：python de176.py [anchor|fit|resume] [workers]   （Windows 用 python 不用 python3）
import json, math, os, sys, time
import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import de173
import de174
from de173 import (R, DT, find_step, halfrise, creep, tail_rise, steady_ratios,
                   tail_fit_kernel, evolve_x, GX_MID, GX_K, P6X, TAIL_WIN, KCT_T0NOM)
from de174 import load174, windows174, PROTOS

# ---- model60 冻结常数（预注册 §1）----
TEMP = 21.3
EREV = (8314.0 * (273.15 + TEMP) / 96485.0) * math.log(4.0 / 130.0)   # −88.33 mV
P_SHORT, P_PEAK = 2.4744, 779.0634     # 16713003 窗功率（174 注册）
W_TAIL, LAM, HOLD_WIDTH = 4.0, 1e4, 0.005
HRS_DATA = [1669.3, 1583.9, 485.1, 203.8]   # 数据 0.5ms boxcar（预注册 §5-L5）
F11 = [1.9800e-4, 0.0593, 7.1688e-5, 0.0493, 0.1048, 0.0139, 0.0038, 0.0360]
GKR0 = 0.1351
A0 = (0.5, 0.44)                        # α_amp, α_τ 起点（干净旋钮 sizing 在案）
M4_AMIN = 0.8 * 0.0579                  # 数据锚：0.04632
M4_TAU_WIN = (0.35, 1.05)               # 数据 τ=0.704 × (0.5, 1.5)
TAIL_SSE_MAX = 87.0
R2_BARS = {"steady_activation": 0.945, "sine_wave": 0.955, "ap": 0.760, "deactivation": 0.80}

NPOP76, DE_MAXITER76 = 132, 60
NM_MAXITER76, NM_MAXFEV76 = 1500, 1700
DE_SEED76 = 20260910
DECIM176 = {pr: 5 for pr in PROTOS}
WALL_BUDGET = 3600.0
WALL_STOP = WALL_BUDGET

# ---- 锚（预注册 §7：沙箱注册后回填；None=拒跑）----
J0_ANCHOR_176 = 88304.3176
SSE0_ANCHOR_176 = 39647.892837


@njit(cache=True)
def _hh_cn(V, dt_ms, P0, P1, P2, P3, P4, P5, P6, P7, vr, gkr):
    """官方 HH 环 CN 步进（对 expm 验收 1.34e-4nA；Cramer x3 项 c32 已修在案）。"""
    n = len(V); out = np.empty(n)
    y1, y2, y3 = 0., 0., 0.
    for i in range(n):
        v = V[i]
        k32 = P4 * np.exp(P5 * v);  k23 = P6 * np.exp(-P7 * v)
        k43 = P0 * np.exp(P1 * v);  k34 = P2 * np.exp(-P3 * v)
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


def sim60(V, x11):
    """x11 = [log10 P1..P8, log10 GKr, log10 α_amp, log10 α_τ] → 电流 nA。"""
    P = 10.0 ** np.asarray(x11[:8], float)
    gkr = 10.0 ** x11[8]
    amp, tsc = 10.0 ** x11[9], 10.0 ** x11[10]
    I = _hh_cn(np.asarray(V, float), DT * 1000.0, *P, EREV, gkr)
    xs = evolve_x(np.asarray(V, float), DT, P6X[0], P6X[1], P6X[2], P6X[3] * tsc, P6X[4])
    gx = 1.0 / (1.0 + np.exp((np.asarray(V, float) - GX_MID) / GX_K))
    return I + amp * P6X[5] * xs * gx * (np.asarray(V, float) - EREV)


X_START = np.array([math.log10(v) for v in F11] + [math.log10(GKR0),
                   math.log10(A0[0]), math.log10(A0[1])])
BOUNDS = ([(x - 0.7, x + 0.7) for x in X_START[:8]]
          + [(X_START[8] - 0.5, X_START[8] + 0.5), (-1.0, 1.0), (-1.3, 0.3)])


def resid60(x11, data, keep, decim):
    rd, raw = {}, None
    for proto, (v, c) in data.items():
        try:
            m = sim60(np.asarray(v, float), x11)
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


def sse60(x11, data, keep):
    out = resid60(x11, data, keep, None)
    if out is None:
        return 1e30
    rd, _ = out
    tot = 0.0
    for pr, r in rd.items():
        tot += float(r @ r)
        if not np.isfinite(tot):
            return 1e30
    return tot


def obj60_factory(wins):
    ISH, IPK, IHD, IWT = wins['ISH'], wins['IPK'], wins['IHD'], wins['IWT']
    HOLD_C, HOLD_T = wins['HOLD_C'], wins['HOLD_TARGET']

    def obj(x11, data, keep, lam, decim):
        out = resid60(x11, data, keep, decim)
        if out is None:
            return 1e12
        rd, raw = out
        sse_full = 0.0
        for pr, r in rd.items():
            sse_full += float(r @ r)
        if not np.isfinite(sse_full):
            return 1e12
        sse_tail = float(raw[IWT[0]:IWT[1]] @ raw[IWT[0]:IWT[1]])
        sse_short = float(raw[ISH[0]:ISH[1]] @ raw[ISH[0]:ISH[1]])
        sse_peak = float(raw[IPK[0]:IPK[1]] @ raw[IPK[0]:IPK[1]])
        hold_m = HOLD_C + float(np.mean(raw[IHD[0]:IHD[1]]))
        hold_pen = ((hold_m - HOLD_T) / HOLD_WIDTH) ** 2
        return sse_full + W_TAIL * sse_tail + lam * (sse_short / P_SHORT
                                                     + sse_peak / P_PEAK + hold_pen)
    return obj


# ---- L1-L8 判线块（de 落盘与 judge176 同式唯一事实源）----
def llines60(x11, data2, keep, wins):
    vd, cd = data2["deactivation"]
    vsa, csa = data2["steady_activation"]
    ISH, IPK, IHD, IWT = wins['ISH'], wins['IPK'], wins['IHD'], wins['IWT']
    n1 = int(9.1612 / DT)
    m1 = sim60(np.asarray(vd[:n1], float), x11)
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
    msa = sim60(np.asarray(vsa, float), x11)
    NOM = [17.15, 25.40, 33.66, 41.92]; SKIP = [0.6, 0.6, 0.05, 0.05]
    hrs = [halfrise(msa, find_step(np.asarray(vsa, float), t), s) for t, s in zip(NOM, SKIP)]
    crs = [creep(msa, find_step(np.asarray(vsa, float), t)) for t in [17.15, 25.40]]
    trise = tail_rise(msa, find_step(np.asarray(vsa, float), 41.92))
    r40, r60 = steady_ratios(msa)[:2]
    L5 = bool(np.all(np.isfinite(hrs)) and
              all(0.4 * d <= h <= 2.5 * d for h, d in zip(hrs, HRS_DATA)))
    # L7 R² 榜
    r2 = {}
    for pr in PROTOS:
        v_, c_ = data2[pr]
        m_ = sim60(np.asarray(v_, float), x11)
        km_ = keep[pr]
        pw_ = float(np.asarray(c_, float)[km_] @ np.asarray(c_, float)[km_])
        r2[pr] = 1 - float(((m_ - np.asarray(c_, float))[km_] ** 2).sum()) / pw_
    L7 = bool(all(r2[pr] >= R2_BARS[pr] for pr in PROTOS))
    return dict(pk=pk, sh=sh, hold=hold, sse_tail=sse_tail, a_fit=a_fit, tau_fit=tau_fit,
                L1=L1, L2=L2, L3=L3, L4=L4, L5=L5, L7=L7, r2=r2, msa=msa,
                hrs=hrs, crs=crs, trise=trise, r40=r40, r60=r60)


def fit176(mode, workers):
    from concurrent.futures import ThreadPoolExecutor
    from scipy.optimize import minimize, differential_evolution

    data2, keep = load174()
    vd, cd = data2["deactivation"]
    wins = windows174(vd, cd)
    obj = obj60_factory(wins)
    print(f"[176] model60 合成卡：窗 t0={wins['t0']:.4f} 数据峰={wins['data_pk']:.4f} "
          f"短步={wins['data_sh']:.4f}", flush=True)

    tt = time.time()
    J0 = obj(X_START, data2, keep, LAM, None)
    t_j0 = time.time() - tt
    sse0 = sse60(X_START, data2, keep)
    print(f"[176] 锚：sse0={sse0:.6f}（锚 {SSE0_ANCHOR_176}）J0={J0:.4f}（锚 {J0_ANCHOR_176}）"
          f" 单评估实测 {t_j0:.2f}s", flush=True)
    if mode == "anchor":
        rec = {"date": "2026-09-10", "card": "job_176_model60", "model": "model60",
               "cell": "16713003", "basis": "F11+X(0.5,0.44) 起点 @176 口径沙箱注册",
               "EREV": EREV, "SSE0_ANCHOR_176": sse0, "J0_ANCHOR_176": J0,
               "eval_seconds_sandbox": t_j0}
        json.dump(rec, open(os.path.join(HERE, "anchor176_注册锚_2026-09-10.json"), "w",
                            encoding='utf-8'), ensure_ascii=False, indent=1)
        print("[176] 锚注册落盘 anchor176_注册锚_2026-09-10.json——回填 de176.py 常量后方可 fit",
              flush=True)
        return
    if J0_ANCHOR_176 is None or SSE0_ANCHOR_176 is None:
        print("[176] 锚未注册（常量 None），按纪律拒跑——先 python de176.py anchor", flush=True)
        return
    if abs(sse0 - SSE0_ANCHOR_176) > 1e-6 * SSE0_ANCHOR_176 or abs(J0 - J0_ANCHOR_176) > 1e-3:
        print("[176] 锚未中，拒跑（内核/坐标一致性事故，原样记录）", flush=True)
        return

    rng = np.random.default_rng(DE_SEED76)
    lo = np.array([b[0] for b in BOUNDS]); hi = np.array([b[1] for b in BOUNDS])
    pop = np.empty((NPOP76, 11))
    if mode == "resume" and os.path.exists(os.path.join(HERE, "ckpt176_de.json")):
        ck = json.load(open(os.path.join(HERE, "ckpt176_de.json"), encoding='utf-8'))
        xc = np.asarray(ck["x_best"], float); done = int(ck["gen"])
        print(f"[176] 续跑：自 ckpt gen={done} best 重撒种群（RNG 不续，双录）", flush=True)
    else:
        xc, done = X_START, 0
    pop[0] = xc
    for j in range(1, NPOP76):
        pop[j] = np.clip(xc + rng.normal(0, 0.1, 11), lo, hi)

    tt0 = time.time()
    nfev_de = [0]
    tp = ThreadPoolExecutor(workers)

    def f_de(x):
        nfev_de[0] += 1
        return obj(x, data2, keep, LAM, DECIM176)

    state = {"gen": [done]}

    def cb_de(xk, convergence=0):
        state["gen"][0] += 1
        el = time.time() - tt0
        fk = float(obj(xk, data2, keep, LAM, DECIM176))
        json.dump({"gen": state["gen"][0], "x_best": list(map(float, xk)), "J_best_dec5": fk,
                   "wall_s": el},
                  open(os.path.join(HERE, "ckpt176_de.json"), "w", encoding='utf-8'))
        print(f"  [176] DE gen {state['gen'][0]}/{DE_MAXITER76} J(dec5)={fk:.1f} ({el:.0f}s)",
              flush=True)
        if el > WALL_BUDGET:
            print("[176] wall 超 1h 硬墙（双录）", flush=True)

    rde = differential_evolution(
        f_de, BOUNDS, maxiter=DE_MAXITER76 - done, tol=1e-6, strategy="rand1bin",
        mutation=0.7, recombination=0.9, seed=DE_SEED76, polish=False,
        init=pop, workers=tp.map, updating="deferred", callback=cb_de)
    tp.shutdown()
    de_rec = {"maxiter": DE_MAXITER76, "npop": NPOP76, "seed": DE_SEED76, "data": "dec5",
              "nfev": int(rde.nfev), "J": float(rde.fun), "message": str(rde.message),
              "resumed_from_gen": done}
    print(f"[176] DE 完：J={rde.fun:.1f} nfev={rde.nfev} ({time.time()-tt0:.0f}s)", flush=True)
    if rde.fun >= 1e11:
        print("[176] DE 全墙——早停条款判死记录，不进 NM", flush=True)
        dump176(np.asarray(rde.x, float), data2, keep, wins, nfev_de[0], de_rec, None,
                "de_allwall", tt0, stage="de_allwall")
        return

    nfev_nm = [0]
    best = {"f": None, "x": np.asarray(rde.x, float)}
    state2 = {"it": 0, "fchk": [None, 0]}

    class _EarlyStop(Exception):
        pass

    def f_nm(x):
        nfev_nm[0] += 1
        v = obj(x, data2, keep, LAM, None)
        if best["f"] is None or v < best["f"]:
            best["f"] = v; best["x"] = np.array(x).copy()
        if nfev_nm[0] % 50 == 0:
            json.dump({"nfev": nfev_nm[0], "J_best": best["f"],
                       "x_best": list(map(float, best["x"])), "wall_s": time.time() - tt0},
                      open(os.path.join(HERE, "ckpt176_nm.json"), "w", encoding='utf-8'))
        return v

    def cb_nm(xk):
        state2["it"] += 1
        if state2["it"] % 200 == 0:
            f_now = best["f"]
            if state2["fchk"][0] is not None and f_now is not None:
                gain = (state2["fchk"][0] - f_now) / max(abs(state2["fchk"][0]), 1e-12)
                state2["fchk"][1] = state2["fchk"][1] + 1 if gain < 1e-3 else 0
                if state2["fchk"][1] >= 2:
                    raise _EarlyStop
            if f_now is not None:
                state2["fchk"][0] = f_now
        if time.time() - tt0 > WALL_STOP:
            print("[176] wall 超 1h 硬墙，自动停并双录", flush=True)
            raise _EarlyStop

    msg, stopped = "", False
    try:
        rnm = minimize(f_nm, np.asarray(rde.x, float), method="Nelder-Mead", callback=cb_nm,
                       options={"maxiter": NM_MAXITER76, "maxfev": NM_MAXFEV76,
                                "xatol": 1e-7, "fatol": 1e-4})
        x_stage, msg = np.asarray(rnm.x, float), str(rnm.message)
    except _EarlyStop:
        x_stage, msg, stopped = best["x"], "early_stop(nm|wall|plateau)", True
    nm_rec = {"maxiter": NM_MAXITER76, "maxfev": NM_MAXFEV76, "nfev": nfev_nm[0],
              "J_best": best["f"], "early_stopped": stopped, "message": msg}
    print(f"[176] NM 完：bestJ={best['f']:.2f} nfev={nfev_nm[0]} ({time.time()-tt0:.0f}s) {msg}",
          flush=True)
    dump176(np.asarray(x_stage, float), data2, keep, wins, nfev_de[0] + nfev_nm[0],
            de_rec, nm_rec, msg, tt0, stage="final")


def dump176(x11, data2, keep, wins, nfev, de_rec, nm_rec, msg, tt0, stage="final"):
    quick = (stage != "final")
    ll = llines60(x11, data2, keep, wins)
    sse_full = sse60(x11, data2, keep)
    # L6 逃逸/flat + L8 机制连续
    esc = []
    for j, (lo_, hi_) in enumerate(BOUNDS):
        if x11[j] <= lo_ + 0.001 * (hi_ - lo_) or x11[j] >= hi_ - 0.001 * (hi_ - lo_):
            esc.append(f"槽{j} x={x11[j]:.4g} 贴/越界[{lo_},{hi_}]")
    drift = np.abs(x11 - X_START) / 1.0   # log10 槽：|Δdex|
    drift8 = float(np.max(np.abs(x11[:8] - X_START[:8])))
    if quick:
        improve, flat, L6 = None, None, None
        L8 = None; branch = stage
    else:
        sse0c = sse60(X_START, data2, keep)
        improve = (sse0c - sse_full) / sse0c if sse0c > 0 else np.nan
        flat = bool(np.max(np.abs(10.0 ** x11 - 10.0 ** X_START)
                           / np.maximum(np.abs(10.0 ** X_START), 1e-12)) > 0.5
                    and improve < 0.05)
        L6 = bool(len(esc) == 0 and not flat)
        L8 = bool(drift8 <= 0.5)
        Ls = [ll['L1'], ll['L2'], ll['L3'], ll['L4'], ll['L5'], L6, ll['L7'], L8]
        branch = "全过" if all(Ls) else "未全过"
    n_pass = None if quick else int(sum([ll['L1'], ll['L2'], ll['L3'], ll['L4'], ll['L5'],
                                         L6, ll['L7'], L8]))
    preds = None if quick else {
        "Q1_L7de由灭转过(de>=0.80)": [ll['r2']['deactivation'], 0.80,
                                    bool(ll['r2']['deactivation'] >= 0.80)],
        "Q2_L2短步由灭转过": [[float(ll['sh']), wins['data_sh']],
                           bool(0.5 <= ll['sh'] / wins['data_sh'] <= 1.5)],
        "Q3_L1L4保持": [bool(ll['L1']), bool(ll['L4'])],
        "Q4_L8机制连续保持": [drift8, bool(drift8 <= 0.5)],
        "Q5_慢爬读数(数据+55.4/+35.2)": ll['crs'],
        "Q6_11槽漂移榜(dex)": sorted([(f"x{j}", float(drift[j])) for j in range(11)],
                                    key=lambda t: -t[1]),
    }
    verdict = {
        "tag": "176", "model": "model60", "stage": stage, "cell": "16713003",
        "architecture": "官方HH4态环+X慢件两旋钮（11参）",
        "x11": list(map(float, x11)),
        "P1_P8": list(map(float, 10.0 ** x11[:8])), "GKr": float(10.0 ** x11[8]),
        "alpha_amp": float(10.0 ** x11[9]), "alpha_tau": float(10.0 ** x11[10]),
        "EREV": EREV,
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
        "predictions": preds,
        "de": de_rec, "nm": nm_rec, "nfev": nfev, "message": msg,
        "wall_s": time.time() - tt0, "branch": branch, "n_pass": n_pass,
    }
    out = os.path.join(HERE, "counter_176_model60.json")
    json.dump(verdict, open(out, "w", encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[176] 落盘 {out}（stage={stage}）", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    assert mode in ("anchor", "fit", "resume"), "用法：de176.py [anchor|fit|resume] [workers]"
    fit176(mode, workers)
