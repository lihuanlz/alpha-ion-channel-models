# de174.py — 代码174 model53 减维 7 槽重拟合卡 @16713003（预注册 2026-09-10 分支 B）
# 自由槽 p7/p8/p9/p10/p11/p19/p21（30 维坐标槽位 J7=[7,8,9,10,11,14,16]）；其余继承 173 终点。
# 口径：de173 obj53 同式（sse_full+4·sse_tail+1e4·(sse_short+sse_peak+hold_pen+u+gate)），
#   16713003 数据（无 SHIFT 无伪影窗）、g_L=1.333e-4 斜率锚、HOLD_TARGET 数据派生 −0.000797nA。
# 用法：
#   python3 de174.py anchor          # 锚注册模式：算 J0/sse0 落 anchor174 json 后退出
#   python3 de174.py fit [workers]   # 拟合模式（锚未回填拒跑）；workers 默认 3
#   python3 de174.py resume [workers] # 断点续跑（ckpt174_de.json 末代 best 重撒种群，RNG 不续双录）
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import de173
from de173 import (sim53_trace, u_dyn, evolve_x, unpack53, x_from_params53, de_bounds53,
                   walls53, FREE53, LG53, LG53_EXTRA, R, DT, EK_FROZEN, P6X, GX_MID, GX_K,
                   find_step, halfrise, creep, tail_rise, steady_ratios, escape_check32,
                   tail_fit_kernel, gate165, gatepar47, RA, gv_midpoint53, u_anchor_pen,
                   W_TAIL, HOLD_WIDTH, M3_WIN, TAIL_WIN, TAIL_SSE_MAX, KCT_TAIL_WIN,
                   KCT_AMIN, KCT_T0NOM)

MODEL_NAME = "model53"
GL174 = 1.3330e-4                     # 16713003 斜率口径锚（transfer174 注册）
LAM = 1e4                             # λ 冻结（de173 同式）
NPOP74 = 70                           # 10 × 7 维（预注册 §4）
DE_MAXITER74 = 50
NM_MAXITER74 = 800
NM_MAXFEV74 = 1000
DE_SEED74 = 20260910
FREE7_PHYS = [7, 8, 9, 10, 11, 19, 21]
J7 = [7, 8, 9, 10, 11, 14, 16]        # 30 维坐标槽位（预注册 §1）
PROTOS = ["steady_activation", "deactivation", "sine_wave", "ap"]
DECIM174 = {pr: 5 for pr in PROTOS}   # DE 段 dec5（de173 同式）
WALL_BUDGET = 2.0 * 3600.0            # 硬墙 2h（预注册 §5）
WALL_STOP = WALL_BUDGET

# M5 自建基线（预注册 §6：transfer174 落盘读数，判官同式同阈值）
KC2_BASE174 = [True, True, False, True]   # hrs1 1482.1✓ / hrs0 2048.4✓ / trise 37.5✗ / crs✓
KS_BASE174 = [True, True]
KG_BASE174 = False
AP_BASE174 = True
HRS1_M5_MAX_174 = 1482.1 * 1.05           # =1556.205，替代 007 派生 2034.7

# ---- 锚（预注册 §8：anchor174_注册锚_2026-09-10.json 沙箱注册 2026-09-10 填入；None=拒跑）----
J0_ANCHOR_174 = 334098.8218
SSE0_ANCHOR_174 = 29142.854436957103      # transfer174 落盘


# ---- 内核：simT（transfer174 逐字，g_L 显式化）----
def simT(p, V, dt):
    tr = sim53_trace(p, V, dt, EK_FROZEN)
    m = tr[:, 0]
    if not np.all(np.isfinite(m)):
        return m
    u = u_dyn(p, V, dt)
    xs = evolve_x(V, dt, P6X[0], P6X[1], P6X[2], P6X[3], P6X[4])
    gx = 1.0 / (1.0 + np.exp((V - GX_MID) / GX_K))
    return m * u + p[7] * P6X[5] * xs * gx * (V - EK_FROZEN) + GL174 * (V - EK_FROZEN)


def load174():
    """16713003 四协议 keep 口径：无 SHIFT、无伪影窗（transfer174 load_keep_cell 同式）。"""
    data_raw = R.load_cell("16713003", PROTOS)
    data = {}
    for k, (v, c) in data_raw.items():
        vs = de173.shift_protocol_1pt(np.asarray(v, float))
        n = min(len(vs), len(c))
        data[k] = (vs[:n], np.asarray(c, float)[:n])
    keep = {}
    for k, (v, c) in data.items():
        if k == "ap":
            km = np.ones(len(v), dtype=bool)
            dv = np.abs(np.diff(v))
            jumps = np.nonzero(dv > 10.0)[0]
            for j in jumps:
                if j * DT < 0.5:
                    km[j + 1:j + 1 + 50] = False
            keep[k] = km
        else:
            keep[k] = R.make_keep_mask(v, 50)
    return data, keep


def start_params174():
    p = np.asarray(json.load(open(os.path.join(HERE, "counter_173_model53.json"),
                                  encoding='utf-8'))["params"], float)
    tj = json.load(open(os.path.join(HERE, "transfer174_16713003_第一判_2026-09-09.json"),
                        encoding='utf-8'))
    p[7] = tj["g_star"]
    return p


X30_START = None  # fit 时初始化


def x30_of(x7):
    x = X30_START.copy()
    x[J7] = np.asarray(x7, float)
    return x


def unpack74(x7):
    return unpack53(x30_of(x7))[0]


def walls74(x7):
    return walls53(x30_of(x7))


# ---- 残差与目标（resid53_pair/obj53 同式，simT 换 sim53）----
def _sse_proto174(p, v, c):
    try:
        m = simT(p, v, DT)
    except (ZeroDivisionError, OverflowError, FloatingPointError):
        return None
    if not np.all(np.isfinite(m)):
        return None
    return m - c


def resid74(p, data, keep, decim):
    rd, raw = {}, None
    for proto, (v, c) in data.items():
        r = _sse_proto174(p, np.asarray(v, float), np.asarray(c, float))
        if r is None:
            return None
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


def sse174(p, data, keep=None, serial=True):
    out = resid74(p, data, keep, None)
    if out is None:
        return 1e30
    rd, _ = out
    tot = 0.0
    for pr, r in rd.items():
        tot += float(r @ r)
        if not np.isfinite(tot):
            return 1e30
    return tot


def obj74_factory(wins):
    ISH, IPK, IHD, IWT = wins['ISH'], wins['IPK'], wins['IHD'], wins['IWT']
    HOLD_C, HOLD_T = wins['HOLD_C'], wins['HOLD_TARGET']

    def obj(x7, data, keep, lam, decim):
        if not walls74(x7):
            return 1e12
        p = unpack74(x7)
        out = resid74(p, data, keep, decim)
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
        gp = gatepar47.gate_pen_par47(R, p[:32], None)
        up = u_anchor_pen(p[:32])
        return sse_full + W_TAIL * sse_tail + lam * (sse_short + sse_peak + hold_pen + up + gp)
    return obj


def windows174(vd, cd):
    t0 = find_step(np.asarray(vd, float), KCT_T0NOM)
    ISH = (int(0.415 / DT), int(0.460 / DT))
    IPK = (int(round((t0 + 0.002) / DT)), int(round((t0 + 0.080) / DT)))
    IHD = (int(0.48 / DT), int(0.60 / DT))
    IWT = (int(round((t0 + TAIL_WIN[0]) / DT)), int(round((t0 + TAIL_WIN[1]) / DT)))
    HOLD_C = float(np.mean(np.asarray(cd, float)[IHD[0]:IHD[1]]))
    return dict(t0=t0, ISH=ISH, IPK=IPK, IHD=IHD, IWT=IWT,
                HOLD_C=HOLD_C, HOLD_TARGET=HOLD_C,
                data_pk=float(np.min(np.asarray(cd, float)[IPK[0]:IPK[1]])),
                data_sh=float(np.min(np.asarray(cd, float)[ISH[0]:ISH[1]])))


# ---- M1-M6 判线块（预注册 §6；de 落盘与 judge174 复算同式唯一事实源）----
def mlines74(p, data2, keep, wins):
    vd, cd = data2["deactivation"]
    vsa, csa = data2["steady_activation"]
    ISH, IPK, IHD, IWT = wins['ISH'], wins['IPK'], wins['IHD'], wins['IWT']
    n1 = int(9.1612 / DT)
    m1 = simT(p, np.asarray(vd[:n1], float), DT)
    c1 = np.asarray(cd[:n1], float)
    finite1 = bool(np.all(np.isfinite(m1)))
    pk = float(np.min(m1[IPK[0]:IPK[1]])) if finite1 else np.nan
    sh = float(np.min(m1[ISH[0]:ISH[1]])) if finite1 else np.nan
    hold = float(np.mean(m1[IHD[0]:IHD[1]])) if finite1 else np.nan
    seg = m1[IWT[0]:IWT[1]] if finite1 else np.full(IWT[1] - IWT[0], np.nan)
    sse_tail = float(np.sum((seg - c1[IWT[0]:IWT[1]]) ** 2)) if finite1 else np.inf
    a_fit, tau_fit = tail_fit_kernel(m1, IWT[0], IWT[1], DT) if finite1 else (None, None)
    M1 = bool(finite1 and abs(pk - wins['data_pk']) / abs(wins['data_pk']) <= 0.10)
    M2 = bool(finite1 and 0.5 <= sh / wins['data_sh'] <= 1.5)
    M3 = bool(finite1 and M3_WIN[0] <= hold <= M3_WIN[1])
    M4 = bool(finite1 and sse_tail <= TAIL_SSE_MAX and tau_fit is not None
              and KCT_TAIL_WIN[0] <= tau_fit <= KCT_TAIL_WIN[1] and abs(a_fit) >= KCT_AMIN)
    msa = simT(p, np.asarray(vsa, float), DT)
    NOM = [17.15, 25.40, 33.66, 41.92]; SKIP = [0.6, 0.6, 0.05, 0.05]
    hrs = [halfrise(msa, find_step(np.asarray(vsa, float), t), s) for t, s in zip(NOM, SKIP)]
    crs = [creep(msa, find_step(np.asarray(vsa, float), t)) for t in [17.15, 25.40]]
    trise = tail_rise(msa, find_step(np.asarray(vsa, float), 41.92))
    r40, r60 = steady_ratios(msa)[:2]
    kc2_now = [bool(hrs[1] <= 1653.6), bool(hrs[0] <= 2499.9),
               bool(90.0 <= trise <= 275.0), bool(crs[0] > 0 and crs[1] > 0)]
    ks_now = [bool(0.05 <= r40 <= 0.30), bool(r60 <= 0.8 * r40)]
    return dict(pk=pk, sh=sh, hold=hold, sse_tail=sse_tail, a_fit=a_fit, tau_fit=tau_fit,
                M1=M1, M2=M2, M3=M3, M4=M4, msa=msa, hrs=hrs, crs=crs, trise=trise,
                r40=r40, r60=r60, kc2_now=kc2_now, ks_now=ks_now)


def fit174(mode, workers):
    global X30_START
    from concurrent.futures import ThreadPoolExecutor
    from scipy.optimize import minimize, differential_evolution

    data2, keep = load174()
    vd, cd = data2["deactivation"]
    wins = windows174(vd, cd)
    p_start = start_params174()
    X30_START = x_from_params53(p_start)
    x7_start = X30_START[J7].copy()
    obj = obj74_factory(wins)
    print(f"[174] 窗：t0={wins['t0']:.4f} IPK={wins['IPK']} ISH={wins['ISH']} IHD={wins['IHD']} "
          f"IWT={wins['IWT']} HOLD_C={wins['HOLD_C']:.6f} 数据峰={wins['data_pk']:.4f} "
          f"短步={wins['data_sh']:.4f}", flush=True)

    # —— 锚注册/校验（裸调用，不计 nfev，兼实测报价计时）——
    tt = time.time()
    J0 = obj(x7_start, data2, keep, LAM, None)
    t_j0 = time.time() - tt
    sse0 = sse174(p_start, data2, keep=keep)
    print(f"[174] 锚：sse0={sse0:.6f}（锚 {SSE0_ANCHOR_174}）J0={J0:.4f}（锚 {J0_ANCHOR_174}）"
          f" 单评估实测 {t_j0:.1f}s", flush=True)
    if mode == "anchor":
        rec = {"date": "2026-09-10", "card": "job_174_model53", "model": MODEL_NAME,
               "cell": "16713003", "basis": "转移点（173 终点+g*）@174 七槽口径沙箱注册",
               "EK_FROZEN": EK_FROZEN, "GL174": GL174,
               "HOLD_TARGET_174": wins['HOLD_C'], "data_pk": wins['data_pk'],
               "data_sh": wins['data_sh'], "t0": wins['t0'],
               "SSE0_ANCHOR_174": sse0, "J0_ANCHOR_174": J0,
               "eval_seconds_sandbox": t_j0}
        json.dump(rec, open(os.path.join(HERE, "anchor174_注册锚_2026-09-10.json"), "w",
                            encoding='utf-8'), ensure_ascii=False, indent=1)
        print("[174] 锚注册落盘 anchor174_注册锚_2026-09-10.json——回填 de174.py 常量后方可 fit",
              flush=True)
        return
    if J0_ANCHOR_174 is None:
        print("[174] J0 锚未注册（常量 None），按纪律拒跑——先 python3 de174.py anchor", flush=True)
        return
    if abs(sse0 - SSE0_ANCHOR_174) > 1e-6 * SSE0_ANCHOR_174 or abs(J0 - J0_ANCHOR_174) > 1e-3:
        print("[174] 锚未中，拒跑（内核/坐标一致性事故，原样记录）", flush=True)
        return

    # —— 种群 ——
    b53 = de_bounds53(X30_START)
    bounds = [b53[j] for j in J7]
    rng = np.random.default_rng(DE_SEED74)
    lo = np.array([bb[0] for bb in bounds]); hi = np.array([bb[1] for bb in bounds])
    pop = np.empty((NPOP74, 7))
    if mode == "resume" and os.path.exists(os.path.join(HERE, "ckpt174_de.json")):
        ck = json.load(open(os.path.join(HERE, "ckpt174_de.json"), encoding='utf-8'))
        x7c = np.asarray(ck["x_best"], float)
        done = int(ck["gen"])
        print(f"[174] 续跑：自 ckpt gen={done} best 重撒种群（RNG 不续，双录）", flush=True)
    else:
        x7c, done = x7_start, 0
    pop[0] = x7c
    for j in range(1, NPOP74):
        x = x7c.copy()
        for jj in range(7):
            i = FREE7_PHYS[jj]
            if i in LG53 or i in LG53_EXTRA:
                x[jj] = x[jj] + rng.normal(0, 0.1)
            else:
                x[jj] = x[jj] * (1.0 + rng.normal(0, 0.1))
        pop[j] = np.clip(x, lo, hi)

    tt0 = time.time()
    nfev_de = [0]
    tp = ThreadPoolExecutor(workers)

    def f_de(x):
        nfev_de[0] += 1
        return obj(x, data2, keep, LAM, DECIM174)

    state = {"gen": [done]}
    best_ck = {"f": [None]}

    def cb_de(xk, convergence=0):
        state["gen"][0] += 1
        el = time.time() - tt0
        fk = float(obj(xk, data2, keep, LAM, DECIM174))
        best_ck["f"][0] = fk
        json.dump({"gen": state["gen"][0], "x_best": list(map(float, xk)), "J_best_dec5": fk,
                   "wall_s": el},
                  open(os.path.join(HERE, "ckpt174_de.json"), "w", encoding='utf-8'))
        print(f"  [174] DE gen {state['gen'][0]}/{DE_MAXITER74} J(dec5)={fk:.1f} ({el:.0f}s)",
              flush=True)
        if el > WALL_BUDGET:
            print(f"  [174] wall 超 2h 硬墙（DE 段 scipy deferred 不可优雅中断，双录在案）",
                  flush=True)

    rde = differential_evolution(
        f_de, bounds, maxiter=DE_MAXITER74 - done, tol=1e-6, strategy="rand1bin",
        mutation=0.7, recombination=0.9, seed=DE_SEED74, polish=False,
        init=pop, workers=tp.map, updating="deferred", callback=cb_de)
    tp.shutdown()
    de_rec = {"maxiter": DE_MAXITER74, "npop": NPOP74, "seed": DE_SEED74, "data": "dec5",
              "nfev": int(rde.nfev), "J": float(rde.fun), "message": str(rde.message),
              "resumed_from_gen": done}
    print(f"[174] DE 完：J={rde.fun:.1f} nfev={rde.nfev} ({time.time()-tt0:.0f}s) "
          f"msg={rde.message}", flush=True)
    x_de = x30_of(np.asarray(rde.x, float))
    p_de = unpack53(x_de)[0]
    if rde.fun >= 1e11:
        print("[174] DE 末代最佳 J≥1e11（全墙）——早停条款判死记录，不进 NM", flush=True)
        dump174(p_de, data2, keep, wins, nfev_de[0], de_rec, None,
                "de_allwall|" + str(rde.message), tt0, stage="de_allwall")
        return

    # —— NM 全量 ——
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
                      open(os.path.join(HERE, "ckpt174_nm.json"), "w", encoding='utf-8'))
        return v

    def cb_nm(xk):
        state2["it"] += 1
        if state2["it"] % 100 == 0:
            print(f"  [174] NM iter {state2['it']} bestJ={best['f']:.2f} "
                  f"({time.time()-tt0:.0f}s)", flush=True)
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
            print("[174] wall 超 2h 硬墙，自动停并双录", flush=True)
            raise _EarlyStop

    msg, stopped = "", False
    try:
        rnm = minimize(f_nm, np.asarray(rde.x, float), method="Nelder-Mead", callback=cb_nm,
                       options={"maxiter": NM_MAXITER74, "maxfev": NM_MAXFEV74,
                                "xatol": 1e-6, "fatol": 1e-3})
        x7_stage, msg = np.asarray(rnm.x, float), str(rnm.message)
    except _EarlyStop:
        x7_stage, msg, stopped = best["x"], "early_stop(nm|wall)", True
    nm_rec = {"maxiter": NM_MAXITER74, "maxfev": NM_MAXFEV74, "nfev": nfev_nm[0],
              "J_best": best["f"], "early_stopped": stopped, "message": msg}
    print(f"[174] NM 完：bestJ={best['f']:.2f} nfev={nfev_nm[0]} ({time.time()-tt0:.0f}s) {msg}",
          flush=True)
    p_stage = unpack53(x30_of(x7_stage))[0]
    dump174(p_stage, data2, keep, wins, nfev_de[0] + nfev_nm[0], de_rec, nm_rec, msg, tt0,
            stage="final")


def dump174(p, data2, keep, wins, nfev, de_rec, nm_rec, msg, tt0, stage="final"):
    """counter_174_model53.json 落盘（判线块 mlines74 同式；判官 judge174 独立复算比对）。"""
    quick = (stage != "final")
    ml = mlines74(p, data2, keep, wins)
    sse_full = sse174(p, data2, keep=keep)
    decomp = {}
    if not quick:
        for pr in PROTOS:
            v_, c_ = data2[pr]
            m_ = simT(p, np.asarray(v_, float), DT)
            r_ = (m_ - np.asarray(c_, float))[keep[pr]]
            decomp[pr] = float(r_ @ r_)
        ap_ok = bool(decomp["ap"] <= 5290.0)
        gm = gatepar47.gate_metrics47(R, p[:32])
        gvm = gv_midpoint53(p)
        kg_ok, kg_det = (False, "门控仿真None→KG灭") if gm is None else gate165.kg_verdict(gm, gvm)
        new_fails = []
        for i in range(4):
            if (not ml["kc2_now"][i]) and KC2_BASE174[i]:
                new_fails.append(f"KC2[{i}] 新增灭")
        for i in range(2):
            if (not ml["ks_now"][i]) and KS_BASE174[i]:
                new_fails.append(f"KS[{i}] 新增灭")
        if (not kg_ok) and KG_BASE174:
            new_fails.append("KG 新增灭")
        if (not ap_ok) and AP_BASE174:
            new_fails.append("KC-U②ap 新增灭")
        hrs1_ok = bool(ml["hrs"][1] <= HRS1_M5_MAX_174)
        M5 = bool(len(new_fails) == 0 and hrs1_ok)
    else:
        decomp, ap_ok, gm, gvm = {}, None, None, None
        kg_ok, kg_det, new_fails, hrs1_ok, M5 = None, "quick中间件未复算", None, None, None
    # M6：逃逸+7 槽贴界+flat
    esc = escape_check32(p[:32])
    x30 = x_from_params53(p)
    b53 = de_bounds53(X30_START)
    for jj, j in enumerate(J7):
        lo, hi = b53[j]
        v = x30[j]
        if v <= lo + 0.001 * (hi - lo) or v >= hi - 0.001 * (hi - lo):
            esc.append(f"自由槽 p{FREE7_PHYS[jj]} x={v:.4g} 贴/越界[{lo},{hi}]")
    p0 = start_params174()
    rel7 = np.abs(p[FREE7_PHYS] - p0[FREE7_PHYS]) / np.maximum(np.abs(p0[FREE7_PHYS]), 1e-12)
    if quick:
        improve, flat, M6 = None, None, None
        ok_all, branch = None, stage
    else:
        sse0c = sse174(p0, data2, keep=keep)
        improve = (sse0c - sse_full) / sse0c if sse0c > 0 else np.nan
        flat = bool(rel7.max() > 0.5 and improve < 0.05)
        M6 = bool(len(esc) == 0 and not flat)
        M1, M2, M3, M4 = ml["M1"], ml["M2"], ml["M3"], ml["M4"]
        ok_all = bool(M1 and M2 and M3 and M4 and M5 and M6)
        if ok_all:
            branch = "A"
        elif M5 and M3 and M4 and (not M1 or not M2):
            branch = "B"
        elif not M5:
            branch = "C"
        else:
            branch = "未分类（原样记录）"
    n_pass = None if quick else int(sum([ml['M1'], ml['M2'], ml['M3'], ml['M4'], M5, M6]))
    preds = None if quick else {
        "Q1_sse_full低于转移点x0.9": [sse_full, 0.9 * SSE0_ANCHOR_174,
                                    bool(sse_full < 0.9 * SSE0_ANCHOR_174)],
        "Q2_M1或M4由灭转过且其余不倒退": [
            [bool(ml['M1']), bool(ml['M4'])],
            "M1∨M4 过 且 M2∧M3∧M6 过 且 M5 无新增灭",
            bool((ml['M1'] or ml['M4']) and ml['M2'] and ml['M3'] and M6 and M5)],
        "Q3_终点g在带[0.125,0.150]": [float(p[7]), bool(0.125 <= p[7] <= 0.150)],
        "Q4_M4尾流幅度读数": ml["a_fit"],
        "Q5_M3保持读数": ml["hold"],
        "Q6_7槽漂移榜": sorted([(f"p{FREE7_PHYS[j]}", float(rel7[j])) for j in range(7)],
                              key=lambda t: -t[1]),
    }
    verdict = {
        "tag": "174", "model": MODEL_NAME, "stage": stage, "cell": "16713003",
        "params": list(map(float, p)), "EK_used": EK_FROZEN, "gL_frozen": GL174,
        "x6_frozen": list(map(float, P6X)),
        "free_slots_phys": FREE7_PHYS, "free_slots_j": J7,
        "sse_full": sse_full, "sse_tail": ml["sse_tail"],
        "J_obj_final": None if nm_rec is None else nm_rec.get("J_best"),
        "decomp": decomp,
        "M1_peak_nA": ml["pk"], "M2_step_nA": ml["sh"], "M3_hold_nA": ml["hold"],
        "M4_tau_s": ml["tau_fit"], "M4_a_nA": ml["a_fit"],
        "data_pk": wins["data_pk"], "data_sh": wins["data_sh"],
        "M1": ml["M1"], "M2": ml["M2"], "M3": ml["M3"], "M4": ml["M4"],
        "M5": M5, "M6": M6,
        "M5_new_fails": new_fails, "hrs1_within_5pct": hrs1_ok,
        "half": ml["hrs"], "creep": ml["crs"], "trise": ml["trise"],
        "r40": ml["r40"], "r60": ml["r60"],
        "KC2": ml["kc2_now"], "KS": ml["ks_now"], "KG": (None if kg_ok is None else bool(kg_ok)),
        "kg_detail": kg_det,
        "gate": None if gm is None else list(map(float, gm)), "gv_midpoint": gvm,
        "KC_U_ap5290": ap_ok, "escape": esc, "flat": flat,
        "improve_vs_start": improve, "drift_max_pct_vs_start": (None if quick else
                                                              float(rel7.max()) * 100),
        "predictions": preds,
        "de": de_rec, "nm": nm_rec, "nfev": nfev, "message": msg,
        "wall_s": time.time() - tt0, "branch": branch, "ok_all": ok_all,
    }
    out = os.path.join(HERE, "counter_174_model53.json")
    json.dump(verdict, open(out, "w", encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[174] 落盘 {out}（stage={stage}）", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    assert mode in ("anchor", "fit", "resume"), "用法：de174.py [anchor|fit|resume] [workers]"
    fit174(mode, workers)
