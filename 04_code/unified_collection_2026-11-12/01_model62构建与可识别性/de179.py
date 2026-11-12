# de179.py — 代码179 model62 锚定收口卡 @16713003（预注册 2026-09-10）
# model62 = model61 + xs 稳态 sigmoid VH/KX 解冻（16 参）；目标函数加 hrs 数据窗罚。
# 拟合在用户本地/集群跑（沙箱只做锚注册/烟测/判官）。用法：python de179.py [anchor|fit|resume] [workers]
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import de173
import de174
import de176
import de177
import de178
from de173 import DT, find_step, halfrise, tail_fit_kernel, evolve_x, GX_K, P6X, TAIL_WIN, KCT_T0NOM
from de174 import load174, windows174, PROTOS
from de176 import EREV, HRS_DATA, M4_AMIN, M4_TAU_WIN, TAIL_SSE_MAX, R2_BARS
from de177 import proto_powers
from de178 import _hh61_cn

LAM_A, LAM_T, LAM_H = 1.0, 1.0, 10.0
A_DATA, TAU_DATA = 0.0579, 0.704
NOM = [17.15, 25.40, 33.66, 41.92]; SKIP = [0.6, 0.6, 0.05, 0.05]
HRS_LO, HRS_HI = math.log10(0.4), math.log10(2.5)
NPOP79, DE_MAXITER79 = 160, 60
NM_MAXFEV79 = 2500
DE_SEED79 = 20260910
DECIM179 = {pr: 5 for pr in PROTOS}
WALL_BUDGET = 3600.0
WALL_STOP = WALL_BUDGET

# ---- 锚（预注册 §7：沙箱注册后回填；None=拒跑）----
J0_ANCHOR_179 = 0.302676
SSE0_ANCHOR_179 = 14239.989563


def sim62(V, x16):
    """x16 = [log10 P1..P8, log10 GKr, log10 Pf1, Pf2, log10 α_amp, log10 α_τ,
              GX_MID, VH, log10 KX]。"""
    P = 10.0 ** np.asarray(x16[:8], float)
    gkr = 10.0 ** x16[8]
    pf1 = 10.0 ** x16[9]
    pf2 = x16[10]
    amp, tsc = 10.0 ** x16[11], 10.0 ** x16[12]
    gmid, VH, KX = x16[13], x16[14], 10.0 ** x16[15]
    I = _hh61_cn(np.asarray(V, float), DT * 1000.0, *P, pf1, pf2, EREV, gkr)
    Vv = np.asarray(V, float)
    xs = evolve_x(Vv, DT, P6X[0], VH, KX, P6X[3] * tsc, P6X[4])
    gx = 1.0 / (1.0 + np.exp((Vv - gmid) / GX_K))
    return I + amp * P6X[5] * xs * gx * (Vv - EREV)


def _load_start():
    """起跑点 = 178 终点 x14 + (VH=−46.175, KX=15.714)（探针定点在案）。"""
    c178 = json.load(open(os.path.join(HERE, "counter_178_model61.json"), encoding='utf-8'))
    x14 = np.asarray(c178["x14"], float)
    return np.concatenate([x14, [-46.175, math.log10(15.7136)]])


X_START62 = None
BOUNDS79 = None


def _mk_bounds(x0):
    b = [(x - 0.5, x + 0.5) for x in x0[:8]]
    b += [(x0[8] - 0.5, x0[8] + 0.5), (x0[9] - 0.7, x0[9] + 0.7), (0.05, 0.13),
          (x0[11] - 0.4, min(0.3, x0[11] + 0.7)), (x0[12] - 0.65, x0[12] + 0.65),
          (-90.0, -45.0), (-66.175, -26.175),
          (x0[15] - 0.18, x0[15] + 0.18)]
    return b


def resid62(x16, data, keep, decim):
    rd, raw = {}, None
    for proto, (v, c) in data.items():
        try:
            m = sim62(np.asarray(v, float), x16)
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


def sse62(x16, data, keep):
    out = resid62(x16, data, keep, None)
    if out is None:
        return 1e30
    rd, _ = out
    tot = 0.0
    for pr, r in rd.items():
        tot += float(r @ r)
        if not np.isfinite(tot):
            return 1e30
    return tot


def _hrs_of(msa, vsa):
    return [halfrise(msa, find_step(np.asarray(vsa, float), t), s) for t, s in zip(NOM, SKIP)]


def _hrs_pen(hrs):
    pen = 0.0
    for h, d in zip(hrs, HRS_DATA):
        if not np.isfinite(h):
            return None
        lr = math.log10(h / d)
        if lr < HRS_LO:
            pen += (HRS_LO - lr) ** 2
        elif lr > HRS_HI:
            pen += (lr - HRS_HI) ** 2
    return pen


def obj79_factory(wins, PPR, vsa):
    IWT = wins['IWT']

    def obj(x16, data, keep, decim):
        out = resid62(x16, data, keep, decim)
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
        m1 = sim62(np.asarray(vd_[:n1], float), x16)
        if not np.all(np.isfinite(m1)):
            return 1e12
        a_, t_ = tail_fit_kernel(m1, IWT[0], IWT[1], DT)
        if a_ is None:
            return 1e12
        msa = sim62(np.asarray(vsa, float), x16)
        hp = _hrs_pen(_hrs_of(msa, vsa))
        if hp is None:
            return 1e12
        return (tot + LAM_A * ((abs(a_) - A_DATA) / A_DATA) ** 2
                  + LAM_T * ((t_ - TAU_DATA) / TAU_DATA) ** 2 + LAM_H * hp)
    return obj


# ---- L1-L8 判线块（de 落盘与 judge179 同式唯一事实源）----
def llines62(x16, data2, keep, wins):
    vd, cd = data2["deactivation"]
    vsa, csa = data2["steady_activation"]
    ISH, IPK, IHD, IWT = wins['ISH'], wins['IPK'], wins['IHD'], wins['IWT']
    n1 = int(9.1612 / DT)
    m1 = sim62(np.asarray(vd[:n1], float), x16)
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
    msa = sim62(np.asarray(vsa, float), x16)
    hrs = _hrs_of(msa, vsa)
    crs = [de173.creep(msa, find_step(np.asarray(vsa, float), t)) for t in [17.15, 25.40]]
    trise = de173.tail_rise(msa, find_step(np.asarray(vsa, float), 41.92))
    r40, r60 = de173.steady_ratios(msa)[:2]
    L5 = bool(np.all(np.isfinite(hrs)) and
              all(0.4 * d <= h <= 2.5 * d for h, d in zip(hrs, HRS_DATA)))
    r2 = {}
    for pr in PROTOS:
        v_, c_ = data2[pr]
        m_ = sim62(np.asarray(v_, float), x16)
        km_ = keep[pr]
        pw_ = float(np.asarray(c_, float)[km_] @ np.asarray(c_, float)[km_])
        r2[pr] = 1 - float(((m_ - np.asarray(c_, float))[km_] ** 2).sum()) / pw_
    L7 = bool(all(r2[pr] >= R2_BARS[pr] for pr in PROTOS))
    return dict(pk=pk, sh=sh, hold=hold, sse_tail=sse_tail, a_fit=a_fit, tau_fit=tau_fit,
                L1=L1, L2=L2, L3=L3, L4=L4, L5=L5, L7=L7, r2=r2, msa=msa,
                hrs=hrs, crs=crs, trise=trise, r40=r40, r60=r60)


def fit179(mode, workers):
    global X_START62, BOUNDS79
    from concurrent.futures import ThreadPoolExecutor
    from scipy.optimize import minimize, differential_evolution

    X_START62 = _load_start()
    BOUNDS79 = _mk_bounds(X_START62)
    data2, keep = load174()
    vd, cd = data2["deactivation"]
    wins = windows174(vd, cd)
    vsa, _ = data2["steady_activation"]
    PPR = proto_powers(data2, keep)
    obj = obj79_factory(wins, PPR, vsa)
    print("[179] model62 锚定收口卡启动", flush=True)

    tt = time.time()
    J0 = obj(X_START62, data2, keep, None)
    sse0 = sse62(X_START62, data2, keep)
    print(f"[179] 锚：sse0={sse0:.6f}（锚 {SSE0_ANCHOR_179}）J0={J0:.6f}（锚 {J0_ANCHOR_179}）"
          f" 单评估实测 {time.time()-tt:.2f}s", flush=True)
    if mode == "anchor":
        rec = {"date": "2026-09-10", "card": "job_179_model62", "model": "model62",
               "cell": "16713003", "basis": "178终点+(VH,KX) 起点 @179 口径沙箱注册",
               "EREV": EREV, "SSE0_ANCHOR_179": sse0, "J0_ANCHOR_179": J0}
        json.dump(rec, open(os.path.join(HERE, "anchor179_注册锚_2026-09-10.json"), "w",
                            encoding='utf-8'), ensure_ascii=False, indent=1)
        print("[179] 锚注册落盘 anchor179_注册锚_2026-09-10.json——回填常量后方可 fit", flush=True)
        return
    if J0_ANCHOR_179 is None or SSE0_ANCHOR_179 is None:
        print("[179] 锚未注册，拒跑——先 python de179.py anchor", flush=True)
        return
    if abs(sse0 - SSE0_ANCHOR_179) > 1e-6 * SSE0_ANCHOR_179 or abs(J0 - J0_ANCHOR_179) > 1e-6 * abs(J0_ANCHOR_179):
        print("[179] 锚未中，拒跑", flush=True)
        return

    rng = np.random.default_rng(DE_SEED79)
    lo = np.array([b[0] for b in BOUNDS79]); hi = np.array([b[1] for b in BOUNDS79])
    pop = np.empty((NPOP79, 16))
    if mode == "resume" and os.path.exists(os.path.join(HERE, "ckpt179_de.json")):
        ck = json.load(open(os.path.join(HERE, "ckpt179_de.json"), encoding='utf-8'))
        xc = np.asarray(ck["x_best"], float); done = int(ck["gen"])
        print(f"[179] 续跑：自 ckpt gen={done}（RNG 不续，双录）", flush=True)
    else:
        xc, done = X_START62, 0
    pop[0] = xc
    for j in range(1, NPOP79):
        x = xc + rng.normal(0, 0.1, 16)
        x[10] = xc[10] + rng.normal(0, 0.005)
        x[13] = xc[13] + rng.normal(0, 2.0)
        x[14] = xc[14] + rng.normal(0, 2.0)
        pop[j] = np.clip(x, lo, hi)

    tt0 = time.time()
    nfev_de = [0]
    tp = ThreadPoolExecutor(workers)

    def f_de(x):
        nfev_de[0] += 1
        return obj(x, data2, keep, DECIM179)

    state = {"gen": [done]}

    def cb_de(xk, convergence=0):
        state["gen"][0] += 1
        el = time.time() - tt0
        fk = float(obj(xk, data2, keep, DECIM179))
        json.dump({"gen": state["gen"][0], "x_best": list(map(float, xk)), "J_best_dec5": fk,
                   "wall_s": el},
                  open(os.path.join(HERE, "ckpt179_de.json"), "w", encoding='utf-8'))
        print(f"  [179] DE gen {state['gen'][0]}/{DE_MAXITER79} J(dec5)={fk:.4f} ({el:.0f}s)",
              flush=True)
        if el > WALL_BUDGET:
            print("[179] wall 超 1h 硬墙（双录）", flush=True)

    rde = differential_evolution(
        f_de, BOUNDS79, maxiter=DE_MAXITER79 - done, tol=1e-6, strategy="rand1bin",
        mutation=0.7, recombination=0.9, seed=DE_SEED79, polish=False,
        init=pop, workers=tp.map, updating="deferred", callback=cb_de)
    tp.shutdown()
    de_rec = {"maxiter": DE_MAXITER79, "npop": NPOP79, "seed": DE_SEED79, "data": "dec5",
              "nfev": int(rde.nfev), "J": float(rde.fun), "message": str(rde.message),
              "resumed_from_gen": done}
    print(f"[179] DE 完：J={rde.fun:.4f} nfev={rde.nfev} ({time.time()-tt0:.0f}s)", flush=True)
    if rde.fun >= 1e11:
        print("[179] DE 全墙——早停条款判死记录", flush=True)
        dump179(np.asarray(rde.x, float), data2, keep, wins, nfev_de[0], de_rec, None,
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
                      open(os.path.join(HERE, "ckpt179_nm.json"), "w", encoding='utf-8'))
        return v

    try:
        rpw = minimize(f_pw, np.asarray(rde.x, float), method="Powell", bounds=BOUNDS79,
                       options={"maxfev": NM_MAXFEV79, "xtol": 1e-7, "ftol": 1e-7})
        x_stage, msg = np.asarray(rpw.x, float), str(rpw.message)
    except Exception as e:
        x_stage, msg = best["x"], f"polish_exception:{e}"
    nm_rec = {"method": "Powell_bounded", "maxfev": NM_MAXFEV79, "nfev": nfev_nm[0],
              "J_best": best["f"], "message": msg}
    print(f"[179] polish 完：bestJ={best['f']:.4f} nfev={nfev_nm[0]} ({time.time()-tt0:.0f}s) {msg}",
          flush=True)
    dump179(np.asarray(x_stage, float), data2, keep, wins, nfev_de[0] + nfev_nm[0],
            de_rec, nm_rec, msg, tt0, stage="final")


def dump179(x16, data2, keep, wins, nfev, de_rec, nm_rec, msg, tt0, stage="final"):
    quick = (stage != "final")
    ll = llines62(x16, data2, keep, wins)
    sse_full = sse62(x16, data2, keep)
    esc = []
    for j, (lo_, hi_) in enumerate(BOUNDS79):
        if x16[j] <= lo_ + 0.001 * (hi_ - lo_) or x16[j] >= hi_ - 0.001 * (hi_ - lo_):
            esc.append(f"槽{j} x={x16[j]:.4g} 贴/越界[{lo_},{hi_}]")
    drift8 = float(np.max(np.abs(x16[:8] - X_START62[:8])))
    if quick:
        improve, flat, L6 = None, None, None
        L8 = None; branch = stage
    else:
        sse0c = sse62(X_START62, data2, keep)
        improve = (sse0c - sse_full) / sse0c if sse0c > 0 else np.nan
        idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15]
        rel = np.abs(10.0 ** x16[idx] - 10.0 ** X_START62[idx]) \
              / np.maximum(np.abs(10.0 ** X_START62[idx]), 1e-12)
        flat = bool(rel.max() > 0.5 and improve < 0.05)
        L6 = bool(len(esc) == 0 and not flat)
        L8 = bool(drift8 <= 0.5)
        Ls = [ll['L1'], ll['L2'], ll['L3'], ll['L4'], ll['L5'], L6, ll['L7'], L8]
        branch = "全过" if all(Ls) else "未全过"
    n_pass = None if quick else int(sum([ll['L1'], ll['L2'], ll['L3'], ll['L4'], ll['L5'],
                                         L6, ll['L7'], L8]))
    verdict = {
        "tag": "179", "model": "model62", "stage": stage, "cell": "16713003",
        "architecture": "官方HH4态环(k43双指数)+X慢件(α,GX_MID,VH,KX)（16参）",
        "x16": list(map(float, x16)), "bounds": [list(b) for b in BOUNDS79],
        "P1_P8": list(map(float, 10.0 ** x16[:8])), "GKr": float(10.0 ** x16[8]),
        "Pf1": float(10.0 ** x16[9]), "Pf2": float(x16[10]),
        "alpha_amp": float(10.0 ** x16[11]), "alpha_tau": float(10.0 ** x16[12]),
        "GX_MID": float(x16[13]), "VH": float(x16[14]), "KX": float(10.0 ** x16[15]),
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
        "de": de_rec, "nm": nm_rec, "nfev": nfev, "message": msg,
        "wall_s": time.time() - tt0, "branch": branch, "n_pass": n_pass,
    }
    out = os.path.join(HERE, "counter_179_model62.json")
    json.dump(verdict, open(out, "w", encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[179] 落盘 {out}（stage={stage}）", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    assert mode in ("anchor", "fit", "resume"), "用法：de179.py [anchor|fit|resume] [workers]"
    fit179(mode, workers)
