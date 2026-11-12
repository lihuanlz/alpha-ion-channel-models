# de177.py — 代码177 model60 特征锚定卡 @16713003（预注册 2026-09-10）
# 模型=176 同架构（官方环+X 件，sim60 不动）；唯一改动=目标函数（协议等权+慢尾锚）与界盒硬化。
# 用法：python de177.py [anchor|fit|resume] [workers]   （Windows 用 python 不用 python3）
import json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import de173
import de174
import de176
from de173 import DT, find_step, tail_fit_kernel, TAIL_WIN, KCT_T0NOM
from de174 import load174, windows174, PROTOS
from de176 import sim60, X_START, EREV, llines60

LAM_A, LAM_T = 1.0, 1.0          # 慢尾锚权重（预注册 §2）
A_DATA, TAU_DATA = 0.0579, 0.704 # 数据慢尾实测（复位核查在案）
NPOP77, DE_MAXITER77 = 132, 60
NM_MAXFEV77 = 2000
DE_SEED77 = 20260910
DECIM177 = {pr: 5 for pr in PROTOS}
WALL_BUDGET = 3600.0
WALL_STOP = WALL_BUDGET
# 界盒硬化：P1-P8 ±0.5dex（L8 内置）、GKr ±0.5、α_amp (−0.7,+0.3)、α_τ (−1.0,+0.3)
BOUNDS77 = ([(x - 0.5, x + 0.5) for x in X_START[:8]]
            + [(X_START[8] - 0.5, X_START[8] + 0.5),
               (max(-1.0, X_START[9] - 0.4), min(1.0, X_START[9] + 0.7)),
               (max(-1.3, X_START[10] - 0.65), min(0.3, X_START[10] + 0.65))])

# ---- 锚（预注册 §7：沙箱注册后回填；None=拒跑）----
J0_ANCHOR_177 = 0.620374
SSE0_ANCHOR_177 = 39647.892837


def proto_powers(data2, keep):
    """协议等权分母：各协议 keep 口径数据功率，现场算+锚注册。"""
    P = {}
    for pr in PROTOS:
        v_, c_ = data2[pr]
        c_ = np.asarray(c_, float)
        P[pr] = float(c_[keep[pr]] @ c_[keep[pr]])
    return P


def obj77_factory(wins, PPR):
    IWT = wins['IWT']
    vd_len = None

    def obj(x11, data, keep, decim):
        # 协议等权 sse
        out = de176.resid60(x11, data, keep, decim)
        if out is None:
            return 1e12
        rd, raw = out
        tot = 0.0
        for pr, r in rd.items():
            pwr = PPR[pr] / (1 if decim is None else 1)  # decim 残差已 ×√ds 保持功率尺度
            tot += float(r @ r) / pwr
        if not np.isfinite(tot):
            return 1e12
        # 慢尾锚（判段全量仿真，dec5 段也用全量判段——口径注册）
        vd_, _ = data['deactivation']
        n1 = int(9.1612 / DT)
        m1 = sim60(np.asarray(vd_[:n1], float), x11)
        if not np.all(np.isfinite(m1)):
            return 1e12
        a_, t_ = tail_fit_kernel(m1, IWT[0], IWT[1], DT)
        if a_ is None:
            return 1e12
        pa = ((abs(a_) - A_DATA) / A_DATA) ** 2
        pt = ((t_ - TAU_DATA) / TAU_DATA) ** 2
        return tot + LAM_A * pa + LAM_T * pt
    return obj


def fit177(mode, workers):
    from concurrent.futures import ThreadPoolExecutor
    from scipy.optimize import minimize, differential_evolution

    data2, keep = load174()
    vd, cd = data2["deactivation"]
    wins = windows174(vd, cd)
    PPR = proto_powers(data2, keep)
    obj = obj77_factory(wins, PPR)
    print(f"[177] 特征锚定卡：协议功率 " + " ".join(f"{k}={v:.3g}" for k, v in PPR.items()), flush=True)

    tt = time.time()
    J0 = obj(X_START, data2, keep, None)
    t_j0 = time.time() - tt
    sse0 = de176.sse60(X_START, data2, keep)
    print(f"[177] 锚：sse0={sse0:.6f}（锚 {SSE0_ANCHOR_177}）J0={J0:.6f}（锚 {J0_ANCHOR_177}）"
          f" 单评估实测 {t_j0:.2f}s", flush=True)
    if mode == "anchor":
        rec = {"date": "2026-09-10", "card": "job_177_model60", "model": "model60",
               "cell": "16713003", "basis": "F11+X(0.5,0.44) 起点 @177 特征锚定口径沙箱注册",
               "EREV": EREV, "proto_powers": PPR,
               "SSE0_ANCHOR_177": sse0, "J0_ANCHOR_177": J0,
               "eval_seconds_sandbox": t_j0}
        json.dump(rec, open(os.path.join(HERE, "anchor177_注册锚_2026-09-10.json"), "w",
                            encoding='utf-8'), ensure_ascii=False, indent=1)
        print("[177] 锚注册落盘 anchor177_注册锚_2026-09-10.json——回填常量后方可 fit", flush=True)
        return
    if J0_ANCHOR_177 is None or SSE0_ANCHOR_177 is None:
        print("[177] 锚未注册（常量 None），按纪律拒跑——先 python de177.py anchor", flush=True)
        return
    if abs(sse0 - SSE0_ANCHOR_177) > 1e-6 * SSE0_ANCHOR_177 or abs(J0 - J0_ANCHOR_177) > 1e-6 * abs(J0_ANCHOR_177):
        print("[177] 锚未中，拒跑（内核/坐标一致性事故，原样记录）", flush=True)
        return

    rng = np.random.default_rng(DE_SEED77)
    lo = np.array([b[0] for b in BOUNDS77]); hi = np.array([b[1] for b in BOUNDS77])
    pop = np.empty((NPOP77, 11))
    if mode == "resume" and os.path.exists(os.path.join(HERE, "ckpt177_de.json")):
        ck = json.load(open(os.path.join(HERE, "ckpt177_de.json"), encoding='utf-8'))
        xc = np.asarray(ck["x_best"], float); done = int(ck["gen"])
        print(f"[177] 续跑：自 ckpt gen={done} best 重撒种群（RNG 不续，双录）", flush=True)
    else:
        xc, done = X_START, 0
    pop[0] = xc
    for j in range(1, NPOP77):
        pop[j] = np.clip(xc + rng.normal(0, 0.1, 11), lo, hi)

    tt0 = time.time()
    nfev_de = [0]
    tp = ThreadPoolExecutor(workers)

    def f_de(x):
        nfev_de[0] += 1
        return obj(x, data2, keep, DECIM177)

    state = {"gen": [done]}

    def cb_de(xk, convergence=0):
        state["gen"][0] += 1
        el = time.time() - tt0
        fk = float(obj(xk, data2, keep, DECIM177))
        json.dump({"gen": state["gen"][0], "x_best": list(map(float, xk)), "J_best_dec5": fk,
                   "wall_s": el},
                  open(os.path.join(HERE, "ckpt177_de.json"), "w", encoding='utf-8'))
        print(f"  [177] DE gen {state['gen'][0]}/{DE_MAXITER77} J(dec5)={fk:.4f} ({el:.0f}s)",
              flush=True)
        if el > WALL_BUDGET:
            print("[177] wall 超 1h 硬墙（双录）", flush=True)

    rde = differential_evolution(
        f_de, BOUNDS77, maxiter=DE_MAXITER77 - done, tol=1e-6, strategy="rand1bin",
        mutation=0.7, recombination=0.9, seed=DE_SEED77, polish=False,
        init=pop, workers=tp.map, updating="deferred", callback=cb_de)
    tp.shutdown()
    de_rec = {"maxiter": DE_MAXITER77, "npop": NPOP77, "seed": DE_SEED77, "data": "dec5",
              "nfev": int(rde.nfev), "J": float(rde.fun), "message": str(rde.message),
              "resumed_from_gen": done}
    print(f"[177] DE 完：J={rde.fun:.4f} nfev={rde.nfev} ({time.time()-tt0:.0f}s)", flush=True)
    if rde.fun >= 1e11:
        print("[177] DE 全墙——早停条款判死记录，不进 polish", flush=True)
        de176.dump176.__wrapped__ if False else None
        dump177(np.asarray(rde.x, float), data2, keep, wins, nfev_de[0], de_rec, None,
                "de_allwall", tt0, stage="de_allwall")
        return

    # —— 有界 Powell 抛光（176 挂账"NM 无界出界"兑现）——
    nfev_nm = [0]
    best = {"f": None, "x": np.asarray(rde.x, float)}

    def f_pw(x):
        nfev_nm[0] += 1
        v = obj(x, data2, keep, None)
        if best["f"] is None or v < best["f"]:
            best["f"] = v; best["x"] = np.array(x).copy()
        if nfev_nm[0] % 50 == 0:
            json.dump({"nfev": nfev_nm[0], "J_best": best["f"],
                       "x_best": list(map(float, best["x"])), "wall_s": time.time() - tt0},
                      open(os.path.join(HERE, "ckpt177_nm.json"), "w", encoding='utf-8'))
        return v

    try:
        rpw = minimize(f_pw, np.asarray(rde.x, float), method="Powell", bounds=BOUNDS77,
                       options={"maxfev": NM_MAXFEV77, "xtol": 1e-7, "ftol": 1e-7})
        x_stage, msg = np.asarray(rpw.x, float), str(rpw.message)
    except Exception as e:
        x_stage, msg = best["x"], f"polish_exception:{e}"
    nm_rec = {"method": "Powell_bounded", "maxfev": NM_MAXFEV77, "nfev": nfev_nm[0],
              "J_best": best["f"], "message": msg}
    print(f"[177] polish 完：bestJ={best['f']:.4f} nfev={nfev_nm[0]} ({time.time()-tt0:.0f}s) {msg}",
          flush=True)
    dump177(np.asarray(x_stage, float), data2, keep, wins, nfev_de[0] + nfev_nm[0],
            de_rec, nm_rec, msg, tt0, stage="final")


def dump177(x11, data2, keep, wins, nfev, de_rec, nm_rec, msg, tt0, stage="final"):
    """counter_177_model60.json——判线块 llines60 同式（判官 judge176 沿用）。"""
    quick = (stage != "final")
    ll = llines60(x11, data2, keep, wins)
    sse_full = de176.sse60(x11, data2, keep)
    esc = []
    for j, (lo_, hi_) in enumerate(BOUNDS77):
        if x11[j] <= lo_ + 0.001 * (hi_ - lo_) or x11[j] >= hi_ - 0.001 * (hi_ - lo_):
            esc.append(f"槽{j} x={x11[j]:.4g} 贴/越界[{lo_},{hi_}]")
    drift8 = float(np.max(np.abs(x11[:8] - X_START[:8])))
    if quick:
        improve, flat, L6 = None, None, None
        L8 = None; branch = stage
    else:
        sse0c = de176.sse60(X_START, data2, keep)
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
    verdict = {
        "tag": "177", "model": "model60", "stage": stage, "cell": "16713003",
        "architecture": "官方HH4态环+X慢件两旋钮（11参）特征锚定口径",
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
        "hrs": ll["hrs"], "hrs_data_smooth": de176.HRS_DATA, "creep": ll["crs"],
        "trise": ll["trise"], "r40": ll["r40"], "r60": ll["r60"],
        "escape": esc, "flat": flat, "drift8_dex_max": drift8,
        "improve_vs_start": improve,
        "de": de_rec, "nm": nm_rec, "nfev": nfev, "message": msg,
        "wall_s": time.time() - tt0, "branch": branch, "n_pass": n_pass,
    }
    out = os.path.join(HERE, "counter_177_model60.json")
    json.dump(verdict, open(out, "w", encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[177] 落盘 {out}（stage={stage}）", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    assert mode in ("anchor", "fit", "resume"), "用法：de177.py [anchor|fit|resume] [workers]"
    fit177(mode, workers)
