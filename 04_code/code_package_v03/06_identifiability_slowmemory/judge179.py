# judge179.py — 代码179 model62 锚定收口卡判官（预注册 §5 判线独立复算件）
# 用法：python judge179.py [counter_179_model62.json 路径]
# 双通道：拟合器 CN，判官 expm 精确路径；逃逸读 counter 存档界盒。
import json, math, os, sys
import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import de173
from de173 import (DT, find_step, halfrise, creep, tail_rise, steady_ratios,
                   tail_fit_kernel, evolve_x, GX_K, P6X, TAIL_WIN, KCT_T0NOM)
from de174 import load174, windows174, PROTOS
import de179
from de179 import (EREV, HRS_DATA, M4_AMIN, M4_TAU_WIN, TAIL_SSE_MAX, R2_BARS,
                   sse62, NOM, SKIP, _hrs_of)

TOL_REL = 1e-6
TOL_TRACE = 2e-3


def sim62_expm(V, x16):
    P = 10.0 ** np.asarray(x16[:8], float)
    gkr = 10.0 ** x16[8]
    pf1 = 10.0 ** x16[9]
    pf2 = x16[10]
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
        k12, k21, k41, k14 = k43, k34, k32, k23
        M = np.array([[-(k12 + k14), k21, 0., k41], [k12, -(k21 + k23), k32, 0.],
                      [0., k23, -(k32 + k34), k43], [k14, 0., k34, -(k43 + k41)]])
        y = expm(M * dt_ms) @ y
        out[i] = gkr * y[2] * (v - EREV)
    xs = evolve_x(Vv, DT, P6X[0], VH, KX, P6X[3] * tsc, P6X[4])
    gx = 1.0 / (1.0 + np.exp((Vv - gmid) / GX_K))
    return out + amp * P6X[5] * xs * gx * (Vv - EREV)


def _close(a, b, tag, diffs, tol=TOL_REL):
    if a is None or b is None:
        if a is not b:
            diffs.append(f"{tag}: None 不对称（落盘={a} 复算={b}）")
        return
    if not (np.isfinite(a) and np.isfinite(b)):
        if not (np.isfinite(a) == np.isfinite(b)):
            diffs.append(f"{tag}: 有限性不对称（落盘={a} 复算={b}）")
        return
    if abs(a - b) > tol * max(abs(b), 1e-9):
        diffs.append(f"{tag}: 落盘={a} 复算={b} 超容差{tol}")


def judge179(counter_path):
    v = json.load(open(counter_path, encoding='utf-8'))
    assert v['model'] == 'model62' and v.get('stage') == 'final', \
        f"非 final 落盘件（stage={v.get('stage')}）——中间件不判"
    assert v.get('cell') == '16713003', "细胞不符——非 179 口径件，拒判"
    x16 = np.asarray(v['x16'], float)
    assert len(x16) == 16
    bounds = [tuple(b) for b in v['bounds']]
    X_START62 = de179._load_start()
    data2, keep = load174()
    vd, cd = data2['deactivation']
    vsa, csa = data2['steady_activation']
    wins = windows174(vd, cd)
    ISH, IPK, IHD, IWT = wins['ISH'], wins['IPK'], wins['IHD'], wins['IWT']
    # —— 全量复算（expm 路径）——
    sse_full = sse62(x16, data2, keep)
    n1 = int(9.1612 / DT)
    m1 = sim62_expm(np.asarray(vd[:n1], float), x16)
    c1 = np.asarray(cd[:n1], float)
    finite1 = bool(np.all(np.isfinite(m1)))
    pk = float(np.min(m1[IPK[0]:IPK[1]]))
    sh = float(np.min(m1[ISH[0]:ISH[1]]))
    hold = float(np.mean(m1[IHD[0]:IHD[1]]))
    seg = m1[IWT[0]:IWT[1]]
    sse_tail = float(np.sum((seg - c1[IWT[0]:IWT[1]]) ** 2))
    a_fit, tau_fit = tail_fit_kernel(m1, IWT[0], IWT[1], DT)
    msa = sim62_expm(np.asarray(vsa, float), x16)
    hrs = _hrs_of(msa, vsa)
    crs = [creep(msa, find_step(np.asarray(vsa, float), t)) for t in [17.15, 25.40]]
    trise = tail_rise(msa, find_step(np.asarray(vsa, float), 41.92))
    r40, r60 = steady_ratios(msa)[:2]
    r2 = {}
    for pr in PROTOS:
        v_, c_ = data2[pr]
        m_ = sim62_expm(np.asarray(v_, float), x16)
        km_ = keep[pr]
        pw_ = float(np.asarray(c_, float)[km_] @ np.asarray(c_, float)[km_])
        r2[pr] = 1 - float(((m_ - np.asarray(c_, float))[km_] ** 2).sum()) / pw_
    # —— 判线（预注册 §5 逐字）——
    L1 = bool(finite1 and abs(pk - wins['data_pk']) / abs(wins['data_pk']) <= 0.10)
    L2 = bool(finite1 and 0.5 <= sh / wins['data_sh'] <= 1.5)
    L3 = bool(finite1 and -0.015 <= hold <= 0.005)
    L4 = bool(finite1 and sse_tail <= TAIL_SSE_MAX and tau_fit is not None
              and M4_TAU_WIN[0] <= tau_fit <= M4_TAU_WIN[1] and abs(a_fit) >= M4_AMIN)
    L5 = bool(np.all(np.isfinite(hrs)) and
              all(0.4 * d <= h <= 2.5 * d for h, d in zip(hrs, HRS_DATA)))
    L7 = bool(all(r2[pr] >= R2_BARS[pr] for pr in PROTOS))
    esc = []
    for j, (lo_, hi_) in enumerate(bounds):
        if x16[j] <= lo_ + 0.001 * (hi_ - lo_) or x16[j] >= hi_ - 0.001 * (hi_ - lo_):
            esc.append(f"槽{j} x={x16[j]:.4g} 贴/越界[{lo_},{hi_}]")
    sse0c = sse62(X_START62, data2, keep)
    improve = (sse0c - sse_full) / sse0c if sse0c > 0 else np.nan
    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15]
    rel = np.abs(10.0 ** x16[idx] - 10.0 ** X_START62[idx]) \
          / np.maximum(np.abs(10.0 ** X_START62[idx]), 1e-12)
    flat = bool(rel.max() > 0.5 and improve < 0.05)
    L6 = bool(len(esc) == 0 and not flat)
    drift8 = float(np.max(np.abs(x16[:8] - X_START62[:8])))
    L8 = bool(drift8 <= 0.5)
    Ls = [L1, L2, L3, L4, L5, L6, L7, L8]
    branch = "全过" if all(Ls) else "未全过"
    n_pass = int(sum(Ls))
    Q = {
        "Q1_L5由灭转过": [hrs, HRS_DATA, L5],
        "Q2_L2保持过": [float(sh / wins['data_sh']), L2],
        "Q3_L1L3L4保持": [L1, L3, L4],
        "Q4_L7全榜过": [r2, L7],
        "Q5_终点读数带": {"KX": float(10.0 ** x16[15]), "VH": float(x16[14]),
                        "Pf2": float(x16[10])},
        "Q6_16槽漂移榜": sorted([(f"x{j}", float(abs(x16[j] - X_START62[j])))
                                for j in range(16)], key=lambda t: -t[1]),
    }
    # —— 落盘值逐项比对 ——
    diffs = []
    _close(v['sse_full'], sse_full, "sse_full", diffs)
    _close(v['sse_tail'], sse_tail, "sse_tail", diffs)
    for tag, a, b in [("L1_peak", v['L1_peak_nA'], pk), ("L2_step", v['L2_step_nA'], sh),
                      ("L3_hold", v['L3_hold_nA'], hold)]:
        _close(a, b, tag, diffs, TOL_TRACE)
    for i in range(4):
        _close(v['hrs'][i], hrs[i], f"hrs[{i}]", diffs, TOL_TRACE)
    for pr in PROTOS:
        _close(v['r2'][pr], r2[pr], f"r2[{pr}]", diffs, TOL_TRACE)
    Lj = [bool(v[k]) for k in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8']]
    if Lj != Ls:
        diffs.append(f"判线布尔不对称：落盘 {Lj} 复算 {Ls}")
    verdict = {
        "branch": branch, "n_pass": n_pass, "L": Ls,
        "sse_full": sse_full, "sse_tail": sse_tail, "r2": r2,
        "pk": pk, "sh": sh, "hold": hold, "tau_fit": tau_fit, "a_fit": a_fit,
        "hrs": hrs, "r40": r40, "r60": r60, "trise": trise,
        "escape": esc, "flat": flat, "drift8_dex_max": drift8,
        "improve_vs_start": improve, "predictions": Q,
        "judge_path": "expm 精确路径（拟合器 CN——数值方法双通道）",
        "replay_diffs": diffs,
    }
    v['judge179'] = verdict
    s = json.dumps(v, ensure_ascii=False, indent=1)
    open(counter_path, 'w', encoding='utf-8').write(s)
    print(f"[judge179] 复算完，写回 {counter_path}", flush=True)
    print(f"[judge179] L1峰={pk:.3f}:{'过' if L1 else '灭'} L2短步比={sh/wins['data_sh']:.2f}:"
          f"{'过' if L2 else '灭'} L3保持={hold:.4f}:{'过' if L3 else '灭'} "
          f"L4慢尾 a={None if a_fit is None else round(a_fit,4)}/"
          f"τ={None if tau_fit is None else round(tau_fit,3)}:{'过' if L4 else '灭'}", flush=True)
    print(f"[judge179] L5hrs:{'过' if L5 else '灭'} L6:{'过' if L6 else '灭'} "
          f"L7R²榜:{'过' if L7 else '灭'}(sa={r2['steady_activation']:.3f} de={r2['deactivation']:.3f} "
          f"sine={r2['sine_wave']:.3f} ap={r2['ap']:.3f}) "
          f"L8机制连续:{'过' if L8 else '灭'}(drift8={drift8:.3f}dex)", flush=True)
    print(f"[judge179] 判决：{branch} 过数={n_pass}/8 复算对照差异={len(diffs)} 起"
          f"{'' if not diffs else '：' + '；'.join(diffs)}", flush=True)
    print(f"[judge179] Q 组对账：{json.dumps(Q, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    judge179(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "counter_179_model62.json"))
