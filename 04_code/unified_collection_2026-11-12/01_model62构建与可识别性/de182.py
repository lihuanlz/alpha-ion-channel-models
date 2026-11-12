# de182.py — 代码182 model62 可识别性谱卡·执行侧（预注册 2026-09-11）
# 硬墙：不重拟合 / 不改架构 / 不改数据 / 不跨协议混噪声 / Fisher 走 expm 验收通道 / 判线不改。
# 用法：python de182.py sanity | group | fisher | dirs | all
import json, math, os, sys, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import de173
import de179
import judge179                       # expm 验收通道（硬墙 5）
from de173 import DT, tail_fit_kernel, steady_ratios
from de174 import load174, windows174, PROTOS
from de179 import EREV, R2_BARS, _hrs_of

OUT = os.path.join(HERE, "model62_identifiability")
os.makedirs(OUT, exist_ok=True)
LOG_SLOTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]   # log10 域
LIN_SLOTS = [10]                                     # pf2 线性域
H_DEX = 1e-3                                         # 预注册 §二.1
NWORK = 24


def f6(x):
    if x is None or not np.isfinite(x):
        return None
    return float(f"{x:.6g}")


def arr6(a):
    return [f6(float(v)) for v in np.asarray(a).ravel()]


def load_x16():
    v = json.load(open(os.path.join(HERE, "counter_181_model62_判官复核件.json"), encoding='utf-8'))
    assert v.get('tag') == '181' and v.get('stage') == 'final' and v.get('cell') == '16713003'
    return np.asarray(v['x16'], float), v


def step_h(x16, j):
    return 1e-3 * abs(x16[j]) if j in LIN_SLOTS else H_DEX


# ---- 并行 expm 仿真引擎（worker 只收电压数组与参数，不触数据盘） ----

def _sim_job(job):
    """job = (key, x16 list, v array, xmax_mult) → (key, 电流数组)"""
    key, xl, v, xmax_mult = job
    import judge179 as JG
    saved = JG.P6X
    if xmax_mult != 1.0:
        JG.P6X = saved.copy(); JG.P6X[0] = saved[0] * xmax_mult
    try:
        m = JG.sim62_expm(np.asarray(v, float), np.asarray(xl, float))
    finally:
        JG.P6X = saved
    return key, m


def run_jobs(jobs, nwork=NWORK):
    with Pool(nwork) as p:
        return dict(p.map(_sim_job, jobs))


def base_eval(x16, data2):
    return {pr: judge179.sim62_expm(np.asarray(data2[pr][0], float), x16) for pr in PROTOS}


def r2_of(m, c, km):
    c_k = np.asarray(c, float)[km]
    return 1.0 - float(((m - np.asarray(c, float))[km] ** 2).sum()) / float(c_k @ c_k)


def sigma_maps(data2, keep, base):
    """预注册 §二.2：keep 连续段分段，段内残差均方；保底=协议汇总×1e-2。"""
    sig = {}
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
        floor = pooled * 1e-2
        s2 = np.full(n, pooled)
        for (a, b) in segs:
            s2[a:b] = max(float((res[a:b] ** 2).mean()), floor)
        sig[pr] = {"s2": s2, "pooled": pooled, "floor": floor, "n_seg": len(segs)}
    return sig


# ---------------- sanity ----------------

def do_sanity():
    x16, v = load_x16()
    data2, keep = load174()
    base = base_eval(x16, data2)
    # 参照 = 正本 judge181 块（expm 验收通道，硬墙 5 同侧；勘误在案：初稿误用拟合侧 CN 的 r2 落盘值）
    ref = v['judge181']['r2']
    ok = True
    for pr in PROTOS:
        r2 = r2_of(base[pr], data2[pr][1], keep[pr])
        d = abs(r2 - ref[pr])
        print(f"[sanity] {pr}: R²={r2:.8f} vs 判官正本 {ref[pr]:.8f} |Δ|={d:.2e}", flush=True)
        ok &= d < 1e-6
    print("[sanity] " + ("逐位恒等：过" if ok else "超差——停卡"), flush=True)
    return ok


# ---------------- 步骤1：乘性标度群排查 ----------------

def do_group():
    x16, _ = load_x16()
    data2, keep = load174()
    base = base_eval(x16, data2)
    # —— 候选 A：g_s 群（20 仿真并行）——
    jobs = []
    for s in [1.01, 1.05, 1.1, 1.5, 2.0]:
        xs = x16.copy()
        ls = math.log10(s)
        xs[0:8] += ls; xs[9] += ls          # P1..P8、pf1 ×s
        xs[8] -= ls; xs[11] -= ls; xs[12] -= ls   # GKr、α_amp、α_τ /s
        for pr in PROTOS:
            jobs.append(((s, pr), list(xs), data2[pr][0], 1.0))
    res = run_jobs(jobs)
    acc = {}
    for (s, pr), m in res.items():
        km = keep[pr]
        d = m[km] - base[pr][km]
        delta = float(np.sqrt(d @ d) / np.sqrt(base[pr][km] @ base[pr][km]))
        np.save(os.path.join(OUT, f"scale_group_A_{pr}_s{s}.npy"), d.astype(np.float32))
        acc.setdefault(pr, {})[str(s)] = {
            "s": f6(s), "Delta_s": f6(delta),
            "逐点残差_npy": f"scale_group_A_{pr}_s{s}.npy",
            "res_max": f6(float(np.abs(d).max())), "res_mean": f6(float(np.abs(d).mean()))}
        print(f"[groupA] s={s} {pr}: Δ={delta:.5f}", flush=True)
    for pr in PROTOS:
        json.dump({"群作用": "P1..P8、pf1 ×s；GKr、α_amp、α_τ /s；pf2 与冻结槽不动",
                   "判线L1": "s=1.1 时 Δ<0.01→近似简并；>0.05→被打破；中间=部分简并",
                   "读数": acc[pr]},
                  open(os.path.join(OUT, f"scale_group_A_{pr}.json"), "w", encoding='utf-8'),
                  ensure_ascii=False, indent=1)
    # —— 候选 B：双指数分量简并（解析，免仿真）——
    do_group_b(x16)
    # —— 候选 C：X 件幅度简并 ——
    do_group_c(x16, data2, keep, base)


def do_group_b(x16=None):
    """候选 B 单跑入口（勘误复算用）：P2 取 10**x16[1]（x16[1] 是 log10 域，勿当线性值）。"""
    if x16 is None:
        x16, _ = load_x16()
    P1, P2 = 10.0 ** x16[0], 10.0 ** x16[1]
    pf1, pf2 = 10.0 ** x16[9], x16[10]
    vv = np.linspace(-120.0, 40.0, 1601)
    rv = pf1 * np.exp(pf2 * vv) / (P1 * np.exp(P2 * vv))
    rec = {"r_v_min": f6(float(rv.min())), "r_v_max": f6(float(rv.max())),
           "r_v_倍数": f6(float(rv.max() / rv.min())),
           "P2": f6(P2), "pf2": f6(pf2),
           "P2_pf2_相对差": f6(abs(P2 - pf2) / max(abs(P2), 1e-30)),
           "判线L2": "倍数<2→简并；>5→可分离；中间登记"}
    json.dump(rec, open(os.path.join(OUT, "double_exp_degeneracy.json"), "w", encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f"[groupB] r(v) 倍数={rv.max()/rv.min():.3f}，|P2−pf2|相对差={rec['P2_pf2_相对差']}", flush=True)


def do_group_c(x16, data2, keep, base):
    # —— 候选 C：X 件幅度简并（XMAX×2 ∧ α_amp/2；4 仿真并行）——
    xc = x16.copy(); xc[11] -= math.log10(2.0)
    jobs = [(pr, list(xc), data2[pr][0], 2.0) for pr in PROTOS]
    res = run_jobs(jobs)
    out = {"变换": "XMAX×2 ∧ α_amp/2（乘积 α_amp·ALPHA·XMAX 不变）", "读数": {}}
    for pr in PROTOS:
        km = keep[pr]
        d = res[pr][km] - base[pr][km]
        out["读数"][pr] = {"Delta": f6(float(np.sqrt(d @ d) / np.sqrt(base[pr][km] @ base[pr][km]))),
                          "res_max": f6(float(np.abs(d).max()))}
        print(f"[groupC] {pr}: Δ={out['读数'][pr]['Delta']}", flush=True)
    json.dump(out, open(os.path.join(OUT, "X_amp_degeneracy.json"), "w", encoding='utf-8'),
              ensure_ascii=False, indent=1)


def _cache_path(kind, key):
    tag = "_".join(str(x) for x in (key if isinstance(key, tuple) else (key,)))
    return os.path.join(OUT, f"cache_{kind}_{tag}.npy")


def run_jobs_cached(kind, jobs, nwork=NWORK):
    """逐件落盘缓存，重跑自动续（imap_unordered 每完成一件即存）。"""
    out, todo = {}, []
    for job in jobs:
        fp = _cache_path(kind, job[0])
        if os.path.exists(fp):
            out[job[0]] = np.load(fp)
        else:
            todo.append(job)
    if todo:
        with Pool(nwork) as p:
            for key, m in p.imap_unordered(_sim_job, todo):
                np.save(_cache_path(kind, key), m)
                out[key] = m
                print(f"  [cache] {kind} {key} 落盘（{len(todo)} 件批次）", flush=True)
    return out


# ---------------- 步骤2：Fisher 矩阵（104 仿真并行） ----------------

def do_fisher():
    x16, _ = load_x16()
    data2, keep = load174()
    bfp = os.path.join(OUT, "cache_base.npy")
    if os.path.exists(bfp):
        base = {pr: np.load(bfp, allow_pickle=True).item()[pr] for pr in PROTOS}
    else:
        base = base_eval(x16, data2)
        np.save(bfp, np.array(base, dtype=object))
    sig = sigma_maps(data2, keep, base)
    jobs = []
    for j in range(13):
        h = step_h(x16, j)
        for sgn in (+1, -1):
            xj = x16.copy(); xj[j] += sgn * h
            for pr in PROTOS:
                jobs.append(((j, sgn, pr), list(xj), data2[pr][0], 1.0))
    res = run_jobs_cached("fisher", jobs)
    Ftot = np.zeros((13, 13))
    slot_names = ["log10P1", "log10P2", "log10P3", "log10P4", "log10P5", "log10P6",
                  "log10P7", "log10P8", "log10GKr", "log10Pf1", "Pf2", "log10α_amp",
                  "log10α_τ"]
    for pr in PROTOS:
        km = keep[pr]
        w = 1.0 / sig[pr]["s2"][km]
        J = np.zeros((int(km.sum()), 13))
        for j in range(13):
            h = step_h(x16, j)
            J[:, j] = (res[(j, 1, pr)][km] - res[(j, -1, pr)][km]) / (2.0 * h)
        WJ = J * w[:, None]
        Fpr = J.T @ WJ
        Ftot += Fpr
        np.save(os.path.join(OUT, f"jacobian_{pr}.npy"), J.astype(np.float32))
        json.dump({"protocol": pr, "sigma2_pooled": f6(sig[pr]["pooled"]),
                   "n_keep": int(km.sum()), "n_seg": sig[pr]["n_seg"],
                   "matrix": [[f6(v) for v in row] for row in Fpr]},
                  open(os.path.join(OUT, f"fisher_per_protocol_{pr}.json"), "w", encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print(f"[fisher] {pr} 协议 Fisher 组装完毕", flush=True)
    json.dump({"槽序": slot_names, "步长": "log域 h=1e-3 dex；pf2 线性域 h=1e-3×|pf2|",
               "通道": "expm 验收通道（硬墙5）", "matrix": [[f6(v) for v in row] for row in Ftot]},
              open(os.path.join(OUT, "fisher_matrix_13x13.json"), "w", encoding='utf-8'),
              ensure_ascii=False, indent=1)
    lam, vec = np.linalg.eigh(Ftot)
    cond = float(lam[-1] / lam[0]) if lam[0] > 0 else float("inf")
    np.save(os.path.join(OUT, "fisher_matrix_raw.npy"), Ftot)
    np.save(os.path.join(OUT, "fisher_eigen_raw.npy"), np.concatenate([lam, vec.ravel()]))
    rec = {"特征值_升序": arr6(lam), "λmax": f6(float(lam[-1])), "λmin": f6(float(lam[0])),
           "条件数χ": f6(cond), "零方向数_阈λmax×1e-3": int((lam < lam[-1] * 1e-3).sum()),
           "特征向量_列对应特征值": [[f6(v) for v in vec[:, k]] for k in range(13)],
           "判线L3": "χ>1e3→显著简并；χ<1e2→基本可识别；中间=中度病态"}
    json.dump(rec, open(os.path.join(OUT, "fisher_eigen.json"), "w", encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f"[fisher] χ={cond:.4g}，零方向数={rec['零方向数_阈λmax×1e-3']}，"
          f"λmax={lam[-1]:.4g} λmin={lam[0]:.4g}", flush=True)


# ---------------- 步骤3：简并方向与不变量 ----------------

def _Iss_nullvec(x16, v):
    """解析稳态：M(v) 零向量 → I_ss(v)=GKr·y3·(v−EREV)（免协议仿真）。"""
    P = 10.0 ** np.asarray(x16[:8], float)
    gkr = 10.0 ** x16[8]
    pf1, pf2 = 10.0 ** x16[9], x16[10]
    P0, P1, P2, P3, P4, P5, P6, P7 = P
    k32 = P4 * math.exp(P5 * v); k23 = P6 * math.exp(-P7 * v)
    k43 = P0 * math.exp(P1 * v) + pf1 * math.exp(pf2 * v); k34 = P2 * math.exp(-P3 * v)
    k12, k21, k41, k14 = k43, k34, k32, k23
    M = np.array([[-(k12 + k14), k21, 0., k41], [k12, -(k21 + k23), k32, 0.],
                  [0., k23, -(k32 + k34), k43], [k14, 0., k34, -(k43 + k41)]])
    _, _, vt = np.linalg.svd(M)
    y = vt[-1]
    y = y / y.sum()
    return gkr * y[2] * (v - EREV)


def do_dirs():
    x16, _ = load_x16()
    data2, keep = load174()
    Ftot = np.load(os.path.join(OUT, "fisher_matrix_raw.npy"))
    lam, vec = np.linalg.eigh(Ftot)
    zero_idx = [k for k in range(13) if lam[k] < lam[-1] * 1e-3]
    slot_names = ["log10P1", "log10P2", "log10P3", "log10P4", "log10P5", "log10P6",
                  "log10P7", "log10P8", "log10GKr", "log10Pf1", "Pf2", "log10α_amp",
                  "log10α_τ"]
    # —— R_ss 口径电压：从 sa 协议实际窗电位读取（de169.steady_ratios 逐字窗）——
    vsa = np.asarray(data2["steady_activation"][0], float)
    v_r40 = (float(vsa[int(45.919 / DT)]), float(vsa[int(29.403 / DT)]))
    v_r60 = (float(vsa[int(54.177 / DT)]), float(vsa[int(29.403 / DT)]))
    # —— R_ss 对 13 槽梯度（解析稳态中心差分，预注册 §二.3）——
    def rss(x, pair):
        return _Iss_nullvec(x, pair[0]) / _Iss_nullvec(x, pair[1])
    grad = {}
    for name, pair in [("r40", v_r40), ("r60", v_r60)]:
        g = np.zeros(13)
        for j in range(13):
            h = step_h(x16, j)
            xp = x16.copy(); xp[j] += h
            xm = x16.copy(); xm[j] -= h
            g[j] = (rss(xp, pair) - rss(xm, pair)) / (2 * h)
        grad[name] = g
    # —— 九细胞检验：引用 judgeA 块已落盘 r40/r60（预注册 §二.3，不重算）——
    cells9 = ["16713003", "16715049", "16708016", "16708060", "16713110",
              "16708118", "16704007", "16704047", "16707014"]
    nine = {}
    for c in cells9:
        fp = "counter_181_model62_判官复核件.json" if c == "16713003" else f"counter_A_model62_{c}.json"
        blk = json.load(open(os.path.join(HERE, fp), encoding='utf-8'))
        ja = blk.get("judgeA", blk)
        r40, r60 = ja["r40"], ja["r60"]
        nine[c] = {"r40": f6(r40), "r60": f6(r60),
                   "过线": bool(0.05 <= r40 <= 0.30 and r60 <= 0.8 * r40)}
    n9 = sum(1 for v in nine.values() if v["过线"])
    # —— 沿零方向的直接扰动验证（每个零方向 ±1% log 域扰动，测功能量响应）——
    jobs = []
    for k in zero_idx:
        u = vec[:, k]
        for sgn in (+1, -1):
            xz = x16.copy(); xz[:13] += sgn * 0.01 * u
            for pr in ("deactivation", "steady_activation"):
                jobs.append(((k, sgn, pr), list(xz), data2[pr][0], 1.0))
    res = run_jobs(jobs) if jobs else {}
    dirs = []
    for k in zero_idx:
        u = vec[:, k]
        dom = sorted(range(13), key=lambda j: -abs(u[j]))[:3]
        rec = {"特征值": f6(float(lam[k])), "λ/λmax": f6(float(lam[k] / lam[-1])),
               "主导参数": [slot_names[j] for j in dom],
               "特征向量": [f6(v) for v in u]}
        for name in ("r40", "r60"):
            g = grad[name]
            rec[f"Rss投影_{name}"] = f6(float(abs(g @ u) / max(np.linalg.norm(g), 1e-30)))
        if (k, 1, "deactivation") in res:
            vd = np.asarray(data2["deactivation"][0], float)
            mp = res[(k, 1, "deactivation")]; mm = res[(k, -1, "deactivation")]
            wins = windows174(vd, np.asarray(data2["deactivation"][1], float))
            ap_, tp_ = tail_fit_kernel(mp, wins['IWT'][0], wins['IWT'][1], DT)
            am_, tm_ = tail_fit_kernel(mm, wins['IWT'][0], wins['IWT'][1], DT)
            msa_p = res[(k, 1, "steady_activation")]; msa_m = res[(k, -1, "steady_activation")]
            hrs_p = _hrs_of(msa_p, vsa); hrs_m = _hrs_of(msa_m, vsa)
            rec["沿向扰动±1%响应"] = {
                "τ_de_相对变化": f6(abs(tp_ - tm_) / max(abs(tp_ + tm_) / 2, 1e-30)),
                "hrs比_相对变化": f6(float(np.max(np.abs(np.array(hrs_p) - np.array(hrs_m))
                                                  / np.maximum((np.abs(np.array(hrs_p))
                                                                + np.abs(np.array(hrs_m))) / 2, 1e-30))))}
        dirs.append(rec)
    out = {"零方向阈": "λ<λmax×1e-3", "λmax": f6(float(lam[-1])), "零方向数": len(zero_idx),
           "Rss口径电压": {"r40窗": v_r40, "r60窗": v_r60},
           "九细胞检验_引自judgeA块": nine, "九细胞过线数": n9,
           "跨细胞贴界登记": {c: json.load(open(os.path.join(HERE, f"counter_A_model62_{c}.json"),
                                                 encoding='utf-8'))["judgeA"]["escape"]
                             for c in cells9[1:]},
           "简并方向": dirs,
           "候选不变量": ["稳态量比 R_ss(r40/r60)", "时间常数比 τ_de/τ_act", "激活半升时间比"]}
    json.dump(out, open(os.path.join(OUT, "degenerate_directions.json"), "w", encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f"[dirs] 零方向数={len(zero_idx)}，九细胞 R_ss 过线={n9}/9", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "sanity"
    t0 = time.time()
    if mode in ("sanity", "all"):
        if not do_sanity():
            sys.exit(1)
    if mode in ("group", "all"):
        do_group()
    if mode in ("fisher", "all"):
        do_fisher()
    if mode in ("dirs", "all"):
        do_dirs()
    print(f"[de182:{mode}] 用时 {time.time()-t0:.0f}s", flush=True)
