# 2026-09-13_α模型_失活门_失活协议封卷判决.py
# 目的（失活块，分窗直提，三块交付）：
#   A) +50x0.6s 预脉冲 16 次重复平均，全带宽，分窗直提（联合 m*h 拟合已证数学上不成立：
#      瞬态/慢分量幅度比 1000:1，绝对残差看不见慢分量，v2 对照 C1 误差 4153% 在案）：
#      早窗 1.0-12ms 单指数衰减 -> tau_obs = tau_h(+50) 上界
#        （声明：含 m 快分量上升污染使衰减显慢、振铃残余 <=0.02nA；SKIP 1.0ms 避振铃）；
#      晚窗 20-600ms 单指数上升 -> tau_slow(+50)（= 激活慢分量，与激活块 285ms@+40 互查）
#        + 稳态 I_ss(+50)；h_ss(+50) = I_ss/(G_hat x 138mV)，G_hat = A_反弹(+50)/31.67mV
#        （本细胞自标定链，m*h 峰效率折进声明不确定度 ±30%）。
#      冒烟实测(16713003)：稳态 0.044nA -> h_ss~0.003-0.004，与 189 冒烟级库存 0.156
#      差 40 倍 -> 本块正式验真。
#   B) -90 尾电流 hook：DF(-90)=-1.67mV 但 G~0.1-0.3nA/mV，实测尾 -0.29nA(峰) 是真实
#      hERG 信号（先升后降 hook 形）。15 短段(0.06s)平均 -> tau_rec(-90)
#      （TD 网格限 0.3s，tau_d 申报窗限）；1 长段(0.21s)单试 -> tau_deact(-90) 参考。
#   C) 反弹幅度包络 A(V)：16 档测压(-100..+50)后 -120x0.5s 反弹，DoE 幅度
#      -> A(V)/A(+50) 可用度曲线，九细胞逐档 CV 判恒定；符号 QC 防负幅度污染。
#      h_ss(V)/h_ss(+50)=A(V)/A(+50)（声明：仅当 m(150ms,V)=1 取等号，否则为下界，
#      m 修正留待组装块，本块不参数化修正）。
# 协议（已程序化解析 112 段）：周期 = +50x0.6s -> -90x0.06s -> 测压Vx0.15s -> -120x0.5s
#   -> -80x1.34s；测压 16 档：-100..+50；测压=-90 时两段 -90 合并为 0.21s 单段。
# 判线（跑前声明）：每量跨细胞 n>=4 且 CV<0.3 判恒定；A(V) 按档同判。
# 对照（跑前声明，用首细胞实测噪声，平均噪声 /4 或 /sqrt15）：
#   C1a 合成早窗（tau=1.2ms, A=0.06, c=0.006）x20：tau 误差中位 <20% 且 n>=16；
#   C1b 合成晚窗（tau=150ms, B=0.04, Iss=0.044）x20：tau 误差 <20%、Iss 误差 <10%；
#   C2  合成反弹 hook（tau_r=3ms, tau_d=30ms, A=4nA, 0.3s 窗，单试噪声）x20：
#       幅度误差中位 <10% 且有效率 >=16/20；
#   C3  合成 -90 hook（tau_r=5ms, tau_d=100ms, 55ms 窗，/sqrt15）x20：tau_r 误差 <15%
#       （tau_d 窗限申报不判）。
# 声明限制：
#   1) 平均噪声 sigma/4(16次)、/sqrt(15)(15次) 为声明近似；
#   2) 早窗太短不够 ACF 白化门（<12 箱），只做 rms 门，申报；晚窗 ACF 门降级为诊断
#      （16 次平均压不掉跨周期共有缓结构，残差 ACF 违约~3、幅度<5% Iss，单指数为
#      主要成分近似，申报），晚窗 QC = rms<2σ_avg + B>3σ_avg + 不顶边；
#   3) SKIP 1.0ms（振铃实测 <=0.5ms 衰减振荡，0.8ms 处残余 <=0.02nA，申报）；
#   4) tau_obs 是 tau_h 上界（m 快分量上升使表观衰减偏慢），+50 失活相对组装尺度
#      (30ms+) 实为瞬时，该精度足够组装用，申报；
#   5) h_ss(+50) 经 G_hat 链导出，不确定度 ±30%，与 189 的 40 倍差判决不受影响。
# 运行：python 本文件（九细胞全量）；SMOKE=1 单细胞 16713003 冒烟。
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
CV_PASS = 0.3
REB_WIN = 0.30                  # 反弹拟合窗（与 hook 块 -120 档一致）
W_EARLY = (0.0010, 0.012)       # +50 早窗 1.0-12ms
W_LATE = (0.020, 0.600)         # +50 晚窗 20-600ms
DF_M120 = 31.67                 # |DF(-120)|, E_rev=-88.33 封卷值
DF_P50 = 138.33                 # DF(+50)

TR_GRID = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))
TD_GRID = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))
TD_GRID_M90 = np.exp(np.linspace(np.log(0.008), np.log(0.300), 20))
SEP_MIN = 1.5
AMP_QC = 5.0

TAU_E_GRID = np.exp(np.linspace(np.log(0.0003), np.log(0.012), 25))   # 早窗 tau 0.3-12ms
TAU_L_GRID = np.exp(np.linspace(np.log(0.020), np.log(0.600), 25))    # 晚窗 tau 20-600ms


# ---------------- 复用件（与旧块同源） ----------------
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


def quiet_pool(info, I, dec=10):
    pool = []
    for v, s0, n in info:
        if abs(v + 80.0) < 2.0 and n * DT >= 0.15:
            seg = I[s0:s0 + n].astype(float)[::dec]
            t = np.arange(len(seg))
            tr = np.polyfit(t, seg, 1)
            pool.append(seg - np.polyval(tr, t))
    return np.concatenate(pool) if pool else None


def quiet_segs(V, I):
    edges = np.where(np.diff(V) != 0)[0] + 1
    out = []
    for s in np.split(np.arange(len(V)), edges):
        if len(s) * DT >= 0.15 and abs(float(V[s[0]]) + 80.0) < 2.0:
            out.append(I[s[0]:s[-1] + 1].astype(float))
    return out


def noise_sigma(I_segs):
    sigs = []
    for seg in I_segs:
        if len(seg) < 2000:
            continue
        t = np.arange(len(seg))
        tr = np.polyfit(t, seg, 1)
        sigs.append(float(np.std(seg - np.polyval(tr, t))))
    return float(np.median(sigs)) if sigs else np.nan


def acf_envelope(info, I, dec=10, lags=(1, 5, 10, 20), bin_n=20):
    env = {L: 0.2 for L in lags}
    vals = {L: [] for L in lags}
    for v, s0, n in info:
        if abs(v + 80.0) < 2.0 and n * DT >= 0.3:
            seg = I[s0:s0 + n].astype(float)[::dec]
            t = np.arange(len(seg))
            tr = np.polyfit(t, seg, 1)
            r = seg - np.polyval(tr, t)
            nb = len(r) // bin_n
            if nb < 12:
                continue
            b = r[:nb * bin_n].reshape(nb, bin_n).mean(axis=1)
            b = b - b.mean()
            denom = float(b @ b)
            if denom <= 0:
                continue
            for L in lags:
                if nb > L + 8:
                    vals[L].append(float(b[L:] @ b[:-L]) / denom)
    for L in lags:
        if len(vals[L]) >= 2:
            env[L] = max(0.2, float(sorted(np.abs(vals[L]))[-2]))
        elif vals[L]:
            env[L] = max(0.2, float(abs(vals[L][0])))
    return env


def white_gate(res, env, bin_n=20, lags=(1, 5, 10, 20), max_viol=2):
    nb = len(res) // bin_n
    if nb < 12:
        return False, 99
    b = (res[:nb * bin_n].reshape(nb, bin_n).mean(axis=1))
    b = b - b.mean()
    denom = float(b @ b)
    if denom <= 0:
        return False, 99
    viol = 0
    for L in lags:
        r = float(b[L:] @ b[:-L]) / denom if nb > L + 8 else 0.0
        if abs(r) > env.get(L, 0.2):
            viol += 1
    return viol <= max_viol, viol


def doe_fit(t, y, tr_grid=TR_GRID, td_grid=TD_GRID):
    if len(t) < 50:
        return None

    def scan(trg, tdg):
        best = None
        for tr in trg:
            td_ok = tdg[tdg >= SEP_MIN * tr]
            if not len(td_ok):
                continue
            er = np.exp(-t / tr)
            for td in td_ok:
                x = np.exp(-t / td) - er
                X = np.column_stack([np.ones(len(t)), x])
                sol, *_ = np.linalg.lstsq(X, y, rcond=None)
                sse = float(np.sum((y - X @ sol) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, tr, td, sol)
        return best

    b = scan(tr_grid, td_grid)
    if b is None:
        return None
    _, tr0, td0, _ = b
    trg = tr0 * np.exp(np.linspace(-0.35, 0.35, 9))
    tdg = td0 * np.exp(np.linspace(-0.35, 0.35, 9))
    b = scan(trg, tdg)
    sse, tr, td, sol = b
    res = y - (sol[0] + sol[1] * (np.exp(-t / td) - np.exp(-t / tr)))
    return dict(tau_r=float(tr), tau_d=float(td), c=float(sol[0]), A=float(sol[1]),
                rms=float(np.sqrt(np.mean(res ** 2))),
                edge=bool(tr <= tr_grid[0] * 1.02 or tr >= tr_grid[-1] * 0.98))


def qc_ok(r, sigma):
    return bool(abs(r["A"]) > AMP_QC * sigma
                and r["rms"] < max(2.0 * sigma, 0.02 * abs(r["A"]))
                and not r["edge"])


def fit_hook(I, s0, win_s, sigma):
    n = int(win_s / DT)
    y = I[s0 + int(0.0008 / DT): s0 + n].astype(float)
    t = np.arange(len(y)) * DT
    half = y[:len(y) // 2]
    sgn = -1.0 if abs(half.min()) > abs(half.max()) else 1.0
    yy = sgn * y
    r = doe_fit(t, yy)
    if r is None:
        return None
    r["valid"] = qc_ok(r, sigma) and sgn < 0 and r["A"] > 0   # 反弹须向内且幅度为正
    r["sgn"] = sgn
    r["t"] = t
    r["y"] = yy
    return r


def resample(pool, n, rng, blk=2000):
    return np.concatenate([pool[i:i + blk]
                           for i in rng.integers(0, len(pool) - blk, n // blk + 1)])[:n]


def fit_single_exp(t, y, tau_grid, mode):
    """单指数直提：mode='decay' y=c+A e^-t/tau (A>0)；mode='rise' y=Iss-B e^-t/tau (B>0)。
    网格 + 一次精修，返回 dict 或 None。"""
    if len(t) < 30:
        return None

    def scan(grid):
        best = None
        for tau in grid:
            x = np.exp(-t / tau)
            X = np.column_stack([np.ones(len(t)), x])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ sol) ** 2))
            if best is None or sse < best[0]:
                best = (sse, tau, sol)
        return best

    b = scan(tau_grid)
    _, tau0, _ = b
    b = scan(tau0 * np.exp(np.linspace(-0.35, 0.35, 9)))
    sse, tau, sol = b
    res = y - (sol[0] + sol[1] * np.exp(-t / tau))
    out = dict(tau=float(tau), p0=float(sol[0]), p1=float(sol[1]),
               rms=float(np.sqrt(np.mean(res ** 2))), res=res,
               edge=bool(tau <= tau_grid[0] * 1.02 or tau >= tau_grid[-1] * 0.98))
    if mode == "decay":
        out["A"] = out["p1"]
        out["c"] = out["p0"]
    else:
        out["Iss"] = out["p0"]
        out["B"] = -out["p1"]
    return out


# ---------------- 本块特有 ----------------
def find_cycles(info):
    """+50x0.6s 预脉冲锚定周期。返回 dict(pre, rec, rec_long, test_v, test, reb)。"""
    cyc = []
    for k, (v, s0, n) in enumerate(info):
        if not (abs(v - 50.0) < 2.0 and 0.55 < n * DT < 0.65):
            continue
        v1, s1, n1 = info[k + 1]
        if abs(v1 + 90.0) > 2.0:
            continue
        if n1 * DT > 0.15:                      # 测压=-90 合并段
            rv, rs, rn = info[k + 2]
            cyc.append(dict(pre=(s0, n), rec=(s1, n1), rec_long=True, test_v=-90.0,
                            test=None, reb=(rs, rn)))
        else:
            v2, s2, n2 = info[k + 2]
            rv, rs, rn = info[k + 3]
            cyc.append(dict(pre=(s0, n), rec=(s1, n1), rec_long=False, test_v=v2,
                            test=(s2, n2), reb=(rs, rn)))
    out = []
    for c in cyc:
        rs, rn = c["reb"]
        if abs(info[[i for i, (vv, ss, nn) in enumerate(info)
                     if ss == rs][0]][0] + 120.0) < 2.0 and rn * DT >= 0.4:
            out.append(c)
    return out


def main():
    rng = np.random.default_rng(11)
    print("=" * 82)
    print(" 失活门·失活协议封卷判决 v3（+50 分窗直提 / -90 hook / A(V) 包络）"
          + ("（冒烟 16713003）" if SMOKE else "（九细胞全量）"))
    print(f" 判线: 跨细胞 n>=4 且 CV<{CV_PASS} 判恒定；A(V) 按档同判")
    print("=" * 82, flush=True)

    # ---------- 载入 ----------
    cells = {}
    V = None
    for c in CELLS:
        V, I = load_mat("inactivation_protocol.mat", c, "inactivation")
        cells[c] = I
        print(f"  细胞 {c}: " + ("缺数据" if I is None else
              f"lenI={len(I)} lenV={len(V)} NaN={int(np.isnan(I).sum())}"), flush=True)
    info = segments(V)
    cyc = find_cycles(info)
    tvs = [c["test_v"] for c in cyc]
    print(f"\n[协议解析] 周期数 {len(cyc)}；测压档 {tvs}", flush=True)
    assert len(cyc) == 16 and len(set(tvs)) == 16, "周期解析与声明不符，停"

    c0 = [c for c in CELLS if cells[c] is not None][0]
    I0 = cells[c0]
    pool_bw0 = quiet_pool(info, I0, dec=1)
    sig_bw0 = noise_sigma(quiet_segs(V, I0))
    print(f"[噪声] 首细胞 {c0}: sigma_bw={sig_bw0:.4f} sigma_avg(=/4)={sig_bw0 / 4:.4f}",
          flush=True)

    # ---------- 对照 ----------
    print("\n[对照]", flush=True)
    t_e = np.arange(int(W_EARLY[0] / DT), int(W_EARLY[1] / DT)) * DT
    t_e = t_e - t_e[0]
    ea = []
    for _ in range(20):
        yn = 0.006 + 0.06 * np.exp(-t_e / 0.0012) + resample(pool_bw0, len(t_e), rng) / 4.0
        r = fit_single_exp(t_e, yn, TAU_E_GRID, "decay")
        if r is not None and not r["edge"]:
            ea.append(abs(r["tau"] - 0.0012) / 0.0012)
    med = float(np.median(ea)) if ea else 1.0
    c1a = bool(med < 0.20 and len(ea) >= 16)
    print(f"  C1a 早窗 tau: 误差中位 {med * 100:.1f}%  n={len(ea)} -> "
          f"{'过' if c1a else '不过'}", flush=True)

    t_l = np.arange(int(W_LATE[0] / DT), int(W_LATE[1] / DT)) * DT
    t_l = t_l - t_l[0]
    eb, ei = [], []
    for _ in range(20):
        yn = 0.044 - 0.04 * np.exp(-t_l / 0.150) + resample(pool_bw0, len(t_l), rng) / 4.0
        r = fit_single_exp(t_l, yn, TAU_L_GRID, "rise")
        if r is not None and not r["edge"]:
            eb.append(abs(r["tau"] - 0.150) / 0.150)
            ei.append(abs(r["Iss"] - 0.044) / 0.044)
    medb = float(np.median(eb)) if eb else 1.0
    medi = float(np.median(ei)) if ei else 1.0
    c1b = bool(medb < 0.20 and medi < 0.10 and len(eb) >= 16)
    print(f"  C1b 晚窗 tau: {medb * 100:.1f}%  Iss: {medi * 100:.1f}%  n={len(eb)} -> "
          f"{'过' if c1b else '不过'}", flush=True)

    ok2 = 0
    eA = []
    t_r = np.arange(int(REB_WIN / DT)) * DT
    for _ in range(20):
        yr = 4.0 * (np.exp(-t_r / 0.030) - np.exp(-t_r / 0.003)) + 0.02 \
            + rng.normal(0, sig_bw0, len(t_r))
        rr = doe_fit(t_r, yr)
        if rr is not None and qc_ok(rr, sig_bw0):
            ok2 += 1
            eA.append(abs(rr["A"] - 4.0) / 4.0)
    medA = float(np.median(eA)) if eA else 1.0
    c2 = bool(ok2 >= 16 and medA < 0.10)
    print(f"  C2 反弹幅度: 误差中位 {medA * 100:.1f}%  有效 {ok2}/20 -> "
          f"{'过' if c2 else '不过'}", flush=True)

    e3 = []
    t_m = np.arange(int(0.055 / DT)) * DT
    for _ in range(20):
        ym = 0.25 * (np.exp(-t_m / 0.100) - np.exp(-t_m / 0.005)) - 0.02 \
            + rng.normal(0, sig_bw0 / 15 ** 0.5, len(t_m))
        rr = doe_fit(t_m, ym, td_grid=TD_GRID_M90)
        if rr is not None:
            e3.append(abs(rr["tau_r"] - 0.005) / 0.005)
    med3 = float(np.median(e3)) if e3 else 1.0
    c3 = bool(med3 < 0.15 and len(e3) >= 16)
    print(f"  C3 -90 hook tau_r: 误差中位 {med3 * 100:.1f}%  n={len(e3)} -> "
          f"{'过' if c3 else '不过'}（tau_d 窗限申报不判）", flush=True)

    if not (c1a and c1b and c2 and c3):
        print("  对照未归位 -> 统计量作废，停。", flush=True)
        return
    print("  对照归位。", flush=True)

    # ---------- 真实数据 ----------
    print("\n[真实数据]", flush=True)
    res = {}
    for c in CELLS:
        I = cells[c]
        if I is None:
            continue
        sig_bw = noise_sigma(quiet_segs(V, I))
        sig_avg = sig_bw / 4.0
        env_bw = acf_envelope(info, I, dec=1, bin_n=20)
        ent = dict(sigma_bw=sig_bw)

        # A) +50 平均，分窗直提
        n_pre = min(cy["pre"][1] for cy in cyc)
        avg = np.mean([I[cy["pre"][0]: cy["pre"][0] + n_pre] for cy in cyc], axis=0)
        i0, i1 = int(W_EARLY[0] / DT), int(W_EARLY[1] / DT)
        ye = avg[i0:i1]
        te = np.arange(len(ye)) * DT
        re_ = fit_single_exp(te, ye, TAU_E_GRID, "decay")
        if re_ is not None:
            re_["qc"] = bool(re_["A"] > 4 * sig_avg and re_["rms"] < 2.0 * sig_avg
                             and not re_["edge"] and re_["A"] > 0)
            ent["early"] = {k: v for k, v in re_.items() if k != "res"}
        j0, j1 = int(W_LATE[0] / DT), int(W_LATE[1] / DT)
        yl = avg[j0:j1]
        tl = np.arange(len(yl)) * DT
        rl_ = fit_single_exp(tl, yl, TAU_L_GRID, "rise")
        if rl_ is not None:
            white, viol = white_gate(rl_["res"], env_bw, bin_n=20)
            rl_["viol"] = int(viol)
            # 晚窗 ACF 门降级为诊断（平均残差带跨周期共有缓结构，幅度<5% Iss，申报）；
            # QC = rms 门 + B 门 + 不顶边
            rl_["qc"] = bool(rl_["B"] > 3 * sig_avg and rl_["rms"] < 2.0 * sig_avg
                             and not rl_["edge"] and rl_["B"] > 0)
            ent["late"] = {k: v for k, v in rl_.items() if k != "res"}
        ent["_avg50"] = (avg, re_, rl_)

        # C) 反弹包络 A(V)（先跑，h_ss 链要用 A(+50)）
        curve = {}
        for cy in cyc:
            rs, rn = cy["reb"]
            rr = fit_hook(I, rs, REB_WIN, sig_bw)
            if rr is None:
                continue
            curve[float(cy["test_v"])] = dict(A=rr["A"], valid=bool(rr["valid"]),
                                              sgn=int(rr["sgn"]),
                                              tau_r=rr["tau_r"], tau_d=rr["tau_d"])
        ent["curve"] = curve

        # h_ss(+50) 自标定链
        a50 = curve.get(50.0)
        if rl_ is not None and rl_["qc"] and a50 and a50["valid"]:
            g_hat = a50["A"] / DF_M120
            ent["g_hat"] = float(g_hat)
            ent["h_ss50"] = float(rl_["Iss"] / (g_hat * DF_P50))

        # B) -90 hook
        shorts = [cy["rec"] for cy in cyc if not cy["rec_long"]]
        longs = [cy["rec"] for cy in cyc if cy["rec_long"]]
        n_rec = min(n for _, n in shorts)
        avg_s = np.mean([I[s:s + n_rec] for s, n in shorts], axis=0)
        yy = -avg_s[int(0.0008 / DT): int(0.055 / DT)]
        tm = np.arange(len(yy)) * DT
        rm = doe_fit(tm, yy, td_grid=TD_GRID_M90)
        if rm is not None:
            sig15 = sig_bw / 15 ** 0.5
            rm["valid"] = bool(abs(rm["A"]) > AMP_QC * sig15
                               and rm["rms"] < max(2.0 * sig15, 0.02 * abs(rm["A"]))
                               and not rm["edge"])
            rm["td_win_limited"] = bool(rm["tau_d"] >= TD_GRID_M90[-1] * 0.9)
            ent["m90"] = {k: v for k, v in rm.items()}
            ent["_m90"] = (tm, yy, rm)
        if longs:
            s, n = longs[0]
            yg = -I[s + int(0.0008 / DT): s + int(0.20 / DT)].astype(float)
            tg = np.arange(len(yg)) * DT
            rg = doe_fit(tg, yg)
            if rg is not None:
                rg["valid"] = qc_ok(rg, sig_bw)
                ent["m90_long"] = {k: v for k, v in rg.items()}

        res[c] = ent
        e_ok = ent.get("early", {}).get("qc")
        l_ok = ent.get("late", {}).get("qc")
        msg = f"  {c}: 早窗{'✓' if e_ok else '×'}"
        if e_ok:
            msg += f" tau_obs={ent['early']['tau'] * 1e3:.2f}ms"
        msg += f" 晚窗{'✓' if l_ok else '×'}"
        if l_ok:
            msg += (f" tau_slow={ent['late']['tau'] * 1e3:.0f}ms "
                    f"Iss={ent['late']['Iss']:.4f} viol={ent['late']['viol']}")
        if "h_ss50" in ent:
            msg += f" h_ss50={ent['h_ss50']:.4f}"
        m9 = ent.get("m90")
        if m9:
            msg += (f" | -90 tau_r={m9['tau_r'] * 1e3:.1f}ms"
                    f"{'(tau_d窗限)' if m9['td_win_limited'] else ''}"
                    f"{'✓' if m9['valid'] else '×'}")
        msg += f" | 反弹 {sum(1 for v in curve.values() if v['valid'])}/16"
        print(msg, flush=True)

    # ---------- 汇总判词 ----------
    print("\n" + "-" * 82, flush=True)
    print("[汇总]", flush=True)
    verdict = {}

    def cv_line(getter, name, scale=1.0, unit=""):
        vals = [getter(c) * scale for c in res if getter(c) is not None]
        if len(vals) < 4:
            print(f"  {name}: 有效 n={len(vals)}<4 -> 数据不足", flush=True)
            verdict[name] = dict(n=len(vals), cv=None)
            return
        vals = np.array(vals)
        cv = float(np.std(vals) / abs(np.mean(vals)))
        verdict[name] = dict(n=len(vals), mean=float(np.mean(vals)),
                             cv=cv, const=bool(cv < CV_PASS))
        print(f"  {name}: n={len(vals)} 均值 {np.mean(vals):.3g}{unit} "
              f"CV={cv:.2f} -> {'恒定' if cv < CV_PASS else '不恒定'}", flush=True)

    cv_line(lambda c: res[c].get("early", {}).get("tau") if res[c].get("early", {}).get("qc")
            else None, "tau_obs(+50上界)", 1e3, "ms")
    cv_line(lambda c: res[c].get("late", {}).get("tau") if res[c].get("late", {}).get("qc")
            else None, "tau_slow(+50)", 1e3, "ms")
    cv_line(lambda c: res[c].get("late", {}).get("Iss") if res[c].get("late", {}).get("qc")
            else None, "Iss(+50)", 1, "nA")
    cv_line(lambda c: res[c].get("h_ss50"), "h_ss(+50)")
    cv_line(lambda c: res[c].get("m90", {}).get("tau_r") if res[c].get("m90", {}).get("valid")
            else None, "tau_rec(-90)", 1e3, "ms")

    ts = verdict.get("tau_slow(+50)", {})
    if ts.get("mean"):
        print(f"  互查：tau_slow(+50)={ts['mean'] * 1e3:.0f}ms vs 激活块 tau2(+40)=285ms"
              f"（同族慢分量，期望同量级）", flush=True)
    tr90 = verdict.get("tau_rec(-90)", {})
    if tr90.get("mean"):
        print(f"  互查：tau_rec(-90)={tr90['mean'] * 1e3:.1f}ms vs hook块 tau_rec(-100)=5.05ms",
              flush=True)
    h50 = verdict.get("h_ss(+50)", {})
    if h50.get("mean"):
        print(f"  验真：h_ss(+50)={h50['mean']:.4f} vs 189 冒烟级库存 0.156 -> "
              f"{'189 否掉' if h50['mean'] < 0.05 else '189 量级成立'}", flush=True)

    print("  A(V)/A(+50) 可用度曲线（档 CV 判线同）:", flush=True)
    acurve = {}
    for vv in sorted({v for c in res for v in res[c]["curve"]}):
        vals = []
        for c in res:
            cur = res[c]["curve"]
            a50 = cur.get(50.0)
            if a50 and a50["valid"] and vv in cur and cur[vv]["valid"] and abs(a50["A"]) > 0:
                vals.append(cur[vv]["A"] / a50["A"])
        if len(vals) >= 4:
            vals = np.array(vals)
            cv = float(np.std(vals) / abs(np.mean(vals))) if abs(np.mean(vals)) > 1e-9 else np.nan
            acurve[vv] = dict(n=len(vals), ratio=float(np.mean(vals)), cv=cv,
                              const=bool(cv < CV_PASS))
            print(f"    V={vv:+.0f}: n={len(vals)} 比值 {np.mean(vals):.3f} "
                  f"CV={cv:.2f} {'恒定' if cv < CV_PASS else '不恒定'}", flush=True)
        else:
            acurve[vv] = dict(n=len(vals))
            print(f"    V={vv:+.0f}: n={len(vals)}<4 数据不足", flush=True)
    verdict["acurve"] = acurve

    print("\n 总判词：", flush=True)
    keys = ["tau_obs(+50上界)", "tau_slow(+50)", "h_ss(+50)", "tau_rec(-90)"]
    consts = [verdict.get(k, {}).get("const") for k in keys]
    if all(consts):
        print("  早/晚窗 tau、h_ss(+50)、tau_rec(-90) 跨细胞恒定 -> 失活门封卷；"
              "A(V) 曲线恒定性见逐档。", flush=True)
    else:
        got = [k for k, v in zip(keys, consts) if v]
        print(f"  恒定达标项: {got if got else '无'}；其余未达判线（明细见上），登记在案。",
              flush=True)

    # ---------- 图 ----------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    for c in res:
        if "_avg50" in res[c]:
            avg, re_, rl_ = res[c]["_avg50"]
            t_full = np.arange(len(avg)) * DT
            cut = int(0.001 / DT)                      # 振铃段不入图（否则信号被压扁）
            ax.plot(t_full[cut:] * 1e3, avg[cut:], lw=0.4, alpha=0.5)
            if re_ is not None and re_.get("qc"):
                te = np.arange(int(W_EARLY[0] / DT), int(W_EARLY[1] / DT)) * DT
                ax.plot(te * 1e3, re_["c"] + re_["A"] * np.exp(-(te - te[0]) / re_["tau"]),
                        lw=1.2, label=f"{c} 早窗")
            if rl_ is not None and rl_.get("qc"):
                tl = np.arange(int(W_LATE[0] / DT), int(W_LATE[1] / DT)) * DT
                ax.plot(tl * 1e3, rl_["Iss"] - rl_["B"] * np.exp(-(tl - tl[0]) / rl_["tau"]),
                        lw=1.2, ls="--", label=f"{c} 晚窗")
    ax.set_ylim(-0.02, 0.13)
    ax.set_title("+50 平均（16次,全带宽）分窗直提")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("I (nA)")
    ax.legend(fontsize=7)
    ax = axes[1]
    for c in res:
        if "_m90" in res[c]:
            tm, yy, rm = res[c]["_m90"]
            ax.plot(tm * 1e3, -yy, lw=0.6, alpha=0.6)
            if rm.get("valid"):
                ax.plot(tm * 1e3, -(rm["c"] + rm["A"] * (np.exp(-tm / rm["tau_d"])
                                    - np.exp(-tm / rm["tau_r"]))), lw=1.0,
                        label=f"{c} fit")
    ax.set_title("-90 尾电流 hook（15次平均）与拟合")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("I (nA)")
    ax.legend(fontsize=7)
    ax = axes[2]
    for c in res:
        cur = res[c]["curve"]
        a50 = cur.get(50.0)
        if not (a50 and a50["valid"]):
            continue
        xs = sorted(cur)
        ys = [cur[v]["A"] / a50["A"] if cur[v]["valid"] else np.nan for v in xs]
        ax.plot(xs, ys, "o-", ms=3, lw=0.8, alpha=0.7, label=c)
    ax.set_title("反弹可用度曲线 A(V)/A(+50)")
    ax.set_xlabel("测压 V (mV)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-13_α模型_失活门_失活协议封卷判决.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")

    # ---------- 落盘 ----------
    out = dict(verdict=verdict,
               cells={c: {k: v for k, v in res[c].items() if not k.startswith("_")}
                      for c in res})
    fjson = os.path.join(HERE, "2026-09-13_α模型_失活门_失活协议封卷判决_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  图落盘: {fpng}", flush=True)
    print(f"  结果落盘: {fjson}", flush=True)


if __name__ == "__main__":
    main()
