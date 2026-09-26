# -*- coding: utf-8 -*-
# 2026-09-13_α模型_激活时程_稳态激活上升沿判决.py
# 目的：激活块补救测量（B 路线）——steady_activation 测试脉冲上升沿直接测激活时程。
#   背景：包络法两档有效 n=3<4 数据不足。本协议九细胞全有、7 档电压（-60..+60）、5-6s 长窗。
#   段结构（已程序化核实）：每周期 -120x50ms 预脉冲 -> -80x0.2s -> 测试档 5-6s
#   -> -40x1s 尾 -> -120x0.5s -> -80x1.508s。测试开始时 h(0)=1（双保险全恢复），
#   m(0)=慢层残留（m0 自由，预期 0-0.15）。
#   h 处理（三轮冒烟定稿，跑前钉死）：模型显式含 h，但 h 是否可识别由贡献门判决——
#     h_contrib = max_t |G·m(t)·(1-h_ss)·e^{-t/τ_h}| > 8σ_dec 才报告 h_ss/τ_h。
#     冒烟实测：h-dip 在多数档落于 m≈0 窗（贡献 <8σ，不可识别，不报）；
#     +60 档 h-bump 贡献 >8σ（可识别，门开，τ1 无偏）。无 h 版拟合在 +60 会把
#     h-bump 吃成 m0 虚高 -> τ1 系统性偏快 22%（偏差非散布，散布仅 4.4%）。
#   模型（预注册）：
#     I(t) = G_eff · [m0 + (1-m0)·R(t)] · [h_ss + (1-h_ss)·e^{-t/τ_h}]
#     R(t)  = 1 - s1·e^{-t/τ1} - (1-s1)·e^{-t/τ2}     （R(0)=0, R(∞)=1；s1<0 => S 形）
#     参数: G_eff(线性投影), m0∈[0,0.5], s1∈[-5,1), τ1∈[5ms,10s], τ2≥3τ1,
#           h_ss∈[0,1], τ_h∈[1ms,0.5s]。基线=测试前 -80x0.2s 段末 100ms 均值（先减）。
#     拟合: 1ms 抽取，去段首 5ms；粗网格(τ1xτ2xτ_h)线性评估取 top3
#           -> least_squares 变量投影（τ2≥3τ1 罚约束）。
#   冒烟数据实测（重要，模型据此定稿）：
#     ① 各档起始有大尖峰（瞬时电流 = G·m0·h(0)·DF）：复位序列清不掉慢层，
#        m0≈0.13-0.15 非零（m0 自由的设计被证实必要）；② 正压 h 衰减巨大
#        （+40 摆幅 ~2nA）可测；③ 稳态 h_ss 在 5s 尺度极深（0mV 仅 ~0.02），
#        m 骑乘幅度 = G·h_ss·(1-m0) 仅 0.05-0.2 nA -> 行 QC 必须按 m 骑乘
#        可视幅度 >4σ 卡（不是 G>8σ：G 外推自近乎全失活的地板，会虚高）；
#     ④ -60 档含 NaN 点（掩码处理，有效点<80% 则弃）。
#   QC（预注册）：① τ1/τ2/m0 不顶边（τ_h 顶边只强制 h 不报告，不废行）、
#     ② RMS/σ_dec<1.5、③ 噪声包络门白化（20ms 分箱 ACF 滞后 1/5/10/20，违约≤2，
#     包络=本细胞静默段每阶第二大幅值、下限 0.2——v5 已立方法）、
#     ④ m 骑乘幅度 m_ride = G·h_ss_fit·(1-m0) > 4σ_dec（4σ 工作点由 C6 认证）。
#   判线（预注册）：每电压档：有效 n≥4 且 CV(τ1)、CV(τ2) 均<0.3 => 该档激活模态
#     =离散常数；部分档过 => 分级入表；全不过 => 另开新卡。
#     h_ss/τ_h 仅在贡献门开放档报告（为失活块登记），本卡不判 h。
#   对照（预注册；噪声=全细胞静默残差中位细胞块自助重排，保 ACF）：
#     C1 +40样(G=4,m0=0.05,s1=-0.6,τ1=80ms,τ2=400ms,h_ss=0.16,τ_h=12ms) x20：
#       τ1/τ2 反演误差中位各<15%；
#     C2 单分量m+h(s1=0,τ2=100ms,h同C1) x20：正确塌缩(min(|s1|,|1-s1|)<0.08 或顶边)≥16；
#     C3 深h不可见档(G=1.5,s1=-0.5,τ1=300ms,τ2=1.5s,h_ss=0.44,τ_h=13.4ms) x20：
#       τ1/τ2 误差中位各<15% 且 h 报告被门拒 ≥16/20；
#     C4 紧分离+60样(G=12,m0=0.05,s1=-0.5,τ1=40ms,τ2=160ms,h_ss=0.156,τ_h=8ms) x20：
#       τ1/τ2 误差中位各<20%（紧分离档判线 20%，跑前声明）；
#     C5 无h(h_ss=1，余同C1) x20：h 报告被门拒 ≥16/20 且 τ1 误差中位<15%。
#     C6 低SNR档（G=1.3，m 骑乘 ~0.2nA ≈ 4σ，余同C1）x20：有效≥16 且
#       τ1/τ2 误差中位各<15%（认证 4σ 行 QC 工作点）。
# 运行：python 本文件（九细胞全量）；SMOKE=1 单细胞 16713003 冒烟。
import os
import json
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
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
DEC = 10                       # 抽取到 1ms
SKIP_MS = 5                    # 去段首 5ms
CV_PASS = 0.3
H_GATE = 8.0                   # h 贡献门（单位 σ_dec）
V_STEPS = [-60, -40, -20, 0, 20, 40, 60]
LN3 = float(np.log(3.0))
BND = dict(m0=(0.0, 0.5), s1=(-5.0, 0.999), u1=(np.log(0.005), np.log(10.0)),
           u2=(np.log(0.005), np.log(10.0)), h=(0.0, 1.0), uh=(np.log(0.001), np.log(0.5)))


def load_cell(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/steady_activation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/steady_activation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def segments(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    return [(float(V[s[0]]), int(s[0]), len(s)) for s in np.split(np.arange(len(V)), edges)]


def find_steps(info):
    """测试档段：时长≥4s，且前两段为 -80x0.2s 与 -120x50ms 预脉冲"""
    out = []
    for k, (v, s0, n) in enumerate(info):
        if n * DT >= 4.0 and any(abs(v - vv) < 2.0 for vv in V_STEPS) and k >= 2:
            pv, ps, pn = info[k - 1]
            ppv, pps, ppn = info[k - 2]
            if abs(pv + 80.0) < 2.0 and 0.1 < pn * DT < 0.4 and abs(ppv + 120.0) < 2.0:
                out.append(dict(v=v, s0=s0, n=n, base_s=ps, base_n=pn))
    return out


def quiet_pool(info, I):
    """-80 静默段残差（1ms 抽取，线性去趋势）"""
    pool = []
    for v, s0, n in info:
        if abs(v + 80.0) < 2.0 and n * DT >= 0.15:
            seg = I[s0:s0 + n].astype(float)[::DEC]
            t = np.arange(len(seg))
            tr = np.polyfit(t, seg, 1)
            pool.append(seg - np.polyval(tr, t))
    return np.concatenate(pool) if pool else None


def acf_envelope(info, I, lags=(1, 5, 10, 20), bin_n=20):
    """每细胞静默段 ACF 包络（1ms 抽取、bin_n 分箱、每阶第二大、下限 0.2）——v5 方法"""
    env = {L: 0.2 for L in lags}
    vals = {L: [] for L in lags}
    for v, s0, n in info:
        if abs(v + 80.0) < 2.0 and n * DT >= 0.3:
            seg = I[s0:s0 + n].astype(float)[::DEC]
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


def model_f(t, m0, s1, t1, t2, h, th):
    R = 1.0 - s1 * np.exp(-t / t1) - (1.0 - s1) * np.exp(-t / t2)
    hh = h + (1.0 - h) * np.exp(-t / th)
    return (m0 + (1.0 - m0) * R) * hh


def m_part(t, m0, s1, t1, t2):
    R = 1.0 - s1 * np.exp(-t / t1) - (1.0 - s1) * np.exp(-t / t2)
    return m0 + (1.0 - m0) * R


def fit_step(t, y):
    """变量投影：G_eff 线性；w 参数化 τ2=3τ1·e^w（w≥0，约束严格）；
    网格初值 -> least_squares。返回参数字典或 None。"""
    scale = max(float(np.ptp(y)), 1e-6)

    def unpack(p):
        m0, s1, u1, w, h, uh = p
        return m0, s1, np.exp(u1), np.exp(u1 + LN3 + w), h, np.exp(uh)

    def proj(p):
        m0, s1, t1, t2, h, th = unpack(p)
        f = model_f(t, m0, s1, t1, t2, h, th)
        num = float(f @ y); den = float(f @ f)
        if den <= 0:
            return None, None
        G = num / den
        return G, y - G * f

    def resid(p):
        G, r = proj(p)
        if G is None or G <= 0:
            r = np.ones_like(y) * scale
        return r

    g1 = [0.015, 0.05, 0.15, 0.5, 1.5]
    g2 = [0.1, 0.3, 0.9, 2.5, 6.0]
    gh = [0.003, 0.010, 0.030, 0.100]
    cand = []
    for a in g1:
        for b in g2:
            if b < 3.0 * a:
                continue
            for hh in gh:
                p0 = np.array([0.05, -0.5, np.log(a), np.log(b / a) - LN3, 0.5, np.log(hh)])
                G, r = proj(p0)
                if G is not None and G > 0:
                    cand.append((float(r @ r), p0))
    if not cand:
        return None
    cand.sort(key=lambda z: z[0])
    lo = np.array([BND['m0'][0], BND['s1'][0], BND['u1'][0], 0.0, BND['h'][0], BND['uh'][0]])
    hi = np.array([BND['m0'][1], BND['s1'][1], BND['u1'][1],
                   BND['u2'][1] - BND['u1'][0] - LN3, BND['h'][1], BND['uh'][1]])
    best = None
    for _, p0 in cand[:3]:
        p0 = np.clip(p0, lo, hi)
        try:
            fq = least_squares(resid, p0, bounds=(lo, hi),
                               xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=3000)
        except Exception:
            continue
        G, r = proj(fq.x)
        if G is None or G <= 0:
            continue
        sse = float(r @ r)
        if best is None or sse < best[0]:
            best = (sse, fq.x, G, r)
    if best is None:
        return None
    sse, p, G, res = best
    m0, s1, t1, t2, h, th = unpack(p)
    m0, s1, t1, t2, h, th = float(m0), float(s1), float(t1), float(t2), float(h), float(th)
    u2 = float(p[2] + LN3 + p[3])
    edge_m = bool(p[2] <= BND['u1'][0] + 0.05 or u2 >= BND['u2'][1] - 0.05
                  or m0 >= BND['m0'][1] - 1e-3)
    h_edge = bool(p[5] <= BND['uh'][0] + 0.05 or p[5] >= BND['uh'][1] - 0.05)
    contrib = float(np.max(np.abs(G * m_part(t, m0, s1, t1, t2) * (1.0 - h) * np.exp(-t / th))))
    return dict(G=G, m0=m0, s1=s1, tau1=t1, tau2=t2, h_ss=h, tau_h=th,
                contrib=contrib, h_edge=h_edge,
                rms=float(np.sqrt(np.mean(res ** 2))), res=res, edge=edge_m)


def synth(t, G, m0, s1, t1, t2, h, th, noise):
    R = 1.0 - s1 * np.exp(-t / t1) - (1.0 - s1) * np.exp(-t / t2)
    hh = h + (1.0 - h) * np.exp(-t / th)
    return G * (m0 + (1.0 - m0) * R) * hh + noise


def resample(pool, n, rng, blk=2000):
    return np.concatenate([pool[i:i + blk]
                           for i in rng.integers(0, len(pool) - blk, n // blk + 1)])[:n]


def main():
    rng = np.random.default_rng(23)
    print("=" * 80)
    print(" 激活时程·稳态激活上升沿判决（I=G·m(t)h(t)，h 贡献门报告制）"
          + ("（冒烟）" if SMOKE else "（九细胞全量）"))
    print(f" 判线: 每电压有效 n>=4 且 CV(tau1),CV(tau2)<{CV_PASS}")
    print("=" * 80, flush=True)

    # ---------- 噪声池与包络（全细胞） ----------
    pools, envs, sigs = {}, {}, {}
    steps_by_cell = {}
    for cell in CELLS:
        V, I = load_cell(cell)
        if I is None:
            print(f"  细胞 {cell}: 文件缺失", flush=True)
            continue
        info = segments(V)
        pools[cell] = quiet_pool(info, I)
        envs[cell] = acf_envelope(info, I)
        sigs[cell] = float(np.std(pools[cell])) if pools[cell] is not None else np.nan
        steps_by_cell[cell] = find_steps(info)
        print(f"  细胞 {cell}: σ_dec={sigs[cell]:.4f}nA  测试档 "
              f"{[s['v'] for s in steps_by_cell[cell]]}", flush=True)
    ok_cells = [c for c in CELLS if c in pools and pools[c] is not None]
    if not ok_cells:
        print("  无可用静默段，停。")
        return
    med_cell = sorted(ok_cells, key=lambda c: sigs[c])[len(ok_cells) // 2]
    pool_med = pools[med_cell]
    sig_med = sigs[med_cell]
    print(f"  对照噪声细胞（中位）: {med_cell}  σ_dec={sig_med:.4f} nA", flush=True)

    # ---------- 对照 ----------
    print("\n[对照]", flush=True)
    T5 = np.arange(int(SKIP_MS), 5000, 1) * 1e-3
    n5 = len(T5)

    def ctrl(truth, nrep=20):
        outs = []
        for _ in range(nrep):
            y = synth(T5, noise=resample(pool_med, n5, rng), **truth)
            outs.append(fit_step(T5, y))
        return outs

    def h_rejected(r):
        return r["h_edge"] or r["contrib"] <= H_GATE * sig_med

    t1e, t2e = [], []
    for r in ctrl(dict(G=4.0, m0=0.05, s1=-0.6, t1=0.080, t2=0.400, h=0.16, th=0.012)):
        if r is not None and not r["edge"]:
            t1e.append(abs(r["tau1"] - 0.080) / 0.080)
            t2e.append(abs(r["tau2"] - 0.400) / 0.400)
    c1 = len(t1e) >= 16 and np.median(t1e) < 0.15 and np.median(t2e) < 0.15
    print(f"  C1 +40样 x20: 有效{len(t1e)}  τ1误差中位 "
          f"{np.median(t1e) * 100:.1f}% τ2 {np.median(t2e) * 100:.1f}%（<15%） {'过' if c1 else '不过'}", flush=True)
    col = 0
    for r in ctrl(dict(G=4.0, m0=0.05, s1=0.0, t1=0.030, t2=0.100, h=0.16, th=0.012)):
        if r is not None and (min(abs(r["s1"]), abs(1.0 - r["s1"])) < 0.08 or r["edge"]):
            col += 1
    c2 = col >= 16
    print(f"  C2 单分量m+h x20: 正确塌缩 {col}/20（>=16） {'过' if c2 else '不过'}", flush=True)
    t1g, t2g, rej3 = [], [], 0
    for r in ctrl(dict(G=1.5, m0=0.05, s1=-0.5, t1=0.300, t2=1.500, h=0.44, th=0.0134)):
        if r is not None:
            if h_rejected(r):
                rej3 += 1
            if not r["edge"]:
                t1g.append(abs(r["tau1"] - 0.300) / 0.300)
                t2g.append(abs(r["tau2"] - 1.500) / 1.500)
    c3 = len(t1g) >= 16 and np.median(t1g) < 0.15 and np.median(t2g) < 0.15 and rej3 >= 16
    print(f"  C3 深h不可见档 x20: 有效{len(t1g)}  τ1误差中位 {np.median(t1g) * 100:.1f}% "
          f"τ2 {np.median(t2g) * 100:.1f}%（<15%） h报告被门拒 {rej3}/20（>=16） {'过' if c3 else '不过'}", flush=True)
    t1f, t2f = [], []
    for r in ctrl(dict(G=12.0, m0=0.05, s1=-0.5, t1=0.040, t2=0.160, h=0.156, th=0.008)):
        if r is not None and not r["edge"]:
            t1f.append(abs(r["tau1"] - 0.040) / 0.040)
            t2f.append(abs(r["tau2"] - 0.160) / 0.160)
    c4 = len(t1f) >= 16 and np.median(t1f) < 0.20 and np.median(t2f) < 0.20
    print(f"  C4 紧分离+60样 x20: 有效{len(t1f)}  τ1误差中位 "
          f"{np.median(t1f) * 100:.1f}% τ2 {np.median(t2f) * 100:.1f}%（<20%） {'过' if c4 else '不过'}", flush=True)
    rej5, t1h = 0, []
    for r in ctrl(dict(G=4.0, m0=0.05, s1=-0.6, t1=0.080, t2=0.400, h=1.0, th=0.012)):
        if r is not None:
            if h_rejected(r):
                rej5 += 1
            if not r["edge"]:
                t1h.append(abs(r["tau1"] - 0.080) / 0.080)
    c5 = rej5 >= 16 and len(t1h) >= 16 and np.median(t1h) < 0.15
    print(f"  C5 无h x20: h报告被门拒 {rej5}/20（>=16） τ1误差中位 "
          f"{np.median(t1h) * 100:.1f}%（<15%） {'过' if c5 else '不过'}", flush=True)
    t1x, t2x = [], []
    for r in ctrl(dict(G=1.3, m0=0.05, s1=-0.6, t1=0.080, t2=0.400, h=0.16, th=0.012)):
        if r is not None and not r["edge"]:
            t1x.append(abs(r["tau1"] - 0.080) / 0.080)
            t2x.append(abs(r["tau2"] - 0.400) / 0.400)
    c6 = len(t1x) >= 16 and np.median(t1x) < 0.15 and np.median(t2x) < 0.15
    print(f"  C6 低SNR档(m骑乘~4σ) x20: 有效{len(t1x)}  τ1误差中位 "
          f"{np.median(t1x) * 100:.1f}% τ2 {np.median(t2x) * 100:.1f}%（<15%） {'过' if c6 else '不过'}", flush=True)
    if not (c1 and c2 and c3 and c4 and c5 and c6):
        print("  对照未归位 -> 统计量作废，停。")
        return
    print("  对照过。", flush=True)

    # ---------- 真实数据 ----------
    rows = []
    for cell in ok_cells:
        V, I = load_cell(cell)
        for st in steps_by_cell[cell]:
            s0, n = st["s0"], st["n"]
            base = float(np.mean(I[st["base_s"] + st["base_n"] - int(0.1 / DT):
                                   st["base_s"] + st["base_n"]]))
            idx = np.arange(int(SKIP_MS / 1000 / DT), n, DEC)
            t = (idx - idx[0]) * DT
            y = I[s0 + idx].astype(float) - base
            mask = ~np.isnan(y)
            if mask.mean() < 0.8:
                print(f"  细胞 {cell} @{st['v']:>4}mV: NaN 占比 {1 - mask.mean():.0%}，弃", flush=True)
                continue
            t, y = t[mask], y[mask]
            r = fit_step(t, y)
            if r is None:
                print(f"  细胞 {cell} @{st['v']:>4}mV: 拟合失败", flush=True)
                continue
            white, viol = white_gate(r["res"], envs[cell])
            snr = r["G"] / sigs[cell]
            m_ride = r["G"] * r["h_ss"] * (1.0 - r["m0"])
            h_rep = bool((not r["h_edge"]) and r["contrib"] > H_GATE * sigs[cell])
            valid = bool((not r["edge"]) and r["rms"] < 1.5 * sigs[cell]
                         and white and m_ride > 4.0 * sigs[cell])
            rows.append(dict(cell=cell, v=st["v"], valid=valid, viol=viol, rms=r["rms"],
                             snr=snr, m_ride=m_ride, G=r["G"], m0=r["m0"], s1=r["s1"], tau1=r["tau1"],
                             tau2=r["tau2"], h_report=h_rep, h_ss=(r["h_ss"] if h_rep else None),
                             tau_h=(r["tau_h"] if h_rep else None), contrib=r["contrib"],
                             t=t[::20].tolist(), y=y[::20].tolist(),
                             fit=(r["G"] * model_f(t, r["m0"], r["s1"], r["tau1"],
                                                   r["tau2"], r["h_ss"], r["tau_h"]))[::20].tolist()))
            hs = f"h_ss={r['h_ss']:.2f} τ_h={r['tau_h'] * 1000:.1f}ms" if h_rep else "h=不报告"
            print(f"  细胞 {cell} @{st['v']:>4}mV: τ1={r['tau1'] * 1000:.0f}ms "
                  f"τ2={r['tau2'] * 1000:.0f}ms s1={r['s1']:.2f} m0={r['m0']:.3f} "
                  f"{hs} ride={m_ride:.3f}({m_ride / sigs[cell]:.1f}σ) "
                  f"rms/σ={r['rms'] / sigs[cell]:.2f} 违约{viol} {'过' if valid else '弃'}", flush=True)

    # ---------- 恒定性 ----------
    print("\n" + "-" * 80)
    print("[恒定性] 激活模态跨细胞（仅 QC 有效）")
    summ = {}
    for vv in V_STEPS:
        rs = [r for r in rows if r["v"] == vv and r["valid"]]
        if len(rs) < 4:
            summ[vv] = dict(n=len(rs), ok=False, note="数据不足")
            print(f"  {vv:>4}mV: 有效 n={len(rs)}<4，数据不足")
            continue
        t1 = np.array([r["tau1"] for r in rs]); t2 = np.array([r["tau2"] for r in rs])
        cv1 = float(t1.std(ddof=1) / t1.mean()); cv2 = float(t2.std(ddof=1) / t2.mean())
        ok = cv1 < CV_PASS and cv2 < CV_PASS
        summ[vv] = dict(n=len(rs), t1_med=float(np.median(t1)), cv1=cv1,
                        t2_med=float(np.median(t2)), cv2=cv2, ok=ok)
        print(f"  {vv:>4}mV: n={len(rs)}  τ1 {np.median(t1) * 1000:.0f}ms CV {cv1:.2f} | "
              f"τ2 {np.median(t2) * 1000:.0f}ms CV {cv2:.2f}  {'恒定' if ok else '不恒定'}")

    # ---------- h 登记 ----------
    print("\n[登记] h 贡献门开放档（为失活块备用，本卡不判）")
    for r in rows:
        if r["h_report"]:
            print(f"  {r['cell']} @{r['v']:>4}mV: h_ss={r['h_ss']:.3f} "
                  f"τ_h={r['tau_h'] * 1000:.2f}ms contrib={r['contrib']:.3f}nA")

    print("\n" + "=" * 80)
    print(" 总判词：")
    n_ok = sum(1 for s in summ.values() if s.get("ok"))
    if n_ok == len(V_STEPS):
        final = "激活模态=离散常数（7 档全过）-> 第二块入表"
    elif n_ok > 0:
        final = f"部分档位恒定（{n_ok}/{len(V_STEPS)}）：恒定档入表，其余登记"
    else:
        final = "激活模态非离散常数（或数据不足）-> 另开新卡"
    print("  " + final)
    print("=" * 80)

    # ---------- 图 ----------
    ncell = len(ok_cells)
    fig, axes = plt.subplots(ncell, len(V_STEPS), figsize=(2.4 * len(V_STEPS), 1.8 * ncell),
                             squeeze=False)
    for i, cell in enumerate(ok_cells):
        for j, vv in enumerate(V_STEPS):
            ax = axes[i][j]
            r = next((x for x in rows if x["cell"] == cell and x["v"] == vv), None)
            if r is None:
                ax.axis("off"); continue
            ax.plot(np.array(r["t"]) * 1000, r["y"], ".", ms=1.5, color="gray")
            ax.plot(np.array(r["t"]) * 1000, r["fit"], "-", lw=1.0, color="crimson")
            ax.set_title(f"{cell[-4:]}@{vv} τ1={r['tau1'] * 1000:.0f} τ2={r['tau2'] * 1000:.0f}"
                         f"{'过' if r['valid'] else '弃'}", fontsize=6)
            ax.set_xscale("log")
            ax.tick_params(labelsize=5)
    fig.suptitle("稳态激活上升沿直测 I=G·m(t)h(t)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fp1 = os.path.join(HERE, f"2026-09-13_α模型_激活时程{'_冒烟' if SMOKE else ''}.png")
    fig.savefig(fp1, dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for r in rows:
        if r["valid"]:
            ax2.plot(r["v"], r["tau1"] * 1000, "o", ms=5, color="navy", alpha=0.7)
            ax2.plot(r["v"], r["tau2"] * 1000, "s", ms=5, color="crimson", alpha=0.7)
    ax2.set_yscale("log"); ax2.set_xlabel("mV"); ax2.set_ylabel("τ (ms)")
    ax2.set_title("激活 τ 阶梯：○τ1 □τ2（仅 QC 有效）")
    ax2.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    fp2 = os.path.join(HERE, f"2026-09-13_α模型_激活时程_阶梯{'_冒烟' if SMOKE else ''}.png")
    fig2.savefig(fp2, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    out = dict(summary=summ, final=final, sigma_dec=sigs, med_cell=med_cell,
               rows=[{k: v for k, v in r.items() if k not in ("t", "y", "fit")} for r in rows],
               figs=[fp1, fp2])
    fj = os.path.join(HERE, f"2026-09-13_α模型_激活时程{'_冒烟' if SMOKE else ''}_结果.json")
    json.dump(out, open(fj, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n  图落盘: {fp1}")
    print(f"  图落盘: {fp2}")
    print(f"  结果落盘: {fj}")


if __name__ == "__main__":
    main()
