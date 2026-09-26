# -*- coding: utf-8 -*-
# 2026-09-13_α模型_激活动力学_包络常数判决.py
# 目的：扩展 α 模型第二块——激活动力学的模态结构与跨细胞恒定性。
#   方法（包络法 + α 方法论：直提、跨细胞 CV）：
#     activation_kinetics_1（0 mV）/ _2（+40 mV）：−120×50ms 全恢复预脉冲
#     -> 测试电压 × 变时长 T -> −120×2.5s 反弹尾；反弹幅度 A(T) ∝ m(T)。
#     v3（冒烟暴露：包络形状不是单指数——初段滞后+跨 30ms~1s 多尺度爬升）：
#       改双分量上升 A(T)=A_max − a1·e^(−T/τ1) − a2·e^(−T/τ2)（τ2≥3τ1，2D 网格
#       确定性 + 线性幅度），提取激活模态结构（τ1,τ2）——这是正压段模态的首次直接测量，
#       与去活阶梯外推的对照本身是判决材料。
#       +40 档 A_max 钉全激活锚（deactivation +50→−120 hook 幅度，脚本内自提）；
#       0mV 档锚不可用（m_ss(0)<m_ss(+50)），A_max 自由，顶边则弃并登记。
#   剔除在案：readme 明示 16708016/16708060/16704007 activation_kinetics 过减漏（6/9 入判）。
#   判词（预注册，跑前钉死）：
#     每电压每 τ：有效 n≥4 且 CV<0.3 判恒定；两 τ 全恒定 => "激活模态=离散常数"；
#     仅其一 => 分级入表；全不 => 另开新卡。τ_act(0)vs(+40) 比较仅登记。
#   QC：A_max>5σ 且 RMS<max(2σ,3%A_max) 且 τ 不顶边 且 点≥5 且 跨≥1 个数量级。
#   对照（噪声自校准，判线不动）：每细胞静默残差块自助 MC 实测 σ_A（低 SNR 端
#         A=0.6nA），对照运行点取全细胞 σ_A 中位（旧版挂列表首细胞=任意；
#         更早的固定 σ=0.04 经 CR 证明 15% 判线信息论不可达）。噪声离群细胞
#         (>2x中位) 打印在案，判词时单独登记。
#         C1 双分量(τ1=30ms,τ2=400ms 等权)×20，两 τ 反演误差中位各<15%；
#         C2 单分量(τ=100ms)不得虚报第二分量（小组分 |a|<8%A_max 或顶边）。
# 运行：python 本文件（全量六细胞）；SMOKE=1 单细胞 16713003 冒烟。
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
EXCLUDE = {"16708016", "16708060", "16704007"}          # readme 过减漏在案
CELLS = ["16713003"] if SMOKE else [c for c in
        ["16704007", "16704047", "16707014", "16708016", "16708060",
         "16708118", "16713003", "16713110", "16715049"] if c not in EXCLUDE]
DT = 1e-4
SKIP_S = 0.0005
TAIL_WIN_S = 0.15
CV_PASS = 0.3
AMP_QC = 5.0
PROTOS = [("activation_kinetics_1_protocol.mat", "activation_kinetics_1", 0),
          ("activation_kinetics_2_protocol.mat", "activation_kinetics_2", 40)]

TR_GRID = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))
TD_GRID = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))
TAU_GRID = np.exp(np.linspace(np.log(0.003), np.log(3.000), 24))   # 3ms–3s


def load_mat(proto, cell, tag):
    V = sio.loadmat(f"{DATA}/data/protocols/{proto}")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/{tag}_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def quiet_sigma(V, I):
    edges = np.where(np.diff(V) != 0)[0] + 1
    sigs = []
    for s in np.split(np.arange(len(V)), edges):
        if len(s) * DT >= 0.15 and abs(float(V[s[0]]) + 80.0) < 2.0:
            seg = I[s[0]:s[-1] + 1].astype(float)
            t = np.arange(len(seg))
            tr = np.polyfit(t, seg, 1)
            sigs.append(float(np.std(seg - np.polyval(tr, t))))
    return float(np.median(sigs)) if sigs else np.nan


def doe_fit(t, y):
    if len(t) < 50:
        return None

    def scan(trg, tdg):
        best = None
        for tr in trg:
            td_ok = tdg[tdg >= 1.5 * tr]
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

    b = scan(TR_GRID, TD_GRID)
    if b is None:
        return None
    _, tr0, td0, _ = b
    b = scan(tr0 * np.exp(np.linspace(-0.35, 0.35, 9)),
             td0 * np.exp(np.linspace(-0.35, 0.35, 9)))
    sse, tr, td, sol = b
    res = y - (sol[0] + sol[1] * (np.exp(-t / td) - np.exp(-t / tr)))
    return dict(tau_r=float(tr), tau_d=float(td), c=float(sol[0]), A=float(sol[1]),
                rms=float(np.sqrt(np.mean(res ** 2))),
                edge=bool(tr <= TR_GRID[0] * 1.02 or td >= TD_GRID[-1] * 0.98))


def quiet_residuals(V, I):
    """静默段残差池（−80/−120 稳态段，去段首 0.1s，线性去趋势）——噪声标定用"""
    edges = np.where(np.diff(V) != 0)[0] + 1
    pool = []
    for s in np.split(np.arange(len(V)), edges):
        v = float(V[s[0]])
        if len(s) * DT >= 0.3 and (abs(v + 80.0) < 2.0 or abs(v + 120.0) < 2.0):
            seg = I[s[0] + int(0.1 / DT):s[-1] + 1].astype(float)
            if len(seg) < 2000:
                continue
            t = np.arange(len(seg))
            tr = np.polyfit(t, seg, 1)
            pool.append(seg - np.polyval(tr, t))
    return np.concatenate(pool) if pool else None


def calibrate_sigma_A(V, I, rng, nrep=40):
    """包络点噪声 σ_A 自校准：块自助重排真实静默残差（保 ACF），在低 SNR 端
    （A=0.6 nA）测 doe_fit 幅度提取噪声。对照噪声以此实测值为准，不用拍脑袋常数。"""
    res = quiet_residuals(V, I)
    if res is None or len(res) < 20000:
        return None
    n = int(TAIL_WIN_S / DT); t = np.arange(n) * DT
    A0, blk = 0.6, 2000
    errs = []
    for _ in range(nrep):
        noise = np.concatenate([res[i:i + blk]
                                for i in rng.integers(0, len(res) - blk, n // blk + 1)])[:n]
        r = doe_fit(t, A0 * (np.exp(-t / 0.030) - np.exp(-t / 0.003)) + noise)
        if r is not None:
            errs.append(r["A"] - A0)
    return float(np.std(errs)) if len(errs) >= nrep // 2 else None


def anchor_amp(cell):
    """全激活锚：deactivation 首个 +50→−120 尾的 DoE 幅度（内向翻正）"""
    V, I = load_mat("deactivation_protocol.mat", cell, "deactivation")
    if I is None:
        return None
    edges = np.where(np.diff(V) != 0)[0] + 1
    segs = np.split(np.arange(len(V)), edges)
    info = [(float(V[s[0]]), int(s[0]), len(s)) for s in segs]
    for k, (v, s0, n) in enumerate(info):
        if v < -100.0 and n * DT > 2.0 and k >= 1 and info[k - 1][0] > 30.0:
            y = I[s0 + int(SKIP_S / DT): s0 + int(0.3 / DT)].astype(float)
            t = np.arange(len(y)) * DT
            r = doe_fit(t, -y)
            if r is not None and not r["edge"] and r["A"] > 0:
                return r["A"]
    return None


def parse_pulses(V, vtest):
    edges = np.where(np.diff(V) != 0)[0] + 1
    segs = np.split(np.arange(len(V)), edges)
    info = [(float(V[s[0]]), int(s[0]), len(s)) for s in segs]
    out = []
    for k, (v, s0, n) in enumerate(info):
        if abs(v - vtest) < 2.0 and n * DT < 5.0 and k + 1 < len(info):
            nv, ns, nn = info[k + 1]
            if nv < -100.0 and nn * DT > 1.0:
                out.append((n * DT, ns))
    return out


def rebound_amp(I, tail_start, sigma):
    s0 = tail_start + int(SKIP_S / DT)
    n = int(TAIL_WIN_S / DT)
    y = I[s0:s0 + n].astype(float)
    t = np.arange(len(y)) * DT
    yy = -y
    r = doe_fit(t, yy)
    if r is not None and abs(r["A"]) > AMP_QC * sigma and not r["edge"]:
        return r["A"], r
    k = max(3, int(0.003 / DT) | 1)
    ys = np.convolve(yy, np.ones(k) / k, mode="same")
    a = float(ys.max())
    return (a if a > 3 * sigma else 0.0), r


def env2_fit(T, A, anchor=None):
    """A(T)=A_max − a1 e^{−T/τ1} − a2 e^{−T/τ2}，物理约束 a1+a2=A_max（m(0)=0：
    −120×50ms 全恢复预脉冲后通道全闭，包络必过原点）=> A=A_max(1−e2)−a1(e1−e2)。
    该约束打掉 A_max↔(a2,τ2) 简并（τ2 在 1s 最长脉冲内不封顶，A_max 主要靠
    原点约束确定）。τ2≥3τ1；anchor 时 A_max 钉锚（只剩 a1+两 τ 三个自由量）。
    变量投影：τ 连续优化（网格多初值 + least_squares），幅度线性 lstsq。"""
    T = np.asarray(T, float); A = np.asarray(A, float)
    lo, hi = float(np.log(TAU_GRID[0])), float(np.log(TAU_GRID[-1]))
    LG3 = float(np.log(3.0))

    def lin(t1, t2):
        e1 = np.exp(-T / t1); e2 = np.exp(-T / t2)
        d = e1 - e2                      # a1 的基：A = A_max(1−e2) − a1·d
        b = 1.0 - e2                     # A_max 的基
        if anchor is None:
            X = np.column_stack([b, -d]); Y = A
        else:
            X = (-d).reshape(-1, 1); Y = A - anchor * b
        sol, *_ = np.linalg.lstsq(X, Y, rcond=None)
        pred = (X @ sol) if anchor is None else (anchor * b + (X @ sol))
        return sol, A - pred

    scale = max(float(np.ptp(A)), 1e-6)

    def resid(p):
        u1, u2 = p
        pen = max(0.0, u1 + LG3 - u2)              # τ2≥3τ1 软约束
        _, r = lin(np.exp(u1), np.exp(u2))
        return np.concatenate([r, [20.0 * scale * pen]])

    cand = []
    for t1g in TAU_GRID:
        for t2g in TAU_GRID[TAU_GRID >= 3.0 * t1g]:
            _, r = lin(t1g, t2g)
            cand.append((float(r @ r), float(t1g), float(t2g)))
    if not cand:
        return None
    cand.sort()
    best = None
    for _, t10, t20 in cand[:5]:
        try:
            f = least_squares(resid, [np.log(t10), np.log(t20)],
                              bounds=([lo, lo], [hi, hi]),
                              xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=4000)
        except Exception:
            continue
        u1, u2 = float(f.x[0]), float(f.x[1])
        if u2 < u1 + LG3 - 1e-6:
            continue
        sol, r = lin(np.exp(u1), np.exp(u2))
        sse = float(r @ r)
        if best is None or sse < best[0]:
            best = (sse, float(np.exp(u1)), float(np.exp(u2)), sol, r)
    if best is None:
        # 连续优化全部滑进罚区 -> 退回满足 τ2≥3τ1 的网格最优点（保底非 None）
        _, tg1, tg2 = cand[0]
        sol, r = lin(tg1, tg2)
        best = (float(r @ r), tg1, tg2, sol, r)
    sse, t1, t2, sol, res = best
    if anchor is None:
        a_max, a1 = float(sol[0]), float(sol[1])
    else:
        a_max, a1 = float(anchor), float(sol[0])
    a2 = a_max - a1
    return dict(tau1=float(t1), tau2=float(t2), A_max=a_max, a1=a1, a2=a2,
                rms=float(np.sqrt(np.mean(res ** 2))),
                edge=bool(t1 <= TAU_GRID[0] * 1.05 or t2 >= TAU_GRID[-1] * 0.95))


def main():
    rng = np.random.default_rng(11)
    print("=" * 78)
    print(" 激活动力学包络常数判决 v3（双分量上升）" + ("（冒烟）" if SMOKE else "（六细胞全量）"))
    print(f" 判线: 每电压每 τ n≥4 且 CV<{CV_PASS} 恒定")
    print("=" * 78, flush=True)

    # ---------- 噪声自校准（全细胞标定；对照取中位运行点，判线不动） ----------
    print("\n[噪声自校准]", flush=True)
    sig_map = {}
    for cell in CELLS:
        Vc, Ic = load_mat(PROTOS[0][0], cell, PROTOS[0][1])
        if Ic is None:
            print(f"  细胞 {cell}: 文件缺失，跳过标定", flush=True)
            continue
        sA = calibrate_sigma_A(Vc, Ic, rng)
        if sA is not None:
            sig_map[cell] = sA
            print(f"  细胞 {cell}: σ_A = {sA:.4f} nA", flush=True)
    if not sig_map:
        print("  全部标定失败 -> 对照无噪声锚，停。")
        return
    sigA = float(np.median(list(sig_map.values())))
    noisy = [c for c, s in sig_map.items() if s > 2.0 * sigA]
    print(f"  对照运行点 σ_A(中位) = {sigA:.4f} nA（n={len(sig_map)}）"
          + (f"；噪声离群(>2x中位)登记: {', '.join(noisy)}" if noisy else ""), flush=True)

    # ---------- 对照 ----------
    print("\n[对照]", flush=True)
    T_c = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    A0c = 4.0
    e1s, e2s = [], []
    for rep in range(20):
        A_c = A0c - (A0c / 2) * np.exp(-T_c / 0.030) - (A0c / 2) * np.exp(-T_c / 0.400) \
            + rng.normal(0, sigA, len(T_c))
        r1 = env2_fit(T_c, A_c)
        got = sorted([r1["tau1"], r1["tau2"]])
        e1s.append(abs(got[0] - 0.030) / 0.030)
        e2s.append(abs(got[1] - 0.400) / 0.400)
    print(f"  C1 双分量(30ms,400ms)×20: τ1 误差中位 {np.median(e1s)*100:.1f}%  "
          f"τ2 {np.median(e2s)*100:.1f}%（判线各<15%）", flush=True)
    collapse_ok = 0
    for rep in range(20):
        A_c2 = A0c - A0c * np.exp(-T_c / 0.100) + rng.normal(0, sigA, len(T_c))
        r2 = env2_fit(T_c, A_c2)
        small = min(abs(r2["a1"]), abs(r2["a2"]))
        if small < 0.08 * r2["A_max"] or r2["edge"]:
            collapse_ok += 1
    print(f"  C2 单分量(100ms)×20: 正确塌缩 {collapse_ok}/20（判线 ≥16）", flush=True)
    if np.median(e1s) > 0.15 or np.median(e2s) > 0.15 or collapse_ok < 16:
        print("  对照未归位 -> 统计量作废，停。")
        return
    print("  对照过。", flush=True)

    # ---------- 真实数据 ----------
    rows = []
    tr_register = []
    for cell in CELLS:
        anch = anchor_amp(cell)
        print(f"  细胞 {cell}: 全激活锚 A_anchor={anch:.2f}nA" if anch else
              f"  细胞 {cell}: 锚提取失败", flush=True)
        for proto, tag, vtest in PROTOS:
            V, I = load_mat(proto, cell, tag)
            if I is None:
                print(f"  细胞 {cell} {tag}: 文件缺失")
                continue
            sigma = quiet_sigma(V, I)
            T, A, trs = [], [], []
            for dur, ts in parse_pulses(V, vtest):
                a, r = rebound_amp(I, ts, sigma)
                T.append(dur); A.append(a)
                if r is not None and not r["edge"]:
                    trs.append(r["tau_r"])
            order = np.argsort(T)
            T = [T[i] for i in order]; A = [A[i] for i in order]
            if len(T) < 5:
                print(f"  细胞 {cell} @{vtest}mV: 脉冲点不足（{len(T)}）", flush=True)
                continue
            use_anchor = anch if vtest == 40 else None
            r = env2_fit(np.array(T), np.array(A), anchor=use_anchor)
            mode = "锚定" if use_anchor else "自由"
            span = max(T) / max(min(T), 1e-9)
            valid = bool(abs(r["A_max"]) > AMP_QC * sigma
                         and r["rms"] < max(2 * sigma, 0.03 * abs(r["A_max"]))
                         and not r["edge"] and span >= 10)
            rows.append(dict(cell=cell, v=vtest, T=T, A=A, sigma=sigma, valid=valid,
                             mode=mode, **r))
            if trs:
                tr_register.append(dict(cell=cell, v=vtest, tau_r_med=float(np.median(trs))))
            print(f"  细胞 {cell} @{vtest}mV [{mode}]: τ1={r['tau1']*1000:.1f}ms "
                  f"τ2={r['tau2']*1000:.0f}ms A_max={r['A_max']:.2f} "
                  f"a1={r['a1']:.2f} a2={r['a2']:.2f} {'过' if valid else '弃'}", flush=True)

    # ---------- 恒定性 ----------
    print("\n" + "-" * 78)
    print("[恒定性] 激活模态跨细胞（仅 QC 有效）")
    summ = {}
    verdicts = []
    for vv in (0, 40):
        rs = [r for r in rows if r["v"] == vv and r["valid"]]
        if len(rs) < 4:
            print(f"  {vv:>4}mV: 有效 n={len(rs)}<4，数据不足")
            continue
        t1 = np.array([r["tau1"] for r in rs]); t2 = np.array([r["tau2"] for r in rs])
        cv1 = float(t1.std(ddof=1) / t1.mean()); cv2 = float(t2.std(ddof=1) / t2.mean())
        ok = cv1 < CV_PASS and cv2 < CV_PASS
        verdicts.append(ok)
        summ[vv] = dict(n=len(rs), t1_med=float(np.median(t1)), cv1=cv1,
                        t2_med=float(np.median(t2)), cv2=cv2, ok=ok)
        print(f"  {vv:>4}mV: n={len(rs)}  τ1 {np.median(t1)*1000:.1f}ms CV {cv1:.2f} | "
              f"τ2 {np.median(t2)*1000:.0f}ms CV {cv2:.2f}  {'恒定' if ok else '不恒定'}")
    if tr_register:
        trv = np.array([r["tau_r_med"] for r in tr_register])
        print(f"[登记] 尾 DoE τ_r(−120) 中位 {np.median(trv)*1000:.2f}ms（hook 值 3.04ms，互查）")

    # ---------- 总判词 ----------
    print("\n" + "=" * 78)
    print(" 总判词：")
    print(f"  恒定电压档 {sum(verdicts)}/{len(summ)}")
    if len(summ) == 2 and all(verdicts):
        final = "激活模态=离散常数（0/+40 两档双 τ 全过）-> 第二块入表，可进失活块"
    elif any(verdicts):
        final = "部分档位恒定：恒定档入表，其余登记（扩展模型分段可行）"
    else:
        final = "激活模态非离散常数（或数据不足）-> 第二块前提不成立，另开新卡"
    print("  " + final)
    print("=" * 78)

    # ---------- 图 ----------
    ncell = len(CELLS)
    fig, axes = plt.subplots(ncell, 2, figsize=(11, 2.0 * ncell), squeeze=False)
    for i, cell in enumerate(CELLS):
        for j, vv in enumerate((0, 40)):
            ax = axes[i][j]
            r = next((x for x in rows if x["cell"] == cell and x["v"] == vv), None)
            if r is None:
                ax.axis("off"); continue
            T = np.array(r["T"]); A = np.array(r["A"])
            ax.plot(T * 1000, A, "o", ms=4, color="black", alpha=0.7)
            tt = np.exp(np.linspace(np.log(min(T) * 0.8), np.log(max(T) * 1.2), 200))
            ax.plot(tt * 1000, r["A_max"] - r["a1"] * np.exp(-tt / r["tau1"])
                    - r["a2"] * np.exp(-tt / r["tau2"]), "-", lw=1.3, color="crimson")
            if r["mode"] == "锚定":
                ax.axhline(r["A_max"], lw=0.8, ls="--", color="navy", alpha=0.6)
            ax.set_xscale("log")
            ax.set_title(f"{cell[-4:]} @{vv}mV [{r['mode']}] τ1={r['tau1']*1000:.0f} τ2={r['tau2']*1000:.0f}ms "
                         f"{'过' if r['valid'] else '弃'}", fontsize=8)
            ax.set_xlabel("脉冲时长 ms"); ax.grid(alpha=0.3, which="both")
    fig.suptitle("激活包络双分量拟合 A(T)=A_max−a1e^{−T/τ1}−a2e^{−T/τ2}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fp = os.path.join(HERE, f"2026-09-13_α模型_激活包络{'_冒烟' if SMOKE else ''}.png")
    fig.savefig(fp, dpi=120, bbox_inches="tight")
    plt.close(fig)

    out = dict(summary=summ, final=final, sigma_A=sig_map, sigma_A_med=sigA,
               noisy_cells=noisy,
               rows=[{k: v for k, v in r.items()} for r in rows],
               tr_register=tr_register, fig=fp)
    fj = os.path.join(HERE, f"2026-09-13_α模型_激活包络{'_冒烟' if SMOKE else ''}_结果.json")
    json.dump(out, open(fj, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n  图落盘: {fp}")
    print(f"  结果落盘: {fj}")


if __name__ == "__main__":
    main()
