# -*- coding: utf-8 -*-
# 2026-09-13_α模型_前向验证_尾巴与sine.py
# 目的：α 模型（τ(V) 阶梯 + a(V) 幅度，2026-09-13 封卷）的前向验证，两部分：
#   Part A 尾巴重建：封卷 τ + 每细胞线性幅度，重建 deactivation 长尾，原始 vs 拟合叠图。
#           （与幅度表提取同口径，确定性复现：白化数应=28/33，作为回归检查）
#   Part B sine 预测：把封卷表接成弛豫动力系统，对独立协议 sine_wave 做零拟合预测。
#           动力学（电流单位，免 G_Kr 锚）：
#             dJ_i/dt = ( f_i(V)·g(V)·(V-E_rev) − J_i ) / τ_i(V)，I = Σ J_i
#             g(V)   = steady_activation 每细胞实测 I_ss(V)/(V-E_rev)（查表插值）
#             f_i(V) = 封卷 w 表插值（-70..-40），外推规则见下
#             τ_late(V) = 阶梯四点对数线性外推（封顶 500s）；τ_f=0.234s、τ_m=1.009s 常数
#                        （仅有 -70/-60 实测，常数化是"恒定"封卷结论的直用）
#           对照零模型：I_null = g(V)·(V-E_rev)（瞬态稳态，无动力学）。
#           判据：动态模型 RMS 必须显著小于零模型，否则动力学不挣钱。
#   已知外推假设（照实写进输出）：
#     A1: τ_late 在 V>-40 外推为冻结（>100s），正压快激活/失活不在模型内 -> 正压段预期失败；
#     A2: f_i(V) 在 V>-40 外推（w_s->0.95）是趋势外推，无实测锚；
#     A3: g(V) 的 -60/-40 两点台阶 5-6s 未完全稳态（τ_late 4.4s/24s），负压 g 值偏低估；
#     A4: τ_f/τ_m 全电压常数化，仅 -70/-60 有实测。
# 运行：python 本文件        （全量九细胞）
#       SMOKE=1 python 本文件（单细胞 16713003 冒烟）
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
CELLS_ALL = ["16704007", "16704047", "16707014", "16708016", "16708060",
             "16708118", "16713003", "16713110", "16715049"]
CELLS = ["16713003"] if SMOKE else CELLS_ALL
DT = 1e-4
DS = 10
SKIP_MS = 5.0
MASK_NA = 0.072
DEBOUNCE = 20
BASE_MS = 200.0
T0_S = 0.05
BIN_S = 0.02
WIN_MIN_S = 2.0
H_ACF = 20
VIOL_TOL = 2
ENV = None
E_REV = -80.0

# ---- 封卷表值（2026-09-13 阶梯表/幅度表 JSON，勿改） ----
TAU_F = {-70: 0.19105, -60: 0.27645, -50: 0.6, -40: 0.6}      # -50/-40 共享精修顶边（仅 PartA 复现用）
TAU_M = {-70: 0.81270, -60: 1.20549, -50: 1.8, -40: 0.8}
TAU_L = {-70: 2.05505, -60: 4.41849, -50: 10.14073, -40: 24.0539}
GEARS = [-70, -60, -50, -40]
# sine 动力学的常数时间尺度（-70/-60 实测均值）
TF_C = 0.5 * (TAU_F[-70] + TAU_F[-60])   # 0.234 s
TM_C = 0.5 * (TAU_M[-70] + TAU_M[-60])   # 1.009 s
# f_i(V) 插值结点（w 表封卷值；-40 档负 w_f 截 0 后归一；-30 起外推封顶）
F_V = np.array([-70.0, -60.0, -50.0, -40.0, -30.0])
F_F = np.array([0.294, 0.144, 0.079, 0.000, 0.000])
F_M = np.array([0.486, 0.439, 0.379, 0.275, 0.050])
F_S = np.array([0.219, 0.418, 0.543, 0.725, 0.950])
# τ_late(V) 对数线性外推
_pl = np.polyfit(np.array(GEARS, float), np.log([TAU_L[v] for v in GEARS]), 1)


def tau_late(V):
    return np.clip(np.exp(_pl[0] * np.asarray(V, float) + _pl[1]), 0.01, 500.0)


# ================= Part A：尾巴重建（口径与幅度表提取脚本一致） =================

def load_deact(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/deactivation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/deactivation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def find_tails(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    segs = np.split(np.arange(len(V)), edges)
    info = [(float(V[s[0]]), int(s[0]), len(s)) for s in segs]
    tails = []
    for k, (v, s0, n) in enumerate(info):
        dur = n * DT
        if dur < 2.0 or k < 2:
            continue
        pv, ps, pn = info[k - 1]
        hv, hs, hn = info[k - 2]
        if pv > -10.0 and 0.2 < pn * DT < 5.0:
            b0 = hs + hn - int(BASE_MS / 1000.0 / DT)
            tails.append(dict(v=v, start=s0, n=n, base_idx=(b0, hs + hn)))
    return tails


def noise_segments(V, I):
    edges = np.where(np.diff(V) != 0)[0] + 1
    out = []
    for s in np.split(np.arange(len(V)), edges):
        if len(s) * DT >= 2.0 and abs(float(V[s[0]]) + 80.0) < 2.0:
            out.append(I[s[0]:s[-1] + 1:DS].astype(float))
    return out


def bin_means(y, bin_s, dt_s=1e-3):
    n_per = max(1, int(round(bin_s / dt_s)))
    nb = len(y) // n_per
    if nb < 8:
        return None
    return y[:nb * n_per].reshape(nb, n_per).mean(axis=1)


def acf(x, h):
    r = x - x.mean()
    d = float(np.sum(r ** 2))
    if d <= 0:
        return np.zeros(h)
    return np.array([float(np.sum(r[k:] * r[:-k])) / d for k in range(1, h + 1)])


def build_envelope(all_noise):
    global ENV
    acs = []
    for seg in all_noise:
        bm = bin_means(seg, BIN_S)
        if bm is None:
            continue
        t = np.arange(len(bm))
        tr = np.polyfit(t, bm, 1)
        acs.append(acf(bm - np.polyval(tr, t), H_ACF))
    A = np.abs(np.array(acs))
    env = np.sort(A, axis=0)[-2] if len(acs) >= 2 else A[0]
    ENV = np.maximum(env, 0.2)
    return len(acs)


def white_by_envelope(resid):
    a = np.abs(acf(resid, H_ACF))
    viol = int(np.sum(a > ENV[:len(a)]))
    return viol <= VIOL_TOL, viol


def tail_binned(I, tl):
    s0 = tl["start"] + int(SKIP_MS / 1000.0 / DT)
    y = I[s0: s0 + tl["n"]].astype(float)
    b0, b1 = tl["base_idx"]
    base_seg = I[b0:b1:DS].astype(float)
    base = float(np.median(base_seg))
    xs0 = y - base
    t_full = np.arange(len(xs0)) * DT
    t, x0 = t_full[::DS], xs0[::DS]
    early = x0[:int(0.2 / (t[1] - t[0]))]
    sgn = 1.0 if np.median(early) >= 0 else -1.0
    x = sgn * x0
    dt_s = t[1] - t[0]
    i0 = int(T0_S / dt_s)
    k = max(3, int(15.0 / 1000.0 / dt_s) | 1)
    ys = np.convolve(x, np.ones(k) / k, mode="same")
    if ys[i0] < 0.08:
        return None
    below = ys[i0:] < MASK_NA
    i1 = len(t) - 1
    if below.sum() >= DEBOUNCE:
        run = np.convolve(below.astype(int), np.ones(DEBOUNCE, int), mode="valid")
        hit = np.where(run >= DEBOUNCE)[0]
        if len(hit):
            i1 = i0 + int(hit[0])
    if (i1 - i0) * dt_s < WIN_MIN_S:
        return None
    tt, xx = t[i0:i1], x[i0:i1]
    nbin = max(8, int((tt[-1] - tt[0]) / BIN_S))
    bt, by = [], []
    for b in np.array_split(np.arange(len(tt)), nbin):
        bt.append(float(np.median(tt[b])))
        by.append(float(np.mean(xx[b])))
    return np.array(bt), np.array(by), sgn


def fit_tail(bt, by, vv):
    ef = np.exp(-bt / TAU_F[vv])
    em = np.exp(-bt / TAU_M[vv])
    es = np.exp(-bt / TAU_L[vv]) - 1.0
    X = np.column_stack([np.ones(len(bt)), ef, em, es])
    sol, *_ = np.linalg.lstsq(X, by, rcond=None)
    res = by - X @ sol
    white, viol = white_by_envelope(res)
    return dict(sol=sol, white=bool(white), viol=viol,
                rms=float(np.sqrt(np.mean(res ** 2))))


# ================= Part B：sine 零拟合预测 =================

def load_sine(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/sine_wave_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/sine_wave_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def g_of_V(cell):
    """steady_activation -> g(V)=I_ss/(V-E_rev) 查表；结点 [-80:0, -60..+60, +60外持]"""
    Vp = sio.loadmat(f"{DATA}/data/protocols/steady_activation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/steady_activation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return None, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(Vp))
    Vp, I = Vp[:n], I[:n]
    edges = np.where(np.diff(Vp) != 0)[0] + 1
    knots, gvals = [-80.0], [0.0]
    for s in np.split(np.arange(len(Vp)), edges):
        vv = float(Vp[s[0]])
        dur = len(s) * DT
        if dur >= 4.0 and vv > -70.0:          # 测试台阶（≥4s，非 -80 保持段）
            m = s[int(0.8 * len(s)):]          # 末 20% 当稳态
            iss = float(np.mean(I[m]))
            knots.append(vv)
            gvals.append(iss / (vv - E_REV))
    order = np.argsort(knots)
    return np.array(knots)[order], np.array(gvals)[order]


def simulate_sine(Vs, knots, gvals):
    gV = np.interp(Vs, knots, gvals)
    drive = gV * (Vs - E_REV)                    # 稳态目标电流 g(V)(V-Erev)
    f_f = np.interp(Vs, F_V, F_F)
    f_m = np.interp(Vs, F_V, F_M)
    f_s = np.interp(Vs, F_V, F_S)
    a_f = DT / TF_C
    a_m = DT / TM_C
    a_s = np.clip(DT / tau_late(Vs), 1e-9, 1.0)
    n = len(Vs)
    Jf = np.zeros(n); Jm = np.zeros(n); Js = np.zeros(n)
    jf = f_f[0] * drive[0]; jm = f_m[0] * drive[0]; js = f_s[0] * drive[0]
    Jf[0], Jm[0], Js[0] = jf, jm, js
    tf_t = f_f * drive; tm_t = f_m * drive; ts_t = f_s * drive
    for k in range(1, n):
        jf += (tf_t[k] - jf) * a_f
        jm += (tm_t[k] - jm) * a_m
        js += (ts_t[k] - js) * a_s[k]
        Jf[k], Jm[k], Js[k] = jf, jm, js
    return Jf + Jm + Js, drive                  # drive = 零模型（瞬态稳态）


def main():
    print("=" * 78)
    print(" α 模型前向验证" + ("（冒烟：单细胞）" if SMOKE else "（全量九细胞）"))
    print(" Part A 尾巴重建 = 封卷 τ + 每细胞线性幅度（复现检验：白化应=28/33）")
    print(" Part B sine 零拟合预测（外推假设 A1-A4 见文件头）")
    print("=" * 78, flush=True)

    # ---------- 噪声包络门 ----------
    all_noise = []
    for cell in CELLS_ALL:                        # 包络门恒用全量九细胞（与封卷口径一致）
        V, I = load_deact(cell)
        if I is not None:
            all_noise.extend(noise_segments(V, I))
    nseg = build_envelope(all_noise)
    print(f"  噪声包络门就绪（{nseg} 段静默段，全量口径）", flush=True)

    # ---------- Part A ----------
    print("\n[Part A] 尾巴重建", flush=True)
    recs = []
    for cell in CELLS:
        V, I = load_deact(cell)
        if I is None:
            print(f"  细胞 {cell}: deactivation 文件缺失")
            continue
        for tl in find_tails(V):
            vv = int(round(tl["v"]))
            if vv not in GEARS:
                continue
            tb = tail_binned(I, tl)
            if tb is None:
                continue
            bt, by, sgn = tb
            r = fit_tail(bt, by, vv)
            recs.append(dict(cell=cell, v=vv, bt=bt, by=by, **r))
        print(f"  细胞 {cell}: 完成", flush=True)
    n_white = sum(r["white"] for r in recs)
    n_tot = len(recs)
    rms_med = float(np.median([r["rms"] for r in recs])) if recs else np.nan
    print(f"  复现白化 {n_white}/{n_tot}（封卷值 28/33；不一致=代码口径漂移，需停查）")
    print(f"  残差 RMS 中位 {rms_med*1000:.1f} pA")

    fig, axes = plt.subplots(len(CELLS), 4, figsize=(17, 2.1 * len(CELLS)), squeeze=False)
    for i, cell in enumerate(CELLS):
        for j, vv in enumerate(GEARS):
            ax = axes[i][j]
            r = next((r for r in recs if r["cell"] == cell and r["v"] == vv), None)
            if r is None:
                ax.axis("off")
                continue
            c, af, am, as_ = r["sol"]
            yrec = (c + af * np.exp(-r["bt"] / TAU_F[vv]) + am * np.exp(-r["bt"] / TAU_M[vv])
                    + as_ * (np.exp(-r["bt"] / TAU_L[vv]) - 1.0))
            ax.semilogy(r["bt"], r["by"], ".", ms=2.5, color="black", alpha=0.6, label="数据")
            ax.semilogy(r["bt"], yrec, "-", lw=1.4, color="crimson", label="α 模型")
            ax.set_title(f"{cell[-4:]} @ {vv}mV  RMS={r['rms']*1000:.0f}pA  违约{r['viol']}",
                         fontsize=8)
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Part A 尾巴重建：原始 vs α 模型（白化 {n_white}/{n_tot}）", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fpA = os.path.join(HERE, f"2026-09-13_α模型_前向验证_尾巴重建{'_冒烟' if SMOKE else ''}.png")
    fig.savefig(fpA, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---------- Part B ----------
    print("\n[Part B] sine 零拟合预测", flush=True)
    print(f"  τ_f={TF_C:.3f}s τ_m={TM_C:.3f}s（常数） τ_late(V)=阶梯对数线性外推封顶500s")
    srows = []
    traces = {}
    for cell in CELLS:
        Vs, Is = load_sine(cell)
        if Is is None:
            print(f"  细胞 {cell}: sine 文件缺失")
            continue
        knots, gvals = g_of_V(cell)
        if knots is None:
            print(f"  细胞 {cell}: steady_activation 缺失")
            continue
        Idyn, Inull = simulate_sine(Vs, knots, gvals)
        res_d = Is - Idyn
        res_n = Is - Inull
        neg = Vs < -40.0
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
        row = dict(cell=cell,
                   rms_dyn=rms(res_d), rms_null=rms(res_n),
                   rms_dyn_neg=rms(res_d[neg]), rms_null_neg=rms(res_n[neg]),
                   rms_dyn_pos=rms(res_d[~neg]), rms_null_pos=rms(res_n[~neg]),
                   frac_neg=float(neg.mean()))
        row["ratio"] = row["rms_dyn"] / max(row["rms_null"], 1e-12)
        srows.append(row)
        traces[cell] = (Vs, Is, Idyn, Inull, knots, gvals)
        print(f"  细胞 {cell}: RMS 动态 {row['rms_dyn']:.3f} / 零模型 {row['rms_null']:.3f} nA "
              f"(比值 {row['ratio']:.2f})  负压段 {row['rms_dyn_neg']:.3f}/{row['rms_null_neg']:.3f}  "
              f"正压段 {row['rms_dyn_pos']:.3f}/{row['rms_null_pos']:.3f}", flush=True)

    if srows:
        fig, axes = plt.subplots(4, 3, figsize=(19, 13))
        axp = axes[0, 0]
        Vs0 = traces[srows[0]["cell"]][0]
        axp.plot(np.arange(len(Vs0)) * DT, Vs0, lw=0.6, color="darkgreen")
        axp.set_title("sine 协议 V(t)", fontweight="bold")
        axp.set_ylabel("mV"); axp.grid(alpha=0.3)
        axg = axes[0, 1]
        for cell, (_, _, _, _, knots, gvals) in traces.items():
            axg.plot(knots, gvals, "o-", ms=3, lw=1, alpha=0.7, label=cell[-4:])
        axg.set_title("g(V)=I_ss/(V−E_rev) 每细胞实测", fontweight="bold")
        axg.set_xlabel("mV"); axg.set_ylabel("µS")
        axg.legend(fontsize=6, ncol=3); axg.grid(alpha=0.3)
        axs = axes[0, 2]
        axs.axis("off")
        med_ratio = float(np.median([r["ratio"] for r in srows]))
        lines = ["Part B 汇总", "",
                 f"RMS 动态/零模型 中位比值: {med_ratio:.2f}",
                 "（<1 动力学挣钱；≥1 动力学白搭）", "",
                 "外推假设 A1-A4 见文件头"]
        for k, L in enumerate(lines):
            axs.text(0.02, 0.92 - 0.11 * k, L, fontsize=10,
                     fontweight="bold" if k in (0, 2) else "normal")
        for idx, r in enumerate(srows):
            ax = axes[(idx + 3) // 3][(idx + 3) % 3]
            Vs, Is, Idyn, Inull, _, _ = traces[r["cell"]]
            t = np.arange(len(Vs)) * DT
            dn = 20
            ax.plot(t[::dn], Is[::dn], lw=0.5, color="black", alpha=0.75, label="数据")
            ax.plot(t[::dn], Idyn[::dn], lw=0.8, color="crimson", label="α 动态")
            ax.plot(t[::dn], Inull[::dn], lw=0.5, color="steelblue", alpha=0.6, ls="--", label="零模型")
            ax.set_title(f"{r['cell'][-4:]}  RMS {r['rms_dyn']:.2f}/{r['rms_null']:.2f} nA "
                         f"(比值 {r['ratio']:.2f})", fontsize=8)
            ax.grid(alpha=0.3)
            if idx == 0:
                ax.legend(fontsize=7)
        for k in range(len(srows) + 3, 12):
            axes[k // 3][k % 3].axis("off")
        fig.suptitle("Part B sine 协议：原始 vs α 动态 vs 零模型（零拟合预测）", fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fpB = os.path.join(HERE, f"2026-09-13_α模型_前向验证_sine{'_冒烟' if SMOKE else ''}.png")
        fig.savefig(fpB, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        fpB = None

    # ---------- 判词（预先钉死） ----------
    print("\n" + "=" * 78)
    print(" 总判词：")
    okA = (not SMOKE and n_white == 28 and n_tot == 33) or (SMOKE and n_tot > 0)
    print(f"  Part A: 白化复现 {n_white}/{n_tot} -> {'与封卷一致' if okA else '与封卷不符，先查代码口径'}")
    if srows:
        med_ratio = float(np.median([r["ratio"] for r in srows]))
        med_neg = float(np.median([r["rms_dyn_neg"] / max(r["rms_null_neg"], 1e-12) for r in srows]))
        med_pos = float(np.median([r["rms_dyn_pos"] / max(r["rms_null_pos"], 1e-12) for r in srows]))
        print(f"  Part B: RMS 比值(动态/零模型) 全程中位 {med_ratio:.2f}，"
              f"负压段 {med_neg:.2f}，正压段 {med_pos:.2f}")
        if med_ratio < 0.8:
            vB = "sine 全程预测成立：动力学显著优于瞬态稳态"
        elif med_neg < 0.8 <= med_pos:
            vB = "sine 分压成立：负压段动力学挣钱，正压段失败（A1 预期内：快激活/失活不在模型内）"
        elif med_ratio < 1.0:
            vB = "sine 弱成立：动力学略优于零模型，需逐细胞看残差定位"
        else:
            vB = "sine 不成立：动态模型不优于瞬态稳态查表，α 模型外推失败"
        print("  " + vB)
    print("=" * 78)

    out = dict(partA=dict(n_white=n_white, n_tot=n_tot, rms_med=rms_med,
                          expected="28/33（非冒烟）"),
               partB=dict(rows=srows,
                          assumptions=["A1 τ_late V>-40 外推冻结", "A2 f_i V>-40 趋势外推",
                                       "A3 g(V) -60/-40 未完全稳态", "A4 τ_f/τ_m 常数化"]),
               figA=fpA, figB=fpB)
    fj = os.path.join(HERE, f"2026-09-13_α模型_前向验证{'_冒烟' if SMOKE else ''}_结果.json")
    json.dump(out, open(fj, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n  图A落盘: {fpA}")
    if fpB:
        print(f"  图B落盘: {fpB}")
    print(f"  结果落盘: {fj}")


if __name__ == "__main__":
    main()
