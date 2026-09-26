# -*- coding: utf-8 -*-
# 2026-09-13_α模型_幅度表提取.py
# 目的：τ 表已封卷（阶梯表 2026-09-13），每细胞每电压只解幅度 -> 线性最小二乘。
#   模型：I(t) = c* + a_f·e^{-t/τ_f(V)} + a_m·e^{-t/τ_m(V)} + a_s·(e^{-t/τ_late(V)}-1)
#   （慢分量减 1 使其与常数解耦；y_ss = c* - a_s，y(0) = c* + a_f + a_m）
# 隐藏红利：固定 τ + 只解幅度仍能拟白 = τ 表的前向验证（τ 照抄、只重标幅度 实测成立）。
# τ 来源（v5 判决/鉴定封卷值）：
#   τ_f/τ_m：-70/-60 恒定档实测（不动）；
#            -50/-40 档做"共享 τ 网格精修"（每档一个 τ 横跨九细胞，幅度逐细胞线性解，
#            恒定性约束内置；冒烟暴露 -40 外推 τ_m=2.425s 错误致 w_m<0/w_s>1，故改精修）；
#   τ_late：阶梯表（-70:2.055, -60:4.418, -50:10.141, -40:24.054）。
# 判白：v5 噪声包络门（静默段实测 ACF）。对照：两条已知幅度合成尾，恢复误差>10% 作废。
# 说明：本脚本提取衰减分量幅度表 + 常数项 c（应≈0，基线质检）；
#       稳态表 y_ss(V) 需从 steady_activation 协议另提（下一步）。
# 运行：python 本文件
import os
import json
import numpy as np
import scipy.io as sio
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = ["16704007", "16704047", "16707014", "16708016", "16708060",
         "16708118", "16713003", "16713110", "16715049"]
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

TAU_F = {-70: 0.19105, -60: 0.27645, -50: 0.30892}     # -40 外推
TAU_M = {-70: 0.81270, -60: 1.20549, -50: 1.67193}     # -40 外推
TAU_L = {-70: 2.05505, -60: 4.41849, -50: 10.14073, -40: 24.0539}
GEARS = [-70, -60, -50, -40]


def load(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/deactivation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/deactivation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None, "文件缺失"
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    diag = f"lenI={len(I)} lenV={len(V)} NaN={int(np.isnan(I).sum())}"
    n = min(len(I), len(V))
    return V[:n], I[:n], diag


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
    sig_raw = 1.4826 * float(np.median(np.abs(base_seg - base)))
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
    n_per_bin = max(1, int(round(BIN_S / dt_s)))
    nbin = max(8, int((tt[-1] - tt[0]) / BIN_S))
    bt, by = [], []
    for b in np.array_split(np.arange(len(tt)), nbin):
        bt.append(float(np.median(tt[b])))
        by.append(float(np.mean(xx[b])))
    return np.array(bt), np.array(by), tl["v"], sig_raw / np.sqrt(n_per_bin)


def design(bt, tf, tm, tl_):
    ef = np.exp(-bt / tf)
    em = np.exp(-bt / tm)
    es = np.exp(-bt / tl_) - 1.0
    return np.column_stack([np.ones(len(bt)), ef, em, es])


def fit_amps(bt, by, tf, tm, tl_):
    X = design(bt, tf, tm, tl_)
    sol, *_ = np.linalg.lstsq(X, by, rcond=None)
    c_star, a_f, a_m, a_s = [float(v) for v in sol]
    res = by - X @ sol
    white, viol = white_by_envelope(res)
    cond = float(np.linalg.cond(X))
    return dict(c_star=c_star, a_f=a_f, a_m=a_m, a_s=a_s,
                y_ss=c_star - a_s, y0=c_star + a_f + a_m,
                white=bool(white), viol=viol, cond=cond,
                rms=float(np.sqrt(np.mean(res ** 2))))


def extrapolate_tau():
    vs = np.array(sorted(TAU_F), float)
    pf = np.polyfit(vs, np.log([TAU_F[int(v)] for v in vs]), 1)
    pm = np.polyfit(vs, np.log([TAU_M[int(v)] for v in vs]), 1)
    tf40 = float(np.exp(np.polyval(pf, -40)))
    tm40 = float(np.exp(np.polyval(pm, -40)))
    return tf40, tm40


def control(rng):
    """两条已知幅度合成尾：恢复误差 >10% 或 y_ss 偏差 >0.02nA -> 作废"""
    t = np.arange(0.05, 5.5, DT * DS)
    cases = [dict(tf=0.2, tm=0.8, tl=2.0, c=0.01, af=0.30, am=0.50, as_=0.40),
             dict(tf=0.3, tm=1.7, tl=10.0, c=-0.01, af=0.20, am=0.35, as_=0.60)]
    for i, cs in enumerate(cases):
        y = (cs["c"] + cs["af"] * np.exp(-t / cs["tf"]) + cs["am"] * np.exp(-t / cs["tm"])
             + cs["as_"] * np.exp(-t / cs["tl"]))
        yy = y + rng.normal(0, 0.036, len(t))
        nb = max(8, int((t[-1] - t[0]) / BIN_S))
        bt, by = [], []
        for b in np.array_split(np.arange(len(t)), nb):
            bt.append(float(np.median(t[b])))
            by.append(float(np.mean(yy[b])))
        r = fit_amps(np.array(bt), np.array(by), cs["tf"], cs["tm"], cs["tl"])
        errs = {k: abs(r[k] - cs[kk]) / cs[kk] for k, kk in (("a_f", "af"), ("a_m", "am"), ("a_s", "as_"))}
        yss_err = abs(r["y_ss"] - cs["c"])
        print(f"  对照{i+1}: a_f 误差 {errs['a_f']*100:.1f}%  a_m {errs['a_m']*100:.1f}%  "
              f"a_s {errs['a_s']*100:.1f}%  y_ss 偏差 {yss_err:.3f}nA  违约 {r['viol']}", flush=True)
        if max(errs.values()) > 0.10 or yss_err > 0.02:
            print("  对照未归位 -> 统计量作废，停。")
            return False
    return True


def main():
    rng = np.random.default_rng(11)
    print("=" * 76)
    print(" α 模型幅度表提取（τ 钉死表值，线性最小二乘解幅度）")
    print("=" * 76, flush=True)

    tf40, tm40 = extrapolate_tau()
    TF = dict(TAU_F); TM = dict(TAU_M)
    TF[-40] = tf40; TM[-40] = tm40
    print(f"  τ_f(−40) 外推 = {tf40:.3f}s，τ_m(−40) 外推 = {tm40:.3f}s（-70/-60/-50 对数线性，标记待验）")

    all_noise = []
    for cell in CELLS:
        V, I, _ = load(cell)
        if I is not None:
            all_noise.extend(noise_segments(V, I))
    nseg = build_envelope(all_noise)
    print(f"  噪声包络门就绪（{nseg} 段静默段）", flush=True)

    print("\n[对照]", flush=True)
    if not control(rng):
        return
    print("  对照过。", flush=True)

    # ---------- 真实数据：第一遍收集 ----------
    print("\n[真实数据]", flush=True)
    tails_data = []
    for cell in CELLS:
        V, I, diag = load(cell)
        if I is None:
            print(f"  细胞 {cell}: {diag}")
            continue
        cnt = 0
        for tl in find_tails(V):
            vv = int(round(tl["v"]))
            if vv not in GEARS:
                continue
            tb = tail_binned(I, tl)
            if tb is None:
                continue
            bt, by, _, sig_bin = tb
            tails_data.append(dict(cell=cell, v=vv, bt=bt, by=by))
            cnt += 1
        print(f"  细胞 {cell}: {cnt} 条可用尾  [{diag}]", flush=True)

    # ---------- 第二遍：-50/-40 档共享 τ 精修 ----------
    print("\n[τ 精修] -50/-40 共享 τ_f/τ_m 网格搜索（-70/-60 用封卷值不动）", flush=True)
    edge_flags = {}
    for vv in (-50, -40):
        ents = [e for e in tails_data if e["v"] == vv]
        if len(ents) < 3:
            print(f"  {vv} mV: 数据不足，保留原值")
            continue
        best = None
        for tf in np.linspace(0.15, 0.6, 10):
            for tm in np.linspace(0.8, 3.2, 13):
                s = 0.0
                for e in ents:
                    X = design(e["bt"], tf, tm, TAU_L[vv])
                    sol, *_ = np.linalg.lstsq(X, e["by"], rcond=None)
                    s += float(np.sum((e["by"] - X @ sol) ** 2))
                if best is None or s < best[0]:
                    best = (s, tf, tm)
        edge = best[1] in (0.15, 0.6) or best[2] in (0.8, 3.2)
        edge_flags[vv] = edge
        print(f"  {vv} mV: τ_f {TF[vv]:.3f}->{best[1]:.3f}s  τ_m {TM[vv]:.3f}->{best[2]:.3f}s"
              f"{'  （顶网格边：劈分不可识别，非测量值）' if edge else ''}", flush=True)
        TF[vv], TM[vv] = best[1], best[2]

    # ---------- 第三遍：最终 τ 解幅度 ----------
    rows = []
    n_white = n_tot = 0
    for e in tails_data:
        r = fit_amps(e["bt"], e["by"], TF[e["v"]], TM[e["v"]], TAU_L[e["v"]])
        n_tot += 1
        n_white += int(r["white"])
        a_sum = r["a_f"] + r["a_m"] + r["a_s"]
        w = tuple(r[k] / a_sum if abs(a_sum) > 1e-6 else np.nan for k in ("a_f", "a_m", "a_s"))
        rows.append(dict(cell=e["cell"], v=e["v"], **r, w_f=w[0], w_m=w[1], w_s=w[2]))
    print(f"\n  白化率 {n_white}/{n_tot}（v5 自由 τ 27/32 作参照；此为共享 τ，非逐细胞自由）")

    # ---------- 汇总表 ----------
    print("\n" + "-" * 76)
    print("[幅度表] 九细胞均值 ± SD（只用包络白的尾巴）；每档标可识别等级")
    print(f"  {'V':>5} {'n':>3} | {'a_f':>7} {'a_m':>7} {'a_s':>7} {'y_ss':>7} {'y0':>7} | {'w_f':>5} {'w_m':>5} {'w_s':>5} | 等级")
    table = []
    for vv in GEARS:
        rs = [r for r in rows if r["v"] == vv and r["white"]]
        if len(rs) < 3:
            print(f"  {vv:>5} {len(rs):>3} | 数据不足")
            continue
        def ms(key):
            a = np.array([r[key] for r in rs], float)
            return a.mean(), a.std(ddof=1)
        af = ms("a_f"); am = ms("a_m"); as_ = ms("a_s"); ys = ms("y_ss"); y0 = ms("y0")
        wf = ms("w_f"); wm = ms("w_m"); ws = ms("w_s")
        cond = float(np.median([r["cond"] for r in rs]))
        neg_amp = min(af[0], am[0], as_[0]) < -0.02
        if edge_flags.get(vv, False) or neg_amp:
            note = "仅y0+τ_late（快/中劈分不可识别）"
        elif abs(ys[0]) >= 0.10:
            note = "全分解（y_ss偏高，常数项参考）"
        else:
            note = "全分解"
        table.append(dict(v=vv, n=len(rs),
                          a_f=af, a_m=am, a_s=as_, y_ss=ys, y0=y0,
                          w_f=wf, w_m=wm, w_s=ws, cond=cond, note=note))
        print(f"  {vv:>5} {len(rs):>3} | {af[0]:>7.3f} {am[0]:>7.3f} {as_[0]:>7.3f} {ys[0]:>7.3f} {y0[0]:>7.3f} |"
              f" {wf[0]:>5.2f} {wm[0]:>5.2f} {ws[0]:>5.2f} | {note}")

    c70 = [r["y_ss"] for r in rows if r["white"] and r["v"] == -70]
    c70m = float(np.mean(np.abs(c70))) if c70 else np.nan
    w40 = [r["w_m"] for r in rows if r["v"] == -40 and r["white"]]
    w40m = float(np.mean(w40)) if w40 else np.nan

    # ---------- 判词（预先钉死） ----------
    frac = n_white / max(n_tot, 1)
    print("\n" + "=" * 76)
    print(" 总判词：")
    print(f"  白化 {n_white}/{n_tot}；-70 档 |y_ss| 均值 {c70m:.3f} nA（基线质检，应≈0）；"
          f"-40 档 w_m 均值 {w40m:.2f}（物理化应≥0）")
    if frac >= 0.75 and (not np.isnan(c70m)) and c70m < 0.05:
        nfull = sum(1 for t in table if t["note"].startswith("全分解"))
        nlim = sum(1 for t in table if t["note"].startswith("仅y0"))
        final = (f"τ 表前向验证过（共享 τ 白化 {n_white}/{n_tot}，不低于自由 τ 的 27/32）；"
                 f"幅度表按可识别等级交付：{nfull} 档全分解，{nlim} 档仅 y0+τ_late"
                 "（快/中劈分窗内不可识别是信息极限，非模型错误）")
    elif frac >= 0.60:
        final = "幅度表可用但白化低于预期：列出低白化档位复查共享 τ"
    elif not np.isnan(c70m) and c70m >= 0.05:
        final = "基线质检不过：-70 档常数项系统性偏零，先查基线/漂移再谈幅度表"
    else:
        final = "共享 τ 不可行：白化崩塌，回到逐细胞自由 τ（τ 表恒定性需重审）"
    print("  " + final)
    print("=" * 76)

    # ---------- 图 ----------
    fig, axes = plt.subplots(2, 3, figsize=(20, 9))
    ax = axes[0, 0]
    for cell in CELLS:
        rs = sorted([r for r in rows if r["cell"] == cell and r["white"]], key=lambda r: r["v"])
        if rs:
            ax.plot([r["v"] for r in rs], [r["y0"] for r in rs], "o-", ms=3, lw=1, alpha=0.75, label=cell[-4:])
    ax.set_title("初始幅度 y0(V) 每细胞", fontweight="bold")
    ax.set_xlabel("尾电压 mV"); ax.set_ylabel("nA")
    ax.legend(fontsize=6.5, ncol=3); ax.grid(alpha=0.3)

    for ax, key, ttl in ((axes[0, 1], "w_f", "快分量分数 w_f(V)"),
                         (axes[0, 2], "w_m", "中分量分数 w_m(V)")):
        for trow in table:
            ax.errorbar([trow["v"]], [trow[key][0]], yerr=trow[key][1], fmt="s", ms=7,
                        capsize=4, color="steelblue")
        ax.set_title(ttl + "（均值±SD）", fontweight="bold")
        ax.set_xlabel("尾电压 mV"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for trow in table:
        ax.errorbar([trow["v"]], [trow["w_s"][0]], yerr=trow["w_s"][1], fmt="s", ms=7,
                    capsize=4, color="darkred")
    ax.set_title("慢分量分数 w_s(V)（均值±SD）", fontweight="bold")
    ax.set_xlabel("尾电压 mV"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    shown = 0
    for cell in CELLS:
        rs = [r for r in rows if r["cell"] == cell and r["v"] == -50]
        if not rs or shown >= 3:
            continue
        V, I, _ = load(cell)
        tl = next(t for t in find_tails(V) if int(round(t["v"])) == -50)
        tb = tail_binned(I, tl)
        bt, by, _, _ = tb
        r = rs[0]
        yrec = (r["c_star"] + r["a_f"] * np.exp(-bt / TF[-50]) + r["a_m"] * np.exp(-bt / TM[-50])
                + r["a_s"] * (np.exp(-bt / TAU_L[-50]) - 1))
        ax.semilogy(bt, by, ".", ms=2, alpha=0.5, label=f"{cell[-4:]} 数据")
        ax.semilogy(bt, yrec, "-", lw=1.5, label=f"{cell[-4:]} 重建")
        shown += 1
    ax.set_title("重建示例（-50 mV，固定 τ + 线性幅度）", fontweight="bold")
    ax.set_xlabel("t (s)"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.axis("off")
    lines = ["α 模型幅度表提取", "",
             f"白化: {n_white}/{n_tot}   -70 |y_ss|均值: {c70m:.3f} nA",
             f"τ 精修后: τ_f={TF[-50]:.2f}/{TF[-40]:.2f} τ_m={TM[-50]:.2f}/{TM[-40]:.2f}s(-50/-40)", "", final]
    y0_ = 0.95
    for L in lines:
        ax.text(0.02, y0_, L, fontsize=10,
                fontweight="bold" if (L.startswith("α") or L == final) else "normal", wrap=True)
        y0_ -= 0.10
    fpng = os.path.join(HERE, "2026-09-13_α模型_幅度表提取.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")

    out = dict(tau_used=dict(TF=TF, TM=TM, TAU_L=TAU_L, refined_gears=[-50, -40]),
               n_white=n_white, n_tot=n_tot, y_ss70_abs_mean=c70m, w40_m_mean=w40m,
               table=table, rows=rows, final=final)
    fjson = os.path.join(HERE, "2026-09-13_α模型_幅度表提取_结果.json")
    json.dump(out, open(fjson, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n  图落盘: {fpng}")
    print(f"  结果落盘: {fjson}")


if __name__ == "__main__":
    main()
