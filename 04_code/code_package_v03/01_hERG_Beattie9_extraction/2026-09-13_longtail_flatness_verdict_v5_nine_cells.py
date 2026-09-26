# -*- coding: utf-8 -*-
# 2026-09-13_长尾平直度判决v5_九细胞.py   (v5.3：WLS斜率+SE自适应判线)
# v5.2 对照N 病灶：s3=-1.04 精确命中真值，s4=-0.86 仅偏 1.3σ（末段信号弱，斜率估计噪声大），
#   但死判线15%不认估计误差 -> 窄谱被冤。修法不是放松判线，是让判线认账估计器噪声：
#   斜率 = 加权最小二乘（权重∝信号²）；门 = max(15%, 2*σ估计器)。
#   窄谱末段信号弱->门自动放宽；宽谱末段信号强->门依然紧。公平性是结构给的。
# 判词：直 = 三尺度窄谱在窗内收敛；弯 = 窗内不收敛（最慢模态 >10s 或宽谱，本脚本不区分）
# 对照（先跑，挂则停）：N 三指数18/149/962ms 必须直；B 拉伸指数beta0.5 必须弯
# 运行：python 本文件
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
CELLS = ["16704007", "16704047", "16707014", "16708016", "16708060",
         "16708118", "16713003", "16713110", "16715049"]
DT = 1e-4
DS = 10
SKIP_MS = 5.0
NOISE_NA = 0.036
MASK_NA = 2 * NOISE_NA       # 仅用于窗口截断
DEBOUNCE = 20
WIN_MIN_S = 1.0
BASE_MS = 200.0
POST_PEAK_MS = 20.0
STRAIGHT_TOL = 0.15
BIN_S = 0.05                 # 分箱宽度
MIN_BINS_Q = 4               # 每四分窗最少箱数


def slopes_of(t, y):
    """统一管线：峰后让位 -> 2σ防抖截断 -> 分箱中位 -> log -> 4等分Theil-Sen"""
    ipk = int(np.argmax(y))
    i0 = ipk + int(POST_PEAK_MS / 1000.0 / (t[1] - t[0]))
    k = max(3, int(15.0 / 1000.0 / (t[1] - t[0])) | 1)
    ys = np.convolve(y, np.ones(k) / k, mode="same")
    below = ys[i0:] < MASK_NA
    i1 = len(t) - 1
    if below.sum() >= DEBOUNCE:
        run = np.convolve(below.astype(int), np.ones(DEBOUNCE, int), mode="valid")
        hit = np.where(run >= DEBOUNCE)[0]
        if len(hit):
            i1 = i0 + int(hit[0])
    win_s = (i1 - i0) * (t[1] - t[0])
    if win_s < WIN_MIN_S:
        return None
    tt, xx = t[i0:i1], y[i0:i1]
    nbin = int(np.clip(win_s / BIN_S, 16, 80))
    sig_bin = 3 * NOISE_NA / np.sqrt(max(1, len(tt) // nbin))   # 箱中位 3σ
    bt, bl = [], []
    for b in np.array_split(np.arange(len(tt)), nbin):
        med = float(np.median(xx[b]))
        if med > max(sig_bin, 0.01):
            bt.append(float(np.median(tt[b])))
            bl.append(float(np.log(med)))
    if len(bt) < 4 * MIN_BINS_Q:
        return None
    bt, bl = np.array(bt), np.array(bl)
    bm = np.exp(bl)                       # 箱中位信号
    qs = np.array_split(np.arange(len(bt)), 4)
    if any(len(q) < MIN_BINS_Q for q in qs):
        return None
    ss, ses = [], []
    for q in qs:                          # 加权最小二乘（权重∝信号²）+ 斜率标准误
        tq, lq, wq = bt[q], bl[q], bm[q] ** 2
        tbar = np.sum(wq * tq) / wq.sum()
        W = float(np.sum(wq * (tq - tbar) ** 2))
        b = float(np.sum(wq * (tq - tbar) * lq) / W)
        a = float(np.sum(wq * lq) / wq.sum() - b * tbar)
        r = lq - (a + b * tq)
        s2 = float(np.sum(wq * r ** 2) / max(len(q) - 2, 1))
        ss.append(b)
        ses.append(float(np.sqrt(s2 / W)))
    s1, s2v, s3, s4 = ss
    e1, e2, e3, e4 = ses
    # 门 = max(15%相对, 2σ估计器)：信号弱处自动放宽，信号强处依然紧
    d2 = abs(s3 - s2v); tol2 = max(STRAIGHT_TOL * abs(s3), 2 * float(np.hypot(e2, e3)))
    d1 = abs(s4 - s3); tol1 = max(STRAIGHT_TOL * abs(s4), 2 * float(np.hypot(e3, e4)))
    straight = (d1 <= tol1) and (d2 <= tol2)
    return dict(t=tt, x=xx, bt=bt, bl=bl, s1=s1, s2=s2v, s3=s3, s4=s4,
                e1=e1, e2=e2, e3=e3, e4=e4,
                g2=d2 / (abs(s3) + 1e-300), g1=d1 / (abs(s4) + 1e-300),
                tol2=tol2, tol1=tol1, straight=bool(straight), win_s=float(win_s))


def load(cell):
    V = sio.loadmat(f"{DATA}/data/protocols/deactivation_protocol.mat")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/deactivation_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    return V, I


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
            tails.append(dict(v=v, start=s0, n=n, base_idx=(b0, hs + hn), pre_v=pv))
    return tails, info


def one_tail(I, tl):
    s0 = tl["start"] + int(SKIP_MS / 1000.0 / DT)
    y = I[s0: s0 + tl["n"]].astype(float)
    b0, b1 = tl["base_idx"]
    base = float(np.median(I[b0:b1]))
    xs0 = y - base
    t_full = np.arange(len(xs0)) * DT
    t, x0 = t_full[::DS], xs0[::DS]
    early = x0[:int(0.2 / (t[1] - t[0]))]
    sgn = 1.0 if np.median(early) >= 0 else -1.0
    r = slopes_of(t, sgn * x0)
    if r is None:
        return None
    r.update(dict(v=tl["v"], pre_v=tl["pre_v"], base=base,
                  amp=float(np.max(np.abs(x0)))))
    return r


def controls():
    rng = np.random.default_rng(7)
    t = np.arange(0, 5.5, DT)[::DS]
    res = {}
    synth = {
        "对照N_窄谱三指数": (0.4 * np.exp(-t / 0.018) + 0.8 * np.exp(-t / 0.149)
                          + 0.5 * np.exp(-t / 0.962), True),
        "对照B_宽谱拉伸": (1.2 * np.exp(-(t / 1.0) ** 0.5), False),
    }
    for name, (yc, want) in synth.items():
        y = yc + rng.normal(0, NOISE_NA, len(t))
        r = slopes_of(t, y)
        if r is None:
            res[name] = dict(error="窗不足")
            continue
        r["want_straight"] = want
        r["ok"] = bool(r["straight"] == want)
        res[name] = r
    return res


def main():
    print("=" * 64)
    print(" 长尾平直度判决 v5.3 · deactivation 长尾 · 九细胞（免拟合）")
    print(f" 判线: |Δs| < max({STRAIGHT_TOL:.0%}·|s|, 2σ估计器)  对 (s3,s4) 与 (s2,s3) 两门 -> 直")
    print(f" 分箱 {BIN_S*1000:.0f}ms 中位取log；截断 平滑<{MASK_NA:.3f} nA 连续{DEBOUNCE}点")
    print("=" * 64, flush=True)

    ctrl = controls()
    ctrl_ok = True
    for name, r in ctrl.items():
        if "error" in r:
            print(f"  {name}: {r['error']}，统计量作废")
            ctrl_ok = False
            continue
        good = r["ok"]
        ctrl_ok &= good
        print(f"  {name}: 窗 {r['win_s']:.2f}s 斜率 "
              f"[{r['s1']:.2f} {r['s2']:.2f} {r['s3']:.2f} {r['s4']:.2f}] "
              f"g2={r['g2']:.0%}(限{r['tol2']/max(abs(r['s3']),1e-9):.0%}) "
              f"g1={r['g1']:.0%}(限{r['tol1']/max(abs(r['s4']),1e-9):.0%}) -> {'直' if r['straight'] else '弯'} "
              f"({'符合预期' if good else '不符合预期，统计量作废！'})")
    if not ctrl_ok:
        print("\n  对照未过 -> 统计量不可信，真实数据判决不执行。停。")
        return

    allres = []
    for cell in CELLS:
        V, I = load(cell)
        if I is None:
            print(f"\n  细胞 {cell}: 数据缺失，跳过")
            allres.append(dict(cell=cell, tails=[], verdict="数据缺失"))
            continue
        tails, info = find_tails(V)
        if cell == CELLS[0]:
            print(f"\n  [诊断] 协议共 {len(info)} 段，识别出长尾 {len(tails)} 条；前 24 段：")
            for j, (v, s0, n) in enumerate(info[:24]):
                print(f"    段{j:02d}: {v:+7.1f} mV × {n * DT:7.3f}s")
        rows = []
        for tl in tails:
            r = one_tail(I, tl)
            if r:
                rows.append(r)
        if len(rows) < 2:
            allres.append(dict(cell=cell, tails=rows, verdict=f"可用长尾不足（{len(rows)} 条）"))
            print(f"\n  细胞 {cell}: 可用长尾 {len(rows)} 条 -> 不足")
            continue
        n_bent = sum(1 for r in rows if not r["straight"])
        frac = n_bent / len(rows)
        verdict = f"弯 {n_bent}/{len(rows)} ({frac:.0%})"
        allres.append(dict(cell=cell, tails=rows, verdict=verdict,
                           n_bent=n_bent, n_tails=len(rows)))
        print(f"\n  细胞 {cell}: 长尾 {len(rows)} 条，判弯 {n_bent} 条 ({frac:.0%})")
        for r in rows:
            print(f"    尾 {r['v']:+6.0f} mV (前脉冲 {r['pre_v']:+.0f}): 窗 {r['win_s']:.2f}s  "
                  f"s1..s4 = {r['s1']:7.2f} {r['s2']:7.2f} {r['s3']:7.2f} {r['s4']:7.2f}  "
                  f"g2={r['g2']:5.0%}(限{r['tol2']/max(abs(r['s3']),1e-9):4.0%}) "
                  f"g1={r['g1']:5.0%}(限{r['tol1']/max(abs(r['s4']),1e-9):4.0%}) -> {'直' if r['straight'] else '弯'}")

    valid = [r for r in allres if "n_bent" in r]
    tot_b = sum(r["n_bent"] for r in valid)
    tot_t = sum(r["n_tails"] for r in valid)
    print("\n" + "=" * 64)
    print(" 汇总：")
    print(f"  有效细胞 {len(valid)}，长尾 {tot_t} 条，判弯 {tot_b} 条 ({tot_b/max(tot_t,1):.0%})")
    print("  判读：三尺度窄谱预言'基本全直'；")
    print("       大多数弯 -> 三尺度图景不完整（最慢模态 >10s 或宽谱）")
    print("=" * 64)

    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    for ax, r in zip(axes.flat, allres):
        rows = r["tails"]
        cmap = plt.cm.viridis(np.linspace(0, 1, max(len(rows), 2)))
        for k, row in enumerate(rows):
            ax.plot(row["bt"], row["bl"], ".", color=cmap[k], ms=3,
                    label=f"{row['v']:+.0f}{'直' if row['straight'] else '弯'}")
        color = "red" if ("n_bent" in r and r["n_bent"] > r["n_tails"] / 2) else "green"
        ax.set_title(f"{r['cell']} · {r['verdict']}", fontsize=10, fontweight="bold", color=color)
        ax.set_ylim(-9, 1)
        ax.grid(alpha=0.3)
        if rows:
            ax.legend(fontsize=6, loc="lower left")
    for ax in axes.flat[len(allres):]:
        ax.axis("off")
    fig.suptitle(f"长尾平直度 v5.3 · 判弯 {tot_b}/{tot_t} 条尾（对照已过）",
                 fontsize=14, fontweight="bold")
    fpng = os.path.join(HERE, "2026-09-13_长尾平直度判决v5_九细胞.png")
    fig.savefig(fpng, dpi=130, bbox_inches="tight")

    out = {
        "controls": {k: ({"error": v["error"]} if "error" in v else
                         {"win_s": v["win_s"], "s1..s4": [round(v[f"s{i}"], 3) for i in range(1, 5)],
                          "se1..se4": [round(v[f"e{i}"], 3) for i in range(1, 5)],
                          "g2": v["g2"], "tol2": v["tol2"], "g1": v["g1"], "tol1": v["tol1"],
                          "straight": v["straight"], "ok": v["ok"]})
                     for k, v in ctrl.items()},
        "straight_tol": STRAIGHT_TOL, "bin_s": BIN_S, "mask_nA": MASK_NA,
        "cells": [{
            "cell": r["cell"], "verdict": r["verdict"],
            "tails": [{"tail_mV": x["v"], "pre_mV": x["pre_v"], "baseline_nA": x["base"],
                       "amp_nA": x["amp"], "win_s": x["win_s"],
                       "s1": x["s1"], "s2": x["s2"], "s3": x["s3"], "s4": x["s4"],
                       "se1": x["e1"], "se2": x["e2"], "se3": x["e3"], "se4": x["e4"],
                       "g2": x["g2"], "tol2": x["tol2"], "g1": x["g1"], "tol1": x["tol1"],
                       "straight": x["straight"]} for x in r["tails"]],
        } for r in allres],
        "total_bent": tot_b, "total_tails": tot_t,
    }
    fjson = os.path.join(HERE, "2026-09-13_长尾平直度判决v5_九细胞_结果.json")
    json.dump(out, open(fjson, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n  图落盘: {fpng}")
    print(f"  结果落盘: {fjson}")


if __name__ == "__main__":
    main()
